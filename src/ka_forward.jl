# ka_forward.jl
#
# KernelAbstractions forward kernels for the Reactant + Enzyme path.
#
# The Zygote/CUDA path (flash.jl, similarity.jl, attention.jl) uses hand-written
# `@cuda` kernels with warp shuffles / dynamic shared memory and hand-written
# backward kernels. None of that raises into StableHLO, so Reactant cannot trace
# it. This file provides *portable, raisable* KA forward kernels: one work-item
# per output row, plain window loops, no shuffles / shared memory / sync. When a
# function built on these is compiled with Reactant, the kernel is raised into
# XLA and **Enzyme-MLIR derives the backward automatically** — we do not port the
# hand-written backward kernels to this path.
#
# These functions are always available (KernelAbstractions is a hard dep) and run
# on any KA backend (CPU included), so the forward can be unit-tested without
# Reactant. The Reactant/Enzyme glue that compiles + differentiates them lives in
# ext/CirculantAttentionReactantExt.jl.
#
# Scope: real-valued inputs (Tier 1). Complex inputs use the Zygote path.

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const

# ------------------------------------------------------------------
# portable channel reduction + similarity values
#
# Deliberately NOT the `_strided_reduce` of similarity.jl: that one has a
# `@static libllvm_version >= 17` linear-index NVPTX strength-reduction
# workaround and `@fastmath`, both of which are CUDA-PTX specific and may not
# raise cleanly. Here we use the plain multidim form with ordinary `exp`/`log`.
# x, y are (nrows, C, batch); Ci/Cj index the row dim, b the batch, loop over C.
# ------------------------------------------------------------------
@inline function _ka_reduce(f::F, init, x, y, Ci, Cj, b, C::Int32) where F
    acc = init
    m = Int32(1)
    @inbounds while m <= C
        acc = f(acc, x[Ci, m, b], y[Cj, m, b])
        m += Int32(1)
    end
    return acc
end

@inline function _ka_simval(::RealDotSimilarity, x, y, Ci, Cj, b, C::Int32)
    Ts = real(promote_type(eltype(x), eltype(y)))
    _ka_reduce((s, xv, yv) -> s + real(xv * conj(yv)), zero(Ts), x, y, Ci, Cj, b, C)
end

@inline function _ka_simval(::DistanceSimilarity, x, y, Ci, Cj, b, C::Int32)
    Ts = real(promote_type(eltype(x), eltype(y)))
    s = _ka_reduce((s, xv, yv) -> s - abs2(xv - yv), zero(Ts), x, y, Ci, Cj, b, C)
    return Ts(0.5) * s
end

@inline function _ka_simval(::DotSimilarity, x, y, Ci, Cj, b, C::Int32)
    Ts = promote_type(eltype(x), eltype(y))
    _ka_reduce((s, xv, yv) -> s + xv * conj(yv), zero(Ts), x, y, Ci, Cj, b, C)
end

@inline function _ka_simval(::PIDotSimilarity, x, y, Ci, Cj, b, C::Int32)
    Ts = promote_type(eltype(x), eltype(y))
    s = _ka_reduce((s, xv, yv) -> s + xv * conj(yv), zero(Ts), x, y, Ci, Cj, b, C)
    return abs(s)
end

@inline function _ka_simval(::PIDistanceSimilarity, x, y, Ci, Cj, b, C::Int32)
    Ts = promote_type(eltype(x), eltype(y)); R = real(Ts)
    a = _ka_reduce(
        (a, xv, yv) -> (a[1] + abs2(xv), a[2] + xv * conj(yv), a[3] + abs2(yv)),
        (zero(R), zero(Ts), zero(R)), x, y, Ci, Cj, b, C)
    return -R(0.5) * a[1] + abs(a[2]) - R(0.5) * a[3]
end

# ------------------------------------------------------------------
# flash forward KA kernel (raisable): one work-item per (row, batch).
#
# Mirrors `_flash_attention_row!` (flash.jl:97) but without the CH register
# chunking (a CUDA register-pressure trick, irrelevant to XLA) and with plain
# `exp`/`log`. Online softmax: pass 1 computes (m, l); pass 2 recomputes weights
# and accumulates against v, one output channel at a time.
# ------------------------------------------------------------------
@kernel function ka_flash_fwd_kernel!(
        y, lse, simfun, @Const(q), @Const(k), @Const(v),
        nrows::Int32, K::Int32, C::Int32, Cv::Int32, spatdims, W::Int32, scale)
    t = @index(Global, Linear)
    ti = Int32(t)
    r  = (ti - Int32(1)) % nrows + Int32(1)
    b  = (ti - Int32(1)) ÷ nrows + Int32(1)
    base = (r - Int32(1)) * K
    Ts = typeof(scale)

    # pass 1: online max m and normalizer l = Σ exp(s - m)
    m = typemin(Ts)
    l = zero(Ts)
    κ = Int32(1)
    while κ <= K
        i, _ = cartesian_circulant(base + κ, spatdims, W)
        s = scale * _ka_simval(simfun, q, k, r, i, b, C)
        mnew = max(m, s)
        l = l * exp(m - mnew) + exp(s - mnew)
        m = mnew
        κ += Int32(1)
    end
    linv = inv(l)
    @inbounds lse[r, b] = m + log(l)

    # pass 2: y[r,c,b] = Σ_i (exp(s - m)/l) v[i,c,b]
    Tacc = promote_type(Ts, eltype(v))
    c = Int32(1)
    while c <= Cv
        acc = zero(Tacc)
        κ = Int32(1)
        while κ <= K
            i, _ = cartesian_circulant(base + κ, spatdims, W)
            p = exp(scale * _ka_simval(simfun, q, k, r, i, b, C) - m) * linv
            @inbounds acc = muladd(oftype(acc, p), oftype(acc, v[i, c, b]), acc)
            κ += Int32(1)
        end
        @inbounds y[r, c, b] = acc
        c += Int32(1)
    end
end

# ------------------------------------------------------------------
# host wrappers — always callable (any KA backend); traced by Reactant.
# ------------------------------------------------------------------
"""
    _ka_flash_attention_fwd(simfun, q, k, v, W, scale=true) -> (y, lse)

Portable KernelAbstractions flash-attention forward. Same math as
`_circulant_flash_attention_fwd` (flash.jl) but with a single raisable KA kernel
(no warp/block dispatch), for the Reactant+Enzyme path. Real similarities only.
"""
function _ka_flash_attention_fwd(
        simfun::AbstractSimilarity,
        q::AbstractArray{Tq,N}, k::AbstractArray{Tk,N}, v::AbstractArray{Tv,N},
        W::Int, scale::Real=true) where {Tq, Tk, Tv, N}
    Ts = simval_dtype(simfun, Tq, Tk)
    Ts <: Real || throw(ArgumentError(
        "_ka_flash_attention needs a real-valued similarity; $(typeof(simfun)) on " *
        "($Tq, $Tk) gives $Ts."))
    @assert isodd(W) "window length W=$W must be odd"

    Tacc = promote_type(Ts, Tv)
    y = similar(v, Tacc)
    spatdims, nrows, K, maxidx = _flash_launch_dims(q, W)
    C  = Int32(size(q, N-1))
    Cv = Int32(size(v, N-1))
    lse = similar(q, Ts, (Int(nrows), size(q, N)))

    # kernels index spatial positions linearly: (nrows, channels, batch) views
    yr = reshape(y, :, size(y, N-1), size(y, N))
    qr = reshape(q, :, size(q, N-1), size(q, N))
    kr = reshape(k, :, size(k, N-1), size(k, N))
    vr = reshape(v, :, size(v, N-1), size(v, N))

    sc = Ts(scale)
    backend = KernelAbstractions.get_backend(y)
    ka_flash_fwd_kernel!(backend)(
        yr, lse, simfun, qr, kr, vr, nrows, K, C, Cv, spatdims, Int32(W), sc;
        ndrange = Int(maxidx))
    # Ordering barrier for the eager (non-Reactant) path; under Reactant tracing
    # this is a no-op the compiler handles.
    KernelAbstractions.synchronize(backend)
    return y, lse
end

_ka_flash_attention(simfun::AbstractSimilarity, q, k, v, W::Int, scale::Real=true) =
    first(_ka_flash_attention_fwd(simfun, q, k, v, W, scale))

# scalar loss the Reactant+Enzyme grad differentiates (matches the Zygote losses
# in benchmark.jl: `sum(abs2, circulant_flash_attention(...))`). Lives here (no
# Reactant dep) so the extension only supplies the autodiff call.
_ka_flash_loss(simfun, q, k, v, W, scale) =
    sum(abs2, _ka_flash_attention(simfun, q, k, v, W, scale))

# ------------------------------------------------------------------
# Reactant+Enzyme entry points — methods added by the extension when Reactant
# and Enzyme are loaded. Declared here so callers (benchmark/tests) can reference
# `CircAtt.reactant_flash_grad` unconditionally; calling without the extension
# loaded is a MethodError, which the benchmark guards against (HAS_REACTANT).
# ------------------------------------------------------------------
function reactant_flash_grad end
