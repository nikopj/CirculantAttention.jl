# flash.jl
#
# Fused ("flash") circulant attention, forward and backward.
#
# The composed pipeline (circulant_similarity → softmax → A ⊠ V) materializes
# the circulant-sparse attention matrix A (N·W^d values per batch) three times
# over in global memory. The kernels below never form A.
#
# Forward — each thread owns one (output row, batch) pair and streams its
# window twice:
#   pass 1: online softmax statistics — running max m and normalizer
#           l = Σ exp(s - m), rescaled as the max updates (Milakov & Gimelshein,
#           arXiv:1805.02867). The logsumexp L = m + log l is saved per row
#           for the backward pass.
#   pass 2: softmax weights P = exp(s - L) recomputed on the fly and
#           accumulated against v in register chunks of CH channels, so each
#           output element is written exactly once and no per-thread window
#           buffer is needed.
#
# Backward — FlashAttention-style (arXiv:2205.14135): attention weights are
# recomputed from q, k and the saved logsumexp instead of being stored. With
#   P[r,i] = exp(s_ri - L_r),  y[r,:] = Σ_i P[r,i] v[i,:],
#   ds[r,i] = P[r,i]·(g[r,i] - δ_r),  g = Re⟨Δ[r,:], v[i,:]⟩,  δ_r = Re⟨Δ[r,:], y[r,:]⟩,
# each thread owns one position a and performs two window sweeps:
#   sweep 1 (a as row):    ∂q[a,:] += ds·(α·k[i,:] + β·q[a,:])
#   sweep 2 (a as column — by symmetry of the circulant pattern, the rows
#            whose window contains column a are the columns of row a):
#                          ∂k[a,:] += ds·(ᾱ·q[r,:] + β·k[a,:])
#                          ∂v[a,:] += P·Δ[r,:]
# (α, β) are per-similarity factors (see simgrad_aux). All writes go to
# position a, so no atomics are needed and every output is written once.
#
# Memory traffic is O(N·C) + O(N) for the logsumexp, instead of O(N·W^d).
#
# Three kernel families implement this scheme (selected by _flash_mode):
#   - warp-cooperative (K ≤ 32·_FLASH_WARP_MAX_NE): a sub-warp owns each row,
#     window entries are lane-split with similarities held in registers, and
#     lanes combine via shfl_xor_sync butterflies — one similarity sweep.
#   - block-per-row (larger K, while the window fits in shared memory): the
#     per-entry state is staged in dynamic shared memory by a whole block —
#     one similarity sweep and full occupancy at any window size.
#   - thread-per-row (last resort, and the CPU-testable reference): softmax
#     weights don't fit in one thread's registers, so the similarity sweep is
#     recomputed once per CH-channel chunk; one thread per row also means low
#     occupancy for typical N·B.
#
# Indexing: the host reshapes all tensors to (nrows, channels, batch), so the
# kernels index spatial positions linearly (identical addressing to the
# CartesianIndex form, column-major).
#
# Index convention (matches circulant_similarity! and rrules.jl):
# window entry κ of row r sits at nnz index n = (r-1)K + κ with column
# i = first(cartesian_circulant(n, spatdims, W)), and S[r,i] = simval(q_r, k_i).

# Wrapper similarities renormalize over the full window (sorting/thresholding),
# which cannot be streamed entry-by-entry.
const _UnfusableSimilarity = Union{TopKSimilarity, SparsemaxSimilarity, EntmaxSimilarity}

# fast transcendentals for the device code: on GPU these lower to the fast
# PTX paths (__nv_fast_expf / ex2.approx — which maps -Inf to 0, so the
# softmax sentinels stay correct); on CPU Base.FastMath falls back to the
# precise versions, keeping the CPU validation harness bit-identical.
# NB: do NOT use `@cuda fastmath=true` on these kernels — GPUCompiler sets
# ninf/nnan on every FP instruction, which makes the -Inf sentinels UB.
@inline _fastexp(x) = @fastmath exp(x)
@inline _fastlog(x) = @fastmath log(x)

# ------------------------------------------------------------------
# device-side chunk accumulators
#
# The ntuple closures must live in their own functions with everything passed
# as arguments: capturing a variable that is reassigned in the enclosing loop
# (acc, c0) makes Julia box it, which is uncompilable on GPU.
# ------------------------------------------------------------------

# acc[t] += w * src[idx, c0+t, b] for channels c0+t <= Cmax
@inline function _chunk_muladd(acc::NTuple{CH}, w, src, idx, b::Int32, c0::Int32, Cmax::Int32) where CH
    ntuple(Val(CH)) do t
        c = c0 + Int32(t)
        c <= Cmax ? muladd(w, @inbounds(src[idx, c, b]), acc[t]) : acc[t]
    end
end

# acc[t] += ds * (α*x[Cx, c0+t, b] + β*y[Cy, c0+t, b]) for channels c0+t <= Cmax
@inline function _chunk_simgrad(acc::NTuple{CH}, ds, α, β, x, Cx, y, Cy, b::Int32, c0::Int32, Cmax::Int32) where CH
    ntuple(Val(CH)) do t
        c = c0 + Int32(t)
        c <= Cmax ? acc[t] + ds * (α * @inbounds(x[Cx, c, b]) + β * @inbounds(y[Cy, c, b])) : acc[t]
    end
end

# ------------------------------------------------------------------
# forward device code
#
# Generic over AbstractArray so the same code path is testable on CPU.
# ------------------------------------------------------------------
@inline function _flash_attention_row!(
        y, lse, simfun::AbstractSimilarity, q, k, v,
        r::Int32, b::Int32,
        K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, ::Val{CH},
    ) where CH
    Ts   = simval_dtype(simfun, eltype(q), eltype(k))
    Tacc = promote_type(Ts, eltype(v))
    base = (r - 1i32) * K

    # pass 1: online max and normalizer
    m = typemin(Ts)
    l = zero(Ts)
    for κ in 1i32:K
        i, _ = cartesian_circulant(base + κ, spatdims, W)
        s = scale * simval(simfun, q, k, r, i, b, C)
        mnew = max(m, s)
        l = l * _fastexp(m - mnew) + _fastexp(s - mnew)
        m = mnew
    end
    linv = inv(l)
    @inbounds lse[r, b] = m + _fastlog(l)

    # pass 2: accumulate softmax-weighted v, CH channels at a time
    zacc = zero(Tacc)
    c0 = 0i32
    while c0 < Cv
        acc = ntuple(_ -> zacc, Val(CH))
        for κ in 1i32:K
            i, _ = cartesian_circulant(base + κ, spatdims, W)
            p = _fastexp(scale * simval(simfun, q, k, r, i, b, C) - m) * linv
            acc = _chunk_muladd(acc, p, v, i, b, c0, Cv)
        end
        for t in 1:CH
            c = c0 + Int32(t)
            if c <= Cv
                @inbounds y[r, c, b] = acc[t]
            end
        end
        c0 += Int32(CH)
    end
    return nothing
end

function circulant_flash_attention_kernel!(
        y, lse, simfun::AbstractSimilarity, q, k, v,
        nrows::Int32, K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, maxidx::Int32,
        chunk::Val{CH},
    ) where CH
    tid    = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    while tid <= maxidx
        r = (tid - 1i32) % nrows + 1i32
        b = (tid - 1i32) ÷ nrows + 1i32
        _flash_attention_row!(y, lse, simfun, q, k, v, r, b, K, C, Cv, spatdims, W, scale, chunk)
        tid += stride
    end
    return nothing
end

# ------------------------------------------------------------------
# backward device code
#
# simgrad_aux returns (s, α, β) such that the similarity gradient at entry
# (row r, col i) with cotangent ds is
#   ∂q[r,c] += ds·(α·k[i,c] + β·q[r,c])
#   ∂k[i,c] += ds·(conj(α)·q[r,c] + β·k[i,c])
# These match the circulant_similarity rrules in rrules.jl entrywise.
# ------------------------------------------------------------------
# β is constant per similarity: 0 for dot-types, -1 for distance-types
# (the -q/-k identity term of the squared-distance gradient).
@inline _simgrad_beta(::Union{DotSimilarity, RealDotSimilarity, PIDotSimilarity}, ::Type{Ts}) where Ts = zero(real(Ts))
@inline _simgrad_beta(::Union{DistanceSimilarity, PIDistanceSimilarity}, ::Type{Ts}) where Ts = -one(real(Ts))

@inline function simgrad_aux(sf::Union{DotSimilarity, RealDotSimilarity}, q, k, r, i, b, C::Int32)
    s = simval(RealDotSimilarity(), q, k, r, i, b, C)
    return s, one(s), _simgrad_beta(sf, typeof(s))
end

@inline function simgrad_aux(sf::DistanceSimilarity, q, k, r, i, b, C::Int32)
    s = simval(DistanceSimilarity(), q, k, r, i, b, C)
    return s, one(s), _simgrad_beta(sf, typeof(s))
end

@inline function simgrad_aux(sf::PIDotSimilarity, q, k, r, i, b, C::Int32)
    z = simval(DotSimilarity(), q, k, r, i, b, C)
    s = abs(z)
    return s, sign(z), _simgrad_beta(sf, typeof(s))
end

@inline function simgrad_aux(sf::PIDistanceSimilarity, q, k, r, i, b, C::Int32)
    Ts = promote_type(eltype(q), eltype(k)); R = real(Ts)
    a = _strided_reduce(
        (a, qm, km) -> (a[1] + abs2(qm), a[2] + qm * conj(km), a[3] + abs2(km)),
        (zero(R), zero(Ts), zero(R)), q, k, r, i, b, C)
    s = -R(0.5) * (a[1] + a[3]) + abs(a[2])
    return s, sign(a[2]), _simgrad_beta(sf, R)
end

# g[r,i] = Re⟨Δ[r,:], v[i,:]⟩ — the ∂A entry of the A ⊠ V pullback.
@inline function _flash_gval(Δ, v, r, i, b, Cv::Int32)
    Tg = real(promote_type(eltype(Δ), eltype(v)))
    _strided_reduce((g, Δv, vv) -> g + real(Δv * conj(vv)), zero(Tg), Δ, v, r, i, b, Cv)
end

@inline function _flash_attention_bwd_row!(
        dq, dk, dv, simfun::AbstractSimilarity, q, k, v, Δ, lse, δ,
        a::Int32, b::Int32,
        K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, ::Val{CH},
    ) where CH
    Tqk  = promote_type(eltype(q), eltype(k))
    TΔ   = eltype(Δ)
    base = (a - 1i32) * K
    lse_a = @inbounds lse[a, b]
    δ_a   = @inbounds δ[a, b]

    # sweep 1: a as row — entries (row a, col i) accumulate ∂q[a,:]
    zqk = zero(Tqk)
    zΔ  = zero(TΔ)
    c0 = 0i32
    while c0 < C
        acc = ntuple(_ -> zqk, Val(CH))
        for κ in 1i32:K
            i, _ = cartesian_circulant(base + κ, spatdims, W)
            s, α, β = simgrad_aux(simfun, q, k, a, i, b, C)
            P  = _fastexp(scale * s - lse_a)
            ds = P * (_flash_gval(Δ, v, a, i, b, Cv) - δ_a)
            acc = _chunk_simgrad(acc, scale * ds, α, β, k, i, q, a, b, c0, C)
        end
        for t in 1:CH
            c = c0 + Int32(t)
            c <= C && (@inbounds dq[a, c, b] = acc[t])
        end
        c0 += Int32(CH)
    end

    # sweep 2: a as column — by pattern symmetry, the rows whose window
    # contains column a are exactly the columns of row a. Entries (row r,
    # col a) accumulate ∂k[a,:] and ∂v[a,:].
    Cmax = max(C, Cv)
    c0 = 0i32
    while c0 < Cmax
        acck = ntuple(_ -> zqk, Val(CH))
        accv = ntuple(_ -> zΔ, Val(CH))
        for κ in 1i32:K
            r, _ = cartesian_circulant(base + κ, spatdims, W)
            s, α, β = simgrad_aux(simfun, q, k, r, a, b, C)
            P  = _fastexp(scale * s - @inbounds(lse[r, b]))
            ds = P * (_flash_gval(Δ, v, r, a, b, Cv) - @inbounds(δ[r, b]))
            acck = _chunk_simgrad(acck, scale * ds, conj(α), β, q, r, k, a, b, c0, C)
            accv = _chunk_muladd(accv, P, Δ, r, b, c0, Cv)
        end
        for t in 1:CH
            c = c0 + Int32(t)
            c <= C  && (@inbounds dk[a, c, b] = acck[t])
            c <= Cv && (@inbounds dv[a, c, b] = accv[t])
        end
        c0 += Int32(CH)
    end
    return nothing
end

function circulant_flash_attention_bwd_kernel!(
        dq, dk, dv, simfun::AbstractSimilarity, q, k, v, Δ, lse, δ,
        nrows::Int32, K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, maxidx::Int32,
        chunk::Val{CH},
    ) where CH
    tid    = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    while tid <= maxidx
        a = (tid - 1i32) % nrows + 1i32
        b = (tid - 1i32) ÷ nrows + 1i32
        _flash_attention_bwd_row!(dq, dk, dv, simfun, q, k, v, Δ, lse, δ, a, b, K, C, Cv, spatdims, W, scale, chunk)
        tid += stride
    end
    return nothing
end

# ------------------------------------------------------------------
# warp-cooperative kernels
#
# One sub-warp of WS lanes (WS = min(32, nextpow(2, K)), so small 1D windows
# don't idle 32 lanes) owns one (row, batch) pair. The K window entries are
# split across lanes (entry e of lane λ is κ = λ + (e-1)·WS, NE = ⌈K/WS⌉ per
# lane) and their similarities are computed ONCE and kept in registers — the
# register file is the only storage that scales with K without spilling, and
# this removes the ⌈C/CH⌉ similarity-recompute factor of the thread kernels.
# Softmax statistics and per-channel outputs are combined with shfl_xor_sync
# butterflies (CUDA.jl shuffles handle Complex via shfl_recurse).
#
# Memory access: at each channel step the lanes of a sub-warp touch
# consecutive window columns (contiguous mod wrap) — coalesced — and q[r,c,b]
# is a warp-uniform load. Each sub-warp uses its own thread mask, so tail
# groups that exit the grid-stride loop early never participate in a shuffle
# they aren't named in.
#
# Entries with κ > K use a clamped column index (valid memory) and a zeroed
# weight, so all lanes stay converged through every shuffle.
# ------------------------------------------------------------------

# butterfly reduction: every lane of the WS-wide segment ends with the result
@inline function _warp_reduce(op::F, val, mask::UInt32, ::Val{WS}) where {F, WS}
    δ = Int32(WS) >> 1i32
    while δ > 0i32
        val = op(val, shfl_xor_sync(mask, val, δ, Int32(WS)))
        δ >>= 1i32
    end
    return val
end

@inline function _flash_warp_fwd_group!(
        y, lse, simfun::AbstractSimilarity, q, k, v,
        r::Int32, b::Int32, lane::Int32, submask::UInt32,
        K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, ::Val{WS}, ::Val{NE},
    ) where {WS, NE}
    Ts   = simval_dtype(simfun, eltype(q), eltype(k))
    Tacc = promote_type(Ts, eltype(v))
    base = (r - 1i32) * K

    # this lane's window columns (clamped) and similarities, in registers.
    # neginf is hoisted as a value: calling typemin on a closure-captured type
    # variable widens to DataType under GPUCompiler (dynamic invocation).
    neginf = typemin(Ts)
    cols = ntuple(Val(NE)) do e
        κ = lane + Int32(e-1) * Int32(WS)
        κ <= K ? first(cartesian_circulant(base + κ, spatdims, W)) : 1i32
    end
    svals = ntuple(Val(NE)) do e
        κ = lane + Int32(e-1) * Int32(WS)
        κ <= K ? scale * simval(simfun, q, k, r, cols[e], b, C) : neginf
    end

    # sub-warp softmax statistics
    m_lane = typemin(Ts)
    for e in 1:NE
        m_lane = max(m_lane, svals[e])
    end
    m = _warp_reduce(max, m_lane, submask, Val(WS))
    l_lane = zero(Ts)
    for e in 1:NE
        l_lane += _fastexp(svals[e] - m)
    end
    l = _warp_reduce(+, l_lane, submask, Val(WS))
    linv = inv(l)
    if lane == 1i32
        @inbounds lse[r, b] = m + _fastlog(l)
    end

    wvals = ntuple(e -> _fastexp(svals[e] - m) * linv, Val(NE))

    # per channel: lane partial, butterfly reduce, lane 1 writes
    c = 1i32
    while c <= Cv
        p = zero(Tacc)
        for e in 1:NE
            p = muladd(wvals[e], @inbounds(v[cols[e], c, b]), p)
        end
        p = _warp_reduce(+, p, submask, Val(WS))
        if lane == 1i32
            @inbounds y[r, c, b] = p
        end
        c += 1i32
    end
    return nothing
end

function circulant_flash_attention_warp_kernel!(
        y, lse, simfun::AbstractSimilarity, q, k, v,
        nrows::Int32, K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, ngroups::Int32,
        ws::Val{WS}, ne::Val{NE},
    ) where {WS, NE}
    tid     = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    lane    = (tid - 1i32) % Int32(WS) + 1i32
    g       = (tid - 1i32) ÷ Int32(WS) + 1i32
    gstride = (gridDim().x * blockDim().x) ÷ Int32(WS)
    # blockDim is a multiple of 32, so WS-wide segments never straddle a warp
    hwlane  = (threadIdx().x - 1i32) % 32i32
    submask = WS == 32 ? 0xffffffff : ((UInt32(1) << WS) - UInt32(1)) << ((hwlane ÷ Int32(WS)) * Int32(WS))

    while g <= ngroups
        r = (g - 1i32) % nrows + 1i32
        b = (g - 1i32) ÷ nrows + 1i32
        _flash_warp_fwd_group!(y, lse, simfun, q, k, v, r, b, lane, submask, K, C, Cv, spatdims, W, scale, ws, ne)
        g += gstride
    end
    return nothing
end

@inline function _flash_warp_bwd_group!(
        dq, dk, dv, simfun::AbstractSimilarity, q, k, v, Δ, lse, δ,
        a::Int32, b::Int32, lane::Int32, submask::UInt32,
        K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, ::Val{WS}, ::Val{NE},
    ) where {WS, NE}
    Ts  = simval_dtype(simfun, eltype(q), eltype(k))
    Tqk = promote_type(eltype(q), eltype(k))
    TΔ  = eltype(Δ)
    β   = _simgrad_beta(simfun, Tqk)
    base = (a - 1i32) * K
    lse_a = @inbounds lse[a, b]
    δ_a   = @inbounds δ[a, b]

    # window columns of row a — by pattern symmetry also the rows whose
    # window contains column a
    cols = ntuple(Val(NE)) do e
        κ = lane + Int32(e-1) * Int32(WS)
        κ <= K ? first(cartesian_circulant(base + κ, spatdims, W)) : 1i32
    end

    # ---- sweep 1: a as row — per-entry (ds·α, ds), then ∂q[a,:] ----------
    # ∂q[a,c] = Σ_e ds_e α_e k[i_e,c]  +  β q[a,c] Σ_e ds_e
    sw1 = ntuple(Val(NE)) do e
        κ = lane + Int32(e-1) * Int32(WS)
        s, α, _ = simgrad_aux(simfun, q, k, a, cols[e], b, C)
        P  = _fastexp(scale * s - lse_a)
        sds = scale * (P * (_flash_gval(Δ, v, a, cols[e], b, Cv) - δ_a))
        valid = κ <= K
        (valid ? sds * α : zero(sds * α), valid ? sds : zero(sds))
    end
    ds1_lane = zero(real(Ts))
    for e in 1:NE
        ds1_lane += sw1[e][2]
    end
    ds1 = _warp_reduce(+, ds1_lane, submask, Val(WS))

    c = 1i32
    while c <= C
        p = zero(Tqk)
        for e in 1:NE
            p += sw1[e][1] * @inbounds(k[cols[e], c, b])
        end
        p = _warp_reduce(+, p, submask, Val(WS))
        if lane == 1i32
            @inbounds dq[a, c, b] = p + β * ds1 * @inbounds(q[a, c, b])
        end
        c += 1i32
    end

    # ---- sweep 2: a as column — per-entry (ds·conj(α), P, ds) -------------
    # ∂k[a,c] = Σ_e ds_e conj(α_e) q[r_e,c] + β k[a,c] Σ_e ds_e
    # ∂v[a,c] = Σ_e P_e Δ[r_e,c]
    sw2 = ntuple(Val(NE)) do e
        κ = lane + Int32(e-1) * Int32(WS)
        s, α, _ = simgrad_aux(simfun, q, k, cols[e], a, b, C)
        P  = _fastexp(scale * s - @inbounds(lse[cols[e], b]))
        sds = scale * (P * (_flash_gval(Δ, v, cols[e], a, b, Cv) - @inbounds(δ[cols[e], b])))
        valid = κ <= K
        (valid ? sds * conj(α) : zero(sds * α), valid ? P : zero(P), valid ? sds : zero(sds))
    end
    ds2_lane = zero(real(Ts))
    for e in 1:NE
        ds2_lane += sw2[e][3]
    end
    ds2 = _warp_reduce(+, ds2_lane, submask, Val(WS))

    Cmax = max(C, Cv)
    c = 1i32
    while c <= Cmax
        pk = zero(Tqk)
        pv = zero(TΔ)
        for e in 1:NE
            if c <= C
                pk += sw2[e][1] * @inbounds(q[cols[e], c, b])
            end
            if c <= Cv
                pv = muladd(sw2[e][2], @inbounds(Δ[cols[e], c, b]), pv)
            end
        end
        pk = _warp_reduce(+, pk, submask, Val(WS))
        pv = _warp_reduce(+, pv, submask, Val(WS))
        if lane == 1i32
            c <= C  && (@inbounds dk[a, c, b] = pk + β * ds2 * @inbounds(k[a, c, b]))
            c <= Cv && (@inbounds dv[a, c, b] = pv)
        end
        c += 1i32
    end
    return nothing
end

function circulant_flash_attention_bwd_warp_kernel!(
        dq, dk, dv, simfun::AbstractSimilarity, q, k, v, Δ, lse, δ,
        nrows::Int32, K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, ngroups::Int32,
        ws::Val{WS}, ne::Val{NE},
    ) where {WS, NE}
    tid     = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    lane    = (tid - 1i32) % Int32(WS) + 1i32
    g       = (tid - 1i32) ÷ Int32(WS) + 1i32
    gstride = (gridDim().x * blockDim().x) ÷ Int32(WS)
    hwlane  = (threadIdx().x - 1i32) % 32i32
    submask = WS == 32 ? 0xffffffff : ((UInt32(1) << WS) - UInt32(1)) << ((hwlane ÷ Int32(WS)) * Int32(WS))

    while g <= ngroups
        a = (g - 1i32) % nrows + 1i32
        b = (g - 1i32) ÷ nrows + 1i32
        _flash_warp_bwd_group!(dq, dk, dv, simfun, q, k, v, Δ, lse, δ, a, b, lane, submask, K, C, Cv, spatdims, W, scale, ws, ne)
        g += gstride
    end
    return nothing
end

# ------------------------------------------------------------------
# block-per-row shared-memory kernels
#
# For windows too large for the warp kernels' register budget
# (K > 32·_FLASH_WARP_MAX_NE), a whole block owns each (row, batch) pair and
# the per-entry state lives in dynamic shared memory instead of registers:
#
#   phase 1: threads cooperatively compute the K similarities (entry κ of
#            thread t is κ = t, t+TB, …), stage weights and column indices in
#            shared memory, and combine softmax statistics with a two-level
#            block reduction (warp butterflies + a 32-slot scratch array).
#   phase 2: each warp takes channels c = wid, wid+nwarps, …; its lanes split
#            the window reading weights from shared (conflict-free: lane λ
#            reads κ = λ, λ+32, …) with v coalesced over κ, and combine with a
#            warp butterfly.
#
# The similarity sweep runs exactly once (like the warp kernels) and the
# launch puts TB threads on every row, so occupancy stays high for any K.
# The backward stages ds·α / ds·conj(α) / P the same way, two sweeps as usual.
# ------------------------------------------------------------------

# eltype of the staged backward weights ds·α (host needs it for shmem sizing)
@inline _flash_alpha_type(::Union{DotSimilarity, RealDotSimilarity, DistanceSimilarity}, ::Type{Tqk}) where Tqk = real(Tqk)
@inline _flash_alpha_type(::Union{PIDotSimilarity, PIDistanceSimilarity}, ::Type{Tqk}) where Tqk = Tqk

@inline function _flash_bwd_wtypes(simfun::AbstractSimilarity, ::Type{Tq}, ::Type{Tk}, ::Type{TΔ}, ::Type{Tv}) where {Tq, Tk, TΔ, Tv}
    Ts  = simval_dtype(simfun, Tq, Tk)
    Tds = promote_type(Ts, real(promote_type(TΔ, Tv)))
    Tw  = promote_type(Tds, _flash_alpha_type(simfun, promote_type(Tq, Tk)))
    return Ts, Tds, Tw
end

# two-level block reduction; every thread returns the result. Uniform across
# the block (contains sync_threads); scratch is a 32-slot shared array.
@inline function _block_reduce(op::F, val::T, neutral::T, scratch) where {F, T}
    lane = (threadIdx().x - 1i32) % 32i32 + 1i32
    wid  = (threadIdx().x - 1i32) ÷ 32i32 + 1i32
    val = _warp_reduce(op, val, 0xffffffff, Val(32))
    if lane == 1i32
        @inbounds scratch[wid] = val
    end
    sync_threads()
    nw = (blockDim().x + 31i32) ÷ 32i32
    if wid == 1i32
        u = lane <= nw ? @inbounds(scratch[lane]) : neutral
        u = _warp_reduce(op, u, 0xffffffff, Val(32))
        if lane == 1i32
            @inbounds scratch[1] = u
        end
    end
    sync_threads()
    out = @inbounds scratch[1]
    sync_threads()  # scratch reusable after return
    return out
end

function circulant_flash_attention_block_kernel!(
        y, lse, simfun::AbstractSimilarity, q, k, v,
        nrows::Int32, K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, ngroups::Int32,
    )
    Ts   = simval_dtype(simfun, eltype(q), eltype(k))
    Tacc = promote_type(Ts, eltype(v))
    w_sh    = CuDynamicSharedArray(Ts, K)
    col_sh  = CuDynamicSharedArray(Int32, K, Int(K) * sizeof(Ts))
    scratch = CuStaticSharedArray(Ts, 32)

    tid    = threadIdx().x
    TB     = blockDim().x
    lane   = (tid - 1i32) % 32i32 + 1i32
    wid    = (tid - 1i32) ÷ 32i32 + 1i32
    nwarps = TB ÷ 32i32
    neginf = typemin(Ts)

    g = blockIdx().x
    while g <= ngroups
        r = (g - 1i32) % nrows + 1i32
        b = (g - 1i32) ÷ nrows + 1i32
        base = (r - 1i32) * K

        # phase 1: similarities into shared, block softmax statistics
        mloc = neginf
        κ = tid
        while κ <= K
            i, _ = cartesian_circulant(base + κ, spatdims, W)
            s = scale * simval(simfun, q, k, r, i, b, C)
            @inbounds w_sh[κ]   = s
            @inbounds col_sh[κ] = i
            mloc = max(mloc, s)
            κ += TB
        end
        m = _block_reduce(max, mloc, neginf, scratch)

        lloc = zero(Ts)
        κ = tid
        while κ <= K
            e = _fastexp(@inbounds(w_sh[κ]) - m)
            @inbounds w_sh[κ] = e
            lloc += e
            κ += TB
        end
        l = _block_reduce(+, lloc, zero(Ts), scratch)
        linv = inv(l)
        κ = tid
        while κ <= K
            @inbounds w_sh[κ] *= linv
            κ += TB
        end
        if tid == 1i32
            @inbounds lse[r, b] = m + _fastlog(l)
        end
        sync_threads()

        # phase 2: warps over channels, lanes over window entries
        c = wid
        while c <= Cv
            p = zero(Tacc)
            κ = lane
            while κ <= K
                p = muladd(@inbounds(w_sh[κ]), @inbounds(v[col_sh[κ], c, b]), p)
                κ += 32i32
            end
            p = _warp_reduce(+, p, 0xffffffff, Val(32))
            if lane == 1i32
                @inbounds y[r, c, b] = p
            end
            c += nwarps
        end
        sync_threads()  # shared safe to overwrite for the next group
        g += gridDim().x
    end
    return nothing
end

function circulant_flash_attention_bwd_block_kernel!(
        dq, dk, dv, simfun::AbstractSimilarity, q, k, v, Δ, lse, δ,
        nrows::Int32, K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, ngroups::Int32,
    )
    Tq = eltype(q); Tk = eltype(k); TΔ = eltype(Δ)
    Ts, Tds, Tw = _flash_bwd_wtypes(simfun, Tq, Tk, TΔ, eltype(v))
    Tqk = promote_type(Tq, Tk)
    β   = _simgrad_beta(simfun, Tqk)
    # largest-aligned first: u (Tw), P (Ts), cols (Int32)
    u_sh    = CuDynamicSharedArray(Tw, K)
    P_sh    = CuDynamicSharedArray(Ts, K, Int(K) * sizeof(Tw))
    col_sh  = CuDynamicSharedArray(Int32, K, Int(K) * (sizeof(Tw) + sizeof(Ts)))
    scratch = CuStaticSharedArray(Tds, 32)

    tid    = threadIdx().x
    TB     = blockDim().x
    lane   = (tid - 1i32) % 32i32 + 1i32
    wid    = (tid - 1i32) ÷ 32i32 + 1i32
    nwarps = TB ÷ 32i32

    g = blockIdx().x
    while g <= ngroups
        a = (g - 1i32) % nrows + 1i32
        b = (g - 1i32) ÷ nrows + 1i32
        base = (a - 1i32) * K
        lse_a = @inbounds lse[a, b]
        δ_a   = @inbounds δ[a, b]

        κ = tid
        while κ <= K
            i, _ = cartesian_circulant(base + κ, spatdims, W)
            @inbounds col_sh[κ] = i
            κ += TB
        end
        sync_threads()

        # ---- sweep 1: a as row — stage ds·α, then ∂q[a,:] -----------------
        dsloc = zero(Tds)
        κ = tid
        while κ <= K
            @inbounds i = col_sh[κ]
            s, α, _ = simgrad_aux(simfun, q, k, a, i, b, C)
            P  = _fastexp(scale * s - lse_a)
            sds = scale * (P * (_flash_gval(Δ, v, a, i, b, Cv) - δ_a))
            @inbounds u_sh[κ] = sds * α
            dsloc += sds
            κ += TB
        end
        ds1 = _block_reduce(+, dsloc, zero(Tds), scratch)

        c = wid
        while c <= C
            p = zero(Tqk)
            κ = lane
            while κ <= K
                p += @inbounds(u_sh[κ]) * @inbounds(k[col_sh[κ], c, b])
                κ += 32i32
            end
            p = _warp_reduce(+, p, 0xffffffff, Val(32))
            if lane == 1i32
                @inbounds dq[a, c, b] = p + β * ds1 * @inbounds(q[a, c, b])
            end
            c += nwarps
        end
        sync_threads()  # u_sh reads done before sweep 2 overwrites

        # ---- sweep 2: a as column — stage ds·conj(α) and P, then ∂k, ∂v ---
        dsloc = zero(Tds)
        κ = tid
        while κ <= K
            r = @inbounds col_sh[κ]
            s, α, _ = simgrad_aux(simfun, q, k, r, a, b, C)
            P  = _fastexp(scale * s - @inbounds(lse[col_sh[κ], b]))
            sds = scale * (P * (_flash_gval(Δ, v, r, a, b, Cv) - @inbounds(δ[col_sh[κ], b])))
            @inbounds u_sh[κ] = sds * conj(α)
            @inbounds P_sh[κ] = P
            dsloc += sds
            κ += TB
        end
        ds2 = _block_reduce(+, dsloc, zero(Tds), scratch)

        Cmax = max(C, Cv)
        c = wid
        while c <= Cmax
            pk = zero(Tqk)
            pv = zero(TΔ)
            κ = lane
            while κ <= K
                Cc = @inbounds col_sh[κ]
                if c <= C
                    pk += @inbounds(u_sh[κ]) * @inbounds(q[Cc, c, b])
                end
                if c <= Cv
                    pv = muladd(@inbounds(P_sh[κ]), @inbounds(Δ[Cc, c, b]), pv)
                end
                κ += 32i32
            end
            pk = _warp_reduce(+, pk, 0xffffffff, Val(32))
            pv = _warp_reduce(+, pv, 0xffffffff, Val(32))
            if lane == 1i32
                c <= C  && (@inbounds dk[a, c, b] = pk + β * ds2 * @inbounds(k[a, c, b]))
                c <= Cv && (@inbounds dv[a, c, b] = pv)
            end
            c += nwarps
        end
        sync_threads()
        g += gridDim().x
    end
    return nothing
end

# ------------------------------------------------------------------
# host-side launchers (inputs already scaled)
# ------------------------------------------------------------------

# Channel-chunk size: register accumulators per thread (complex counts double).
_flash_chunk(::Type{T}) where T = T <: Complex ? Val(16) : Val(32)

function _flash_launch_dims(q::AbstractArray{Tq,N}, W::Int) where {Tq,N}
    spatdims = ntuple(i -> Int32(size(q, i)), N-2)
    nrows    = Int32(prod(spatdims))
    K        = Int32(W)^Int32(N-2)
    maxidx   = nrows * Int32(size(q, N))
    return spatdims, nrows, K, maxidx
end

# Sub-warp width and entries-per-lane. Beyond _FLASH_WARP_MAX_NE registers of
# window state per lane the block-per-row kernels take over. A100 measurements
# (128×128×64×2, DistanceSimilarity): NE=8 (K=225) warp 2.0ms vs block 6.2ms,
# NE=20 (K=625) warp 9.6ms vs block 7.7ms — the crossover is register
# pressure, between NE=8 and NE=20.
const _FLASH_WARP_MAX_NE = 8

# dynamic shared memory budget for the block kernels (default 48KB per block,
# minus headroom for the 32-slot static reduction scratch)
const _FLASH_BLOCK_MAX_SHMEM = 47 * 1024
const _FLASH_BLOCK_THREADS = 256

function _flash_warp_dims(K::Int32)
    # Smallest sub-warp width that keeps NE ≤ _FLASH_WARP_MAX_NE: the output
    # stage costs ~K + WS·log2(WS) lane-ops per channel, so narrow sub-warps
    # win for small windows (K=25: WS=8 does 49 lane-ops vs 185 at WS=32).
    # 8 lanes minimum keeps ≥ 8·nrows·B threads of parallelism.
    WS = clamp(nextpow(2, cld(Int(K), _FLASH_WARP_MAX_NE)), 8, 32)
    NE = cld(Int(K), WS)
    return WS, NE
end

# Kernel selection: warp while the window fits in lane registers, then block
# while the staged window fits in shared memory, then thread-per-row.
function _flash_mode(mode::Symbol, NE::Int, shmem::Int)
    mode === :auto || return mode
    NE <= _FLASH_WARP_MAX_NE     && return :warp
    shmem <= _FLASH_BLOCK_MAX_SHMEM && return :block
    return :thread
end

function _flash_block_launch(kernelfn::F, args, ngroups::Int32, shmem::Int) where F
    kernel = @cuda launch=false kernelfn(args...)
    kernel(args...; threads=_FLASH_BLOCK_THREADS, blocks=Int(ngroups), shmem=shmem)
    return nothing
end

# Launch a warp-cooperative kernel: ngroups sub-warps of WS lanes, block size a
# multiple of 32 (so segments never straddle warps), grid-stride over groups.
function _flash_warp_launch(kernelfn::F, args, ngroups::Int32, WS::Int) where F
    kernel = @cuda launch=false kernelfn(args...)
    config = launch_configuration(kernel.fun)
    threads = max(32, min(256, (config.threads ÷ 32) * 32))
    blocks  = cld(Int(ngroups) * WS, threads)
    kernel(args...; threads=threads, blocks=blocks)
    return nothing
end

function _circulant_flash_attention_fwd(
        simfun::AbstractSimilarity,
        q::AnyCuArray{Tq,N}, k::AnyCuArray{Tk,N}, v::AnyCuArray{Tv,N},
        W::Int, scale::Real=true;
        mode::Symbol=:auto,
    ) where {Tq, Tk, Tv, N}
    Ts = simval_dtype(simfun, Tq, Tk)
    Ts <: Real || throw(ArgumentError(
        "circulant_flash_attention needs a real-valued similarity for softmax; " *
        "$(typeof(simfun)) on ($Tq, $Tk) gives $Ts. Use RealDotSimilarity() instead."))
    @assert isodd(W) "window length W=$W must be odd"
    size(q) == size(k) || throw(DimensionMismatch("q $(size(q)) and k $(size(k)) must match"))
    size(v)[1:N-2] == size(q)[1:N-2] && size(v, N) == size(q, N) ||
        throw(DimensionMismatch("v $(size(v)) must match q $(size(q)) in spatial and batch dims"))

    Tacc = promote_type(Ts, Tv)
    y = similar(v, Tacc)

    spatdims, nrows, K, maxidx = _flash_launch_dims(q, W)
    C   = Int32(size(q, N-1))
    Cv  = Int32(size(v, N-1))
    lse = similar(q, Ts, (Int(nrows), size(q, N)))
    WS, NE = _flash_warp_dims(K)
    shmem  = Int(K) * (sizeof(Ts) + sizeof(Int32))
    usemode = _flash_mode(mode, NE, shmem)

    # kernels index spatial positions linearly: (nrows, channels, batch) views
    yr = reshape(y, :, size(y, N-1), size(y, N))
    qr = reshape(q, :, size(q, N-1), size(q, N))
    kr = reshape(k, :, size(k, N-1), size(k, N))
    vr = reshape(v, :, size(v, N-1), size(v, N))

    sc = Ts(scale)
    if usemode === :warp
        args = (yr, lse, simfun, qr, kr, vr, nrows, K, C, Cv, spatdims, Int32(W), sc, maxidx, Val(WS), Val(NE))
        _flash_warp_launch(circulant_flash_attention_warp_kernel!, args, maxidx, WS)
    elseif usemode === :block
        args = (yr, lse, simfun, qr, kr, vr, nrows, K, C, Cv, spatdims, Int32(W), sc, maxidx)
        _flash_block_launch(circulant_flash_attention_block_kernel!, args, maxidx, shmem)
    else
        chunk = _flash_chunk(Tacc)
        args = (yr, lse, simfun, qr, kr, vr, nrows, K, C, Cv, spatdims, Int32(W), sc, maxidx, chunk)
        kernel = @cuda launch=false circulant_flash_attention_kernel!(args...)
        config = launch_configuration(kernel.fun)
        threads = min(maxidx, config.threads)
        blocks  = cld(maxidx, threads)
        kernel(args...; threads=threads, blocks=blocks)
    end
    return y, lse
end

function _circulant_flash_attention(simfun::AbstractSimilarity, q, k, v, W::Int, scale::Real=true)
    first(_circulant_flash_attention_fwd(simfun, q, k, v, W, scale))
end

function ∇circulant_flash_attention(
        simfun::AbstractSimilarity,
        Δ::AnyCuArray{TΔ,N}, y, lse,
        q::AnyCuArray{Tq,N}, k::AnyCuArray{Tk,N}, v::AnyCuArray{Tv,N},
        W::Int, scale::Real=true;
        mode::Symbol=:auto,
        Δlse=nothing,
    ) where {TΔ, Tq, Tk, Tv, N}
    Tqk = promote_type(Tq, Tk)
    dq = similar(q, Tqk)
    dk = similar(k, Tqk)
    dv = similar(v, TΔ)

    # δ_r = Re⟨Δ[r,:], y[r,:]⟩ = Σ_i P[r,i] g[r,i] — the softmax-pullback shift.
    # A logsumexp cotangent λ enters as ds += λ_r·P[r,i] (since ∂L_r/∂s_ri =
    # P_ri), which folds into the same formula as δ → δ - λ.
    δ = reshape(sum(real.(Δ .* conj.(y)); dims=N-1), :, size(Δ, N))
    if Δlse !== nothing
        δ = δ .- Δlse
    end

    spatdims, nrows, K, maxidx = _flash_launch_dims(q, W)
    C  = Int32(size(q, N-1))
    Cv = Int32(size(v, N-1))
    WS, NE = _flash_warp_dims(K)
    Ts, _, Tw = _flash_bwd_wtypes(simfun, Tq, Tk, TΔ, Tv)
    shmem = Int(K) * (sizeof(Tw) + sizeof(Ts) + sizeof(Int32))
    usemode = _flash_mode(mode, NE, shmem)

    Δr = reshape(Δ, :, size(Δ, N-1), size(Δ, N))
    qr = reshape(q, :, size(q, N-1), size(q, N))
    kr = reshape(k, :, size(k, N-1), size(k, N))
    vr = reshape(v, :, size(v, N-1), size(v, N))
    dqr = reshape(dq, :, size(dq, N-1), size(dq, N))
    dkr = reshape(dk, :, size(dk, N-1), size(dk, N))
    dvr = reshape(dv, :, size(dv, N-1), size(dv, N))

    sc = Ts(scale)
    if usemode === :warp
        args = (dqr, dkr, dvr, simfun, qr, kr, vr, Δr, lse, δ, nrows, K, C, Cv, spatdims, Int32(W), sc, maxidx, Val(WS), Val(NE))
        _flash_warp_launch(circulant_flash_attention_bwd_warp_kernel!, args, maxidx, WS)
    elseif usemode === :block
        args = (dqr, dkr, dvr, simfun, qr, kr, vr, Δr, lse, δ, nrows, K, C, Cv, spatdims, Int32(W), sc, maxidx)
        _flash_block_launch(circulant_flash_attention_bwd_block_kernel!, args, maxidx, shmem)
    else
        chunk = _flash_chunk(promote_type(Tqk, TΔ))
        args = (dqr, dkr, dvr, simfun, qr, kr, vr, Δr, lse, δ, nrows, K, C, Cv, spatdims, Int32(W), sc, maxidx, chunk)
        kernel = @cuda launch=false circulant_flash_attention_bwd_kernel!(args...)
        config = launch_configuration(kernel.fun)
        threads = min(maxidx, config.threads)
        blocks  = cld(maxidx, threads)
        kernel(args...; threads=threads, blocks=blocks)
    end
    return dq, dk, dv
end

# ------------------------------------------------------------------
# public API
# ------------------------------------------------------------------

@doc raw"""
    y = circulant_flash_attention(simfun::AbstractSimilarity, q, k, v, W::Int)

Fused (flash) version of [`circulant_attention`](@ref): computes the same
`y = rowsoftmax(S)v` with ``S_{ij} = \mathrm{simfun}(q_i, k_j)`` in a single
kernel using an online softmax, without ever materializing the circulant-sparse
attention matrix — memory traffic is ``O(NC)`` instead of ``O(NW^d)``.
Only `y` is returned; if the adjacency matrix is needed, use
[`circulant_attention`](@ref).

The backward pass is fused as well: attention weights are recomputed from the
logsumexp saved during the forward pass (FlashAttention-style), so no sparse
intermediate exists in either direction.

The similarity must be real-valued (`RealDotSimilarity`, `DistanceSimilarity`,
`PIDotSimilarity`, `PIDistanceSimilarity`, or `DotSimilarity` on real inputs).
Window-renormalizing similarities (`TopKSimilarity`, `SparsemaxSimilarity`,
`EntmaxSimilarity`) are not supported.

See also [`circulant_attention`](@ref), [`circulant_mh_flash_attention`](@ref).
"""
function circulant_flash_attention(simfun::AbstractSimilarity, q::T, k::T, v::T, W::Int) where {Tv, N, T<:AbstractArray{Tv,N}}
    # scaling q,k by τ^(-1/2) ≡ scaling the similarity by 1/τ for every
    # supported simfun (dot- and distance-types are 2-homogeneous), so the
    # scale is folded into the kernels instead of allocating scaled copies
    scale = inv(sqrt(real(Tv)(size(k, N-1))))
    _circulant_flash_attention(simfun, q, k, v, W, scale)
end
circulant_flash_attention(q::T, k::T, v::T, W::Int) where T = circulant_flash_attention(DotSimilarity(), q, k, v, W)

function circulant_flash_attention(simfun::_UnfusableSimilarity, q::T, k::T, v::T, W::Int) where {Tv, N, T<:AbstractArray{Tv,N}}
    throw(ArgumentError(
        "$(typeof(simfun)) renormalizes over the full window and cannot be fused; " *
        "use circulant_attention instead."))
end

@doc raw"""
    y = circulant_mh_flash_attention(simfun::AbstractSimilarity, q, k, v, W::Int, nheads::Int)

Multi-head version of [`circulant_flash_attention`](@ref): performs fused
circulant attention on `nheads` channel groups separately and concatenates the
result along channels. The number of channels must be divisible by `nheads`.

See also [`circulant_mh_attention`](@ref).
"""
function circulant_mh_flash_attention(simfun::AbstractSimilarity, q::T, k::T, v::T, W::Int, nheads::Int) where {Tv, N, T<:AbstractArray{Tv,N}}
    qr, kr, vr = splitheads.((q, k, v), nheads)
    yr = circulant_flash_attention(simfun, qr, kr, vr, W)
    return reshape(yr, size(v)...)
end
circulant_mh_flash_attention(q::T, k::T, v::T, W::Int, nheads::Int) where T = circulant_mh_flash_attention(DotSimilarity(), q, k, v, W, nheads)

# rowwise logsumexp across the branch lse arrays; the max shift cancels
# analytically in the gradient, so plain tracing is exact.
function _joint_lse(lses::NTuple{M}) where M
    Lmax = reduce((a, b) -> max.(a, b), lses)
    s = reduce(+, map(L -> exp.(L .- Lmax), lses))
    return log.(s) .+ Lmax
end

@doc raw"""
    ys = circulant_flash_joint_attention(simfuns, qs, ks, vs, Ws)
    ys = circulant_flash_joint_attention(simfun, q, k, v, Ws::NTuple{M,Int})

Fused (flash) version of joint-softmax attention: each branch ``m`` computes
circulant similarities with window ``W_m``, the softmax is normalized *jointly*
over the union of all branches' window entries per row (as in
[`joint_softmax`](@ref)), and branch outputs ``y_m = P_m^{joint} v_m`` are
returned as a tuple — without materializing any attention matrix.

No additional kernel is involved: the joint softmax decomposes over the
per-branch logsumexps the fused kernels already produce,
``y_m^{joint} = e^{L_m - L^{joint}} \odot y_m`` with
``L^{joint} = \mathrm{logsumexp}_m(L_m)`` rowwise, so this is the per-branch
[`circulant_flash_attention`](@ref) plus dense reweighting. Branches may have
different windows, inputs, and similarities, but must share spatial size and
batch. Inputs are scaled by `sqrt(sqrt(channels))` per branch as in
[`circulant_attention`](@ref).

The convenience form runs `M` branches with shared `simfun, q, k, v` and
windows `Ws`.

See also [`joint_softmax`](@ref), [`circulant_flash_attention`](@ref).
"""
function circulant_flash_joint_attention(
        simfuns::NTuple{M,AbstractSimilarity},
        qs::NTuple{M}, ks::NTuple{M}, vs::NTuple{M}, Ws::NTuple{M,Int},
    ) where M
    any(sf -> sf isa _UnfusableSimilarity, simfuns) && throw(ArgumentError(
        "window-renormalizing similarities cannot be fused; use circulant_similarity + joint_softmax instead."))

    outs = map(simfuns, qs, ks, vs, Ws) do simfun, q, k, v, W
        scale = inv(sqrt(real(eltype(k))(size(k, ndims(k) - 1))))
        _circulant_flash_attention_lse(simfun, q, k, v, W, scale)
    end
    ys   = map(first, outs)
    lses = map(last, outs)
    Lj   = _joint_lse(lses)

    return map(ys, lses) do y, L
        ω = reshape(exp.(L .- Lj), size(y)[1:ndims(y)-2]..., 1, size(y, ndims(y)))
        ω .* y
    end
end

function circulant_flash_joint_attention(simfun::AbstractSimilarity, q::T, k::T, v::T, Ws::NTuple{M,Int}) where {T, M}
    circulant_flash_joint_attention(
        ntuple(_ -> simfun, Val(M)), ntuple(_ -> q, Val(M)),
        ntuple(_ -> k, Val(M)), ntuple(_ -> v, Val(M)), Ws)
end

@doc raw"""
    ys = circulant_mh_flash_joint_attention(simfuns, qs, ks, vs, Ws, nheads::Int)
    ys = circulant_mh_flash_joint_attention(simfun, qs, ks, vs, Ws, nheads::Int)

Multi-head version of [`circulant_flash_joint_attention`](@ref): the joint
softmax is normalized per head over the union of the branches' window entries.
Heads are folded into the batch dimension (as in
[`circulant_mh_flash_attention`](@ref)); the joint normalization is independent
per row, so this is exact. The number of channels must be divisible by `nheads`.
"""
function circulant_mh_flash_joint_attention(
        simfuns::NTuple{M,AbstractSimilarity},
        qs::NTuple{M}, ks::NTuple{M}, vs::NTuple{M}, Ws::NTuple{M,Int}, nheads::Int,
    ) where M
    qrs = map(q -> splitheads(q, nheads), qs)
    krs = map(k -> splitheads(k, nheads), ks)
    vrs = map(v -> splitheads(v, nheads), vs)
    yrs = circulant_flash_joint_attention(simfuns, qrs, krs, vrs, Ws)
    return map((y, v) -> reshape(y, size(v)...), yrs, vs)
end

function circulant_mh_flash_joint_attention(simfun::AbstractSimilarity, qs::NTuple{M}, ks::NTuple{M}, vs::NTuple{M}, Ws::NTuple{M,Int}, nheads::Int) where M
    circulant_mh_flash_joint_attention(ntuple(_ -> simfun, Val(M)), qs, ks, vs, Ws, nheads)
end

# Replicate the batch dimension `G` times, guide-fastest:
# (lead..., B) → (lead..., G·B) with new index gb = g + (b-1)·G. Matches the
# layout of a guide tensor `reshape(v, lead..., G, B)`.
function _replicate_batch(q::AbstractArray{T,N}, G::Integer) where {T,N}
    lead = size(q)[1:N-1]
    B    = size(q, N)
    qe   = reshape(q, lead..., 1, B)
    qr   = repeat(qe, ntuple(_ -> 1, N-1)..., G, 1)   # (lead..., G, B)
    return reshape(qr, lead..., G * B)
end

@doc raw"""
    ξz, ξg = circulant_mh_flash_guided_joint_attention(simfun, qz, kz, vz, Wz,
                                                       kg, vg, Wg, num_guides, nheads)

Joint-softmax flash attention for the guided multi-guide proximal map: a single
self branch (`qz, kz, vz`, window `Wz`) plus `num_guides` guide branches that
**share the self query** and a common window `Wg`, with the guide keys/values
stacked along the batch dimension of `kg`/`vg` as `(guide-fastest, base-batch)`
(`size(kg, end) == num_guides * size(kz, end)`).

Equivalent to [`circulant_mh_flash_joint_attention`](@ref) over the branches
`(self, g₁, …, g_G)` — the softmax is normalized jointly over the union of the
self window and all guide windows — but the guides are attended in **one
batched flash call** instead of one call per guide (and the caller projects
them once). Returns the self output `ξz` and the summed guide output
`ξg = Σ_g ξ_g`, each `(spatial..., Cv, B)`.

See also [`circulant_mh_flash_joint_attention`](@ref).
"""
function circulant_mh_flash_guided_joint_attention(
        simfun::AbstractSimilarity,
        qz::AbstractArray{T,N}, kz::AbstractArray{T,N}, vz::AbstractArray{Tv,N}, Wz::Int,
        kg::AbstractArray{T,N}, vg::AbstractArray{Tv,N}, Wg::Int,
        num_guides::Int, nheads::Int) where {T, Tv, N}
    # self branch (head-folded): yz (sp...,Cvh,nh·B), Lz_flat (nrows, nh·B)
    qzr, kzr, vzr = splitheads.((qz, kz, vz), nheads)
    sz = inv(sqrt(real(T)(size(kzr, N-1))))
    yz, Lz_flat = _circulant_flash_attention_lse(simfun, qzr, kzr, vzr, Wz, sz)

    # guide branch: replicate the shared query across guides and attend them all
    # in one batched flash; yg (sp...,Cvh,nh·G·B), Lg_flat (nrows, nh·G·B)
    qg = _replicate_batch(qz, num_guides)
    qgr, kgr, vgr = splitheads.((qg, kg, vg), nheads)
    sg = inv(sqrt(real(T)(size(kgr, N-1))))
    yg, Lg_flat = _circulant_flash_attention_lse(simfun, qgr, kgr, vgr, Wg, sg)

    nrows = size(Lz_flat, 1)
    B     = size(Lz_flat, 2) ÷ nheads
    sp    = size(yz)[1:N-2]
    Cvh   = size(yz, N-1)

    # joint logsumexp over self + G guides, per (row, head, base-batch).
    # folded batch order is head-fastest then (for guides) guide then base.
    Lz = reshape(Lz_flat, nrows, nheads, B)
    Lg = reshape(Lg_flat, nrows, nheads, num_guides, B)
    m  = max.(Lz, dropdims(maximum(Lg; dims=3); dims=3))             # (nrows,nh,B)
    m4 = reshape(m, nrows, nheads, 1, B)
    Lj = m .+ log.(exp.(Lz .- m) .+ dropdims(sum(exp.(Lg .- m4); dims=3); dims=3))

    # reweight (broadcast over channels) and join heads back
    ωz = reshape(exp.(Lz .- Lj), sp..., 1, nheads * B)
    ωg = reshape(exp.(Lg .- reshape(Lj, nrows, nheads, 1, B)), sp..., 1, nheads * num_guides * B)
    ξz = reshape(yz .* ωz, size(vz)...)

    ξg_f   = yg .* ωg                                                # (sp...,Cvh,nh·G·B)
    ξg_sum = dropdims(sum(reshape(ξg_f, sp..., Cvh, nheads, num_guides, B); dims=N+1); dims=N+1)
    ξg     = reshape(ξg_sum, size(vz)...)
    return ξz, ξg
end

# ------------------------------------------------------------------
# rrules — fused backward
# ------------------------------------------------------------------

# materialize Zero/nothing/Fill tangents into dense CuArrays shaped like ref
# (the backward kernels and CUSPARSE cannot consume lazy tangent types)
function _flash_materialize(Δ, ref)
    Δ = CRC.unthunk(Δ)
    if Δ === nothing || Δ isa CRC.AbstractZero
        return CUDA.zeros(eltype(ref), size(ref)...)
    elseif Δ isa Zygote.FillArrays.AbstractFill
        return CUDA.fill(convert(eltype(ref), Zygote.FillArrays.getindex_value(Δ)), size(ref)...)
    end
    return Δ
end

function CRC.rrule(::typeof(_circulant_flash_attention), simfun::AbstractSimilarity, q, k, v, W::Int, scale::Real)
    y, lse = _circulant_flash_attention_fwd(simfun, q, k, v, W, scale)
    project_q, project_k, project_v = CRC.ProjectTo(q), CRC.ProjectTo(k), CRC.ProjectTo(v)
    function flash_attention_pullback(Δ)
        Δy = _flash_materialize(Δ, y)
        ∂q, ∂k, ∂v = ∇circulant_flash_attention(simfun, Δy, y, lse, q, k, v, W, scale)
        return CRC.NoTangent(), CRC.NoTangent(), project_q(∂q), project_k(∂k), project_v(∂v), CRC.NoTangent(), CRC.NoTangent()
    end
    return y, flash_attention_pullback
end

function CRC.rrule(::typeof(_circulant_flash_attention), simfun::AbstractSimilarity, q, k, v, W::Int)
    y, pb7 = CRC.rrule(_circulant_flash_attention, simfun, q, k, v, W, true)
    flash_attention_pullback6(Δ) = pb7(Δ)[1:6]
    return y, flash_attention_pullback6
end

# (y, lse) variant: exposes the per-row logsumexp as a differentiable output,
# which is what joint normalization needs.
function _circulant_flash_attention_lse(simfun::AbstractSimilarity, q, k, v, W::Int, scale::Real=true)
    _circulant_flash_attention_fwd(simfun, q, k, v, W, scale)
end

function CRC.rrule(::typeof(_circulant_flash_attention_lse), simfun::AbstractSimilarity, q, k, v, W::Int, scale::Real)
    y, lse = _circulant_flash_attention_fwd(simfun, q, k, v, W, scale)
    project_q, project_k, project_v = CRC.ProjectTo(q), CRC.ProjectTo(k), CRC.ProjectTo(v)
    function flash_attention_lse_pullback(Δ)
        Δ  = CRC.unthunk(Δ)
        Δy = _flash_materialize(Δ[1], y)
        Δλ = CRC.unthunk(Δ[2])
        Δλ = (Δλ === nothing || Δλ isa CRC.AbstractZero) ? nothing : _flash_materialize(Δλ, lse)
        ∂q, ∂k, ∂v = ∇circulant_flash_attention(simfun, Δy, y, lse, q, k, v, W, scale; Δlse=Δλ)
        return CRC.NoTangent(), CRC.NoTangent(), project_q(∂q), project_k(∂k), project_v(∂v), CRC.NoTangent(), CRC.NoTangent()
    end
    return (y, lse), flash_attention_lse_pullback
end

function CRC.rrule(::typeof(_circulant_flash_attention_lse), simfun::AbstractSimilarity, q, k, v, W::Int)
    out, pb7 = CRC.rrule(_circulant_flash_attention_lse, simfun, q, k, v, W, true)
    flash_attention_lse_pullback6(Δ) = pb7(Δ)[1:6]
    return out, flash_attention_lse_pullback6
end

# ------------------------------------------------------------------
# transposed flash attention:  y = Γᵀ x  without materializing Γ
#
#   (Γᵀx)_a = Σ_r P_{ra} x_r,   P_{ra} = exp(scale·s_{ra} − L_r),
#   s_{ra} = simval(q_r, k_a),  L_r = logsumexp_i scale·s_{ri}.
#
# This is the adjoint (w.r.t. v) of the forward Γ-apply y = Γv, so it is
# exactly the `dv` output of the fused backward — which depends only on P and
# the cotangent, not on v/y/δ. We therefore reuse the already-GPU-tested
# forward/backward primitives instead of writing a new device kernel:
#   forward:  L = lse from one fwd pass;  Γᵀx = dv from one bwd pass.
#   gradient (derived from y_a = Σ_r exp(scale·s_{ra}−L_r) x_r):
#     ∂x = Γ Δ̄              (forward flash on the cotangent Δ̄)
#     ∂q,∂k = dq,dk of the fused backward with Δ←x, v←Δ̄, y←ȳ=ΓΔ̄
#   (the backward's −δ_r softmax-shift term carries L's dependence on q,k, so
#    the gradient is exact even though L is recomputed rather than threaded).
#
# A dedicated fused kernel would avoid the wasted dq/dk/y compute below; this
# reuse keeps the column-sweep on the tested code path and is used on the
# multigrid subgradient, not the hot per-iteration prox.
# ------------------------------------------------------------------

# Γᵀx given the per-row logsumexp `lse` (value-only, no AD).
function _flash_transposed_from_lse(simfun::AbstractSimilarity, q, k, x, lse, W::Int, scale::Real)
    zr = zero(x)                                   # y and v are irrelevant to dv
    _, _, dv = ∇circulant_flash_attention(simfun, x, zr, lse, q, k, zr, W, scale)
    return dv
end

function _circulant_flash_transposed_attention(simfun::AbstractSimilarity, q, k, x, W::Int, scale::Real=true)
    _, lse = _circulant_flash_attention_fwd(simfun, q, k, x, W, scale)
    return _flash_transposed_from_lse(simfun, q, k, x, lse, W, scale)
end

function CRC.rrule(::typeof(_circulant_flash_transposed_attention), simfun::AbstractSimilarity, q, k, x, W::Int, scale::Real)
    _, lse = _circulant_flash_attention_fwd(simfun, q, k, x, W, scale)
    y = _flash_transposed_from_lse(simfun, q, k, x, lse, W, scale)
    project_q, project_k, project_x = CRC.ProjectTo(q), CRC.ProjectTo(k), CRC.ProjectTo(x)
    function flash_transposed_pullback(Δ)
        Δ̄ = _flash_materialize(Δ, y)
        ȳ = _circulant_flash_attention(simfun, q, k, Δ̄, W, scale)            # Γ Δ̄  ( = ∂x )
        ∂q, ∂k, _ = ∇circulant_flash_attention(simfun, x, ȳ, lse, q, k, Δ̄, W, scale)
        return CRC.NoTangent(), CRC.NoTangent(), project_q(∂q), project_k(∂k), project_x(ȳ), CRC.NoTangent(), CRC.NoTangent()
    end
    return y, flash_transposed_pullback
end

function CRC.rrule(::typeof(_circulant_flash_transposed_attention), simfun::AbstractSimilarity, q, k, x, W::Int)
    y, pb7 = CRC.rrule(_circulant_flash_transposed_attention, simfun, q, k, x, W, true)
    flash_transposed_pullback6(Δ) = pb7(Δ)[1:6]
    return y, flash_transposed_pullback6
end

@doc raw"""
    y = circulant_flash_transposed_attention(simfun::AbstractSimilarity, q, k, x, W::Int)

Fused (flash) version of the *transposed* circulant attention apply
``y = \Gamma^{\!\top} x``, where ``\Gamma = \mathrm{rowsoftmax}(S)`` with
``S_{ij} = \mathrm{simfun}(q_i, k_j)`` is the same row-softmax attention matrix
[`circulant_flash_attention`](@ref) applies as ``\Gamma v``. The transpose is

```math
(\Gamma^{\!\top} x)_a = \sum_r e^{\,s_{ra} - L_r}\, x_r ,\qquad
L_r = \mathrm{logsumexp}_i\, s_{ri},
```

i.e. each source row keeps its own softmax normalizer ``L_r`` — so a plain
``q\leftrightarrow k`` swap does **not** give ``\Gamma^{\!\top}`` (that would
renormalize over the wrong axis). No attention matrix is materialized;
gradients w.r.t. `q`, `k`, `x` are exact.

`q`, `k` must share shape; `x` carries the attended channels. The similarity
must be real-valued (window-renormalizing similarities are not supported).

See also [`circulant_flash_attention`](@ref),
[`circulant_mh_flash_transposed_attention`](@ref).
"""
function circulant_flash_transposed_attention(simfun::AbstractSimilarity, q::T, k::T, x::T, W::Int) where {Tv, N, T<:AbstractArray{Tv,N}}
    scale = inv(sqrt(real(Tv)(size(k, N-1))))
    _circulant_flash_transposed_attention(simfun, q, k, x, W, scale)
end
circulant_flash_transposed_attention(q::T, k::T, x::T, W::Int) where T = circulant_flash_transposed_attention(DotSimilarity(), q, k, x, W)

function circulant_flash_transposed_attention(simfun::_UnfusableSimilarity, q::T, k::T, x::T, W::Int) where {Tv, N, T<:AbstractArray{Tv,N}}
    throw(ArgumentError(
        "$(typeof(simfun)) renormalizes over the full window and cannot be fused; " *
        "form the adjacency with circulant_adjacency and use circulant_mh_transposed_attention instead."))
end

@doc raw"""
    y = circulant_mh_flash_transposed_attention(simfun, q, k, x, W::Int, nheads::Int)

Multi-head version of [`circulant_flash_transposed_attention`](@ref): applies
the fused transposed attention on `nheads` channel groups separately and
concatenates along channels. The number of channels must be divisible by
`nheads`.
"""
function circulant_mh_flash_transposed_attention(simfun::AbstractSimilarity, q::T, k::T, x::T, W::Int, nheads::Int) where {Tv, N, T<:AbstractArray{Tv,N}}
    qr, kr, xr = splitheads.((q, k, x), nheads)
    yr = circulant_flash_transposed_attention(simfun, qr, kr, xr, W)
    return reshape(yr, size(x)...)
end
circulant_mh_flash_transposed_attention(q::T, k::T, x::T, W::Int, nheads::Int) where T = circulant_mh_flash_transposed_attention(DotSimilarity(), q, k, x, W, nheads)
