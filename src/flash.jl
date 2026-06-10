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
# The price is recomputing the similarity sweep once per CH-channel chunk.
#
# Index convention (matches circulant_similarity! and rrules.jl):
# window entry κ of row r sits at nnz index n = (r-1)K + κ with column
# i = first(cartesian_circulant(n, spatdims, W)), and S[r,i] = simval(q_r, k_i).

# Wrapper similarities renormalize over the full window (sorting/thresholding),
# which cannot be streamed entry-by-entry.
const _UnfusableSimilarity = Union{TopKSimilarity, SparsemaxSimilarity, EntmaxSimilarity}

# ------------------------------------------------------------------
# forward device code
#
# Generic over AbstractArray so the same code path is testable on CPU.
# ------------------------------------------------------------------
@inline function _flash_attention_row!(
        y, lse, simfun::AbstractSimilarity, q, k, v,
        r::Int32, b::Int32,
        K::Int32, C::Int32, Cv::Int32,
        spatdims, CartInd, W::Int32, ::Val{CH},
    ) where CH
    Ts   = simval_dtype(simfun, eltype(q), eltype(k))
    Tacc = promote_type(Ts, eltype(v))
    Cr   = @inbounds CartInd[r]
    base = (r - 1i32) * K

    # pass 1: online max and normalizer
    m = typemin(Ts)
    l = zero(Ts)
    for κ in 1i32:K
        i, _ = cartesian_circulant(base + κ, spatdims, W)
        s = simval(simfun, q, k, Cr, @inbounds(CartInd[i]), b, C)
        mnew = max(m, s)
        l = l * exp(m - mnew) + exp(s - mnew)
        m = mnew
    end
    linv = inv(l)
    @inbounds lse[r, b] = m + log(l)

    # pass 2: accumulate softmax-weighted v, CH channels at a time
    c0 = 0i32
    while c0 < Cv
        acc = ntuple(_ -> zero(Tacc), Val(CH))
        for κ in 1i32:K
            i, _ = cartesian_circulant(base + κ, spatdims, W)
            Ci = @inbounds CartInd[i]
            p = exp(simval(simfun, q, k, Cr, Ci, b, C) - m) * linv
            acc = ntuple(Val(CH)) do t
                c = c0 + Int32(t)
                c <= Cv ? muladd(p, @inbounds(v[Ci, c, b]), acc[t]) : acc[t]
            end
        end
        for t in 1:CH
            c = c0 + Int32(t)
            if c <= Cv
                @inbounds y[Cr, c, b] = acc[t]
            end
        end
        c0 += Int32(CH)
    end
    return nothing
end

function circulant_flash_attention_kernel!(
        y, lse, simfun::AbstractSimilarity, q, k, v,
        nrows::Int32, K::Int32, C::Int32, Cv::Int32,
        spatdims, CartInd, W::Int32, maxidx::Int32,
        chunk::Val{CH},
    ) where CH
    tid    = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    while tid <= maxidx
        r = (tid - 1i32) % nrows + 1i32
        b = (tid - 1i32) ÷ nrows + 1i32
        _flash_attention_row!(y, lse, simfun, q, k, v, r, b, K, C, Cv, spatdims, CartInd, W, chunk)
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
@inline function simgrad_aux(::Union{DotSimilarity, RealDotSimilarity}, q, k, Cr, Ci, b, C::Int32)
    s = simval(RealDotSimilarity(), q, k, Cr, Ci, b, C)
    return s, one(s), zero(s)
end

@inline function simgrad_aux(::DistanceSimilarity, q, k, Cr, Ci, b, C::Int32)
    s = simval(DistanceSimilarity(), q, k, Cr, Ci, b, C)
    return s, one(s), -one(s)
end

@inline function simgrad_aux(::PIDotSimilarity, q, k, Cr, Ci, b, C::Int32)
    z = simval(DotSimilarity(), q, k, Cr, Ci, b, C)
    s = abs(z)
    return s, sign(z), zero(s)
end

@inline function simgrad_aux(::PIDistanceSimilarity, q, k, Cr, Ci, b, C::Int32)
    Ts = promote_type(eltype(q), eltype(k))
    s_xx = zero(real(Ts)); s_xy = zero(Ts); s_yy = zero(real(Ts))
    @inbounds for m in 1i32:C
        qm = q[Cr, m, b]; km = k[Ci, m, b]
        s_xx += abs2(qm)
        s_xy += qm * conj(km)
        s_yy += abs2(km)
    end
    s = -real(Ts)(0.5) * (s_xx + s_yy) + abs(s_xy)
    return s, sign(s_xy), -one(real(Ts))
end

# g[r,i] = Re⟨Δ[r,:], v[i,:]⟩ — the ∂A entry of the A ⊠ V pullback.
@inline function _flash_gval(Δ, v, Cr, Ci, b, Cv::Int32)
    g = zero(real(promote_type(eltype(Δ), eltype(v))))
    @inbounds for c in 1i32:Cv
        g += real(Δ[Cr, c, b] * conj(v[Ci, c, b]))
    end
    return g
end

@inline function _flash_attention_bwd_row!(
        dq, dk, dv, simfun::AbstractSimilarity, q, k, v, Δ, lse, δ,
        a::Int32, b::Int32,
        K::Int32, C::Int32, Cv::Int32,
        spatdims, CartInd, W::Int32, ::Val{CH},
    ) where CH
    Tqk  = promote_type(eltype(q), eltype(k))
    TΔ   = eltype(Δ)
    Ca   = @inbounds CartInd[a]
    base = (a - 1i32) * K
    lse_a = @inbounds lse[a, b]
    δ_a   = @inbounds δ[a, b]

    # sweep 1: a as row — entries (row a, col i) accumulate ∂q[a,:]
    c0 = 0i32
    while c0 < C
        acc = ntuple(_ -> zero(Tqk), Val(CH))
        for κ in 1i32:K
            i, _ = cartesian_circulant(base + κ, spatdims, W)
            Ci = @inbounds CartInd[i]
            s, α, β = simgrad_aux(simfun, q, k, Ca, Ci, b, C)
            P  = exp(s - lse_a)
            ds = P * (_flash_gval(Δ, v, Ca, Ci, b, Cv) - δ_a)
            acc = ntuple(Val(CH)) do t
                c = c0 + Int32(t)
                c <= C ? acc[t] + ds * (α * @inbounds(k[Ci, c, b]) + β * @inbounds(q[Ca, c, b])) : acc[t]
            end
        end
        for t in 1:CH
            c = c0 + Int32(t)
            c <= C && (@inbounds dq[Ca, c, b] = acc[t])
        end
        c0 += Int32(CH)
    end

    # sweep 2: a as column — by pattern symmetry, the rows whose window
    # contains column a are exactly the columns of row a. Entries (row r,
    # col a) accumulate ∂k[a,:] and ∂v[a,:].
    Cmax = max(C, Cv)
    c0 = 0i32
    while c0 < Cmax
        acck = ntuple(_ -> zero(Tqk), Val(CH))
        accv = ntuple(_ -> zero(TΔ), Val(CH))
        for κ in 1i32:K
            r, _ = cartesian_circulant(base + κ, spatdims, W)
            Cr = @inbounds CartInd[r]
            s, α, β = simgrad_aux(simfun, q, k, Cr, Ca, b, C)
            P  = exp(s - @inbounds(lse[r, b]))
            ds = P * (_flash_gval(Δ, v, Cr, Ca, b, Cv) - @inbounds(δ[r, b]))
            acck = ntuple(Val(CH)) do t
                c = c0 + Int32(t)
                c <= C ? acck[t] + ds * (conj(α) * @inbounds(q[Cr, c, b]) + β * @inbounds(k[Ca, c, b])) : acck[t]
            end
            accv = ntuple(Val(CH)) do t
                c = c0 + Int32(t)
                c <= Cv ? muladd(P, @inbounds(Δ[Cr, c, b]), accv[t]) : accv[t]
            end
        end
        for t in 1:CH
            c = c0 + Int32(t)
            c <= C  && (@inbounds dk[Ca, c, b] = acck[t])
            c <= Cv && (@inbounds dv[Ca, c, b] = accv[t])
        end
        c0 += Int32(CH)
    end
    return nothing
end

function circulant_flash_attention_bwd_kernel!(
        dq, dk, dv, simfun::AbstractSimilarity, q, k, v, Δ, lse, δ,
        nrows::Int32, K::Int32, C::Int32, Cv::Int32,
        spatdims, CartInd, W::Int32, maxidx::Int32,
        chunk::Val{CH},
    ) where CH
    tid    = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    while tid <= maxidx
        a = (tid - 1i32) % nrows + 1i32
        b = (tid - 1i32) ÷ nrows + 1i32
        _flash_attention_bwd_row!(dq, dk, dv, simfun, q, k, v, Δ, lse, δ, a, b, K, C, Cv, spatdims, CartInd, W, chunk)
        tid += stride
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
    CartInd  = CartesianIndices(spatdims)
    nrows    = Int32(prod(spatdims))
    K        = Int32(W)^Int32(N-2)
    maxidx   = nrows * Int32(size(q, N))
    return spatdims, CartInd, nrows, K, maxidx
end

function _circulant_flash_attention_fwd(
        simfun::AbstractSimilarity,
        q::AnyCuArray{Tq,N}, k::AnyCuArray{Tk,N}, v::AnyCuArray{Tv,N},
        W::Int,
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

    spatdims, CartInd, nrows, K, maxidx = _flash_launch_dims(q, W)
    C   = Int32(size(q, N-1))
    Cv  = Int32(size(v, N-1))
    lse = similar(q, Ts, (Int(nrows), size(q, N)))
    chunk = _flash_chunk(Tacc)

    args = (y, lse, simfun, q, k, v, nrows, K, C, Cv, spatdims, CartInd, Int32(W), maxidx, chunk)
    kernel = @cuda launch=false circulant_flash_attention_kernel!(args...)
    config = launch_configuration(kernel.fun)
    threads = min(maxidx, config.threads)
    blocks  = cld(maxidx, threads)

    kernel(args...; threads=threads, blocks=blocks)
    return y, lse
end

function _circulant_flash_attention(simfun::AbstractSimilarity, q, k, v, W::Int)
    first(_circulant_flash_attention_fwd(simfun, q, k, v, W))
end

function ∇circulant_flash_attention(
        simfun::AbstractSimilarity,
        Δ::AnyCuArray{TΔ,N}, y, lse,
        q::AnyCuArray{Tq,N}, k::AnyCuArray{Tk,N}, v::AnyCuArray{Tv,N},
        W::Int,
    ) where {TΔ, Tq, Tk, Tv, N}
    Tqk = promote_type(Tq, Tk)
    dq = similar(q, Tqk)
    dk = similar(k, Tqk)
    dv = similar(v, TΔ)

    # δ_r = Re⟨Δ[r,:], y[r,:]⟩ = Σ_i P[r,i] g[r,i] — the softmax-pullback shift
    δ = reshape(sum(real.(Δ .* conj.(y)); dims=N-1), :, size(Δ, N))

    spatdims, CartInd, nrows, K, maxidx = _flash_launch_dims(q, W)
    C  = Int32(size(q, N-1))
    Cv = Int32(size(v, N-1))
    chunk = _flash_chunk(promote_type(Tqk, TΔ))

    args = (dq, dk, dv, simfun, q, k, v, Δ, lse, δ, nrows, K, C, Cv, spatdims, CartInd, Int32(W), maxidx, chunk)
    kernel = @cuda launch=false circulant_flash_attention_bwd_kernel!(args...)
    config = launch_configuration(kernel.fun)
    threads = min(maxidx, config.threads)
    blocks  = cld(maxidx, threads)

    kernel(args...; threads=threads, blocks=blocks)
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
    τ = sqrt(Tv(size(k, N-1)))
    _circulant_flash_attention(simfun, q ./ sqrt(τ), k ./ sqrt(τ), v, W)
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

# ------------------------------------------------------------------
# rrule — fused backward
# ------------------------------------------------------------------

function CRC.rrule(::typeof(_circulant_flash_attention), simfun::AbstractSimilarity, q, k, v, W::Int)
    y, lse = _circulant_flash_attention_fwd(simfun, q, k, v, W)
    project_q, project_k, project_v = CRC.ProjectTo(q), CRC.ProjectTo(k), CRC.ProjectTo(v)
    function flash_attention_pullback(Δ)
        Δy = CRC.unthunk(Δ)
        if Δy isa Zygote.FillArrays.AbstractFill
            Δy = CUDA.fill(convert(eltype(y), Zygote.FillArrays.getindex_value(Δy)), size(y)...)
        end
        ∂q, ∂k, ∂v = ∇circulant_flash_attention(simfun, Δy, y, lse, q, k, v, W)
        return CRC.NoTangent(), CRC.NoTangent(), project_q(∂q), project_k(∂k), project_v(∂v), CRC.NoTangent()
    end
    return y, flash_attention_pullback
end
