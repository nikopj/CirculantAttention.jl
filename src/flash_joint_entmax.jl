# flash_joint_entmax.jl
#
# Fused ("flash") JOINT circulant α-entmax / sparsemax attention over M branches,
# and the guided variant built on top of it.
#
# Joint entmax normalizes over the UNION of all branches' window entries per row
# with a single threshold τ (non-flash `joint_entmax`, src/entmax.jl). Unlike the
# softmax joint — which decomposes via logsumexp and reweights independent
# per-branch outputs — entmax's τ is coupled across branches, so it must be solved
# over the concatenation. Only the forward τ-solve is coupled, though: with the
# joint threshold τ_r, joint normalizer usum_r = Σ_{all branches} u_i (u=p^(2-α)),
# and per-branch ỹ_m = (Σ_κ u v_m)/usum_r, the backward decouples —
#   δ_r = (Σ_all u_i g_i)/usum = Σ_m Re⟨ΔY_m, ỹ_m⟩,
# and given the shared (τ_r, δ_r) each branch's ∂q_m,∂k_m,∂v_m are branch-local,
# so the backward reuses the single-branch entmax bwd kernels (_∇entmax_launch)
# per branch. The only new device code is the multi-branch joint FORWARD.
#
# Branches are an NTuple of descriptors br = (sim, q, k, v, W, scale, C, Cv, K)
# with per-branch (heterogeneous) inner similarity; α is shared. The reducers
# below unroll over branches by tuple recursion (Base.tail) — fully inlined on the
# GPU. A parametrized (κstart, κstride) window stride lets the thread (1,1), warp
# (lane,WS) and block (tid,TB) kernels share the same per-branch reducers.

# ------------------------------------------------------------------
# per-branch window reducers (κ strided by κstart:κstride:K)
# ------------------------------------------------------------------
@inline function _brz(zmax, br, r::Int32, b::Int32, spatdims, αm1, κ0::Int32, κs::Int32)
    sim, q, k, _, W, scale, C, _, K = br
    base = (r - 1i32) * K
    κ = κ0
    while κ <= K
        i, _ = cartesian_circulant(base + κ, spatdims, W)
        zmax = max(zmax, αm1 * scale * simval(sim, q, k, r, i, b, C))
        κ += κs
    end
    return zmax
end

@inline function _bracc(acc, br, t, r::Int32, b::Int32, spatdims, αm1, pe, κ0::Int32, κs::Int32)
    sim, q, k, _, W, scale, C, _, K = br
    base = (r - 1i32) * K
    a0, a1, a2 = acc
    κ = κ0
    while κ <= K
        i, _ = cartesian_circulant(base + κ, spatdims, W)
        d0, d1, d2 = _halley_terms(αm1 * scale * simval(sim, q, k, r, i, b, C) - t, pe)
        a0 += d0; a1 += d1; a2 += d2
        κ += κs
    end
    return (a0, a1, a2)
end

@inline function _brusum(usum, br, τ, r::Int32, b::Int32, spatdims, αm1, pe, κ0::Int32, κs::Int32)
    sim, q, k, _, W, scale, C, _, K = br
    base = (r - 1i32) * K
    κ = κ0
    while κ <= K
        i, _ = cartesian_circulant(base + κ, spatdims, W)
        _, u = _entmax_weights(αm1 * scale * simval(sim, q, k, r, i, b, C) - τ, pe)
        usum += u
        κ += κs
    end
    return usum
end

# joint reductions: recurse over the branch tuple, accumulating across branches
@inline _joint_zmax(zmax, ::Tuple{}, r, b, spatdims, αm1, κ0, κs) = zmax
@inline function _joint_zmax(zmax, brs::Tuple, r, b, spatdims, αm1, κ0, κs)
    zmax = _brz(zmax, first(brs), r, b, spatdims, αm1, κ0, κs)
    _joint_zmax(zmax, Base.tail(brs), r, b, spatdims, αm1, κ0, κs)
end

@inline _joint_acc(acc, ::Tuple{}, t, r, b, spatdims, αm1, pe, κ0, κs) = acc
@inline function _joint_acc(acc, brs::Tuple, t, r, b, spatdims, αm1, pe, κ0, κs)
    acc = _bracc(acc, first(brs), t, r, b, spatdims, αm1, pe, κ0, κs)
    _joint_acc(acc, Base.tail(brs), t, r, b, spatdims, αm1, pe, κ0, κs)
end

@inline _joint_usum(usum, ::Tuple{}, τ, r, b, spatdims, αm1, pe, κ0, κs) = usum
@inline function _joint_usum(usum, brs::Tuple, τ, r, b, spatdims, αm1, pe, κ0, κs)
    usum = _brusum(usum, first(brs), τ, r, b, spatdims, αm1, pe, κ0, κs)
    _joint_usum(usum, Base.tail(brs), τ, r, b, spatdims, αm1, pe, κ0, κs)
end

# The joint Halley τ-solve is inlined into each of the three forward kernels
# below (thread / warp / block) rather than abstracted behind a reduction
# callable — passing a capturing closure (submask/scratch) plus the max/+ op as a
# runtime value defeats GPU inlining and allocates. Only the per-branch PARTIAL
# reducers (_joint_zmax/_joint_acc/_joint_usum) are shared; the group reduction
# (identity / _warp_reduce / _block_reduce) is written out per variant.

# ------------------------------------------------------------------
# forward — thread-per-row (generic / CPU-testable reference)
# ------------------------------------------------------------------
@inline _joint_apply_thread!(::Tuple{}, ::Tuple{}, ::Tuple{}, τ, uinv, r::Int32, b::Int32, spatdims, αm1, pe, ::Val{CH}) where CH = nothing
@inline function _joint_apply_thread!(brs::Tuple, ys::Tuple, yts::Tuple, τ, uinv, r::Int32, b::Int32, spatdims, αm1, pe, ::Val{CH}) where CH
    br = first(brs); y_m = first(ys); yt_m = first(yts)
    sim, q, k, v, W, scale, C, Cv, K = br
    Tacc = eltype(y_m)
    base = (r - 1i32) * K
    c0 = 0i32
    while c0 < Cv
        accy = ntuple(_ -> zero(Tacc), Val(CH))
        acct = ntuple(_ -> zero(Tacc), Val(CH))
        for κ in 1i32:K
            i, _ = cartesian_circulant(base + κ, spatdims, W)
            p, u = _entmax_weights(αm1 * scale * simval(sim, q, k, r, i, b, C) - τ, pe)
            accy = _chunk_muladd(accy, p, v, i, b, c0, Cv)
            acct = _chunk_muladd(acct, u * uinv, v, i, b, c0, Cv)
        end
        for t_ in 1:CH
            c = c0 + Int32(t_)
            if c <= Cv
                @inbounds y_m[r, c, b]  = accy[t_]
                @inbounds yt_m[r, c, b] = acct[t_]
            end
        end
        c0 += Int32(CH)
    end
    _joint_apply_thread!(Base.tail(brs), Base.tail(ys), Base.tail(yts), τ, uinv, r, b, spatdims, αm1, pe, Val(CH))
end

@inline function _flash_joint_entmax_row!(brs, ys, yts, tau, r::Int32, b::Int32, spatdims, αm1::T, pe::T, chunk::Val{CH}) where {T, CH}
    zmax = _joint_zmax(typemin(T), brs, r, b, spatdims, αm1, 1i32, 1i32)
    t = zmax - T(0.5); tlo = zmax - one(T); thi = zmax
    for _ in 1:_FLASH_ENTMAX_NITER
        a0, a1, a2 = _joint_acc((zero(T), zero(T), zero(T)), brs, t, r, b, spatdims, αm1, pe, 1i32, 1i32)
        t, tlo, thi = _halley_step(a0, a1, a2, t, tlo, thi, pe)
    end
    τ = t
    usum = _joint_usum(zero(T), brs, τ, r, b, spatdims, αm1, pe, 1i32, 1i32)
    @inbounds tau[r, b] = τ
    _joint_apply_thread!(brs, ys, yts, τ, inv(usum), r, b, spatdims, αm1, pe, chunk)
    return nothing
end

function circulant_flash_joint_entmax_kernel!(brs, ys, yts, tau, nrows::Int32, spatdims, αm1, pe, maxidx::Int32, chunk::Val{CH}) where CH
    tid    = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while tid <= maxidx
        r = (tid - 1i32) % nrows + 1i32
        b = (tid - 1i32) ÷ nrows + 1i32
        _flash_joint_entmax_row!(brs, ys, yts, tau, r, b, spatdims, αm1, pe, chunk)
        tid += stride
    end
    return nothing
end

# ------------------------------------------------------------------
# forward — warp-cooperative (WS lanes per group; joint reduce = warp butterfly)
# entries are streamed lane-strided per branch and recomputed each pass — no
# per-branch register caching, which keeps heterogeneous branches tractable.
# ------------------------------------------------------------------
@inline _joint_apply_warp!(::Tuple{}, ::Tuple{}, ::Tuple{}, τ, uinv, r::Int32, b::Int32, lane::Int32, submask::UInt32, spatdims, αm1, pe, ::Val{WS}) where WS = nothing
@inline function _joint_apply_warp!(brs::Tuple, ys::Tuple, yts::Tuple, τ, uinv, r::Int32, b::Int32, lane::Int32, submask::UInt32, spatdims, αm1, pe, ::Val{WS}) where WS
    br = first(brs); y_m = first(ys); yt_m = first(yts)
    sim, q, k, v, W, scale, C, Cv, K = br
    Tacc = eltype(y_m)
    base = (r - 1i32) * K
    c = 1i32
    while c <= Cv
        py = zero(Tacc); pt = zero(Tacc)
        κ = lane
        while κ <= K
            i, _ = cartesian_circulant(base + κ, spatdims, W)
            p, u = _entmax_weights(αm1 * scale * simval(sim, q, k, r, i, b, C) - τ, pe)
            vv = @inbounds v[i, c, b]
            py = muladd(p, vv, py)
            pt = muladd(u * uinv, vv, pt)
            κ += Int32(WS)
        end
        py = _warp_reduce(+, py, submask, Val(WS))
        pt = _warp_reduce(+, pt, submask, Val(WS))
        if lane == 1i32
            @inbounds y_m[r, c, b]  = py
            @inbounds yt_m[r, c, b] = pt
        end
        c += 1i32
    end
    _joint_apply_warp!(Base.tail(brs), Base.tail(ys), Base.tail(yts), τ, uinv, r, b, lane, submask, spatdims, αm1, pe, Val(WS))
end

@inline function _flash_joint_entmax_warp_group!(brs, ys, yts, tau, r::Int32, b::Int32, lane::Int32, submask::UInt32, spatdims, αm1::T, pe::T, ::Val{WS}) where {T, WS}
    zmax = _warp_reduce(max, _joint_zmax(typemin(T), brs, r, b, spatdims, αm1, lane, Int32(WS)), submask, Val(WS))
    t = zmax - T(0.5); tlo = zmax - one(T); thi = zmax
    for _ in 1:_FLASH_ENTMAX_NITER
        a0l, a1l, a2l = _joint_acc((zero(T), zero(T), zero(T)), brs, t, r, b, spatdims, αm1, pe, lane, Int32(WS))
        a0 = _warp_reduce(+, a0l, submask, Val(WS))
        a1 = _warp_reduce(+, a1l, submask, Val(WS))
        a2 = _warp_reduce(+, a2l, submask, Val(WS))
        t, tlo, thi = _halley_step(a0, a1, a2, t, tlo, thi, pe)
    end
    τ = t
    usum = _warp_reduce(+, _joint_usum(zero(T), brs, τ, r, b, spatdims, αm1, pe, lane, Int32(WS)), submask, Val(WS))
    if lane == 1i32
        @inbounds tau[r, b] = τ
    end
    _joint_apply_warp!(brs, ys, yts, τ, inv(usum), r, b, lane, submask, spatdims, αm1, pe, Val(WS))
    return nothing
end

function circulant_flash_joint_entmax_warp_kernel!(brs, ys, yts, tau, nrows::Int32, spatdims, αm1, pe, ngroups::Int32, ws::Val{WS}) where WS
    tid     = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    lane    = (tid - 1i32) % Int32(WS) + 1i32
    g       = (tid - 1i32) ÷ Int32(WS) + 1i32
    gstride = (gridDim().x * blockDim().x) ÷ Int32(WS)
    hwlane  = (threadIdx().x - 1i32) % 32i32
    submask = WS == 32 ? 0xffffffff : ((UInt32(1) << WS) - UInt32(1)) << ((hwlane ÷ Int32(WS)) * Int32(WS))
    while g <= ngroups
        r = (g - 1i32) % nrows + 1i32
        b = (g - 1i32) ÷ nrows + 1i32
        _flash_joint_entmax_warp_group!(brs, ys, yts, tau, r, b, lane, submask, spatdims, αm1, pe, ws)
        g += gstride
    end
    return nothing
end

# ------------------------------------------------------------------
# forward — block-per-row (TB threads per group; joint reduce = block reduction)
# ------------------------------------------------------------------
@inline _joint_apply_block!(::Tuple{}, ::Tuple{}, ::Tuple{}, τ, uinv, r::Int32, b::Int32, lane::Int32, wid::Int32, nwarps::Int32, spatdims, αm1, pe) = nothing
@inline function _joint_apply_block!(brs::Tuple, ys::Tuple, yts::Tuple, τ, uinv, r::Int32, b::Int32, lane::Int32, wid::Int32, nwarps::Int32, spatdims, αm1, pe)
    br = first(brs); y_m = first(ys); yt_m = first(yts)
    sim, q, k, v, W, scale, C, Cv, K = br
    Tacc = eltype(y_m)
    base = (r - 1i32) * K
    c = wid
    while c <= Cv
        py = zero(Tacc); pt = zero(Tacc)
        κ = lane
        while κ <= K
            i, _ = cartesian_circulant(base + κ, spatdims, W)
            p, u = _entmax_weights(αm1 * scale * simval(sim, q, k, r, i, b, C) - τ, pe)
            vv = @inbounds v[i, c, b]
            py = muladd(p, vv, py)
            pt = muladd(u * uinv, vv, pt)
            κ += 32i32
        end
        py = _warp_reduce(+, py, 0xffffffff, Val(32))
        pt = _warp_reduce(+, pt, 0xffffffff, Val(32))
        if lane == 1i32
            @inbounds y_m[r, c, b]  = py
            @inbounds yt_m[r, c, b] = pt
        end
        c += nwarps
    end
    _joint_apply_block!(Base.tail(brs), Base.tail(ys), Base.tail(yts), τ, uinv, r, b, lane, wid, nwarps, spatdims, αm1, pe)
end

function circulant_flash_joint_entmax_block_kernel!(brs, ys, yts, tau, nrows::Int32, spatdims, αm1::T, pe::T, ngroups::Int32) where T
    scratch = CuStaticSharedArray(T, 32)
    tid    = threadIdx().x
    TB     = blockDim().x
    lane   = (tid - 1i32) % 32i32 + 1i32
    wid    = (tid - 1i32) ÷ 32i32 + 1i32
    nwarps = TB ÷ 32i32
    g = blockIdx().x
    while g <= ngroups
        r = (g - 1i32) % nrows + 1i32
        b = (g - 1i32) ÷ nrows + 1i32
        zmax = _block_reduce(max, _joint_zmax(typemin(T), brs, r, b, spatdims, αm1, tid, TB), typemin(T), scratch)
        t = zmax - T(0.5); tlo = zmax - one(T); thi = zmax
        for _ in 1:_FLASH_ENTMAX_NITER
            a0l, a1l, a2l = _joint_acc((zero(T), zero(T), zero(T)), brs, t, r, b, spatdims, αm1, pe, tid, TB)
            a0 = _block_reduce(+, a0l, zero(T), scratch)
            a1 = _block_reduce(+, a1l, zero(T), scratch)
            a2 = _block_reduce(+, a2l, zero(T), scratch)
            t, tlo, thi = _halley_step(a0, a1, a2, t, tlo, thi, pe)
        end
        τ = t
        usum = _block_reduce(+, _joint_usum(zero(T), brs, τ, r, b, spatdims, αm1, pe, tid, TB), zero(T), scratch)
        if tid == 1i32
            @inbounds tau[r, b] = τ
        end
        sync_threads()
        _joint_apply_block!(brs, ys, yts, τ, inv(usum), r, b, lane, wid, nwarps, spatdims, αm1, pe)
        sync_threads()
        g += gridDim().x
    end
    return nothing
end

# ------------------------------------------------------------------
# host-side launchers (inputs already scaled per branch)
# ------------------------------------------------------------------
function _circulant_flash_joint_entmax_fwd(
        sims::Tuple, α::Real, qs::Tuple, ks::Tuple, vs::Tuple, Ws::Tuple, scales::Tuple;
        mode::Symbol=:auto,
    )
    N  = ndims(qs[1])
    Ts = simval_dtype(sims[1], eltype(qs[1]), eltype(ks[1]))
    Ts <: Real || throw(ArgumentError(
        "joint flash entmax needs real-valued similarities; got $Ts. Use RealDotSimilarity()."))
    spatdims, nrows, _, maxidx = _flash_launch_dims(qs[1], Ws[1])   # spatial + batch shared
    αm1 = Ts(α) - one(Ts); pe = inv(αm1)
    tau = similar(qs[1], Ts, (Int(nrows), size(qs[1], N)))

    r3(x) = reshape(x, :, size(x, N-1), size(x, N))
    ys  = map(v -> similar(v, promote_type(Ts, eltype(v))), vs)
    yts = map(v -> similar(v, promote_type(Ts, eltype(v))), vs)
    brs = map(sims, qs, ks, vs, Ws, scales) do sim, q, k, v, W, sc
        (sim, r3(q), r3(k), r3(v), Int32(W), Ts(sc),
         Int32(size(q, N-1)), Int32(size(v, N-1)), Int32(W)^Int32(N-2))
    end
    ysr  = map(r3, ys)
    ytsr = map(r3, yts)

    Ktot = sum(Int(Int32(W)^Int32(N-2)) for W in Ws)
    WS, NE = _flash_warp_dims(Int32(Ktot))
    usemode = _flash_mode(mode, NE, 0)          # block staging is recompute-based: no dynamic shmem

    if usemode === :warp
        args = (brs, ysr, ytsr, tau, nrows, spatdims, αm1, pe, maxidx, Val(WS))
        _flash_warp_launch(circulant_flash_joint_entmax_warp_kernel!, args, maxidx, WS)
    elseif usemode === :block
        args = (brs, ysr, ytsr, tau, nrows, spatdims, αm1, pe, maxidx)
        _flash_block_launch(circulant_flash_joint_entmax_block_kernel!, args, maxidx, 0)
    else
        Tacc = promote_type(Ts, mapreduce(eltype, promote_type, vs))
        args = (brs, ysr, ytsr, tau, nrows, spatdims, αm1, pe, maxidx, _flash_chunk(Tacc))
        kernel = @cuda launch=false circulant_flash_joint_entmax_kernel!(args...)
        config = launch_configuration(kernel.fun)
        threads = min(maxidx, config.threads)
        blocks  = cld(maxidx, threads)
        kernel(args...; threads=threads, blocks=blocks)
    end
    return ys, tau, yts
end

_circulant_flash_joint_entmax(sims::Tuple, α::Real, qs, ks, vs, Ws, scales) =
    first(_circulant_flash_joint_entmax_fwd(sims, α, qs, ks, vs, Ws, scales))

function ∇circulant_flash_joint_entmax(sims::Tuple, α::Real, Δs::Tuple, ys, tau, yts, qs, ks, vs, Ws, scales; mode::Symbol=:auto)
    # joint normalizer δ_r = Σ_m Re⟨ΔY_m, ỹ_m⟩ (the entmax analog of softmax's
    # per-row δ, summed over branches). Each branch's backward is then the
    # single-branch entmax bwd with the shared joint (τ, δ).
    δ = mapreduce(((Δ, yt),) -> _entmax_delta(Δ, yt), (a, b) -> a .+ b, zip(Δs, yts))
    grads = map(sims, Δs, qs, ks, vs, Ws, scales) do sim, Δ, q, k, v, W, sc
        _∇entmax_launch(sim, α, Δ, tau, δ, q, k, v, W, sc; mode)
    end
    return map(g -> g[1], grads), map(g -> g[2], grads), map(g -> g[3], grads)
end

# ------------------------------------------------------------------
# rrule — tuple output, per-branch cotangents (α not differentiated)
# ------------------------------------------------------------------
function CRC.rrule(::typeof(_circulant_flash_joint_entmax), sims::Tuple, α::Real, qs, ks, vs, Ws, scales)
    ys, tau, yts = _circulant_flash_joint_entmax_fwd(sims, α, qs, ks, vs, Ws, scales)
    pq = map(CRC.ProjectTo, qs); pk = map(CRC.ProjectTo, ks); pv = map(CRC.ProjectTo, vs)
    function joint_entmax_pullback(Δ)
        Δu = CRC.unthunk(Δ)
        Δs = map((y, m) -> _flash_materialize(Δu isa CRC.AbstractZero ? nothing : Δu[m], y),
                 ys, ntuple(identity, length(ys)))
        dqs, dks, dvs = ∇circulant_flash_joint_entmax(sims, α, Δs, ys, tau, yts, qs, ks, vs, Ws, scales)
        return (CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(),
                map((p, g) -> p(g), pq, dqs), map((p, g) -> p(g), pk, dks), map((p, g) -> p(g), pv, dvs),
                CRC.NoTangent(), CRC.NoTangent())
    end
    return ys, joint_entmax_pullback
end

# ------------------------------------------------------------------
# public API — joint / multi-head-joint / guided for entmax & sparsemax
# ------------------------------------------------------------------
_joint_entmax_alpha(sfs::NTuple{M,<:EntmaxSimilarity}) where M = sfs[1].α
_joint_entmax_alpha(sfs::NTuple{M,<:SparsemaxSimilarity}) where M = 2

@doc raw"""
    ys = circulant_flash_joint_attention(sfs::NTuple{M,<:EntmaxSimilarity}, qs, ks, vs, Ws)
    ys = circulant_flash_joint_attention(sf::EntmaxSimilarity, q, k, v, Ws::NTuple)

Fused (flash) **joint** α-entmax / sparsemax attention: the normalization is a
*single* α-entmax over the union of all branches' window entries per row (as in
[`joint_entmax`](@ref) / [`joint_sparsemax`](@ref)), computed without
materializing any attention matrix. Returns a tuple of per-branch outputs. All
branches share α (`sfs[1].α`; α=2 for `SparsemaxSimilarity`); inner similarities
may differ per branch. Gradients w.r.t. `qs,ks,vs` are exact.

See also [`circulant_flash_attention`](@ref), [`joint_entmax`](@ref).
"""
function circulant_flash_joint_attention(sfs::NTuple{M,<:Union{EntmaxSimilarity,SparsemaxSimilarity}}, qs::NTuple{M}, ks::NTuple{M}, vs::NTuple{M}, Ws::NTuple{M,Int}) where M
    α = _joint_entmax_alpha(sfs)
    sims = map(s -> s.sim, sfs)
    if sfs isa NTuple{M,<:EntmaxSimilarity}
        all(s -> s.α == α, sfs) || throw(ArgumentError("joint entmax requires a shared α across branches"))
        abs(Float32(α) - 1f0) < _ENTMAX_EPS && return circulant_flash_joint_attention(sims, qs, ks, vs, Ws)  # softmax joint
    end
    scales = map(k -> inv(sqrt(real(eltype(k))(size(k, ndims(k) - 1)))), ks)
    _circulant_flash_joint_entmax(sims, α, qs, ks, vs, Ws, scales)
end

function circulant_flash_joint_attention(sf::Union{EntmaxSimilarity,SparsemaxSimilarity}, q::AbstractArray, k::AbstractArray, v::AbstractArray, Ws::NTuple{M,Int}) where M
    circulant_flash_joint_attention(ntuple(_ -> sf, Val(M)), ntuple(_ -> q, Val(M)), ntuple(_ -> k, Val(M)), ntuple(_ -> v, Val(M)), Ws)
end

@doc raw"""
    ys = circulant_mh_flash_joint_attention(sfs::NTuple{M,<:EntmaxSimilarity}, qs, ks, vs, Ws, nheads)

Multi-head [`circulant_flash_joint_attention`](@ref) for α-entmax / sparsemax:
the joint normalization is applied per head over the union of the branches'
window entries (heads folded into the batch dimension).
"""
function circulant_mh_flash_joint_attention(sfs::NTuple{M,<:Union{EntmaxSimilarity,SparsemaxSimilarity}}, qs::NTuple{M}, ks::NTuple{M}, vs::NTuple{M}, Ws::NTuple{M,Int}, nheads::Int) where M
    qrs = map(q -> splitheads(q, nheads), qs)
    krs = map(k -> splitheads(k, nheads), ks)
    vrs = map(v -> splitheads(v, nheads), vs)
    yrs = circulant_flash_joint_attention(sfs, qrs, krs, vrs, Ws)
    return map((y, v) -> reshape(y, size(v)...), yrs, vs)
end

function circulant_mh_flash_joint_attention(sf::Union{EntmaxSimilarity,SparsemaxSimilarity}, qs::NTuple{M}, ks::NTuple{M}, vs::NTuple{M}, Ws::NTuple{M,Int}, nheads::Int) where M
    circulant_mh_flash_joint_attention(ntuple(_ -> sf, Val(M)), qs, ks, vs, Ws, nheads)
end

@doc raw"""
    ξz, ξg = circulant_mh_flash_guided_joint_attention(sf::EntmaxSimilarity, qz, kz, vz, Wz,
                                                       kg, vg, Wg, num_guides, nheads)

Guided joint α-entmax / sparsemax flash attention: a self branch plus
`num_guides` guide branches (sharing the self query `qz`, keys/values stacked
guide-fastest in the batch dim of `kg`/`vg`), jointly normalized per row. Returns
the self output and the summed guide output — i.e. `circulant_mh_flash_joint_attention`
over `(self, g₁…g_G)` with the guide outputs summed.
"""
function circulant_mh_flash_guided_joint_attention(sf::Union{EntmaxSimilarity,SparsemaxSimilarity},
        qz::AbstractArray{Tq,N}, kz, vz, Wz::Int, kg, vg, Wg::Int, num_guides::Int, nheads::Int) where {Tq, N}
    sp = size(qz)[1:N-2]; d = size(qz, N-1); B = size(qz, N)
    kgr = reshape(kg, sp..., d, num_guides, B)
    vgr = reshape(vg, sp..., d, num_guides, B)
    gslice(x5, g) = x5[ntuple(_ -> Colon(), N-2)..., :, g, :]
    qs = (qz, ntuple(_ -> qz, num_guides)...)
    ks = (kz, ntuple(g -> gslice(kgr, g), num_guides)...)
    vs = (vz, ntuple(g -> gslice(vgr, g), num_guides)...)
    Ws = (Wz, ntuple(_ -> Wg, num_guides)...)
    ys = circulant_mh_flash_joint_attention(ntuple(_ -> sf, num_guides + 1), qs, ks, vs, Ws, nheads)
    return ys[1], reduce(+, ys[2:end])
end
