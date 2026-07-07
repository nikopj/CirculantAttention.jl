# flash_entmax.jl
#
# Fused ("flash") circulant α-entmax / sparsemax attention, forward and backward,
# following AdaSplash (deep-spin/adasplash, arXiv:2502.12082).
#
# α-entmax normalizes a row's window scores by a threshold τ solved from the
# constraint Σ p = 1, so — unlike softmax — it cannot be streamed with a running
# max/normalizer. AdaSplash's insight is that τ can be found by a *fixed*-iteration
# root-finder over the window scores inside the flash loop. The circulant kernels
# already hold each row's window (registers for the warp kernel, shared memory for
# the block kernel), so the only change vs the softmax flash is iterating a Σ p(τ)
# reduction instead of keeping a single online normalizer.
#
# Working in z = (α−1)·X space with X = scale·s the (scaled) similarity, and
# pe = 1/(α−1):
#
#   forward:  p_i = max(z_i − τ, 0)^pe,   Σ_i p_i = 1,
#             u_i = max(z_i − τ, 0)^(pe−1) = p_i^(2−α)          (aux weight)
#   τ solver: Halley step clamped to a bisection bracket (triton_entmax.py):
#             t₀ = zmax − ½,  [t_lo, t_hi] = [zmax − 1, zmax];  each of N_ITER iters
#             accumulates acc_j = Σ_{z>t}(z−t)^(pe−j), j=0,1,2 and takes
#             t ← t − 2·ff·df / (2·df² − ff·ddf) (ff=acc₀−1, df=−pe·acc₁,
#             ddf=pe(pe−1)acc₂), falling back to the bracket midpoint if the Halley
#             step leaves [t_lo, t_hi]. Cubic convergence → ~10 iters (vs ~50 for
#             plain bisection). Converges to the same root as _entmax_threshold.
#
# The forward saves τ_r and the normalized u-weighted output
#   ỹ_r = (Σ_i u_i v_i) / (Σ_i u_i)
# so the backward's per-row normalizer δ_r is a dense host reduction
#   δ_r = (Σ_i u_i g_i)/(Σ_i u_i) = Re⟨Δ_r, ỹ_r⟩,   g_i = Re⟨Δ_r, v_i⟩,
# exactly like the softmax flash's δ_r = Re⟨Δ_r, y_r⟩ — no separate kernel.
#
# The score cotangent is (triton_entmax.py: `grad = u·(Dy − scalar)`,
# `scalar = Σ(u·Dy)/Σu`; identical to the in-repo _entmax_pullback_z):
#   ds_{rj} = scale · u_{rj} · (g_{rj} − δ_r)
# — the softmax bwd's ds = scale·P·(g−δ) with P → u = p^(2−α). ∂q,∂k reuse the same
# simgrad_aux machinery; ∂v_a = Σ_r p_{ra} Δ_r uses the true weight p (not u).
#
# Sparsemax is α=2: pe=1 (p = max(X−τ,0) linear), u = p^0 = 1, δ = mean over the
# support — the same kernels with α=2.
#
# α is a fixed scalar hyperparameter and is NOT differentiated, matching the
# non-flash EntmaxSimilarity adjacency rrule (src/entmax.jl).

# AdaSplash's default; Halley converges cubically so this is ample for Float32.
const _FLASH_ENTMAX_NITER = 10

# ------------------------------------------------------------------
# device-side entmax helpers
# ------------------------------------------------------------------

# positive-base power d^e via exp2/log2 — the device-mapped form AdaSplash uses
# (triton_entmax.py: tl.exp2(e * tl.log2(·))); maps to __nv_exp2f/__nv_log2f on
# GPU and Base.exp2/log2 on CPU. Callers guard d > 0.
@inline _dpow(d::T, e::T) where T = exp2(e * log2(d))

# (p, u) = (max(d,0)^pe, max(d,0)^(pe-1)) with d = z - τ; u = p^(2-α).
@inline function _entmax_weights(d::T, pe::T) where T
    if d > zero(T)
        return _dpow(d, pe), _dpow(d, pe - one(T))
    else
        return zero(T), zero(T)
    end
end

# Halley accumulation terms acc_j = (z-t)^(pe-j) for j=0,1,2 (0 when z<=t).
@inline function _halley_terms(d::T, pe::T) where T
    if d > zero(T)
        return _dpow(d, pe), _dpow(d, pe - one(T)), _dpow(d, pe - T(2))
    else
        return zero(T), zero(T), zero(T)
    end
end

# One Halley step clamped to the bisection bracket [tlo, thi]; returns the new
# (t, tlo, thi). ff = Σp - 1, df = -pe·acc1, ddf = pe(pe-1)·acc2. A NaN/Inf Halley
# step (e.g. denom≈0) fails the bracket test and falls back to the midpoint, so
# the solver degrades to robust bisection.
@inline function _halley_step(acc0::T, acc1::T, acc2::T, t::T, tlo::T, thi::T, pe::T) where T
    ff    = acc0 - one(T)
    df    = -pe * acc1
    ddf   = pe * (pe - one(T)) * acc2
    new_t = t - (T(2) * ff * df) / (T(2) * df * df - ff * ddf)
    tlo   = ff > zero(T) ? t : tlo
    thi   = ff < zero(T) ? t : thi
    eps   = T(1f-6)
    good  = (new_t > tlo - eps) & (new_t < thi + eps)
    t     = good ? new_t : T(0.5) * (tlo + thi)
    return t, tlo, thi
end

# ------------------------------------------------------------------
# forward — thread-per-row (generic / CPU-testable reference)
# ------------------------------------------------------------------
@inline function _flash_entmax_row!(
        y, tau, ytil, simfun::AbstractSimilarity, q, k, v,
        r::Int32, b::Int32,
        K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, αm1, pe, ::Val{CH},
    ) where CH
    Ts   = simval_dtype(simfun, eltype(q), eltype(k))
    Tacc = promote_type(Ts, eltype(v))
    base = (r - 1i32) * K

    # pass 1: max of z = (α-1)·scale·s
    zmax = typemin(Ts)
    for κ in 1i32:K
        i, _ = cartesian_circulant(base + κ, spatdims, W)
        z = αm1 * scale * simval(simfun, q, k, r, i, b, C)
        zmax = max(zmax, z)
    end

    # τ solver: Halley clamped to bisection bracket, recomputing the window each iter
    t   = zmax - Ts(0.5)
    tlo = zmax - one(Ts)
    thi = zmax
    for _ in 1:_FLASH_ENTMAX_NITER
        a0 = zero(Ts); a1 = zero(Ts); a2 = zero(Ts)
        for κ in 1i32:K
            i, _ = cartesian_circulant(base + κ, spatdims, W)
            z = αm1 * scale * simval(simfun, q, k, r, i, b, C)
            d0, d1, d2 = _halley_terms(z - t, pe)
            a0 += d0; a1 += d1; a2 += d2
        end
        t, tlo, thi = _halley_step(a0, a1, a2, t, tlo, thi, pe)
    end
    τ = t
    @inbounds tau[r, b] = τ

    # usum = Σ u (normalizer for ỹ)
    usum = zero(Ts)
    for κ in 1i32:K
        i, _ = cartesian_circulant(base + κ, spatdims, W)
        z = αm1 * scale * simval(simfun, q, k, r, i, b, C)
        _, u = _entmax_weights(z - τ, pe)
        usum += u
    end
    uinv = inv(usum)

    # pass 2: y = Σ p v and ỹ = (Σ u v)/usum, CH channels at a time
    c0 = 0i32
    while c0 < Cv
        accy = ntuple(_ -> zero(Tacc), Val(CH))
        acct = ntuple(_ -> zero(Tacc), Val(CH))
        for κ in 1i32:K
            i, _ = cartesian_circulant(base + κ, spatdims, W)
            z = αm1 * scale * simval(simfun, q, k, r, i, b, C)
            p, u = _entmax_weights(z - τ, pe)
            accy = _chunk_muladd(accy, p, v, i, b, c0, Cv)
            acct = _chunk_muladd(acct, u * uinv, v, i, b, c0, Cv)
        end
        for t_ in 1:CH
            c = c0 + Int32(t_)
            if c <= Cv
                @inbounds y[r, c, b]    = accy[t_]
                @inbounds ytil[r, c, b] = acct[t_]
            end
        end
        c0 += Int32(CH)
    end
    return nothing
end

function circulant_flash_entmax_kernel!(
        y, tau, ytil, simfun::AbstractSimilarity, q, k, v,
        nrows::Int32, K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, αm1, pe, maxidx::Int32,
        chunk::Val{CH},
    ) where CH
    tid    = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while tid <= maxidx
        r = (tid - 1i32) % nrows + 1i32
        b = (tid - 1i32) ÷ nrows + 1i32
        _flash_entmax_row!(y, tau, ytil, simfun, q, k, v, r, b, K, C, Cv, spatdims, W, scale, αm1, pe, chunk)
        tid += stride
    end
    return nothing
end

# ------------------------------------------------------------------
# forward — warp-cooperative (window held in lane registers)
# ------------------------------------------------------------------
@inline function _flash_entmax_warp_group!(
        y, tau, ytil, simfun::AbstractSimilarity, q, k, v,
        r::Int32, b::Int32, lane::Int32, submask::UInt32,
        K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, αm1, pe, ::Val{WS}, ::Val{NE},
    ) where {WS, NE}
    Ts   = simval_dtype(simfun, eltype(q), eltype(k))
    Tacc = promote_type(Ts, eltype(v))
    base = (r - 1i32) * K
    neginf = typemin(Ts)

    cols = ntuple(Val(NE)) do e
        κ = lane + Int32(e-1) * Int32(WS)
        κ <= K ? first(cartesian_circulant(base + κ, spatdims, W)) : 1i32
    end
    # z-scores in registers (masked lanes → -Inf so they never enter max/support)
    zvals = ntuple(Val(NE)) do e
        κ = lane + Int32(e-1) * Int32(WS)
        κ <= K ? αm1 * scale * simval(simfun, q, k, r, cols[e], b, C) : neginf
    end

    zmax_lane = neginf
    for e in 1:NE
        zmax_lane = max(zmax_lane, zvals[e])
    end
    zmax = _warp_reduce(max, zmax_lane, submask, Val(WS))

    t   = zmax - Ts(0.5)
    tlo = zmax - one(Ts)
    thi = zmax
    for _ in 1:_FLASH_ENTMAX_NITER
        a0 = zero(Ts); a1 = zero(Ts); a2 = zero(Ts)
        for e in 1:NE
            d0, d1, d2 = _halley_terms(zvals[e] - t, pe)
            a0 += d0; a1 += d1; a2 += d2
        end
        a0 = _warp_reduce(+, a0, submask, Val(WS))
        a1 = _warp_reduce(+, a1, submask, Val(WS))
        a2 = _warp_reduce(+, a2, submask, Val(WS))
        t, tlo, thi = _halley_step(a0, a1, a2, t, tlo, thi, pe)
    end
    τ = t

    usum_lane = zero(Ts)
    for e in 1:NE
        _, u = _entmax_weights(zvals[e] - τ, pe)
        usum_lane += u
    end
    usum = _warp_reduce(+, usum_lane, submask, Val(WS))
    uinv = inv(usum)
    if lane == 1i32
        @inbounds tau[r, b] = τ
    end

    pw = ntuple(e -> _entmax_weights(zvals[e] - τ, pe), Val(NE))  # (p, u) per lane-entry

    c = 1i32
    while c <= Cv
        py = zero(Tacc); pt = zero(Tacc)
        for e in 1:NE
            vv = @inbounds v[cols[e], c, b]
            py = muladd(pw[e][1], vv, py)
            pt = muladd(pw[e][2] * uinv, vv, pt)
        end
        py = _warp_reduce(+, py, submask, Val(WS))
        pt = _warp_reduce(+, pt, submask, Val(WS))
        if lane == 1i32
            @inbounds y[r, c, b]    = py
            @inbounds ytil[r, c, b] = pt
        end
        c += 1i32
    end
    return nothing
end

function circulant_flash_entmax_warp_kernel!(
        y, tau, ytil, simfun::AbstractSimilarity, q, k, v,
        nrows::Int32, K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, αm1, pe, ngroups::Int32,
        ws::Val{WS}, ne::Val{NE},
    ) where {WS, NE}
    tid     = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    lane    = (tid - 1i32) % Int32(WS) + 1i32
    g       = (tid - 1i32) ÷ Int32(WS) + 1i32
    gstride = (gridDim().x * blockDim().x) ÷ Int32(WS)
    hwlane  = (threadIdx().x - 1i32) % 32i32
    submask = WS == 32 ? 0xffffffff : ((UInt32(1) << WS) - UInt32(1)) << ((hwlane ÷ Int32(WS)) * Int32(WS))

    while g <= ngroups
        r = (g - 1i32) % nrows + 1i32
        b = (g - 1i32) ÷ nrows + 1i32
        _flash_entmax_warp_group!(y, tau, ytil, simfun, q, k, v, r, b, lane, submask, K, C, Cv, spatdims, W, scale, αm1, pe, ws, ne)
        g += gstride
    end
    return nothing
end

# ------------------------------------------------------------------
# forward — block-per-row (window staged in shared memory)
# ------------------------------------------------------------------
function circulant_flash_entmax_block_kernel!(
        y, tau, ytil, simfun::AbstractSimilarity, q, k, v,
        nrows::Int32, K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, αm1, pe, ngroups::Int32,
    )
    Ts   = simval_dtype(simfun, eltype(q), eltype(k))
    Tacc = promote_type(Ts, eltype(v))
    z_sh    = CuDynamicSharedArray(Ts, K)               # window z-scores
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

        # stage z-scores, block max
        zloc = neginf
        κ = tid
        while κ <= K
            i, _ = cartesian_circulant(base + κ, spatdims, W)
            z = αm1 * scale * simval(simfun, q, k, r, i, b, C)
            @inbounds z_sh[κ]   = z
            @inbounds col_sh[κ] = i
            zloc = max(zloc, z)
            κ += TB
        end
        zmax = _block_reduce(max, zloc, neginf, scratch)

        # τ solver over the staged scores
        t   = zmax - Ts(0.5)
        tlo = zmax - one(Ts)
        thi = zmax
        for _ in 1:_FLASH_ENTMAX_NITER
            a0 = zero(Ts); a1 = zero(Ts); a2 = zero(Ts)
            κ = tid
            while κ <= K
                d0, d1, d2 = _halley_terms(@inbounds(z_sh[κ]) - t, pe)
                a0 += d0; a1 += d1; a2 += d2
                κ += TB
            end
            a0 = _block_reduce(+, a0, zero(Ts), scratch)
            a1 = _block_reduce(+, a1, zero(Ts), scratch)
            a2 = _block_reduce(+, a2, zero(Ts), scratch)
            t, tlo, thi = _halley_step(a0, a1, a2, t, tlo, thi, pe)
        end
        τ = t

        uloc = zero(Ts)
        κ = tid
        while κ <= K
            _, u = _entmax_weights(@inbounds(z_sh[κ]) - τ, pe)
            uloc += u
            κ += TB
        end
        usum = _block_reduce(+, uloc, zero(Ts), scratch)
        uinv = inv(usum)
        if tid == 1i32
            @inbounds tau[r, b] = τ
        end
        sync_threads()

        # warps over channels, lanes over window entries
        c = wid
        while c <= Cv
            py = zero(Tacc); pt = zero(Tacc)
            κ = lane
            while κ <= K
                p, u = _entmax_weights(@inbounds(z_sh[κ]) - τ, pe)
                vv = @inbounds v[col_sh[κ], c, b]
                py = muladd(p, vv, py)
                pt = muladd(u * uinv, vv, pt)
                κ += 32i32
            end
            py = _warp_reduce(+, py, 0xffffffff, Val(32))
            pt = _warp_reduce(+, pt, 0xffffffff, Val(32))
            if lane == 1i32
                @inbounds y[r, c, b]    = py
                @inbounds ytil[r, c, b] = pt
            end
            c += nwarps
        end
        sync_threads()
        g += gridDim().x
    end
    return nothing
end

# ------------------------------------------------------------------
# backward — thread-per-row (generic / CPU-testable reference)
#
# Structurally the softmax flash backward with the softmax weight P replaced by
# u = p^(2-α) in the ds term, the true weight p in the ∂v accumulation, and the
# saved logsumexp replaced by the saved threshold τ (to recompute p, u) plus the
# host-computed normalizer δ.
# ------------------------------------------------------------------
@inline function _flash_entmax_bwd_row!(
        dq, dk, dv, simfun::AbstractSimilarity, q, k, v, Δ, tau, δ,
        a::Int32, b::Int32,
        K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, αm1, pe, ::Val{CH},
    ) where CH
    Tqk = promote_type(eltype(q), eltype(k))
    TΔ  = eltype(Δ)
    base = (a - 1i32) * K
    τ_a = @inbounds tau[a, b]
    δ_a = @inbounds δ[a, b]

    # sweep 1: a as row — entries (row a, col i) accumulate ∂q[a,:]
    zqk = zero(Tqk)
    zΔ  = zero(TΔ)
    c0 = 0i32
    while c0 < C
        acc = ntuple(_ -> zqk, Val(CH))
        for κ in 1i32:K
            i, _ = cartesian_circulant(base + κ, spatdims, W)
            s, α, β = simgrad_aux(simfun, q, k, a, i, b, C)
            _, u = _entmax_weights(αm1 * scale * s - τ_a, pe)
            ds = u * (_flash_gval(Δ, v, a, i, b, Cv) - δ_a)
            acc = _chunk_simgrad(acc, scale * ds, α, β, k, i, q, a, b, c0, C)
        end
        for t_ in 1:CH
            c = c0 + Int32(t_)
            c <= C && (@inbounds dq[a, c, b] = acc[t_])
        end
        c0 += Int32(CH)
    end

    # sweep 2: a as column — entries (row r, col a) accumulate ∂k[a,:], ∂v[a,:]
    Cmax = max(C, Cv)
    c0 = 0i32
    while c0 < Cmax
        acck = ntuple(_ -> zqk, Val(CH))
        accv = ntuple(_ -> zΔ, Val(CH))
        for κ in 1i32:K
            r, _ = cartesian_circulant(base + κ, spatdims, W)
            s, α, β = simgrad_aux(simfun, q, k, r, a, b, C)
            p, u = _entmax_weights(αm1 * scale * s - @inbounds(tau[r, b]), pe)
            ds = u * (_flash_gval(Δ, v, r, a, b, Cv) - @inbounds(δ[r, b]))
            acck = _chunk_simgrad(acck, scale * ds, conj(α), β, q, r, k, a, b, c0, C)
            accv = _chunk_muladd(accv, p, Δ, r, b, c0, Cv)
        end
        for t_ in 1:CH
            c = c0 + Int32(t_)
            c <= C  && (@inbounds dk[a, c, b] = acck[t_])
            c <= Cv && (@inbounds dv[a, c, b] = accv[t_])
        end
        c0 += Int32(CH)
    end
    return nothing
end

function circulant_flash_entmax_bwd_kernel!(
        dq, dk, dv, simfun::AbstractSimilarity, q, k, v, Δ, tau, δ,
        nrows::Int32, K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, αm1, pe, maxidx::Int32,
        chunk::Val{CH},
    ) where CH
    tid    = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while tid <= maxidx
        a = (tid - 1i32) % nrows + 1i32
        b = (tid - 1i32) ÷ nrows + 1i32
        _flash_entmax_bwd_row!(dq, dk, dv, simfun, q, k, v, Δ, tau, δ, a, b, K, C, Cv, spatdims, W, scale, αm1, pe, chunk)
        tid += stride
    end
    return nothing
end

# ------------------------------------------------------------------
# backward — warp-cooperative
# ------------------------------------------------------------------
@inline function _flash_entmax_bwd_warp_group!(
        dq, dk, dv, simfun::AbstractSimilarity, q, k, v, Δ, tau, δ,
        a::Int32, b::Int32, lane::Int32, submask::UInt32,
        K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, αm1, pe, ::Val{WS}, ::Val{NE},
    ) where {WS, NE}
    Ts  = simval_dtype(simfun, eltype(q), eltype(k))
    Tqk = promote_type(eltype(q), eltype(k))
    TΔ  = eltype(Δ)
    β   = _simgrad_beta(simfun, Tqk)
    base = (a - 1i32) * K
    τ_a = @inbounds tau[a, b]
    δ_a = @inbounds δ[a, b]

    cols = ntuple(Val(NE)) do e
        κ = lane + Int32(e-1) * Int32(WS)
        κ <= K ? first(cartesian_circulant(base + κ, spatdims, W)) : 1i32
    end

    # sweep 1: a as row — ∂q[a,c] = Σ_e ds_e α_e k[i_e,c] + β q[a,c] Σ_e ds_e
    sw1 = ntuple(Val(NE)) do e
        κ = lane + Int32(e-1) * Int32(WS)
        s, α, _ = simgrad_aux(simfun, q, k, a, cols[e], b, C)
        _, u = _entmax_weights(αm1 * scale * s - τ_a, pe)
        ds = scale * (u * (_flash_gval(Δ, v, a, cols[e], b, Cv) - δ_a))
        valid = κ <= K
        (valid ? ds * α : zero(ds * α), valid ? ds : zero(ds))
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

    # sweep 2: a as column — ∂k[a,c], ∂v[a,c]
    sw2 = ntuple(Val(NE)) do e
        κ = lane + Int32(e-1) * Int32(WS)
        s, α, _ = simgrad_aux(simfun, q, k, cols[e], a, b, C)
        pw, u = _entmax_weights(αm1 * scale * s - @inbounds(tau[cols[e], b]), pe)
        ds = scale * (u * (_flash_gval(Δ, v, cols[e], a, b, Cv) - @inbounds(δ[cols[e], b])))
        valid = κ <= K
        (valid ? ds * conj(α) : zero(ds * α), valid ? pw : zero(pw), valid ? ds : zero(ds))
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

function circulant_flash_entmax_bwd_warp_kernel!(
        dq, dk, dv, simfun::AbstractSimilarity, q, k, v, Δ, tau, δ,
        nrows::Int32, K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, αm1, pe, ngroups::Int32,
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
        _flash_entmax_bwd_warp_group!(dq, dk, dv, simfun, q, k, v, Δ, tau, δ, a, b, lane, submask, K, C, Cv, spatdims, W, scale, αm1, pe, ws, ne)
        g += gstride
    end
    return nothing
end

# ------------------------------------------------------------------
# backward — block-per-row (shared-memory staging)
# ------------------------------------------------------------------
function circulant_flash_entmax_bwd_block_kernel!(
        dq, dk, dv, simfun::AbstractSimilarity, q, k, v, Δ, tau, δ,
        nrows::Int32, K::Int32, C::Int32, Cv::Int32,
        spatdims, W::Int32, scale, αm1, pe, ngroups::Int32,
    )
    Tq = eltype(q); Tk = eltype(k); TΔ = eltype(Δ)
    Ts, Tds, Tw = _flash_bwd_wtypes(simfun, Tq, Tk, TΔ, eltype(v))
    Tqk = promote_type(Tq, Tk)
    β   = _simgrad_beta(simfun, Tqk)
    u_sh    = CuDynamicSharedArray(Tw, K)               # ds·α  (sweep1) / ds·conj(α) (sweep2)
    p_sh    = CuDynamicSharedArray(Ts, K, Int(K) * sizeof(Tw))
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
        τ_a = @inbounds tau[a, b]
        δ_a = @inbounds δ[a, b]

        κ = tid
        while κ <= K
            i, _ = cartesian_circulant(base + κ, spatdims, W)
            @inbounds col_sh[κ] = i
            κ += TB
        end
        sync_threads()

        # sweep 1: stage ds·α, then ∂q[a,:]
        dsloc = zero(Tds)
        κ = tid
        while κ <= K
            @inbounds i = col_sh[κ]
            s, α, _ = simgrad_aux(simfun, q, k, a, i, b, C)
            _, u = _entmax_weights(αm1 * scale * s - τ_a, pe)
            ds = scale * (u * (_flash_gval(Δ, v, a, i, b, Cv) - δ_a))
            @inbounds u_sh[κ] = ds * α
            dsloc += ds
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
        sync_threads()

        # sweep 2: stage ds·conj(α) and p, then ∂k, ∂v
        dsloc = zero(Tds)
        κ = tid
        while κ <= K
            r = @inbounds col_sh[κ]
            s, α, _ = simgrad_aux(simfun, q, k, r, a, b, C)
            pw, u = _entmax_weights(αm1 * scale * s - @inbounds(tau[col_sh[κ], b]), pe)
            ds = scale * (u * (_flash_gval(Δ, v, r, a, b, Cv) - @inbounds(δ[col_sh[κ], b])))
            @inbounds u_sh[κ] = ds * conj(α)
            @inbounds p_sh[κ] = pw
            dsloc += ds
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
                    pv = muladd(@inbounds(p_sh[κ]), @inbounds(Δ[Cc, c, b]), pv)
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
function _circulant_flash_entmax_fwd(
        simfun::AbstractSimilarity, α::Real,
        q::AnyCuArray{Tq,N}, k::AnyCuArray{Tk,N}, v::AnyCuArray{Tv,N},
        W::Int, scale::Real=true;
        mode::Symbol=:auto,
    ) where {Tq, Tk, Tv, N}
    Ts = simval_dtype(simfun, Tq, Tk)
    Ts <: Real || throw(ArgumentError(
        "circulant flash entmax needs a real-valued similarity; " *
        "$(typeof(simfun)) on ($Tq, $Tk) gives $Ts. Use RealDotSimilarity() instead."))
    @assert isodd(W) "window length W=$W must be odd"
    size(q) == size(k) || throw(DimensionMismatch("q $(size(q)) and k $(size(k)) must match"))
    size(v)[1:N-2] == size(q)[1:N-2] && size(v, N) == size(q, N) ||
        throw(DimensionMismatch("v $(size(v)) must match q $(size(q)) in spatial and batch dims"))

    Tacc = promote_type(Ts, Tv)
    y    = similar(v, Tacc)
    ytil = similar(v, Tacc)

    spatdims, nrows, K, maxidx = _flash_launch_dims(q, W)
    C   = Int32(size(q, N-1))
    Cv  = Int32(size(v, N-1))
    tau = similar(q, Ts, (Int(nrows), size(q, N)))
    WS, NE = _flash_warp_dims(K)
    shmem  = Int(K) * (sizeof(Ts) + sizeof(Int32))
    usemode = _flash_mode(mode, NE, shmem)

    yr  = reshape(y,    :, size(y, N-1), size(y, N))
    ytr = reshape(ytil, :, size(ytil, N-1), size(ytil, N))
    qr  = reshape(q,    :, size(q, N-1), size(q, N))
    kr  = reshape(k,    :, size(k, N-1), size(k, N))
    vr  = reshape(v,    :, size(v, N-1), size(v, N))

    sc  = Ts(scale)
    αm1 = Ts(α) - one(Ts)
    pe  = inv(αm1)
    if usemode === :warp
        args = (yr, tau, ytr, simfun, qr, kr, vr, nrows, K, C, Cv, spatdims, Int32(W), sc, αm1, pe, maxidx, Val(WS), Val(NE))
        _flash_warp_launch(circulant_flash_entmax_warp_kernel!, args, maxidx, WS)
    elseif usemode === :block
        args = (yr, tau, ytr, simfun, qr, kr, vr, nrows, K, C, Cv, spatdims, Int32(W), sc, αm1, pe, maxidx)
        _flash_block_launch(circulant_flash_entmax_block_kernel!, args, maxidx, shmem)
    else
        chunk = _flash_chunk(Tacc)
        args = (yr, tau, ytr, simfun, qr, kr, vr, nrows, K, C, Cv, spatdims, Int32(W), sc, αm1, pe, maxidx, chunk)
        kernel = @cuda launch=false circulant_flash_entmax_kernel!(args...)
        config = launch_configuration(kernel.fun)
        threads = min(maxidx, config.threads)
        blocks  = cld(maxidx, threads)
        kernel(args...; threads=threads, blocks=blocks)
    end
    return y, tau, ytil
end

function _circulant_flash_entmax(simfun::AbstractSimilarity, α::Real, q, k, v, W::Int, scale::Real=true)
    first(_circulant_flash_entmax_fwd(simfun, α, q, k, v, W, scale))
end

# entmax δ_r = (Σ_i u_i g_i)/(Σ_i u_i) = Re⟨Δ_r, ỹ_r⟩ — dense host reduction over
# the saved u-weighted output (the entmax analog of softmax's δ = Re⟨Δ, y⟩). For
# joint normalization this is summed across branches (see ∇circulant_flash_joint_entmax).
_entmax_delta(Δ::AbstractArray{TΔ,N}, ytil) where {TΔ,N} =
    reshape(sum(real.(Δ .* conj.(ytil)); dims=N-1), :, size(Δ, N))

function ∇circulant_flash_entmax(
        simfun::AbstractSimilarity, α::Real,
        Δ::AnyCuArray{TΔ,N}, y, tau, ytil,
        q::AnyCuArray{Tq,N}, k::AnyCuArray{Tk,N}, v::AnyCuArray{Tv,N},
        W::Int, scale::Real=true;
        mode::Symbol=:auto,
    ) where {TΔ, Tq, Tk, Tv, N}
    _∇entmax_launch(simfun, α, Δ, tau, _entmax_delta(Δ, ytil), q, k, v, W, scale; mode)
end

# Backward kernel launch given the (possibly joint) threshold `tau` and normalizer
# `δ` — both per-row arrays shared across branches in the joint case. This is the
# single-branch entmax backward; the joint backward calls it once per branch.
function _∇entmax_launch(
        simfun::AbstractSimilarity, α::Real,
        Δ::AnyCuArray{TΔ,N}, tau, δ,
        q::AnyCuArray{Tq,N}, k::AnyCuArray{Tk,N}, v::AnyCuArray{Tv,N},
        W::Int, scale::Real=true;
        mode::Symbol=:auto,
    ) where {TΔ, Tq, Tk, Tv, N}
    Tqk = promote_type(Tq, Tk)
    dq = similar(q, Tqk)
    dk = similar(k, Tqk)
    dv = similar(v, TΔ)

    spatdims, nrows, K, maxidx = _flash_launch_dims(q, W)
    C  = Int32(size(q, N-1))
    Cv = Int32(size(v, N-1))
    WS, NE = _flash_warp_dims(K)
    Ts, _, Tw = _flash_bwd_wtypes(simfun, Tq, Tk, TΔ, Tv)
    shmem = Int(K) * (sizeof(Tw) + sizeof(Ts) + sizeof(Int32))
    usemode = _flash_mode(mode, NE, shmem)

    Δr  = reshape(Δ, :, size(Δ, N-1), size(Δ, N))
    qr  = reshape(q, :, size(q, N-1), size(q, N))
    kr  = reshape(k, :, size(k, N-1), size(k, N))
    vr  = reshape(v, :, size(v, N-1), size(v, N))
    dqr = reshape(dq, :, size(dq, N-1), size(dq, N))
    dkr = reshape(dk, :, size(dk, N-1), size(dk, N))
    dvr = reshape(dv, :, size(dv, N-1), size(dv, N))

    sc  = Ts(scale)
    αm1 = Ts(α) - one(Ts)
    pe  = inv(αm1)
    if usemode === :warp
        args = (dqr, dkr, dvr, simfun, qr, kr, vr, Δr, tau, δ, nrows, K, C, Cv, spatdims, Int32(W), sc, αm1, pe, maxidx, Val(WS), Val(NE))
        _flash_warp_launch(circulant_flash_entmax_bwd_warp_kernel!, args, maxidx, WS)
    elseif usemode === :block
        args = (dqr, dkr, dvr, simfun, qr, kr, vr, Δr, tau, δ, nrows, K, C, Cv, spatdims, Int32(W), sc, αm1, pe, maxidx)
        _flash_block_launch(circulant_flash_entmax_bwd_block_kernel!, args, maxidx, shmem)
    else
        chunk = _flash_chunk(promote_type(Tqk, TΔ))
        args = (dqr, dkr, dvr, simfun, qr, kr, vr, Δr, tau, δ, nrows, K, C, Cv, spatdims, Int32(W), sc, αm1, pe, maxidx, chunk)
        kernel = @cuda launch=false circulant_flash_entmax_bwd_kernel!(args...)
        config = launch_configuration(kernel.fun)
        threads = min(maxidx, config.threads)
        blocks  = cld(maxidx, threads)
        kernel(args...; threads=threads, blocks=blocks)
    end
    return dq, dk, dv
end

# ------------------------------------------------------------------
# rrule — fused backward (α is a fixed hyperparameter, not differentiated)
# ------------------------------------------------------------------
function CRC.rrule(::typeof(_circulant_flash_entmax), simfun::AbstractSimilarity, α::Real, q, k, v, W::Int, scale::Real)
    y, tau, ytil = _circulant_flash_entmax_fwd(simfun, α, q, k, v, W, scale)
    project_q, project_k, project_v = CRC.ProjectTo(q), CRC.ProjectTo(k), CRC.ProjectTo(v)
    function flash_entmax_pullback(Δ)
        Δy = _flash_materialize(Δ, y)
        ∂q, ∂k, ∂v = ∇circulant_flash_entmax(simfun, α, Δy, y, tau, ytil, q, k, v, W, scale)
        return (CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(),
                project_q(∂q), project_k(∂k), project_v(∂v), CRC.NoTangent(), CRC.NoTangent())
    end
    return y, flash_entmax_pullback
end

function CRC.rrule(::typeof(_circulant_flash_entmax), simfun::AbstractSimilarity, α::Real, q, k, v, W::Int)
    y, pb8 = CRC.rrule(_circulant_flash_entmax, simfun, α, q, k, v, W, true)
    flash_entmax_pullback7(Δ) = pb8(Δ)[1:7]
    return y, flash_entmax_pullback7
end

# ------------------------------------------------------------------
# public API
# ------------------------------------------------------------------
@doc raw"""
    y = circulant_flash_attention(es::EntmaxSimilarity, q, k, v, W::Int)
    y = circulant_flash_attention(ss::SparsemaxSimilarity, q, k, v, W::Int)

Fused (flash) circulant **α-entmax** / **sparsemax** attention: computes
`y = P v` where `P` is the row-wise α-entmax (resp. sparsemax = α-entmax at α=2)
of the windowed similarities ``S_{ij} = \mathrm{simfun}(q_i, k_j)``, without ever
materializing the circulant-sparse attention matrix. Following AdaSplash
(arXiv:2502.12082), the per-row entmax threshold is solved by a fixed-iteration
Halley/bisection root-finder inside the fused loop; memory traffic is ``O(NC)``
instead of ``O(NW^d)``.

The forward returns only `y`; the backward is fused as well (`q`, `k`, `v`
gradients are exact — `α` is a fixed hyperparameter and is not differentiated, as
in the non-flash [`EntmaxSimilarity`](@ref) path). The inner similarity must be
real-valued (see [`circulant_flash_attention`](@ref)).

See also [`circulant_attention`](@ref), [`EntmaxSimilarity`](@ref),
[`SparsemaxSimilarity`](@ref).
"""
function circulant_flash_attention(es::EntmaxSimilarity, q::AbstractArray{Tq,N}, k::AbstractArray{Tk,N}, v::AbstractArray{Tv,N}, W::Int) where {Tq, Tk, Tv, N}
    # α ≈ 1 is softmax — reuse the (cheaper) softmax flash path.
    abs(Float32(es.α) - 1f0) < _ENTMAX_EPS && return circulant_flash_attention(es.sim, q, k, v, W)
    scale = inv(sqrt(real(Tk)(size(k, N-1))))
    _circulant_flash_entmax(es.sim, es.α, q, k, v, W, scale)
end

function circulant_flash_attention(ss::SparsemaxSimilarity, q::AbstractArray{Tq,N}, k::AbstractArray{Tk,N}, v::AbstractArray{Tv,N}, W::Int) where {Tq, Tk, Tv, N}
    scale = inv(sqrt(real(Tk)(size(k, N-1))))
    _circulant_flash_entmax(ss.sim, 2, q, k, v, W, scale)
end

@doc raw"""
    y = circulant_mh_flash_attention(es::EntmaxSimilarity, q, k, v, W::Int, nheads::Int)
    y = circulant_mh_flash_attention(ss::SparsemaxSimilarity, q, k, v, W::Int, nheads::Int)

Multi-head [`circulant_flash_attention`](@ref) for α-entmax / sparsemax: the
normalization is applied per head over each head's window entries (heads folded
into the batch dimension). The number of channels must be divisible by `nheads`.
"""
function circulant_mh_flash_attention(es::EntmaxSimilarity, q::AbstractArray{Tq,N}, k::AbstractArray{Tk,N}, v::AbstractArray{Tv,N}, W::Int, nheads::Int) where {Tq, Tk, Tv, N}
    qr, kr, vr = splitheads.((q, k, v), nheads)
    yr = circulant_flash_attention(es, qr, kr, vr, W)
    return reshape(yr, size(v)...)
end

function circulant_mh_flash_attention(ss::SparsemaxSimilarity, q::AbstractArray{Tq,N}, k::AbstractArray{Tk,N}, v::AbstractArray{Tv,N}, W::Int, nheads::Int) where {Tq, Tk, Tv, N}
    qr, kr, vr = splitheads.((q, k, v), nheads)
    yr = circulant_flash_attention(ss, qr, kr, vr, W)
    return reshape(yr, size(v)...)
end
