# benchmark/tiled_bwd_validate.jl
#
# CPU correctness oracle for the `:tiled` recompute-once backward — step 1 of the
# rollout in TILED_BACKWARD_DESIGN.md. It proves the reorganization
#
#     compute each pair's (ds, P, α) ONCE, then apply to BOTH ∂q and ∂k/∂v
#
# is mathematically identical to the existing two-sweep backward, BEFORE any GPU
# shared-memory kernel is written. Everything here runs on CPU Arrays: the existing
# per-row backward (`_flash_attention_bwd_row!`) and a naive windowed-softmax forward
# are generic scalar code, so no GPU is needed to validate the schedule.
#
# For real + complex(q/k) × {RealDot, Distance, PIDistance} it checks:
#   (1) tiled  ≈  the existing two-sweep `_flash_attention_bwd_row!`   [the reorg]
#   (2) tiled  ≈  finite differences of the naive forward (real case)  [the math]
#
# Note (stash width): the tiled stash is (ds, P) for Dot/RealDot/Distance (α≡1), but
# (ds, P, α) for PIDot/PIDistance — α = sign(...) is a per-pair COMPLEX factor needed
# by both ∂q and ∂k. Still independent of C. This is the refinement over the design
# note's "2 scalars/pair" (which holds only for the α≡1 similarities).
#
# Run:  julia --project benchmark/tiled_bwd_validate.jl

using CirculantAttention, Printf, Random
const CA = CirculantAttention
using CirculantAttention: cartesian_circulant, simval, simgrad_aux, _flash_gval,
                          _fastexp, _flash_attention_bwd_row!, simval_dtype

const N1, N2, W = 5, 5, 3            # 25 rows, 3×3 window (K=9)
const NR, K = N1 * N2, W * W
const C, Cv, B = 3, 3, 2

_cart(a, κ) = Int(first(cartesian_circulant(Int32((a - 1) * K + κ), (Int32(N1), Int32(N2)), Int32(W))))

# ── naive windowed-softmax forward on (NR,C,B) → y (NR,Cv,B), lse (NR,B) ──────
function naive_fwd(sf, q, k, v, scale)
    Ts = simval_dtype(sf, eltype(q), eltype(k))
    y   = zeros(promote_type(Ts, eltype(v)), NR, Cv, B)
    lse = zeros(real(Ts), NR, B)
    for b in 1:B, a in 1:NR
        sv = [scale * real(simval(sf, q, k, Int32(a), Int32(_cart(a, κ)), Int32(b), Int32(C))) for κ in 1:K]
        m  = maximum(sv); l = sum(exp.(sv .- m)); lse[a, b] = m + log(l)
        for κ in 1:K
            i = _cart(a, κ); p = exp(sv[κ] - m) / l
            for c in 1:Cv; y[a, c, b] += p * v[i, c, b]; end
        end
    end
    return y, lse
end

_delta(Δ, y) = [sum(real(Δ[a, c, b] * conj(y[a, c, b])) for c in 1:Cv) for a in 1:NR, b in 1:B]

# ── existing two-sweep backward (the oracle) ─────────────────────────────────
function oracle_bwd(sf, q, k, v, Δ, lse, δ, scale)
    Tqk = promote_type(eltype(q), eltype(k)); TΔ = eltype(Δ)
    dq = zeros(Tqk, NR, C, B); dk = zeros(Tqk, NR, C, B); dv = zeros(TΔ, NR, Cv, B)
    for b in 1:B, a in 1:NR
        _flash_attention_bwd_row!(dq, dk, dv, sf, q, k, v, Δ, lse, δ,
            Int32(a), Int32(b), Int32(K), Int32(C), Int32(Cv),
            (Int32(N1), Int32(N2)), Int32(W), scale, Val(2))
    end
    return dq, dk, dv
end

# ── tiled recompute-once backward: stash (ds,P,α) per pair, apply to both dirs ─
function tiled_bwd(sf, q, k, v, Δ, lse, δ, scale)
    Tqk = promote_type(eltype(q), eltype(k)); TΔ = eltype(Δ)
    dq = zeros(Tqk, NR, C, B); dk = zeros(Tqk, NR, C, B); dv = zeros(TΔ, NR, Cv, B)
    Tds = real(Tqk)
    # Phase 1 — compute (ds, P, α, β) ONCE per (center a, col i), keyed by (a,i,b)
    D = Dict{NTuple{3, Int}, Tuple{Tds, Tds, Tqk, Tds}}()
    for b in 1:B, a in 1:NR, κ in 1:K
        i = _cart(a, κ)
        s, α, β = simgrad_aux(sf, q, k, Int32(a), Int32(i), Int32(b), Int32(C))
        P  = _fastexp(scale * s - lse[a, b])
        g  = _flash_gval(Δ, v, Int32(a), Int32(i), Int32(b), Int32(Cv))
        ds = P * (g - δ[a, b])
        D[(a, i, b)] = (Tds(ds), Tds(P), Tqk(α), Tds(β))
    end
    # Phase 2 — ∂q, owner = center a  (reads D[(a,i,·)])
    for b in 1:B, a in 1:NR, κ in 1:K
        i = _cart(a, κ); ds, _, α, β = D[(a, i, b)]
        for c in 1:C; dq[a, c, b] += scale * ds * (α * k[i, c, b] + β * q[a, c, b]); end
    end
    # Phase 3 — ∂k, ∂v, owner = col a; rows r whose window contains a  (reads D[(r,a,·)])
    for b in 1:B, a in 1:NR, κ in 1:K
        r = _cart(a, κ); ds, P, α, β = D[(r, a, b)]
        for c in 1:C;  dk[a, c, b] += scale * ds * (conj(α) * q[r, c, b] + β * k[a, c, b]); end
        for c in 1:Cv; dv[a, c, b] += P * Δ[r, c, b]; end
    end
    return dq, dk, dv
end

# ── atomic-scatter schedule (mode=:scatter): row-parallel, ∂q single-writer,
#    ∂k/∂v scattered. Mirrors the GPU kernel EXACTLY, incl. the interleaved real
#    reinterpret for complex targets (re(i)→2i-1, im(i)→2i) — validates that indexing.
#    Plain += stands in for the device atomicAdd (same sums). ───────────────────
_relin(A, i, c, b) = (real(A[2i-1, c, b]), A) # helper unused; kept for clarity
function scatter_bwd(sf, q, k, v, Δ, lse, δ, scale)
    Tqk = promote_type(eltype(q), eltype(k)); TΔ = eltype(Δ)
    DKC = Tqk <: Complex; DVC = TΔ <: Complex
    dq  = zeros(Tqk, NR, C, B)
    dkr = zeros(real(Tqk), DKC ? 2NR : NR, C, B)      # interleaved re/im if complex
    dvr = zeros(real(TΔ),  DVC ? 2NR : NR, Cv, B)
    scat!(A, i, c, b, val, ::Val{true})  = (A[2i-1, c, b] += real(val); A[2i, c, b] += imag(val))
    scat!(A, i, c, b, val, ::Val{false}) = (A[i, c, b] += real(val))
    for b in 1:B, r in 1:NR, κ in 1:K
        i = _cart(r, κ)
        s, α, β = simgrad_aux(sf, q, k, Int32(r), Int32(i), Int32(b), Int32(C))
        P  = _fastexp(scale * s - lse[r, b])
        ds = P * (_flash_gval(Δ, v, Int32(r), Int32(i), Int32(b), Int32(Cv)) - δ[r, b])
        sds = scale * ds
        for c in 1:C
            qv = q[r, c, b]; kv = k[i, c, b]
            dq[r, c, b] += sds * (α * kv + β * qv)
            scat!(dkr, i, c, b, sds * (conj(α) * qv + β * kv), Val(DKC))
        end
        for c in 1:Cv
            scat!(dvr, i, c, b, P * Δ[r, c, b], Val(DVC))
        end
    end
    dk = DKC ? [dkr[2i-1, c, b] + im * dkr[2i, c, b] for i in 1:NR, c in 1:C, b in 1:B] : dkr
    dv = DVC ? [dvr[2i-1, c, b] + im * dvr[2i, c, b] for i in 1:NR, c in 1:Cv, b in 1:B] : dvr
    return dq, dk, dv
end

# ── finite-difference backward of the naive forward (real case only) ─────────
function fd_bwd(sf, q, k, v, Δ, scale; ε = 1e-6)
    L(qq, kk, vv) = sum(Δ .* first(naive_fwd(sf, qq, kk, vv, scale)))
    function grad(x, f)
        g = zero(x)
        for idx in eachindex(x)
            xp = copy(x); xp[idx] += ε
            xm = copy(x); xm[idx] -= ε
            g[idx] = (f(xp) - f(xm)) / 2ε
        end
        return g
    end
    return grad(q, qq -> L(qq, k, v)), grad(k, kk -> L(q, kk, v)), grad(v, vv -> L(q, k, vv))
end

maxerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps())

function run(name, sf, Tqk, Tv; fd = false)
    rng = MersenneTwister(0)
    q = randn(rng, Tqk, NR, C, B); k = randn(rng, Tqk, NR, C, B); v = randn(rng, Tv, NR, Cv, B)
    scale = 1 / sqrt(real(Tqk)(C))
    y, lse = naive_fwd(sf, q, k, v, scale)
    Δ = randn(rng, eltype(y), NR, Cv, B); δ = _delta(Δ, y)
    dqo, dko, dvo = oracle_bwd(sf, q, k, v, Δ, lse, δ, scale)
    dqt, dkt, dvt = tiled_bwd(sf, q, k, v, Δ, lse, δ, scale)
    dqs, dks, dvs = scatter_bwd(sf, q, k, v, Δ, lse, δ, scale)
    eo = max(maxerr(dqt, dqo), maxerr(dkt, dko), maxerr(dvt, dvo))
    es = max(maxerr(dqs, dqo), maxerr(dks, dko), maxerr(dvs, dvo))
    @printf("  %-34s  tiled: dq %.1e dk %.1e dv %.1e  %s   scatter: dq %.1e dk %.1e dv %.1e  %s\n",
            name, maxerr(dqt, dqo), maxerr(dkt, dko), maxerr(dvt, dvo), eo < 1e-6 ? "✓" : "✗",
            maxerr(dqs, dqo), maxerr(dks, dko), maxerr(dvs, dvo), es < 1e-6 ? "✓" : "✗ FAIL")
    if fd
        dqf, dkf, dvf = fd_bwd(sf, q, k, v, Δ, scale)
        ef = max(maxerr(dqt, dqf), maxerr(dkt, dkf), maxerr(dvt, dvf))
        @printf("  %-34s  tiled-vs-FD:     dq %.1e dk %.1e dv %.1e  %s\n",
                "", maxerr(dqt, dqf), maxerr(dkt, dkf), maxerr(dvt, dvf), ef < 1e-4 ? "✓" : "✗ FAIL")
    end
end

println("Tiled recompute-once backward — CPU validation ($(N1)×$(N2), W=$W, C=$C, B=$B)\n")
println(" real (Float64):")
run("RealDotSimilarity",    RealDotSimilarity(),    Float64, Float64; fd = true)
run("DistanceSimilarity",   DistanceSimilarity(),   Float64, Float64; fd = true)
run("PIDistanceSimilarity", PIDistanceSimilarity(), Float64, Float64; fd = true)
println("\n complex q/k, real v (GroupCDL config):")
run("RealDot  (C64 q/k, F64 v)",    RealDotSimilarity(),    ComplexF64, Float64)
run("Distance (C64 q/k, F64 v)",    DistanceSimilarity(),   ComplexF64, Float64)
run("PIDistance (C64 q/k, F64 v)",  PIDistanceSimilarity(), ComplexF64, Float64)
