# benchmark/guided_profile.jl
#
# Decompose the guided-joint flash attention backward to find where time goes.
# The guided prox (Sljiva GuidedGroupFlashThresholding{true,true}) calls
# `circulant_mh_flash_guided_joint_attention`, whose backward = two hand-written
# `_circulant_flash_attention_lse` rrules (self + all-guides-batched) PLUS a
# host-side joint-softmax reweight that is differentiated by Zygote source-to-source.
#
# This script times, forward-only and forward+backward:
#   (a) the full guided-joint attention           — the real thing
#   (b) the self flash branch alone (W=1)         — hand-written rrule
#   (c) the guide flash branch alone (W=9, G·B)   — hand-written rrule
#   (d) the host reweight in isolation            — the Zygote-differentiated math
# and reports the implied reweight overhead so we can decide whether a custom
# rrule for the combine is worth writing.
#
# Config matches the arch benchmark's guided nets: 256×256, Mh=64, nh=1, complex
# q/k, real value energies, self window 1, guide window 9, 3 guides.
#
# Run:  julia --project benchmark/guided_profile.jl        (B via ENV GUIDED_B, default 4)

using CirculantAttention, CUDA, Zygote, Printf
const CA = CirculantAttention

CUDA.functional() || error("guided_profile requires a functional CUDA device.")

const H   = parse(Int, get(ENV, "GUIDED_H", "256"))
const Wd  = H
const Mh  = 64
const nh  = 1
const Wz  = 1                 # self window
const Wg  = 9                 # guide window
const G   = 3                 # num guides
const B   = parse(Int, get(ENV, "GUIDED_B", "4"))
const Tqk = ComplexF32        # complex queries/keys (projections of complex z)
const Tv  = Float32           # real value energies (abs2 of complex projections)
const sf  = PIDistanceSimilarity()

# ── GPU timing: warmup, then median of N synced reps ─────────────────────────
function gputime(f; N=30, warm=5)
    for _ in 1:warm; f(); end
    CUDA.synchronize()
    ts = Float64[]
    for _ in 1:N
        CUDA.synchronize()
        t0 = time_ns()
        f()
        CUDA.synchronize()
        push!(ts, (time_ns() - t0) / 1e6)   # ms
    end
    sort!(ts)
    return ts[cld(N, 2)]
end

newq() = CUDA.randn(Tqk, H, Wd, Mh, B)
newv() = CUDA.rand(Tv,  H, Wd, Mh, B)

# scale factors as the guided-joint computes them (per branch, from channel count)
scale(k) = inv(sqrt(real(Tqk)(size(k, ndims(k) - 1))))

# ── (a) full guided-joint attention ──────────────────────────────────────────
function bench_full()
    qz, kz, vz = newq(), newq(), newv()
    kg = CUDA.randn(Tqk, H, Wd, Mh, G*B)
    vg = CUDA.rand(Tv,  H, Wd, Mh, G*B)
    fwd() = CA.circulant_mh_flash_guided_joint_attention(sf, qz, kz, vz, Wz, kg, vg, Wg, G, nh)
    lossgrad() = Zygote.gradient((a,b,c,d,e) -> begin
        ξz, ξg = CA.circulant_mh_flash_guided_joint_attention(sf, a, b, c, Wz, d, e, Wg, G, nh)
        sum(abs2, ξz) + sum(abs2, ξg)
    end, qz, kz, vz, kg, vg)
    return gputime(fwd), gputime(lossgrad)
end

# ── (b) self flash branch, (c) guide flash branch ────────────────────────────
function bench_branch(nbatch, Wwin)
    q, k, v = CUDA.randn(Tqk,H,Wd,Mh,nbatch), CUDA.randn(Tqk,H,Wd,Mh,nbatch), CUDA.rand(Tv,H,Wd,Mh,nbatch)
    s = scale(k)
    fwd() = CA._circulant_flash_attention_lse(sf, q, k, v, Wwin, s)
    # backward exercised through BOTH outputs (y and lse) as the joint combine does
    lossgrad() = Zygote.gradient((a,b,c) -> begin
        y, lse = CA._circulant_flash_attention_lse(sf, a, b, c, Wwin, s)
        sum(abs2, y) + sum(abs2, lse)
    end, q, k, v)
    return gputime(fwd), gputime(lossgrad)
end

# ── (d) host reweight in isolation — exact math from flash.jl:1114-1135 ───────
# Pure function of the branch outputs (yz, yg) and per-row logsumexps (Lz, Lg),
# so Zygote differentiates ONLY the reweight, matching what it does inside the
# full call. Shapes as produced by the two flash-lse calls (nh folded = nh·B).
function reweight(yz, yg, Lz_flat, Lg_flat, sp, Cvh)
    nrows = size(Lz_flat, 1)
    Lz = reshape(Lz_flat, nrows, nh, B)
    Lg = reshape(Lg_flat, nrows, nh, G, B)
    m  = max.(Lz, dropdims(maximum(Lg; dims=3); dims=3))
    m4 = reshape(m, nrows, nh, 1, B)
    Lj = m .+ log.(exp.(Lz .- m) .+ dropdims(sum(exp.(Lg .- m4); dims=3); dims=3))
    ωz = reshape(exp.(Lz .- Lj), sp..., 1, nh * B)
    ωg = reshape(exp.(Lg .- reshape(Lj, nrows, nh, 1, B)), sp..., 1, nh * G * B)
    ξz = yz .* ωz
    ξg_f   = yg .* ωg
    ξg_sum = dropdims(sum(reshape(ξg_f, sp..., Cvh, nh, G, B); dims=ndims(yg)+1); dims=ndims(yg)+1)
    return ξz, ξg_sum
end

function bench_reweight()
    sp  = (H, Wd); Cvh = Mh; nrows = H * Wd
    yz  = CUDA.rand(Tv, H, Wd, Cvh, nh*B)
    yg  = CUDA.rand(Tv, H, Wd, Cvh, nh*G*B)
    Lz  = CUDA.randn(Tv, nrows, nh*B)
    Lg  = CUDA.randn(Tv, nrows, nh*G*B)
    fwd() = reweight(yz, yg, Lz, Lg, sp, Cvh)
    lossgrad() = Zygote.gradient((a,b,c,d) -> begin
        ξz, ξg = reweight(a, b, c, d, sp, Cvh)
        sum(abs2, ξz) + sum(abs2, ξg)
    end, yz, yg, Lz, Lg)
    return gputime(fwd), gputime(lossgrad)
end

# ── run ──────────────────────────────────────────────────────────────────────
@printf("\nGuided-joint flash decomposition  (H=%d Mh=%d nh=%d B=%d | Wz=%d Wg=%d G=%d | %s)\n\n",
        H, Mh, nh, B, Wz, Wg, G, Tqk)

af, ab = bench_full()
bf, bb = bench_branch(B,   Wz)      # self
cf, cb = bench_branch(G*B, Wg)      # guides (batched)
df, db = bench_reweight()

row(name, f, b) = @printf("  %-26s fwd %8.3f ms   fwd+bwd %8.3f ms   bwd %8.3f ms\n", name, f, b, max(b-f,0))
row("(a) full guided-joint",     af, ab)
row("(b) self flash  W=1",       bf, bb)
row("(c) guide flash W=9 (G·B)", cf, cb)
row("(d) host reweight only",    df, db)

# ── mode sweep on the guide branch (the bottleneck) ──────────────────────────
# The warp/block/thread crossover (_FLASH_WARP_MAX_NE=8) was tuned on REAL
# DistanceSimilarity; complex entries cost 2× the registers, so :auto may be
# picking a spilling warp kernel where :block wins. Call the fwd/bwd kernels
# directly to force each mode. Skips modes that error (e.g. shmem over budget).
function bench_modes(nbatch, Wwin, label)
    q, k, v = CUDA.randn(Tqk,H,Wd,Mh,nbatch), CUDA.randn(Tqk,H,Wd,Mh,nbatch), CUDA.rand(Tv,H,Wd,Mh,nbatch)
    s = scale(k)
    y, lse = CA._circulant_flash_attention_fwd(sf, q, k, v, Wwin, s)   # :auto reference outputs
    Δ  = CUDA.rand(Tv, size(y)...)
    Δl = CUDA.rand(Tv, size(lse)...)
    _, _, K, _ = CA._flash_launch_dims(q, Wwin)
    WS, NE = CA._flash_warp_dims(K)
    @printf("\n  mode sweep — %s  (K=%d, auto: WS=%d NE=%d → %s)\n", label, Int(K), WS, NE,
            CA._flash_mode(:auto, NE, Int(K)*(sizeof(Float32)+sizeof(Int32))))
    for m in (:warp, :block, :thread)
        fwdok = try; CA._circulant_flash_attention_fwd(sf, q, k, v, Wwin, s; mode=m); true
                catch e; @printf("    %-7s fwd  ERR (%s)\n", m, sprint(showerror,e)[1:min(end,42)]); false end
        fwdok || continue
        tf = gputime(() -> CA._circulant_flash_attention_fwd(sf, q, k, v, Wwin, s; mode=m))
        tb = gputime(() -> CA.∇circulant_flash_attention(sf, Δ, y, lse, q, k, v, Wwin, s; mode=m, Δlse=Δl))
        @printf("    %-7s fwd %8.3f ms   bwd %8.3f ms   fwd+bwd %8.3f ms\n", m, tf, tb, tf+tb)
    end
end
bench_modes(G*B, Wg, "guide W=9")
bench_modes(B,   Wz, "self  W=1")

@printf("\n  branch bwd sum  (b)+(c)          = %8.3f ms\n", (bb-bf)+(cb-cf))
@printf("  measured reweight bwd  (d)        = %8.3f ms\n", db-df)
@printf("  full bwd (a)                      = %8.3f ms\n", ab-af)
@printf("  implied reweight+glue bwd (a-b-c) = %8.3f ms   (%.0f%% of full bwd)\n",
        (ab-af)-((bb-bf)+(cb-cf)), 100*((ab-af)-((bb-bf)+(cb-cf)))/max(ab-af,eps()))
println()
