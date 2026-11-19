# benchmark/nsys_flash.jl
#
# Targeted device-kernel profiler for circulant flash attention FORWARD and
# BACKWARD, at realistic GroupCDL sizes. Complements profile_flash.jl (which pins
# ONE kernel for ncu): this one warms up, then issues the fwd and bwd launches
# inside a CUDA.@profile region so you get an attributable per-kernel table —
# integrated (standalone) or via nsys (external). The fwd and bwd CUDA kernels have
# distinct names (`circulant_flash_attention_*_kernel!` vs `..._bwd_*_kernel!`), so
# `nsys stats --report gpukernsum` separates the two directions with no NVTX needed.
#
# Why nsys (not the script's own ms): under nsys, per-kernel DEVICE durations
# (gpukernsum) are accurate, but wall-clock is inflated by CUPTI — worst for these
# launch-heavy passes. Take the reference ms below from a run WITHOUT nsys; take the
# kernel breakdown from nsys.
#
# Config via env vars (all optional):
#   FLASH_HW      spatial side           (default 128)
#   FLASH_C       channels = Mh          (default 64)
#   FLASH_B       batch                  (default 4)
#   FLASH_WS      window size (odd)      (default 9;  GroupCDL single-grid uses 35)
#   FLASH_NHEADS  attention heads        (default 1)
#   FLASH_ELTYPE  f32 | c32              (default f32; c32 ⇒ complex q/k, REAL v, as GroupCDL)
#   FLASH_SIM     dot|realdot|distance|pidot|pidistance
#                                        (default: dot for f32, pidistance for c32)
#   FLASH_DIR     fwd | bwd | both       (default both)
#   FLASH_MODE    auto|thread|warp|block|all
#                                        (default auto = the production path used by GroupCDL)
#   FLASH_REPS    launches inside capture(default 1)
#   FLASH_WARMUP  warmup launches        (default 5)
#
# Standalone (integrated CUPTI table):
#   julia --project benchmark/nsys_flash.jl
#   FLASH_ELTYPE=c32 FLASH_WS=35 FLASH_MODE=all julia --project benchmark/nsys_flash.jl
#
# NOTE: the standalone integrated CUPTI table above ALREADY reports accurate per-kernel
# DEVICE times (see the "Device-side activity" tables) — for a simple "which kernel, how
# long" question it is sufficient and cleaner than nsys. Reach for nsys when you want the
# full timeline, kernel-overlap, or memory traces.
#
# Under nsys (accurate device breakdown; capture only the warm region). Profile ONE
# direction per run so there's a single capture region, and add --capture-range-end=stop
# so nsys stops collecting but lets Julia EXIT NORMALLY (the default stop-shutdown SIGTERMs
# the app the instant the region closes — the report is still valid, just a noisy exit).
# The reference-ms gputime loop runs OUTSIDE the region, so nsys ignores it:
#   FLASH_DIR=bwd nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
#        --trace=cuda --sample=none -o flash_c32_w35_bwd julia --project benchmark/nsys_flash.jl
#   nsys stats --report gpukernsum flash_c32_w35_bwd.nsys-rep | head -40
# For DIR=both or MODE=all (multiple capture regions) use --capture-range-end=repeat.

using CUDA, NNlib, CirculantAttention, Zygote, Printf
const CA = CirculantAttention
CUDA.allowscalar(false)

# ── config ───────────────────────────────────────────────────────────────────
const HW     = parse(Int, get(ENV, "FLASH_HW",     "128"))
const C      = parse(Int, get(ENV, "FLASH_C",      "64"))
const Bn     = parse(Int, get(ENV, "FLASH_B",      "4"))
const WS     = parse(Int, get(ENV, "FLASH_WS",     "9"))
const NHEADS = parse(Int, get(ENV, "FLASH_NHEADS", "1"))
const ELTY   = get(ENV, "FLASH_ELTYPE", "f32") == "c32" ? ComplexF32 : Float32
const DIR    = Symbol(get(ENV, "FLASH_DIR",  "both"))
const MODE   = Symbol(get(ENV, "FLASH_MODE", "auto"))
const REPS   = parse(Int, get(ENV, "FLASH_REPS",   "1"))
const WARMUP = parse(Int, get(ENV, "FLASH_WARMUP", "5"))

const SIMNAME = get(ENV, "FLASH_SIM", ELTY <: Complex ? "pidistance" : "dot")
const SIM = SIMNAME == "pidistance" ? PIDistanceSimilarity() :
            SIMNAME == "pidot"      ? PIDotSimilarity()      :
            SIMNAME == "distance"   ? DistanceSimilarity()   :
            SIMNAME == "realdot"    ? RealDotSimilarity()    : DotSimilarity()

@assert isodd(WS) "FLASH_WS=$WS must be odd"
@assert C % NHEADS == 0 "FLASH_C=$C must be divisible by FLASH_NHEADS=$NHEADS"
@assert !(ELTY <: Complex && SIM isa DotSimilarity) "DotSimilarity is real-only; use pidistance/pidot/distance for c32"

# GroupCDL: complex q/k (projected latent), REAL v (=|zα|²). Real elty ⇒ all real.
const Tqk = ELTY
const Tv  = real(ELTY)
q = CUDA.randn(Tqk, HW, HW, C, Bn)
k = CUDA.randn(Tqk, HW, HW, C, Bn)
v = CUDA.randn(Tv,  HW, HW, C, Bn)

# ── production path (exactly as GroupCDL calls it: public API, mode=:auto) ─────
fwd_prod()   = circulant_mh_flash_attention(SIM, q, k, v, WS, NHEADS)
loss(a,b,c)  = sum(abs2, circulant_mh_flash_attention(SIM, a, b, c, WS, NHEADS))
bwd_prod()   = Zygote.gradient(loss, q, k, v)

# ── forced-mode path (internal launchers on per-head slices) for the mode sweep ─
const qh, kh, vh = NHEADS == 1 ? (q, k, v) : CA.splitheads.((q, k, v), NHEADS)
fwd_mode(m) = first(CA._circulant_flash_attention_fwd(SIM, qh, kh, vh, WS; mode = m))
function bwd_mode(m)
    y, lse = CA._circulant_flash_attention_fwd(SIM, qh, kh, vh, WS; mode = m)
    Δ = CUDA.randn(eltype(y), size(y)...)
    return CA.∇circulant_flash_attention(SIM, Δ, y, lse, qh, kh, vh, WS; mode = m)
end

# ── mode-selection dry-run ────────────────────────────────────────────────────
# What :auto picks (fwd + bwd) and how far the window is from the block→thread
# CLIFF — an ~11× bwd slowdown (block ≈ 98 ms vs thread ≈ 1080 ms at W=35, c32).
# Pure host arithmetic reusing CircAtt's exact _flash_warp_dims / _flash_bwd_wtypes /
# _flash_mode, so it MATCHES the launcher. shmem = K·(bytes/entry) is independent of
# C and B — only the window W and the dtype sizes move the cliff.
_Ts()  = CA.simval_dtype(SIM, Tqk, Tqk)
_TΔ()  = promote_type(_Ts(), Tv)                       # fwd output type = bwd Δ type
_Tw()  = last(CA._flash_bwd_wtypes(SIM, Tqk, Tqk, _TΔ(), Tv))
_perentry_bwd() = sizeof(_Tw()) + sizeof(_Ts()) + sizeof(Int32)
_shmem_fwd(K) = K * (sizeof(_Ts()) + sizeof(Int32))
_shmem_bwd(K) = K * _perentry_bwd()

function _mode_row(W, N)
    K = W^(N - 2)
    _, NE = CA._flash_warp_dims(Int32(K))
    (; W, K, NE, sf = _shmem_fwd(K), sb = _shmem_bwd(K),
       mf = CA._flash_mode(:auto, NE, _shmem_fwd(K)),
       mb = CA._flash_mode(:auto, NE, _shmem_bwd(K)))
end

function report_mode_selection(N)
    budget = CA._FLASH_BLOCK_MAX_SHMEM
    Kmax   = budget ÷ _perentry_bwd()                  # largest K whose bwd staging fits block
    Wc     = floor(Int, Kmax^(1 / (N - 2)))
    Wcliff = iseven(Wc) ? Wc - 1 : Wc                  # last odd W still on block (bwd)
    while Wcliff^(N - 2) > Kmax; Wcliff -= 2; end

    println("\n══ :auto mode-selection dry-run  (eltype=$ELTY  sim=$SIMNAME  $(N-2)D) ══")
    @printf("   block budget %.1f KB · warp iff NE≤%d · bwd/entry=%dB (Ts=%s, Tw=%s)\n",
            budget/1024, CA._FLASH_WARP_MAX_NE, _perentry_bwd(), _Ts(), _Tw())
    r = _mode_row(WS, N)
    @printf("   configured W=%d:  fwd→%-6s  bwd→%-6s   (bwd staging %.1f/%.1f KB = %.0f%% of budget)\n",
            WS, r.mf, r.mb, r.sb/1024, budget/1024, 100r.sb/budget)
    if r.mb === :thread
        println("   ⚠  bwd is ALREADY on the THREAD path here — ~11× slower than block.")
    else
        @printf("   block→thread cliff: block up to W=%d, thread from W=%d.  Headroom: %d→%d.\n",
                Wcliff, Wcliff+2, WS, Wcliff)
    end
    println("   W-sweep (fwd/bwd mode | bwd staging):")
    hi = max(WS + 6, Wcliff + 6)
    for W in 3:2:hi
        row  = _mode_row(W, N)
        prev = W > 3 ? _mode_row(W-2, N).mb : row.mb
        mark = W == WS ? "  ← configured" :
               (row.mb === :thread && prev !== :thread) ? "  ← CLIFF (11× bwd)" : ""
        @printf("     W=%2d  NE=%3d  fwd=%-6s bwd=%-6s  %.1f KB%s\n",
                W, row.NE, row.mf, row.mb, row.sb/1024, mark)
    end
end

# ── helpers ───────────────────────────────────────────────────────────────────
function gputime(f; N = 30, warm = 5)
    for _ in 1:warm; f(); end; CUDA.synchronize()
    ts = Float64[]
    for _ in 1:N
        CUDA.synchronize(); t = time_ns(); f(); CUDA.synchronize(); push!(ts, (time_ns() - t) / 1e6)
    end
    sort!(ts); ts[cld(N, 2)]
end

# warm, then capture REPS launches in a CUDA.@profile region. Standalone prints the
# integrated per-kernel table; under nsys/ncu it defers (and cudaProfilerStart/Stop
# lets --capture-range=cudaProfilerApi grab only this warm region).
function capture(label, thunk)
    for _ in 1:WARMUP; CUDA.@sync thunk(); end
    CUDA.synchronize()
    println("\n── capture: $label ──")
    res = CUDA.@profile begin
        for _ in 1:REPS; CUDA.@sync thunk(); end
    end
    res === nothing || display(res)
    println()
end

# ── run ────────────────────────────────────────────────────────────────────────
println("== nsys_flash: $(HW)² C=$C B=$Bn W=$WS nheads=$NHEADS $(ELTY) $(SIMNAME) ",
        "| dir=$DIR mode=$MODE | GPU ", CUDA.name(CUDA.device()), " ==")

report_mode_selection(ndims(q))   # what :auto picks + distance to the 11× bwd cliff

# reference wall-clock (trust ONLY without nsys; inflated under CUPTI)
tf = gputime(fwd_prod)
tb = gputime(bwd_prod)
println("  reference (no-nsys) wall-clock:  fwd ", round(tf; digits = 3),
        " ms   fwd+bwd ", round(tb; digits = 3), " ms   (bwd ",
        round(max(tb - tf, 0); digits = 3), " ms)  [inflated under nsys]")

if MODE === :all
    for m in (:thread, :warp, :block)
        DIR in (:fwd, :both) && capture("FWD  mode=$m", () -> fwd_mode(m))
        DIR in (:bwd, :both) && capture("BWD  mode=$m", () -> bwd_mode(m))
    end
elseif MODE === :auto
    # production path — public API fwd + Zygote bwd, the exact GroupCDL call
    DIR in (:fwd, :both) && capture("FWD  (production, auto)", fwd_prod)
    DIR in (:bwd, :both) && capture("BWD  (production, auto)", bwd_prod)
else
    DIR in (:fwd, :both) && capture("FWD  mode=$MODE", () -> fwd_mode(MODE))
    DIR in (:bwd, :both) && capture("BWD  mode=$MODE", () -> bwd_mode(MODE))
end

println("""
done. The Device-side tables above are accurate per-kernel times (often all you need).
For an nsys timeline, profile ONE direction per run and add --capture-range-end=stop so
Julia exits cleanly (default stop-shutdown SIGTERMs the app — report is still valid):
  FLASH_DIR=bwd nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \\
       --trace=cuda --sample=none -o flash_bwd julia --project benchmark/nsys_flash.jl
  nsys stats --report gpukernsum flash_bwd.nsys-rep | head -40
  (DIR=both / MODE=all ⇒ use --capture-range-end=repeat for the extra regions)
Fwd kernels: circulant_flash_attention_{,warp_,block_}kernel!
Bwd kernels: circulant_flash_attention_bwd_{,warp_,block_}kernel!
Set FLASH_MODE=all to compare thread/warp/block at this (W,C,eltype).""")
