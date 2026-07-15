# benchmark/profile_flash.jl
#
# Minimal harness for profiling ONE flash-attention kernel launch under Nsight
# Compute (ncu). It warms up (compile + a few steady-state runs) so ncu can skip
# those, then issues the launch(es) to capture. Nothing else GPU-heavy runs, so a
# kernel-name filter + --launch-skip pins ncu to exactly the kernel we want, and ncu
# reports its roofline / occupancy / warp-stall breakdown (the "is this kernel near
# the hardware limit or leaving 2-4× on the floor" question the timing can't answer).
#
# Config via env vars (all optional):
#   FLASH_DIR     fwd | bwd                          (default bwd)
#   FLASH_MODE    auto | warp | block | thread       (default block)
#   FLASH_WS      window size (odd)                  (default 35 — GroupCDL single-grid)
#   FLASH_ELTYPE  f32 | c32                          (default c32; c32 ⇒ complex q/k, REAL v)
#   FLASH_SIM     dot|realdot|distance|pidot|pidistance
#                                                    (default: dot for f32, pidistance for c32)
#   FLASH_HW / FLASH_C / FLASH_B  spatial / channels / batch   (default 128 / 64 / 4)
#   FLASH_PROFILER  ncu (default, bare launch for an external ncu) | cuda (built-in CUPTI table)
#
# Defaults reproduce the GroupCDL-complex BLOCK BACKWARD kernel (the ~98 ms cost that
# dominates group-net training). The target kernel launches WARMUP+1 times, so ncu with
# `--launch-skip 3 --launch-count 1` captures the last (warm) launch.
#
# ── ncu, roofline + occupancy + stalls (the block bwd kernel) ─────────────────
#   ncu --set full --launch-skip 3 --launch-count 1 \
#       --kernel-name regex:'circulant_flash_attention_bwd_block_kernel' \
#       -o flash_bwd_block -f julia --project benchmark/profile_flash.jl
#   ncu-ui flash_bwd_block.ncu-rep      # or: ncu --import flash_bwd_block.ncu-rep --page details
# `--set full` replays the kernel many times (slow but complete). For a fast first look:
#   ncu --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
#       --section WarpStateStats --section SchedulerStats --section LaunchStats \
#       --launch-skip 3 --launch-count 1 \
#       --kernel-name regex:'circulant_flash_attention_bwd_block_kernel' \
#       julia --project benchmark/profile_flash.jl

using CUDA, NNlib, CirculantAttention
const CA = CirculantAttention
CUDA.allowscalar(false)

const WARMUP = 3
const WS   = parse(Int, get(ENV, "FLASH_WS", "35"))
const MODE = Symbol(get(ENV, "FLASH_MODE", "block"))
const DIR  = Symbol(get(ENV, "FLASH_DIR",  "bwd"))
const ELTY = get(ENV, "FLASH_ELTYPE", "c32") == "c32" ? ComplexF32 : Float32
const H  = parse(Int, get(ENV, "FLASH_HW", "128"))
const C  = parse(Int, get(ENV, "FLASH_C",  "64"))
const Bn = parse(Int, get(ENV, "FLASH_B",  "4"))

const SIMNAME = get(ENV, "FLASH_SIM", ELTY <: Complex ? "pidistance" : "dot")
const simfun = SIMNAME == "pidistance" ? PIDistanceSimilarity() :
               SIMNAME == "pidot"      ? PIDotSimilarity()      :
               SIMNAME == "distance"   ? DistanceSimilarity()   :
               SIMNAME == "realdot"    ? RealDotSimilarity()    : DotSimilarity()

@assert isodd(WS) "FLASH_WS=$WS must be odd"
@assert !(ELTY <: Complex && simfun isa DotSimilarity) "DotSimilarity is real-only; use pidistance for c32"

# GroupCDL: complex q/k (projected latent), REAL v (=|zα|²). Real elty ⇒ all real.
q = CUDA.randn(ELTY,       H, H, C, Bn)
k = CUDA.randn(ELTY,       H, H, C, Bn)
v = CUDA.randn(real(ELTY), H, H, C, Bn)

run_fwd() = CA._circulant_flash_attention_fwd(simfun, q, k, v, WS; mode = MODE)

# exact CUDA kernel name for this (dir,mode), for the ncu --kernel-name filter
_suffix(m) = m === :warp ? "warp_" : m === :block ? "block_" : ""   # thread ⇒ ""
kernel_name = DIR === :fwd ?
    "circulant_flash_attention_$(_suffix(MODE))kernel" :
    "circulant_flash_attention_bwd_$(_suffix(MODE))kernel"
kernel_regex = MODE === :auto ?     # unknown statically ⇒ match the whole direction
    (DIR === :fwd ? "circulant_flash_attention_(warp_|block_)?kernel" :
                    "circulant_flash_attention_bwd_") : kernel_name

println("== profile_flash: dir=$DIR mode=$MODE ws=$WS $(ELTY) $(SIMNAME) size=($H,$H,$C,$Bn) ==")
println("   target kernel: $kernel_name")
println("   ncu filter:    --kernel-name regex:'$kernel_regex' --launch-skip $WARMUP --launch-count 1")

if DIR === :fwd
    target = run_fwd
else
    y, lse = run_fwd()                       # one fwd launch (filtered out by the bwd regex)
    Δ = CUDA.randn(eltype(y), size(y)...)
    target = () -> CA.∇circulant_flash_attention(simfun, Δ, y, lse, q, k, v, WS; mode = MODE)
end

for _ in 1:WARMUP                            # ncu --launch-skip skips these
    CUDA.@sync target()
end
CUDA.synchronize()

# FLASH_PROFILER=ncu (default): bare launch — an external ncu process wraps this.
# FLASH_PROFILER=cuda: CUDA.jl's built-in CUPTI profiler (no external tool); prints a
#   per-kernel table with time + registers + shared memory (a quick sanity check).
if get(ENV, "FLASH_PROFILER", "ncu") == "cuda"
    display(CUDA.@profile CUDA.@sync target())
    println()
else
    CUDA.@sync target()                      # <- the launch ncu captures
    CUDA.synchronize()
end

println("done (target kernel launched $(WARMUP+1)× total; skip $WARMUP, capture 1)")
