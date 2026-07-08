# benchmark/profile_flash.jl
#
# Minimal harness for profiling ONE flash-attention kernel launch under Nsight
# Compute (ncu). It warms up (compile + a few steady-state runs) so ncu can skip
# those, then issues the launch(es) to capture. Nothing else GPU-heavy runs, so
# a kernel-name filter + --launch-skip pins ncu to exactly the kernel we want.
#
# Config via env vars (all optional):
#   FLASH_DIR   fwd | bwd            (default fwd)
#   FLASH_MODE  auto | warp | block | thread   (default auto)
#   FLASH_WS    window size          (default 5)
#
# The target kernel launches exactly WARMUP+1 times (default 3+1), so ncu with
# `--launch-skip 3 --launch-count 1` captures the last (warm) launch. See the ncu
# commands printed at the end of this file's run.

using CUDA, NNlib, CirculantAttention
const CA = CirculantAttention
CUDA.allowscalar(false)

const WARMUP = 3
ws   = parse(Int, get(ENV, "FLASH_WS", "5"))
mode = Symbol(get(ENV, "FLASH_MODE", "auto"))
dir  = Symbol(get(ENV, "FLASH_DIR", "fwd"))

H, W, C, B = 128, 128, 64, 2
simfun = DotSimilarity()
q = CUDA.randn(Float32, H, W, C, B)
k = CUDA.randn(Float32, H, W, C, B)
v = CUDA.randn(Float32, H, W, C, B)

run_fwd() = CA._circulant_flash_attention_fwd(simfun, q, k, v, ws; mode)

println("== profile_flash: dir=$dir mode=$mode ws=$ws size=($H,$W,$C,$B) ==")

if dir === :fwd
    target = run_fwd
else
    y, lse = run_fwd()                       # one fwd launch (different kernel name)
    Δ = CUDA.randn(Float32, size(y)...)
    target = () -> CA.∇circulant_flash_attention(simfun, Δ, y, lse, q, k, v, ws; mode)
end

for _ in 1:WARMUP                            # ncu --launch-skip skips these
    CUDA.@sync target()
end
CUDA.synchronize()

# FLASH_PROFILER=ncu (default): bare launch — an external ncu process wraps this.
# FLASH_PROFILER=cuda: use CUDA.jl's built-in CUPTI profiler (no external tool);
#   prints a per-kernel table with time + registers + shared memory.
if get(ENV, "FLASH_PROFILER", "ncu") == "cuda"
    display(CUDA.@profile CUDA.@sync target())
    println()
else
    CUDA.@sync target()                      # <- the launch ncu captures
    CUDA.synchronize()
end

println("done (target kernel launched $(WARMUP+1)× total; skip $WARMUP, capture 1)")
