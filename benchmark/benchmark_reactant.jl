# benchmark/benchmark_reactant.jl
#
# Quick comparison of the two AD paths for single-head circulant flash attention:
#   • zygote   — the CUDA flash kernels + ChainRules rrule (existing path)
#   • reactant — the array-op shift forward (@reactant_overlay) differentiated by
#                Enzyme-MLIR under Reactant (the Lux+Reactant training path)
#
# Reports forward and forward+backward median times (ms). Reactant compile time is
# excluded: each function is @compile'd once, warmed up, then timed. Scalar-valued
# compiled functions are used so reading the result forces XLA to finish (cheap
# sync, no large device→host copy).
#
# Run on the CUDA box:  julia --project benchmark/benchmark_reactant.jl

using CUDA, BenchmarkTools, Printf
using Zygote, CirculantAttention
using Reactant, Enzyme

CUDA.allowscalar(false)

const SIMFUN      = DistanceSimilarity()
const TENSORSIZE  = (128, 128, 64, 2)      # H, W, C, B
const WINDOWSIZES = (5, 9, 15)             # keep small: Reactant graph grows with W²
const SAMPLES     = 30

# ── timing helpers ────────────────────────────────────────────────────────────
# force XLA completion: scalar Reactant result → Julia number; array → host copy.
force(x::Number) = Float64(x)
force(x)         = (Array(x); nothing)

function bench(thunk; samples=SAMPLES, warmup=3)
    for _ in 1:warmup; thunk(); end
    t = @benchmark $thunk() samples=samples evals=1 gcsample=true
    return median(t).time / 1e6
end

# ── scalar-valued entry points (so a compiled call is cheap to sync) ───────────
_fwd_norm(simfun, q, k, v, ws)  = sum(abs2, circulant_flash_attention(simfun, q, k, v, ws))
function _grad_norm(simfun, q, k, v, ws)
    g = Enzyme.gradient(Enzyme.Reverse,
            (a, b, c) -> sum(abs2, circulant_flash_attention(simfun, a, b, c, ws)),
            q, k, v)
    return sum(abs2, g[1]) + sum(abs2, g[2]) + sum(abs2, g[3])
end

# ── run ───────────────────────────────────────────────────────────────────────
println("device: ", CUDA.name(CUDA.device()))
println("simfun: ", SIMFUN, "   tensor: ", TENSORSIZE)
@printf("\n%-4s | %10s %10s %6s | %10s %10s %6s\n",
        "W", "zyg_fwd", "rea_fwd", "x", "zyg_bwd", "rea_bwd", "x")
println("-"^66)

for ws in WINDOWSIZES
    x = CUDA.randn(Float32, TENSORSIZE...)
    y = CUDA.randn(Float32, TENSORSIZE...)
    z = CUDA.randn(Float32, TENSORSIZE...)
    xr, yr, zr = Reactant.to_rarray.((x, y, z))

    # forward
    zyg_fwd = bench(() -> CUDA.@sync circulant_flash_attention(SIMFUN, x, y, z, ws))
    fwd_c   = let ws=ws; @compile _fwd_norm(SIMFUN, xr, yr, zr, ws); end
    rea_fwd = bench(() -> force(fwd_c(SIMFUN, xr, yr, zr, ws)))

    # forward + backward
    zyg_bwd = bench(() -> CUDA.@sync Zygote.gradient(
                    (q, k, v) -> sum(abs2, circulant_flash_attention(SIMFUN, q, k, v, ws)), x, y, z))
    grad_c  = let ws=ws; @compile _grad_norm(SIMFUN, xr, yr, zr, ws); end
    rea_bwd = bench(() -> force(grad_c(SIMFUN, xr, yr, zr, ws)))

    @printf("%-4d | %10.3f %10.3f %6.2f | %10.3f %10.3f %6.2f\n",
            ws, zyg_fwd, rea_fwd, zyg_fwd/rea_fwd, zyg_bwd, rea_bwd, zyg_bwd/rea_bwd)

    x = y = z = xr = yr = zr = nothing
    GC.gc(true); CUDA.reclaim()
end
println("\n(x = zygote/reactant ratio; >1 means reactant is faster)")
