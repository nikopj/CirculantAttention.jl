# benchmark/check_warp_ws.jl
#
# Correctness of the warp/block flash kernels vs the CPU-validated THREAD kernel,
# at window sizes that select each sub-warp width WS (the ws=5 test suite only
# ever exercises WS=8). _flash_warp_dims: WS=8 for K≲64, 16 for K≲128, 32 above.
# 2-D windows: ws=5→WS8, ws=9→WS16, ws=13→WS32 (all NE≤8 ⇒ warp mode).

using CUDA, NNlib, CirculantAttention
const CA = CirculantAttention
CUDA.allowscalar(false)

N, d, B = 20, 8, 2
maxdiff(a, b) = maximum(abs.(Array(a) .- Array(b)))

for ws in (5, 9, 13)
    K = ws^2
    WS, NE = CA._flash_warp_dims(Int32(K))
    println("\n── ws=$ws  K=$K  →  WS=$WS  NE=$NE ──")
    for sim in (DotSimilarity(), DistanceSimilarity())
        q = CUDA.randn(Float32, N, N, d, B)
        k = CUDA.randn(Float32, N, N, d, B)
        v = CUDA.randn(Float32, N, N, d, B)
        Δ = CUDA.randn(Float32, N, N, d, B)

        yt, lt = CA._circulant_flash_attention_fwd(sim, q, k, v, ws; mode=:thread)
        gt = CA.∇circulant_flash_attention(sim, Δ, yt, lt, q, k, v, ws; mode=:thread)

        for m in (:warp, :block)
            ym, lm = CA._circulant_flash_attention_fwd(sim, q, k, v, ws; mode=m)
            gm = CA.∇circulant_flash_attention(sim, Δ, ym, lm, q, k, v, ws; mode=m)
            fwd = max(maxdiff(ym, yt), maxdiff(lm, lt))
            bwd = maximum(maxdiff(a, b) for (a, b) in zip(gm, gt))
            flag = (fwd < 1e-3 && bwd < 1e-3) ? "ok " : "FAIL"
            println("  $flag  $(nameof(typeof(sim)))  mode=$m :  fwd_err=$(round(fwd; sigdigits=3))  bwd_err=$(round(bwd; sigdigits=3))")
        end
    end
end
