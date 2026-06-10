using CUDA
using BenchmarkTools
using NNlib
using Zygote
using CirculantAttention
using CSV
using DataFrames
using Dates

CUDA.allowscalar(false)

# --------------------------
# utils
# --------------------------

function git_commit()
    try
        readchomp(`git rev-parse --short HEAD`)
    catch
        "unknown"
    end
end

function bench_gpu(f; samples=50)
    CUDA.synchronize()

    trial = @benchmark begin
        CUDA.@sync $f()
    end samples=samples evals=1

    CUDA.synchronize()

    time_ms = median(trial).time / 1e6
    return time_ms 
end

# --------------------------
# problem setup
# --------------------------

function make_data(elty, tensorsize, ws)
    x = CUDA.randn(elty, tensorsize...)
    y = CUDA.randn(elty, tensorsize...)
    A = circulant_adjacency(DistanceSimilarity(), x, y, ws)
    return x, y, A
end

# --------------------------
# run benchmarks
# --------------------------

results = DataFrame(
    commit = String[],
    date = String[],
    gpu_name = String[],
    eltype = String[],
    tensorsize = Tuple[],
    windowsize = Int[],
    function_name = String[],
    time_ms = Float64[],
    gflops = Float64[],
)

commit = git_commit()
date = string(Dates.now())
dev_name = CUDA.name(CUDA.device())

for elty in (Float32, ComplexF32), tensorsize in ((128, 128, 64, 2),), windowsize in 5:10:45
    H, W, C, B = tensorsize
    N = H*W

    for _ in 1:3 # warmup
        global x, y, A
        x, y, A = make_data(elty, tensorsize, windowsize)
    end

    # similarity
    t = bench_gpu(() -> circulant_similarity(DistanceSimilarity(), x, y, windowsize))
    flops = B * N * (2*C - 1) * (windowsize^2)
    gf = flops / (t / 1e3) / 1e9
    push!(results, (commit, date, dev_name, string(elty), tensorsize, windowsize, "circulant_similarity", t, gf))

    # attention
    t = bench_gpu(() -> circulant_attention(A, x))
    flops = B * N * C * (2*windowsize^2 - 1)
    gf = flops / (t / 1e3) / 1e9
    push!(results, (commit, date, dev_name, string(elty), tensorsize, windowsize, "circulant_attention", t, gf))

    # end-to-end: composed pipeline (similarity → softmax → ⊠) vs fused flash kernel
    flops = B * N * (2*C - 1) * (windowsize^2) + B * N * C * (2*windowsize^2 - 1)
    t = bench_gpu(() -> circulant_attention(DistanceSimilarity(), x, y, x, windowsize))
    gf = flops / (t / 1e3) / 1e9
    push!(results, (commit, date, dev_name, string(elty), tensorsize, windowsize, "circulant_attention_pipeline", t, gf))

    t = bench_gpu(() -> circulant_flash_attention(DistanceSimilarity(), x, y, x, windowsize))
    gf = flops / (t / 1e3) / 1e9
    push!(results, (commit, date, dev_name, string(elty), tensorsize, windowsize, "circulant_flash_attention", t, gf))

    # forward + backward: composed vs fused gradients (nominal flops ≈ 3x forward)
    gradflops = 3 * flops
    t = bench_gpu(() -> Zygote.gradient((q, k, v) -> sum(abs2, first(circulant_attention(DistanceSimilarity(), q, k, v, windowsize))), x, y, x))
    gf = gradflops / (t / 1e3) / 1e9
    push!(results, (commit, date, dev_name, string(elty), tensorsize, windowsize, "circulant_attention_gradient", t, gf))

    t = bench_gpu(() -> Zygote.gradient((q, k, v) -> sum(abs2, circulant_flash_attention(DistanceSimilarity(), q, k, v, windowsize)), x, y, x))
    gf = gradflops / (t / 1e3) / 1e9
    push!(results, (commit, date, dev_name, string(elty), tensorsize, windowsize, "circulant_flash_attention_gradient", t, gf))

    # softmax
    t = bench_gpu(() -> NNlib.softmax(A))
    flops = B * N * C * 15 * (2*windowsize^2 - 1 + 1) # 15 bc exp, +1 bc division in softmax
    gf = flops / (t / 1e3) / 1e9
    push!(results, (commit, date, dev_name, string(elty), tensorsize, windowsize, "softmax", t, gf))

    # joint softmax
    t = bench_gpu(() -> CircAtt.joint_softmax(A, A))
    flops = B * N * C * 15 * (2 * 2*windowsize^2 - 1 + 1) # 15 bc exp, +1 bc division in softmax
    gf = flops / (t / 1e3) / 1e9
    push!(results, (commit, date, dev_name, string(elty), tensorsize, windowsize, "joint_softmax", t, gf))
end

CSV.write("benchmark/benchmark_results.csv", results; append=true)
println("Saved benchmark_results.csv")
