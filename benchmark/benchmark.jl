using CUDA
using BenchmarkTools
using NNlib
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
    for _ in 1:3 # warmup
        global x, y, A
        x, y, A = make_data(elty, tensorsize, windowsize)
    end

    # similarity
    t = bench_gpu(() -> circulant_similarity(DistanceSimilarity(), x, y, windowsize))
    flops = tensorsize[4] * tensorsize[1] * tensorsize[2] * (2*tensorsize[3] - 1) * (windowsize^2)
    gf = flops / (t / 1e3) / 1e9
    push!(results, (commit, date, dev_name, string(elty), tensorsize, windowsize, "circulant_similarity", t, gf))

    # attention
    t = bench_gpu(() -> circulant_attention(A, x))
    flops = tensorsize[4] * tensorsize[1] * tensorsize[2] * tensorsize[3] * (2*windowsize^2 - 1)
    gf = flops / (t / 1e3) / 1e9
    push!(results, (commit, date, dev_name, string(elty), tensorsize, windowsize, "circulant_attention", t, gf))

    # softmax
    t = bench_gpu(() -> NNlib.softmax(A))
    flops = tensorsize[4] * tensorsize[1] * tensorsize[2] * tensorsize[3] * 15 * (2*windowsize^2 - 1 + 1) # 15 bc exp, +1 bc division in softmax
    gf = flops / (t / 1e3) / 1e9
    push!(results, (commit, date, dev_name, string(elty), tensorsize, windowsize, "softmax", t, gf))
end

CSV.write("benchmark/benchmark_results.csv", results; append=true)
println("Saved benchmark_results.csv")
