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
    eltype = String[],
    tensorsize = Tuple[],
    windowsize = Int[],
    function_name = String[],
    time_ms = Float64[],
)

commit = git_commit()
date = string(Dates.now())

for elty in (Float32, ComplexF32), tensorsize in ((128, 128, 64, 2),), windowsize in 5:10:45
    for _ in 1:3 # warmup
        global x, y, A
        x, y, A = make_data(elty, tensorsize, windowsize)
    end

    # similarity
    t = bench_gpu(() -> circulant_similarity(DistanceSimilarity(), x, y, windowsize))
    push!(results, (commit, date, string(elty), tensorsize, windowsize, "circulant_similarity", t))

    # attention
    t = bench_gpu(() -> circulant_attention(A, x))
    push!(results, (commit, date, string(elty), tensorsize, windowsize, "circulant_attention", t))

    # softmax
    t = bench_gpu(() -> NNlib.softmax(A))
    push!(results, (commit, date, string(elty), tensorsize, windowsize, "softmax", t))
end

CSV.write("benchmark/benchmark_results.csv", results; append=true)
println("Saved benchmark_results.csv")
