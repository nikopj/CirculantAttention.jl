using CSV, DataFrames, Plots
using StatsPlots

# Read and process the CSV
df = CSV.read("benchmark/benchmark_results.csv", DataFrame, header=false)
rename!(df, [:commit, :timestamp, :gpu_name, :dtype, :shape, :windowsize, :function_name, :time_ms])

# Filter for circulant_similarity and Float32 only
filtered = filter(row -> row.function_name == "circulant_similarity" && row.dtype == "Float32", df)

commits = unique(filtered[:, :commit])
window_sizes = sort(unique(filtered[:, :windowsize]))

# Build a matrix: rows = window sizes, cols = commits
time_matrix = [
    filtered[(filtered.commit .== c) .& (filtered.windowsize .== w), :time_ms][1]
    for w in window_sizes, c in commits
]

# Plot grouped bars
groupedbar(window_sizes, time_matrix,
    xlabel="Window Size",
    ylabel="Time (ms)",
    title="circulant_similarity (Float32)",
    label=reshape(commits, 1, :),
    bar_width=4,
    xticks=window_sizes,
    legend=:topleft,
)

savefig("benchmark/circulant_similarity_float32.png")
