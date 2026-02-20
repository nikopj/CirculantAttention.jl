using CSV, DataFrames, Plots

# Read and process the CSV
df = CSV.read("benchmark/benchmark_results.csv", DataFrame, header=false)
rename!(df, [:commit, :timestamp, :dtype, :shape, :windowsize, :function_name, :time_ms])

# Filter for circulant_similarity and Float32 only
filtered = filter(row -> row.function_name == "circulant_similarity" && row.dtype == "Float32", df)

# Plot
bar(filtered.windowsize, filtered.time_ms,
    xlabel="Window Size",
    ylabel="Time (ms)",
    title="circulant_similarity (Float32)",
    legend=false,
    bar_width=4,
    xticks=filtered.windowsize,
    color=:steelblue,
)

savefig("benchmark/circulant_similarity_float32.png")
