using CSV, DataFrames, Plots
using StatsPlots

# Read and process the CSV
df = CSV.read("benchmark/benchmark_results.csv", DataFrame, header=false)
rename!(df, [:commit, :timestamp, :gpu_name, :dtype, :shape, :windowsize, :function_name, :time_ms, :gflops])

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

# Filter for circulant_similarity and Float32 only
filtered = filter(row -> row.function_name == "circulant_attention" && row.dtype == "Float32", df)

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
    title="circulant_attention (Float32)",
    label=reshape(commits, 1, :),
    bar_width=4,
    xticks=window_sizes,
    legend=:topleft,
)

savefig("benchmark/circulant_attention_float32.png")

# ------------------------------------------------------------------
# standard-vs-flash comparisons (forward, backward, single/multi-head).
# Uses the latest entry per (function, windowsize) so results from
# older commits don't mix into the comparison.
# ------------------------------------------------------------------
function comparison_plot(df, impls, impl_labels, titlestr, outprefix)
    for dtype in unique(df.dtype)
        sub = filter(row -> row.function_name in impls && row.dtype == dtype, df)
        isempty(sub) && continue

        sort!(sub, :timestamp)
        latest = combine(groupby(sub, [:function_name, :windowsize])) do g
            g[end:end, :]
        end

        wsizes = sort(unique(latest.windowsize))
        tmat = [
            begin
                rows = latest[(latest.function_name .== f) .& (latest.windowsize .== w), :time_ms]
                isempty(rows) ? NaN : rows[1]
            end
            for w in wsizes, f in impls
        ]

        groupedbar(wsizes, tmat,
            xlabel="Window Size",
            ylabel="Time (ms)",
            title="$titlestr ($dtype)",
            label=impl_labels,
            bar_width=4,
            xticks=wsizes,
            legend=:topleft,
        )

        savefig("benchmark/$(outprefix)_$(lowercase(dtype)).png")
    end
end

# forward, single-head
comparison_plot(df,
    ["circulant_attention_pipeline", "circulant_flash_attention", "circulant_flash_attention_thread"],
    ["standard" "flash" "flash (thread)"],
    "forward: standard vs flash",
    "flash_comparison")

# forward + backward (Zygote.gradient), single-head
comparison_plot(df,
    ["circulant_attention_gradient", "circulant_flash_attention_gradient"],
    ["standard" "flash"],
    "forward+backward: standard vs flash",
    "flash_gradient_comparison")

# forward, multi-head (nheads=4)
comparison_plot(df,
    ["circulant_mh_attention_pipeline", "circulant_mh_flash_attention"],
    ["standard" "flash"],
    "multi-head forward: standard vs flash",
    "flash_mh_comparison")

# forward + backward, multi-head (nheads=4)
comparison_plot(df,
    ["circulant_mh_attention_gradient", "circulant_mh_flash_attention_gradient"],
    ["standard" "flash"],
    "multi-head forward+backward: standard vs flash",
    "flash_mh_gradient_comparison")
