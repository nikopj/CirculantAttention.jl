using CSV, DataFrames, Plots
using StatsPlots

# Read and process the CSV
df = CSV.read("benchmark/benchmark_results.csv", DataFrame, header=false)
rename!(df, [:commit, :timestamp, :gpu_name, :dtype, :shape, :windowsize, :function_name, :time_ms, :gflops])

# Log-scale bar plots need a positive lower y-limit (bars are drawn from 0,
# which is -Inf in log space); pad to the surrounding decades.
function log_ylims(tmat)
    finite = filter(x -> !isnan(x) && x > 0, vec(collect(Float64, tmat)))
    isempty(finite) && return (1e-2, 1e2)
    return (10.0^floor(log10(minimum(finite) / 1.2)), 10.0^ceil(log10(maximum(finite) * 1.2)))
end

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
    # yscale=:log10,
    # ylims=log_ylims(time_matrix),
    grid=true,
    minorgrid=true,
    gridalpha=0.4,
    minorgridalpha=0.15,
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
    # yscale=:log10,
    # ylims=log_ylims(time_matrix),
    grid=true,
    minorgrid=true,
    gridalpha=0.4,
    minorgridalpha=0.15,
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
            # yscale=:log10,
            # ylims=log_ylims(tmat),
            grid=true,
            minorgrid=true,
            gridalpha=0.4,
            minorgridalpha=0.15,
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
    ["circulant_mh_attention_pipeline", "circulant_mh_flash_attention"],    ["standard" "flash"],
    "multi-head forward: standard vs flash",
    "flash_mh_comparison")

# forward + backward, multi-head (nheads=4)
comparison_plot(df,
    ["circulant_mh_attention_gradient", "circulant_mh_flash_attention_gradient"],
    ["standard" "flash"],
    "multi-head forward+backward: standard vs flash",
    "flash_mh_gradient_comparison")

# joint-softmax attention over windows (5, windowsize)
comparison_plot(df,
    ["circulant_joint_attention_pipeline", "circulant_flash_joint_attention"],
    ["standard" "flash"],
    "joint-softmax forward: standard vs flash",
    "flash_joint_comparison")

comparison_plot(df,
    ["circulant_joint_attention_gradient", "circulant_flash_joint_attention_gradient"],
    ["standard" "flash"],
    "joint-softmax forward+backward: standard vs flash",
    "flash_joint_gradient_comparison")

# guided multi-guide joint attention (nheads=4, 3 guides)
comparison_plot(df,
    ["guided_pipeline", "guided_flash_tuple", "guided_flash_batched"],
    ["standard" "flash (per-guide)" "flash (batched)"],
    "guided multi-guide forward: standard vs flash",
    "flash_guided_comparison")

comparison_plot(df,
    ["guided_pipeline_gradient", "guided_flash_batched_gradient"],
    ["standard" "flash (batched)"],
    "guided multi-guide forward+backward: standard vs flash",
    "flash_guided_gradient_comparison")
