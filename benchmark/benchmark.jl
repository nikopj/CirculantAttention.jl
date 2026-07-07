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

function bench_gpu(f; samples=50, warmup=3)
    # warm up each function on its own (compile + first-call costs) before timing
    for _ in 1:warmup
        CUDA.@sync f()
    end
    CUDA.synchronize()

    # gcsample=true runs GC before every sample (not counted in the timing) so
    # dead CuArrays are finalized and the pool can be reused — without it 50
    # samples of a large allocation pile up in the pool and OOM mid-trial.
    trial = @benchmark begin
        CUDA.@sync $f()
    end samples=samples evals=1 gcsample=true

    CUDA.synchronize()

    time_ms = median(trial).time / 1e6
    return time_ms
end

# --------------------------
# problem setup
# --------------------------

function make_data(elty, tensorsize, ws)
    x = CUDA.randn(elty, tensorsize...)   # q
    y = CUDA.randn(elty, tensorsize...)   # k
    z = CUDA.randn(elty, tensorsize...)   # v (distinct from q,k)
    A = circulant_adjacency(DistanceSimilarity(), x, y, ws)
    return x, y, z, A
end

# composed joint-softmax attention over two windows (reference for the fused
# circulant_flash_joint_attention); mirrors its τ-scaling
function joint_pipeline(simfun, q, k, v, W1, W2)
    τ = sqrt(eltype(k)(size(k, ndims(k) - 1)))
    S1 = circulant_similarity(simfun, q ./ sqrt(τ), k ./ sqrt(τ), W1)
    S2 = circulant_similarity(simfun, q ./ sqrt(τ), k ./ sqrt(τ), W2)
    A1, A2 = joint_softmax(S1, S2)
    return A1 ⊗ v, A2 ⊗ v
end

# guide tensors stacked guide-fastest as (lead..., G·B); slice guide g → (lead..., B)
_gslice(x5, g) = x5[ntuple(_ -> Colon(), ndims(x5) - 2)..., g, :]

# STANDARD (non-batched) guided multi-guide joint-softmax attention (mh): one
# self branch + G guide branches sharing the query, jointly normalized — the
# Γ-materializing path (mirrors GuidedGroupThreshold), with per-guide slices.
function guided_pipeline(simfun, qz, kz, vz, kg, vg, Wz, Wg, G, B, nheads)
    lead = size(kg)[1:ndims(kg)-1]
    kgr  = reshape(kg, lead..., G, B)
    vgr  = reshape(vg, lead..., G, B)
    Sz = CircAtt.circulant_mh_similarity(simfun, qz, kz, Wz, nheads)
    Sg = ntuple(g -> CircAtt.circulant_mh_similarity(simfun, qz, _gslice(kgr, g), Wg, nheads), G)
    As = joint_softmax(Sz, Sg...)
    yz = As[1] ⨷ vz
    yg = mapreduce(+, 1:G) do g
        As[g+1] ⨷ _gslice(vgr, g)
    end
    return yz, yg
end

# FLASH but NON-batched: per-guide slices fed to the tuple joint flash kernel
# (G+1 separate flash launches). Isolates the cost the batched path removes.
function guided_flash_tuple(simfun, qz, kz, vz, kg, vg, Wz, Wg, G, B, nheads)
    lead = size(kg)[1:ndims(kg)-1]
    kgr  = reshape(kg, lead..., G, B)
    vgr  = reshape(vg, lead..., G, B)
    qs = (qz, ntuple(_ -> qz, G)...)
    ks = (kz, ntuple(g -> _gslice(kgr, g), G)...)
    vs = (vz, ntuple(g -> _gslice(vgr, g), G)...)
    Ws = (Wz, ntuple(_ -> Wg, G)...)
    return circulant_mh_flash_joint_attention(simfun, qs, ks, vs, Ws, nheads)
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

const ELTYPES = (Float32, ComplexF32)
const TENSORSIZES = ((128, 128, 64, 2),)
const WINDOWSIZES = 5:10:45
const NITER = length(ELTYPES) * length(TENSORSIZES) * length(WINDOWSIZES)
const t_start = time()
iter = 0

for elty in ELTYPES, tensorsize in TENSORSIZES, windowsize in WINDOWSIZES
    global iter += 1
    H, W, C, B = tensorsize
    N = H*W

    elapsed = round(Int, time() - t_start)
    println("\n[$iter/$NITER] elty=$elty  W=$windowsize  size=$tensorsize   (elapsed $(elapsed)s)")

    # build q (x), k (y), v (z) and adjacency A once; per-function warmup now
    # happens inside bench_gpu so every benchmarked function is warmed itself.
    global x, y, z, A
    x, y, z, A = make_data(elty, tensorsize, windowsize)

    # bench `f`, record under `name` with the given nominal flops, and print a
    # progress line (so a long run shows steady output). Closure captures the
    # loop vars + globals so call sites stay short.
    rec = function (name, flops, f)
        print("    ", rpad(name, 38), " … "); flush(stdout)
        t  = bench_gpu(f)
        gf = flops / (t / 1e3) / 1e9
        push!(results, (commit, date, dev_name, string(elty), tensorsize, windowsize, name, t, gf))
        println(lpad(round(t; digits=3), 9), " ms   ", lpad(round(gf; digits=1), 8), " GFLOP/s")
        return t
    end

    sim_flops  = B * N * (2*C - 1) * (windowsize^2)
    att_flops  = B * N * C * (2*windowsize^2 - 1)
    e2e_flops  = sim_flops + att_flops          # composed/flash forward
    grad_flops = 3 * e2e_flops                  # nominal fwd+bwd ≈ 3× forward

    rec("circulant_similarity", sim_flops, () -> circulant_similarity(DistanceSimilarity(), x, y, windowsize))
    rec("circulant_attention",  att_flops, () -> circulant_attention(A, z))

    # end-to-end: composed pipeline (similarity → softmax → ⊠) vs fused flash kernel.
    # q=x, k=y, v=z are three distinct tensors.
    rec("circulant_attention_pipeline", e2e_flops, () -> circulant_attention(DistanceSimilarity(), x, y, z, windowsize))
    rec("circulant_flash_attention",    e2e_flops, () -> circulant_flash_attention(DistanceSimilarity(), x, y, z, windowsize))
    rec("circulant_flash_attention_thread", e2e_flops, () -> CircAtt._circulant_flash_attention_fwd(DistanceSimilarity(), x, y, z, windowsize; mode=:thread))

    # forward + backward: composed vs fused gradients
    rec("circulant_attention_gradient", grad_flops, () -> Zygote.gradient((q, k, v) -> sum(abs2, first(circulant_attention(DistanceSimilarity(), q, k, v, windowsize))), x, y, z))
    rec("circulant_flash_attention_gradient", grad_flops, () -> Zygote.gradient((q, k, v) -> sum(abs2, circulant_flash_attention(DistanceSimilarity(), q, k, v, windowsize)), x, y, z))

    # multi-head (nheads=4 → 16 channels per head): the per-head channel count
    # drops, so the adjacency-matrix traffic the flash kernels avoid is a much
    # larger fraction of the runtime than in the single-head benchmark
    nheads = 4
    rec("circulant_mh_attention_pipeline", e2e_flops, () -> circulant_mh_attention(DistanceSimilarity(), x, y, z, windowsize, nheads))
    rec("circulant_mh_flash_attention",    e2e_flops, () -> circulant_mh_flash_attention(DistanceSimilarity(), x, y, z, windowsize, nheads))
    rec("circulant_mh_attention_gradient", grad_flops, () -> Zygote.gradient((q, k, v) -> sum(abs2, first(circulant_mh_attention(DistanceSimilarity(), q, k, v, windowsize, nheads))), x, y, z))
    rec("circulant_mh_flash_attention_gradient", grad_flops, () -> Zygote.gradient((q, k, v) -> sum(abs2, circulant_mh_flash_attention(DistanceSimilarity(), q, k, v, windowsize, nheads)), x, y, z))

    # joint-softmax attention over windows (5, windowsize): composed vs fused
    Wsj = (5, windowsize)
    Ktot = sum(w -> w^2, Wsj)
    joint_flops     = B * N * (2*C - 1) * Ktot + B * N * C * (2*Ktot - 1)
    joint_gradflops = 3 * joint_flops
    rec("circulant_joint_attention_pipeline", joint_flops, () -> joint_pipeline(DistanceSimilarity(), x, y, z, Wsj...))
    rec("circulant_flash_joint_attention",    joint_flops, () -> circulant_flash_joint_attention(DistanceSimilarity(), x, y, z, Wsj))
    rec("circulant_joint_attention_gradient", joint_gradflops, () -> Zygote.gradient((q, k, v) -> begin
        ya, yb = joint_pipeline(DistanceSimilarity(), q, k, v, Wsj...); sum(abs2, ya) + sum(abs2, yb)
    end, x, y, z))
    rec("circulant_flash_joint_attention_gradient", joint_gradflops, () -> Zygote.gradient((q, k, v) -> begin
        ya, yb = circulant_flash_joint_attention(DistanceSimilarity(), q, k, v, Wsj); sum(abs2, ya) + sum(abs2, yb)
    end, x, y, z))

    # guided multi-guide joint attention (nheads=4, G guides, all windows =
    # windowsize): standard composed (Γ-materialized) vs flash-tuple (G+1
    # separate launches) vs flash-batched (guides in one flash call).
    nhg = 4
    G   = 3
    global kg, vg
    kg = CUDA.randn(elty, tensorsize[1:end-1]..., G*B)
    vg = CUDA.randn(elty, tensorsize[1:end-1]..., G*B)
    g_flops     = (G + 1) * (B * N * (2*C - 1) * windowsize^2 + B * N * C * (2*windowsize^2 - 1))
    g_gradflops = 3 * g_flops
    # The standard (Γ-materializing) guided pipeline blows up with window² × G;
    # only run it at smaller windows. The flash variants never form Γ, so they
    # cover the full window sweep.
    if windowsize < 25
        rec("guided_pipeline", g_flops, () -> guided_pipeline(DistanceSimilarity(), x, y, z, kg, vg, windowsize, windowsize, G, B, nhg))
        rec("guided_pipeline_gradient", g_gradflops, () -> Zygote.gradient((qz, kz, vz, kgg, vgg) -> begin
            ya, yb = guided_pipeline(DistanceSimilarity(), qz, kz, vz, kgg, vgg, windowsize, windowsize, G, B, nhg); sum(abs2, ya) + sum(abs2, yb)
        end, x, y, z, kg, vg))
    end
    rec("guided_flash_tuple",   g_flops, () -> guided_flash_tuple(DistanceSimilarity(), x, y, z, kg, vg, windowsize, windowsize, G, B, nhg))
    rec("guided_flash_batched", g_flops, () -> circulant_mh_flash_guided_joint_attention(DistanceSimilarity(), x, y, z, windowsize, kg, vg, windowsize, G, nhg))
    rec("guided_flash_batched_gradient", g_gradflops, () -> Zygote.gradient((qz, kz, vz, kgg, vgg) -> begin
        ya, yb = circulant_mh_flash_guided_joint_attention(DistanceSimilarity(), qz, kz, vz, windowsize, kgg, vgg, windowsize, G, nhg); sum(abs2, ya) + sum(abs2, yb)
    end, x, y, z, kg, vg))

    rec("softmax",       B * N * C * 15 * (2*windowsize^2 - 1 + 1), () -> NNlib.softmax(A))
    rec("joint_softmax", B * N * C * 15 * (2 * 2*windowsize^2 - 1 + 1), () -> CircAtt.joint_softmax(A, A))

    # ── reclaim GPU memory before the next (eltype, windowsize) ──────────
    # Drop this iteration's arrays, run finalizers so the dead CuArrays return
    # to the pool, then hand the pooled blocks back to the driver. Without this
    # the pool keeps every window's peak resident and large windows OOM.
    x = y = z = A = kg = vg = nothing
    GC.gc(true)
    CUDA.reclaim()
end

println("\nfinished $NITER configs in $(round(Int, time() - t_start))s")
CSV.write("benchmark/benchmark_results.csv", results; append=true)
println("Saved benchmark_results.csv")
