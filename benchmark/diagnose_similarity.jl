# Diagnose the circulant_similarity slowdown between commits 1db36c6 (fast)
# and d579062+ (slow). The only source change on that path is the simval
# index-order fix, so this script separates:
#   (1) construction (cucirculant + repeat)  vs  similarity kernel time
#   (2) new (row-q/col-k) index order        vs  old (pre-fix) order
#   (3) environment (CUDA.jl / driver / GPU) — printed up front; compare the
#       gpu_name and date columns of old vs new rows in benchmark_results.csv
using CUDA, BenchmarkTools
using CirculantAttention
using CirculantAttention: simval, cartesian_circulant
using CUDA: i32, launch_configuration

CUDA.versioninfo()
println("GPU: ", CUDA.name(CUDA.device()))
println()

# NB: the begin/end block is required — without it, `CUDA.@sync $f() samples=30
# evals=1` lets @sync swallow the @benchmark kwargs as its own arguments.
function bench(f)
    CUDA.@sync f()
    trial = @benchmark begin
        CUDA.@sync $f()
    end samples=30 evals=1
    return median(trial).time / 1e6
end

H, Wd, C, B = 128, 128, 64, 2
x = CUDA.randn(Float32, H, Wd, C, B)
y = CUDA.randn(Float32, H, Wd, C, B)

# clone of circulant_similarity_kernel! with the PRE-d579062 index order
# (x at column i, y at row j) — perf experiment only, values are transposed
function sim_kernel_old_order!(S, simfun, x, y, nnzb, M, spatdims, CartInd, W, maxidx)
    tid    = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    @inbounds while tid <= maxidx
        n = (tid - Int32(1)) % nnzb + Int32(1)
        b = (tid - Int32(1)) ÷ nnzb + Int32(1)
        i, j = cartesian_circulant(n, spatdims, W)
        s = simval(simfun, x, y, CartInd[i], CartInd[j], b, M)
        S.data.nzVal[n, b] = s
        tid += stride
    end
    return nothing
end

function sim_old_order!(A::Circulant, simfun, x::AbstractArray{T,N}, y) where {T,N}
    Ar = reshape(A, :, :, :)
    maxidx   = Ar.data.nnz
    nnzb     = Ar.data.nnz ÷ Int32(size(Ar, 3))
    M        = Int32(size(x, N-1))
    spatdims = ntuple(i -> Int32(size(x, i)), N-2)
    CartInd  = CartesianIndices(spatdims)
    args = (Ar, simfun, x, y, nnzb, M, spatdims, CartInd, Int32(kernel_length(A)), maxidx)
    kernel = @cuda launch=false sim_kernel_old_order!(args...)
    cfg = launch_configuration(kernel.fun)
    threads = min(maxidx, cfg.threads)
    blocks  = cld(maxidx, threads)
    kernel(args...; threads=threads, blocks=blocks)
    return A
end

println(rpad("W", 5), rpad("full (ms)", 12), rpad("construct", 12), rpad("kernel new", 12), rpad("kernel old-order", 18))
for W in 5:10:45
    local A
    A      = circulant_similarity(DistanceSimilarity(), x, y, W)
    tfull  = bench(() -> circulant_similarity(DistanceSimilarity(), x, y, W))
    tcons  = bench(() -> Circulant{Float32}(W, (H, Wd, 1, B)))
    tkern  = bench(() -> CircAtt.circulant_similarity!(A, DistanceSimilarity(), x, y))
    told   = bench(() -> sim_old_order!(A, DistanceSimilarity(), x, y))
    println(rpad(W, 5),
            rpad(round(tfull; digits=3), 12), rpad(round(tcons; digits=3), 12),
            rpad(round(tkern; digits=3), 12), rpad(round(told; digits=3), 18))
    CUDA.unsafe_free!(A)
end

# ==========================================================================
# round 2: separate the remaining agnostic suspects
#   (a) block-size sensitivity of the similarity kernel (occupancy/L2
#       locality changed with the new toolchain's register allocation)
#   (b) allocator overhead (the ~0.5ms construct floor at W=5)
#   (c) integrated profile of one large-window call
# ==========================================================================
using CirculantAttention: circulant_similarity_kernel!

# Compile the real similarity kernel, robust to whichever signature the
# installed source uses: linearized (3D-reshaped x/y, no CartInd) or the
# CartInd form. Returns (kernel, args, maxidx) ready to relaunch.
function compile_sim(W)
    A  = circulant_similarity(DistanceSimilarity(), x, y, W)
    Ar = reshape(A, :, :, :)
    nnzb = Ar.data.nnz ÷ Int32(size(Ar, 3))
    M, maxidx, Wi = Int32(C), Ar.data.nnz, Int32(W)
    spatdims = (Int32(H), Int32(Wd))
    xr = reshape(x, :, C, B); yr = reshape(y, :, C, B)
    # linearized form first (matches the current cluster source)
    args = (Ar, DistanceSimilarity(), xr, yr, nnzb, M, spatdims, Wi, maxidx)
    kern = @cuda launch=false fastmath=true circulant_similarity_kernel!(args...)
    return A, kern, args, Int(maxidx)
end

println()
println("=== (a) similarity kernel block-size sweep ===")
let (A, kern, _, _) = compile_sim(25)
    cfg = launch_configuration(kern.fun)
    println("launch_configuration picks: threads=$(cfg.threads), blocks=$(cfg.blocks), regs=$(CUDA.registers(kern))")
    CUDA.unsafe_free!(A)
end

function bench_sim_threads(W, tpb)
    A, kern, args, maxidx = compile_sim(W)
    t = bench(() -> kern(args...; threads=tpb, blocks=cld(maxidx, tpb)))
    CUDA.unsafe_free!(A)
    return t
end

println(rpad("W", 5), join(rpad.(("tpb=64", "tpb=128", "tpb=256", "tpb=512", "tpb=1024"), 11)))
for W in (25, 35, 45)
    ts = [bench_sim_threads(W, tpb) for tpb in (64, 128, 256, 512, 1024)]
    println(rpad(W, 5), join(rpad.(round.(ts; digits=3), 11)))
end

println()
println("=== (b) allocator microbenchmark (alloc + fill + free) ===")
for sz in (1<<20, 1<<23, 1<<26, 1<<28)  # 1MB..256MB
    t = bench(() -> begin
        a = CuVector{UInt8}(undef, sz)
        CUDA.unsafe_free!(a)
    end)
    println(rpad("$(sz >> 20)MB", 8), round(t; digits=4), " ms")
end

println()
println("=== (c) integrated profile, one W=45 similarity call ===")
CUDA.@profile circulant_similarity(DistanceSimilarity(), x, y, 45)

# ==========================================================================
# round 3: colVal-lookup vs cartesian_circulant in the similarity kernel.
# The column index i is already materialized in A.data.colVal during
# construction; the row is cld(n, K). Replacing the ~15-op cartesian_circulant
# with one coalesced colVal load + a cld trades non-hideable integer division
# for latency-hideable memory. Same simval inner loop, so any win transfers to
# the flash kernels' index math too.
# ==========================================================================
function sim_kernel_colval!(S, simfun, x, y, nnzb, M, Krow, maxidx)
    tid    = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    @inbounds while tid <= maxidx
        n = (tid - Int32(1)) % nnzb + Int32(1)
        b = (tid - Int32(1)) ÷ nnzb + Int32(1)
        i = S.data.colVal[n, b]          # precomputed column index
        j = cld(n, Krow)                 # row = ceil(n / nnz-per-row)
        s = simval(simfun, x, y, j, i, b, M)
        S.data.nzVal[n, b] = s
        tid += stride
    end
    return nothing
end

function compile_sim_colval(W)
    A  = circulant_similarity(DistanceSimilarity(), x, y, W)
    Ar = reshape(A, :, :, :)
    nnzb = Ar.data.nnz ÷ Int32(size(Ar, 3))
    M, maxidx = Int32(C), Ar.data.nnz
    Krow = Int32(W)^Int32(2)             # nnz per row (2D spatial)
    xr = reshape(x, :, C, B); yr = reshape(y, :, C, B)
    args = (Ar, DistanceSimilarity(), xr, yr, nnzb, M, Krow, maxidx)
    kern = @cuda launch=false fastmath=true sim_kernel_colval!(args...)
    return A, kern, args, Int(maxidx)
end

println()
println("=== (d) cartesian_circulant vs colVal-lookup (similarity kernel) ===")
# correctness: both must produce identical nzVal
let
    A1, k1, a1, mi1 = compile_sim(25)
    A2, k2, a2, mi2 = compile_sim_colval(25)
    cfg = launch_configuration(k2.fun)
    tpb = min(mi2, cfg.threads)
    k1(a1...; threads=min(mi1, cfg.threads), blocks=cld(mi1, min(mi1, cfg.threads)))
    k2(a2...; threads=tpb, blocks=cld(mi2, tpb))
    CUDA.synchronize()
    println("colVal-lookup matches cartesian_circulant: ", Array(A1.data.nzVal) ≈ Array(A2.data.nzVal),
            "  (regs cartesian=$(CUDA.registers(k1)), colval=$(CUDA.registers(k2)))")
    CUDA.unsafe_free!(A1); CUDA.unsafe_free!(A2)
end
println(rpad("W", 5), rpad("cartesian (ms)", 18), rpad("colVal (ms)", 15), "speedup")
for W in (25, 35, 45)
    A1, k1, a1, mi1 = compile_sim(W)
    A2, k2, a2, mi2 = compile_sim_colval(W)
    cfg = launch_configuration(k1.fun)
    tpb = min(mi1, cfg.threads)
    t1 = bench(() -> k1(a1...; threads=tpb, blocks=cld(mi1, tpb)))
    t2 = bench(() -> k2(a2...; threads=tpb, blocks=cld(mi2, tpb)))
    CUDA.unsafe_free!(A1); CUDA.unsafe_free!(A2)
    println(rpad(W, 5), rpad(round(t1; digits=3), 18), rpad(round(t2; digits=3), 15), round(t1/t2; digits=2), "x")
end
