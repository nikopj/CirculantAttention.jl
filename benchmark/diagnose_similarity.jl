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

function bench(f)
    CUDA.@sync f()
    trial = @benchmark begin
        CUDA.@sync $f()
    end samples=30 evals=2
    return median(trial).time / 1e6
end

H, Wd, C, B = 128, 128, 64, 2
x = CUDA.randn(Float32, H, Wd, C, B)
y = CUDA.randn(Float32, H, Wd, C, B)

# clone of circulant_similarity_kernel! with the PRE-d579062 index order
# (x at column i, y at row j) — perf experiment only, values are transposed
function sim_kernel_old_order!(S, simfun, x, y, nnzb, M, spatdims, W, maxidx)
    tid    = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    @inbounds while tid <= maxidx
        n = (tid - Int32(1)) % nnzb + Int32(1)
        b = (tid - Int32(1)) ÷ nnzb + Int32(1)
        i, j = cartesian_circulant(n, spatdims, W)
        s = simval(simfun, x, y, i, j, b, M)
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
    xr = reshape(x, :, size(x, N-1), size(x, N))
    yr = reshape(y, :, size(y, N-1), size(y, N))
    args = (Ar, simfun, xr, yr, nnzb, M, spatdims, Int32(kernel_length(A)), maxidx)
    kernel = @cuda launch=false sim_kernel_old_order!(args...)
    cfg = launch_configuration(kernel.fun)
    threads = min(maxidx, cfg.threads)
    blocks  = cld(maxidx, threads)
    kernel(args...; threads=threads, blocks=blocks)
    return A
end

# println(rpad("W", 5), rpad("full (ms)", 12), rpad("construct", 12), rpad("kernel new", 12), rpad("kernel old-order", 18))
# for W in 5:10:45
#     A      = circulant_similarity(DistanceSimilarity(), x, y, W)
#     tfull  = bench(() -> circulant_similarity(DistanceSimilarity(), x, y, W))
#     tcons  = bench(() -> Circulant{Float32}(W, (H, Wd, 1, B)))
#     tkern  = bench(() -> CircAtt.circulant_similarity!(A, DistanceSimilarity(), x, y))
#     told   = bench(() -> sim_old_order!(A, DistanceSimilarity(), x, y))
#     println(rpad(W, 5),
#             rpad(round(tfull; digits=3), 12), rpad(round(tcons; digits=3), 12),
#             rpad(round(tkern; digits=3), 12), rpad(round(told; digits=3), 18))
#     CUDA.unsafe_free!(A)
# end

println()
println(rpad("W", 5), 
        rpad("full (ms)", 12), 
        rpad("softmax", 15), 
        rpad("mult", 12), 
        rpad("flash-auto", 18),
        rpad("flash-thread", 18),
        rpad("flash-warp", 18),
        rpad("flash-block", 18),
       )
for W in 5:10:45
    z, A   = circulant_attention(DistanceSimilarity(), x, y, x, W)
    tfull  = bench(() -> circulant_attention(DistanceSimilarity(), x, y, x, W))
    tsoft  = bench(() -> NNlib.softmax(A))
    tmult  = bench(() -> CircAtt.circulant_attention!(z, A, x))
    tflash = bench(() -> CircAtt.circulant_flash_attention(DistanceSimilarity(), x, y, z, W))
    tflash_t = bench(() -> CircAtt._circulant_flash_attention_fwd(DistanceSimilarity(), x, y, z, W; mode=:thread))
    tflash_w = bench(() -> CircAtt._circulant_flash_attention_fwd(DistanceSimilarity(), x, y, z, W; mode=:warp))
    tflash_b = bench(() -> CircAtt._circulant_flash_attention_fwd(DistanceSimilarity(), x, y, z, W; mode=:block))
    println(rpad(W, 5),
            rpad(round(tfull; digits=3), 12), 
            rpad(round(tsoft; digits=3), 15), 
            rpad(round(tmult; digits=3), 12), 
            rpad(round(tflash; digits=3), 18),
            rpad(round(tflash_t; digits=3), 18),
            rpad(round(tflash_w; digits=3), 18),
            rpad(round(tflash_b; digits=3), 18))
    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(z)
end
