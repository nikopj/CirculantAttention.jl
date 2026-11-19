using CUDA
using BenchmarkTools
using LinearAlgebra
using Printf

CUDA.allowscalar(false)

function run_case(M, N; T=Float32)

    println("\n==============================")
    @printf("M = %d, N = %d\n", M, N)

    A = CUDA.rand(T, M, N)
    x = CUDA.rand(T, N)
    y = similar(A, T, M)

    # warmup (compile)
    matvec!(y, A, x; kernelfun=kernel_warp_impl!)
    matvec!(y, A, x; kernelfun=kernel_block_impl!)
    matvec!(y, A, x; kernelfun=kernel_shwarp_impl!)
    mul!(y, A, x)
    CUDA.synchronize()

    # total FLOPs for matvec
    flops = 2.0 * M * N

    function bench(name, kern)
        t = @belapsed begin
            matvec!($y, $A, $x; kernelfun=$kern)
            CUDA.synchronize()
        end
        gflops = flops / t / 1e9
        @printf("%-14s  %8.4f ms   %8.2f GF/s\n",
                name, t*1e3, gflops)
    end

    function bench_cublas()
        t = @belapsed begin
            mul!($y, $A, $x)
            CUDA.synchronize()
        end
        gflops = flops / t / 1e9
        @printf("%-14s  %8.4f ms   %8.2f GF/s\n",
                "cuBLAS", t*1e3, gflops)
    end

    # bench("warp",        kernel_warp_impl!)
    # bench("block",       kernel_block_impl!)
    # bench("shared-warp", kernel_shwarp_impl!)
    bench("register-blocked", kernel_register_blocked!)
    # bench_cublas()
end


# Suggested test shapes
cases = [
    (4096, 4096),    # square
    (8192, 8192),    # large square
    (65536, 256),    # many short rows
    (4096, 65536),   # very long rows (memory stress)
]

for (M, N) in cases
    run_case(M, N)
end
