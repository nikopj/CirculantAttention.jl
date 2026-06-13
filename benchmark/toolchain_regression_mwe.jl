# toolchain_regression_mwe.jl
#
# Minimal, standalone reproducer for a ~2.3× slowdown observed in a
# reduction-style CUDA kernel between two CUDA.jl toolchains on the same A100:
#
#   April 2026 stack : CUDA.jl 5.x  / GPUCompiler 0.x / LLVM 16 / PTX 8.x
#   June  2026 stack : CUDACore 6.2 / GPUCompiler 1.21 / LLVM 18 / PTX 8.8
#
# No dependency on CirculantAttention — only CUDA + BenchmarkTools. The kernel
# below is the distilled inner loop of a windowed-attention "distance
# similarity": for each (row, window-entry, batch) it reduces over C channels,
#   s = -1/2 Σ_m (x[j,m,b] - y[i,m,b])^2
# with the same data layout the real code uses — (nrows, C, batch), so the
# channel loop strides by `nrows`, the "row" vector x[j,:] is shared by K
# consecutive threads (broadcast/cache), and the "col" vector y[i,:] is
# coalesced across a warp.
#
# HOW TO USE FOR A BUG REPORT
#   1. Run on the suspected-slow toolchain; note the kernel times.
#   2. ] add CUDA@<older-version>   (or use a separate project/manifest)
#   3. Run again; compare. The saxpy control isolates raw bandwidth so a
#      regression in `dist_kernel!` but not saxpy points at codegen, not HW.
#   The block-size sweep shows the slowdown is not a launch-config artifact.

using CUDA, BenchmarkTools, Dates

# Tee all output to a results file (override the path with MWE_OUTFILE=...). The
# default name is timestamped so old-vs-new-toolchain runs don't overwrite.
const OUTFILE = get(ENV, "MWE_OUTFILE",
    joinpath(@__DIR__, "toolchain_regression_$(Dates.format(now(), "yyyymmdd-HHMMSS")).txt"))
const _io = open(OUTFILE, "w")
log(args...) = (println(args...); println(_io, args...); flush(_io))

log(sprint(CUDA.versioninfo))
log("GPU: ", CUDA.name(CUDA.device()))
log("run: ", now())
log()

function bench(f)
    CUDA.@sync f()
    # begin/end is required: otherwise CUDA.@sync swallows the @benchmark kwargs
    trial = @benchmark begin
        CUDA.@sync $f()
    end samples=30 evals=1
    return median(trial).time / 1e6   # ms
end

# --------------------------------------------------------------------------
# the kernel under test
# --------------------------------------------------------------------------
function dist_kernel!(out, x, y, colval, nrows::Int32, C::Int32, K::Int32, maxidx::Int32)
    tid    = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    nnzb   = nrows * K
    @inbounds while tid <= maxidx
        n = (tid - Int32(1)) % nnzb + Int32(1)
        b = (tid - Int32(1)) ÷ nnzb + Int32(1)
        j = cld(n, K)                       # row index (shared across K threads)
        i = colval[n]                       # column index (per-thread, coalesced)
        s = 0.0f0
        @fastmath for m in Int32(1):C
            d = x[j, m, b] - y[i, m, b]
            s = s - d * d
        end
        out[n, b] = 0.5f0 * s
        tid += stride
    end
    return nothing
end

# Manually strength-reduced variant: instead of x[j,m,b] (whose linear offset
# LLVM18 recomputes with a per-iteration `mul.lo.s64`), maintain linear indices
# that step by `nrows` each iteration — handing the compiler the constant-stride
# pointer increment that LLVM16's LSR produced automatically. A/B this against
# dist_kernel! to see whether it dodges the lost-strength-reduction regression.
function dist_kernel_sr!(out, x, y, colval, nrows::Int32, C::Int32, K::Int32, maxidx::Int32)
    tid    = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    nnzb   = nrows * K
    nr     = Int(nrows)
    @inbounds while tid <= maxidx
        n = (tid - Int32(1)) % nnzb + Int32(1)
        b = (tid - Int32(1)) ÷ nnzb + Int32(1)
        j = cld(n, K)
        i = colval[n]
        # linear indices for channel m=1 (column-major (nrows, C, batch))
        base = (Int(b) - 1) * nr * Int(C)
        xl = base + Int(j)
        yl = base + Int(i)
        s = 0.0f0
        @fastmath for _ in 1:C
            d = x[xl] - y[yl]
            s = s - d * d
            xl += nr            # constant stride → pointer-bump, no multiply
            yl += nr
        end
        out[n, b] = 0.5f0 * s
        tid += stride
    end
    return nothing
end

# circulant column pattern (1D, width K): data-independent, captures the
# broadcast-row / coalesced-col access pattern without any library.
function make_colval(nrows::Int, K::Int)
    p   = (K - 1) ÷ 2
    col = Vector{Int32}(undef, nrows * K)
    @inbounds for j in 1:nrows, w in 1:K
        col[(j-1)*K + w] = Int32(mod(j - 1 + (w - 1) - p, nrows) + 1)
    end
    return CuArray(col)
end

# --------------------------------------------------------------------------
# saxpy control: a trivially bandwidth-bound kernel. If this stays constant
# across toolchains while dist_kernel! regresses, the GPU/bandwidth is fine.
# --------------------------------------------------------------------------
function saxpy_kernel!(z, a, x, y, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    @inbounds if i <= n
        z[i] = a * x[i] + y[i]
    end
    return nothing
end

# --------------------------------------------------------------------------
# problem setup — matches the real benchmark: 128×128 spatial, C=64, batch 2
# --------------------------------------------------------------------------
const NROWS, C, B = 128 * 128, 64, 2
x = CUDA.randn(Float32, NROWS, C, B)
y = CUDA.randn(Float32, NROWS, C, B)

log("=== dist_kernel! (the regressed kernel) ===")
log(rpad("W", 4), rpad("K=W^2", 8), rpad("time(ms)", 11), rpad("regs", 7), "eff. GB/s (lower bound)")
for W in (5, 15, 25, 35, 45)
    K       = W * W
    colval  = make_colval(NROWS, K)
    out     = CUDA.zeros(Float32, NROWS * K, B)
    maxidx  = Int32(NROWS * K * B)
    args    = (out, x, y, colval, Int32(NROWS), Int32(C), Int32(K), maxidx)

    kern    = @cuda launch=false fastmath=true dist_kernel!(args...)
    cfg     = launch_configuration(kern.fun)
    threads = min(Int(maxidx), cfg.threads)
    blocks  = cld(Int(maxidx), threads)
    t       = bench(() -> kern(args...; threads=threads, blocks=blocks))

    # lower-bound traffic: each thread touches 2C floats (row reads largely cache)
    gb = Int(maxidx) * 2 * C * 4 / 1e9
    log(rpad(W, 4), rpad(K, 8), rpad(round(t; digits=3), 11),
            rpad(CUDA.registers(kern), 7), round(gb / (t / 1e3); digits=1))
    CUDA.unsafe_free!(colval); CUDA.unsafe_free!(out)
end

log()
log("=== workaround: manual strength-reduction (dist_kernel_sr!) ===")
log("    if sr ≈ old dist_kernel! on LLVM16, the win on LLVM18 = the LSR workaround")
log(rpad("W", 4), rpad("orig(ms)", 11), rpad("sr(ms)", 11), "speedup")
for W in (15, 25, 35, 45)
    K       = W * W
    colval  = make_colval(NROWS, K)
    out     = CUDA.zeros(Float32, NROWS * K, B)
    maxidx  = Int32(NROWS * K * B)
    args    = (out, x, y, colval, Int32(NROWS), Int32(C), Int32(K), maxidx)
    k1 = @cuda launch=false fastmath=true dist_kernel!(args...)
    k2 = @cuda launch=false fastmath=true dist_kernel_sr!(args...)
    c1 = launch_configuration(k1.fun); t1 = bench(() -> k1(args...; threads=min(Int(maxidx), c1.threads), blocks=cld(Int(maxidx), min(Int(maxidx), c1.threads))))
    c2 = launch_configuration(k2.fun); t2 = bench(() -> k2(args...; threads=min(Int(maxidx), c2.threads), blocks=cld(Int(maxidx), min(Int(maxidx), c2.threads))))
    log(rpad(W, 4), rpad(round(t1; digits=3), 11), rpad(round(t2; digits=3), 11),
        "$(round(t1/t2; digits=2))x  (sr regs=$(CUDA.registers(k2)))")
    CUDA.unsafe_free!(colval); CUDA.unsafe_free!(out)
end

log()
log("=== dist_kernel! block-size sweep at W=45 (should be ~flat) ===")
let W = 45
    K      = W * W
    colval = make_colval(NROWS, K)
    out    = CUDA.zeros(Float32, NROWS * K, B)
    maxidx = Int32(NROWS * K * B)
    args   = (out, x, y, colval, Int32(NROWS), Int32(C), Int32(K), maxidx)
    kern   = @cuda launch=false fastmath=true dist_kernel!(args...)
    for tpb in (64, 128, 256, 512, 1024)
        t = bench(() -> kern(args...; threads=tpb, blocks=cld(Int(maxidx), tpb)))
        log("  tpb=$(rpad(tpb,5)) $(round(t; digits=3)) ms")
    end
    CUDA.unsafe_free!(colval); CUDA.unsafe_free!(out)
end

log()
log("=== saxpy control (raw bandwidth sanity; should NOT regress) ===")
let n = 1 << 26    # 64M elements
    xv = CUDA.randn(Float32, n); yv = CUDA.randn(Float32, n); zv = similar(xv)
    args = (zv, 2.0f0, xv, yv, Int32(n))
    kern = @cuda launch=false saxpy_kernel!(args...)
    cfg  = launch_configuration(kern.fun)
    tpb  = min(n, cfg.threads)
    t    = bench(() -> kern(args...; threads=tpb, blocks=cld(n, tpb)))
    gb   = n * 3 * 4 / 1e9   # 2 reads + 1 write
    log("  $(round(t; digits=3)) ms  →  $(round(gb / (t / 1e3); digits=1)) GB/s")
    CUDA.unsafe_free!(xv); CUDA.unsafe_free!(yv); CUDA.unsafe_free!(zv)
end

# Codegen evidence for the report: set MWE_DUMP_CODEGEN=1 to dump the lowered/
# typed/LLVM-IR/PTX/SASS of the W=45 kernel to a directory. Run on each
# toolchain and diff the .ll (LLVM IR — shows the LLVM 16↔18 difference before
# ptxas) and .ptx files; that diff is the actual evidence maintainers need.
if get(ENV, "MWE_DUMP_CODEGEN", "0") == "1"
    K = 45 * 45
    colval = make_colval(NROWS, K)
    out    = CUDA.zeros(Float32, NROWS * K, B)
    args   = (out, x, y, colval, Int32(NROWS), Int32(C), Int32(K), Int32(NROWS * K * B))
    dumpdir = joinpath(@__DIR__, "codegen_$(Dates.format(now(), "yyyymmdd-HHMMSS"))")
    mkpath(dumpdir)
    CUDA.@device_code dir=dumpdir @cuda launch=false fastmath=true dist_kernel!(args...)
    log("codegen dumped to ", dumpdir)
    CUDA.unsafe_free!(colval); CUDA.unsafe_free!(out)
end

close(_io)
println("\nresults written to ", OUTFILE)
