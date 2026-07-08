# benchmark/kernel_info.jl
#
# Reports registers/thread, static+dynamic shared memory, the occupancy-optimal
# block size, and register-limited occupancy for the flash forward and backward
# warp kernels — the numbers CUDA.@profile truncates. No external tools needed.
# Rebuilds the exact kernel-arg tuples the launchers use (see src/flash.jl).

using CUDA, NNlib, CirculantAttention
const CA = CirculantAttention
CUDA.allowscalar(false)

H, W, C, B = 128, 128, 64, 2
ws = parse(Int, get(ENV, "FLASH_WS", "5"))
N, Ts = 4, Float32
sf = DotSimilarity()
q = CUDA.randn(Float32, H, W, C, B); k = CUDA.randn(Float32, H, W, C, B); v = CUDA.randn(Float32, H, W, C, B)

spatdims, nrows, K, maxidx = CA._flash_launch_dims(q, ws)
Cc = Int32(size(q, N-1)); Cv = Int32(size(v, N-1))
WS, NE = CA._flash_warp_dims(K)
r3(x) = reshape(x, :, size(x, N-1), size(x, N))

# A100: 65536 regs/SM, 2048 threads/SM (64 warps), 32-bit reg alloc granularity 256/warp.
const REGS_PER_SM = 65536
const THREADS_PER_SM = 2048
function report(name, kern; shmem_dyn=0)
    reg = CUDA.registers(kern)
    mem = CUDA.memory(kern)
    cfg = launch_configuration(kern.fun; shmem=shmem_dyn)
    threads = max(32, min(256, (cfg.threads ÷ 32) * 32))
    ab  = CUDA.active_blocks(kern.fun, threads; shmem=shmem_dyn)
    occ = ab * threads / THREADS_PER_SM
    regcap = (REGS_PER_SM ÷ reg) / THREADS_PER_SM
    println(rpad(name, 10),
            " regs/thr=", lpad(reg, 3),
            "  static_smem=", mem.shared, "B",
            "  suggested_threads=", cfg.threads,
            "  launch_threads=", threads,
            "  active_blocks/SM=", ab,
            "  occupancy≈", lpad(round(100*occ; digits=1), 5), "%",
            "  (reg-cap ", round(100*min(1.0, regcap); digits=1), "%)")
end

println("== ws=$ws  K=$(Int(K))  WS=$WS  NE=$NE ==")

let  # forward warp kernel
    y = similar(v, Ts); lse = similar(q, Ts, (Int(nrows), B))
    args = (r3(y), lse, sf, r3(q), r3(k), r3(v), nrows, K, Cc, Cv, spatdims, Int32(ws), Ts(1), maxidx, Val(WS), Val(NE))
    kern = @cuda launch=false CA.circulant_flash_attention_warp_kernel!(args...)
    report("fwd warp", kern)
end

let  # backward warp kernel
    dq = similar(q); dk = similar(k); dv = similar(v)
    Δ  = CUDA.randn(Float32, size(v)...); y = CUDA.randn(Float32, size(v)...)
    lse = CUDA.randn(Float32, Int(nrows), B)
    δ  = reshape(sum(real.(Δ .* conj.(y)); dims=N-1), :, B)
    _, _, Tw = CA._flash_bwd_wtypes(sf, Float32, Float32, Float32, Float32)
    args = (r3(dq), r3(dk), r3(dv), sf, r3(q), r3(k), r3(v), r3(Δ), lse, δ,
            nrows, K, Cc, Cv, spatdims, Int32(ws), Ts(1), maxidx, Val(WS), Val(NE))
    kern = @cuda launch=false CA.circulant_flash_attention_bwd_warp_kernel!(args...)
    report("bwd warp", kern)
end
