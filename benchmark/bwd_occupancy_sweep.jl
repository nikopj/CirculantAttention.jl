# benchmark/bwd_occupancy_sweep.jl
#
# Tests whether raising the backward warp kernel's occupancy (it's register-
# limited to 50%) actually speeds it up. Two levers:
#   (1) sub-warp width WS: bigger WS → fewer window entries per lane (NE=⌈K/WS⌉)
#       → fewer window-state registers, at the cost of wider shuffle reductions.
#   (2) launch bounds (minblocks): force the compiler to cap/spill registers to
#       hit a target blocks/SM.
# Reports registers + median kernel time + correctness (vs the default WS=8).

using CUDA, NNlib, CirculantAttention, Statistics
const CA = CirculantAttention
CUDA.allowscalar(false)

H, W, C, B = 128, 128, 64, 2
ws = parse(Int, get(ENV, "FLASH_WS", "5"))
N, Ts = 4, Float32
sf = DotSimilarity()
q = CUDA.randn(Float32, H, W, C, B); k = CUDA.randn(Float32, H, W, C, B); v = CUDA.randn(Float32, H, W, C, B)
spatdims, nrows, K, maxidx = CA._flash_launch_dims(q, ws)
Cc = Int32(size(q, N-1)); Cv = Int32(size(v, N-1))
r3(x) = reshape(x, :, size(x, N-1), size(x, N))
Δ = CUDA.randn(Float32, size(v)...); y = CUDA.randn(Float32, size(v)...)
lse = CUDA.randn(Float32, Int(nrows), B)
δ = reshape(sum(real.(Δ .* conj.(y)); dims=N-1), :, B)

function run_variant(WS, NE; maxregs=nothing)
    dq = similar(q); dk = similar(k); dv = similar(v)
    args = (r3(dq), r3(dk), r3(dv), sf, r3(q), r3(k), r3(v), r3(Δ), lse, δ,
            nrows, K, Cc, Cv, spatdims, Int32(ws), Ts(1), maxidx, Val(WS), Val(NE))
    kern = if maxregs === nothing
        @cuda launch=false CA.circulant_flash_attention_bwd_warp_kernel!(args...)
    else
        @cuda launch=false maxregs=maxregs CA.circulant_flash_attention_bwd_warp_kernel!(args...)
    end
    reg = CUDA.registers(kern)
    cfg = launch_configuration(kern.fun)
    threads = max(32, min(256, (cfg.threads ÷ 32) * 32))
    blocks  = cld(Int(maxidx) * WS, threads)
    launch() = kern(args...; threads, blocks)
    for _ in 1:5; CUDA.@sync launch(); end
    ts = [CUDA.@elapsed launch() for _ in 1:100]
    return (; reg, time_ms = median(ts) * 1e3, dq, dk, dv)
end

println("== backward warp kernel, ws=$ws K=$(Int(K)) ==")
ref = run_variant(8, cld(Int(K), 8))               # default (auto) config = reference
maxdiff(a, b) = maximum(abs.(Array(a) .- Array(b)))
report(tag, r) = println(rpad(tag, 26),
    " regs=", lpad(r.reg, 3), "  time=", lpad(round(r.time_ms; digits=3), 7), " ms",
    "  |Δdq|=", round(maxdiff(r.dq, ref.dq); sigdigits=2),
    " |Δdk|=", round(maxdiff(r.dk, ref.dk); sigdigits=2),
    " |Δdv|=", round(maxdiff(r.dv, ref.dv); sigdigits=2))

report("WS=8  NE=$(cld(Int(K),8)) (default)", ref)
for WS in (16, 32)
    report("WS=$WS NE=$(cld(Int(K),WS))", run_variant(WS, cld(Int(K), WS)))
end
# register caps → target occupancy tiers (256-thread blocks): 51→62.5%, 42→75%, 32→100%
for mr in (51, 42, 32)
    report("WS=8 maxregs=$mr", run_variant(8, cld(Int(K), 8); maxregs=mr))
end
