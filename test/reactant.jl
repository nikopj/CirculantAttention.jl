# test/reactant.jl
#
# Reactant + Enzyme differentiability of circulant flash attention (the Lux+Reactant
# training path). Under Reactant the CUDA flash kernel is replaced by the array-op
# forward (@reactant_overlay → _circ_flash_attention_shift); this checks that
#   (1) that traced forward matches the CUDA forward, and
#   (2) Enzyme-MLIR gradients through it match the Zygote/ChainRules reference.
#
# BOX-ONLY: needs a CUDA GPU + a working Reactant/Enzyme install. Self-skips if
# they fail to load. Tier 1: real (Float32) inputs.

const REACTANT_OK = try
    @eval using Reactant, Enzyme
    true
catch err
    @warn "Reactant/Enzyme unavailable — skipping Reactant+Enzyme tests" exception=err
    false
end

if REACTANT_OK
@testset "Reactant + Enzyme flash" begin
    N, d, B, ws = 8, 4, 2, 5

    for elty in (Float32,)
        # same data on both paths: host arrays → CuArray (reference) and
        # → ConcreteRArray (Reactant).
        qh = randn(elty, N, d, B); kh = randn(elty, N, d, B); vh = randn(elty, N, d, B)
        q, k, v    = CuArray.((qh, kh, vh))
        qr, kr, vr = Reactant.to_rarray.((qh, kh, vh))

        for simfun in (RealDotSimilarity(), DistanceSimilarity(),
                       PIDotSimilarity(), PIDistanceSimilarity())
            tag = string(typeof(simfun))

            @testset "forward $tag" begin
                y_ref = circulant_flash_attention(simfun, q, k, v, ws)   # CUDA kernel
                y_re  = @jit circulant_flash_attention(simfun, qr, kr, vr, ws)  # array-op overlay
                @test Array(y_re) ≈ Array(y_ref)  rtol=1e-4 atol=1e-6
            end

            @testset "gradient $tag" begin
                f = (a, b, c) -> sum(abs2, circulant_flash_attention(simfun, a, b, c, ws))
                g_ref = Zygote.gradient(f, q, k, v)                      # reference
                g_re  = @jit Enzyme.gradient(Enzyme.Reverse, f, qr, kr, vr)
                for (gr, ge) in zip(g_ref, g_re)
                    @test Array(ge) ≈ Array(gr)  rtol=1e-3 atol=1e-5
                end
            end
        end
    end
end
end
