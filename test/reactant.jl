# test/reactant.jl
#
# Reactant + Enzyme backward vs the existing Zygote/ChainRules backward, and the
# raisable KA forward vs the CUDA reference forward. The Zygote path is the
# reference; the Reactant path must match it.
#
# BOX-ONLY: needs a CUDA GPU and a Reactant install whose `@jit` raises the KA
# forward kernels (src/ka_forward.jl) into StableHLO. Cleanly skipped if Reactant
# or Enzyme fail to load. Tier 1 covers real (Float32) inputs only.

const REACTANT_OK = try
    @eval using Reactant, Enzyme
    true
catch err
    @warn "Reactant/Enzyme unavailable — skipping Reactant+Enzyme tests" exception=err
    false
end

if REACTANT_OK
@testset "Reactant + Enzyme" begin
    N, d, B, ws = 8, 4, 2, 5

    for elty in (Float32,)                        # Tier 1: real inputs
        # same data on both paths: host arrays → CuArray (reference) and
        # → ConcreteRArray (Reactant). Avoids any CuArray↔rarray conversion.
        qh = randn(elty, N, d, B)
        kh = randn(elty, N, d, B)
        vh = randn(elty, N, d, B)
        q, k, v    = CuArray.((qh, kh, vh))
        qr, kr, vr = Reactant.to_rarray.((qh, kh, vh))
        scale = inv(sqrt(real(elty)(d)))          # matches circulant_flash_attention

        for simfun in (RealDotSimilarity(), DistanceSimilarity(),
                       PIDotSimilarity(), PIDistanceSimilarity())
            tag = string(typeof(simfun))

            @testset "KA forward $tag" begin
                y_ref = circulant_flash_attention(simfun, q, k, v, ws)
                y_ka  = @jit CircAtt._ka_flash_attention(simfun, qr, kr, vr, ws, scale)
                @test Array(y_ka) ≈ Array(y_ref)  rtol=1e-4 atol=1e-6
            end

            @testset "Enzyme gradient $tag" begin
                g_ref = Zygote.gradient(
                    (q, k, v) -> sum(abs2, circulant_flash_attention(simfun, q, k, v, ws)),
                    q, k, v)
                g_re = @jit CircAtt.reactant_flash_grad(simfun, qr, kr, vr, ws, scale)
                for (gref, gre) in zip(g_ref, g_re)
                    @test Array(gre) ≈ Array(gref)  rtol=1e-3 atol=1e-5
                end
            end
        end
    end
end
end
