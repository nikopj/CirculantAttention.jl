# test/flash.jl
# Fused flash attention vs the composed pipeline (similarity → softmax → ⊠).
# The two paths must agree in forward values and gradients; gradients share the
# same underlying rrules (the flash pullback is checkpointed through the
# composed path), so only forward roundoff separates them.

@testset "Flash Attention" begin
for elty in TEST_ELTYPES, nspatdims in TEST_SPATDIMS
    N  = 8
    d  = 4
    B  = 2
    ws = 5
    spatdims = ntuple(_ -> N, nspatdims)
    tag = "elty=$elty, nspatdims=$nspatdims"

    q = CUDA.randn(elty, spatdims..., d, B)
    k = CUDA.randn(elty, spatdims..., d, B)
    v = CUDA.randn(elty, spatdims..., d, B)

    simfuns = elty <: Complex ?
        (RealDotSimilarity(), DistanceSimilarity(), PIDotSimilarity(), PIDistanceSimilarity()) :
        (DotSimilarity(), RealDotSimilarity(), DistanceSimilarity(), PIDotSimilarity(), PIDistanceSimilarity())

    for simfun in simfuns
        @testset "forward $(typeof(simfun)) [$tag]" begin
            y_ref, _ = circulant_attention(simfun, q, k, v, ws)
            y_fl     = circulant_flash_attention(simfun, q, k, v, ws)
            @test eltype(y_fl) == eltype(y_ref)
            @test Array(y_fl) ≈ Array(y_ref)  rtol=1e-4 atol=1e-6
        end

        @testset "gradient $(typeof(simfun)) [$tag]" begin
            g_ref = Zygote.gradient((q, k, v) -> sum(abs2, first(circulant_attention(simfun, q, k, v, ws))), q, k, v)
            g_fl  = Zygote.gradient((q, k, v) -> sum(abs2, circulant_flash_attention(simfun, q, k, v, ws)), q, k, v)
            for (gr, gf) in zip(g_ref, g_fl)
                @test Array(gf) ≈ Array(gr)  rtol=1e-3 atol=1e-5
            end
        end
    end

    # Fill cotangent path (sum produces a FillArrays.Fill tangent)
    @testset "Fill cotangent [$tag]" begin
        simfun = RealDotSimilarity()
        g_ref = Zygote.gradient((q, k, v) -> sum(real(first(circulant_attention(simfun, q, k, v, ws)))), q, k, v)
        g_fl  = Zygote.gradient((q, k, v) -> sum(real(circulant_flash_attention(simfun, q, k, v, ws))), q, k, v)
        for (gr, gf) in zip(g_ref, g_fl)
            @test Array(gf) ≈ Array(gr)  rtol=1e-3 atol=1e-5
        end
    end

    # default simfun is DotSimilarity (real eltypes only)
    if elty <: Real
        @testset "default simfun [$tag]" begin
            y_ref, _ = circulant_attention(DotSimilarity(), q, k, v, ws)
            @test Array(circulant_flash_attention(q, k, v, ws)) ≈ Array(y_ref)  rtol=1e-4 atol=1e-6
        end
    end

    # channel counts spanning multiple register chunks, incl. partial chunks
    @testset "wide channels [$tag]" begin
        dw = 40
        qw = CUDA.randn(elty, spatdims..., dw, B)
        kw = CUDA.randn(elty, spatdims..., dw, B)
        vw = CUDA.randn(elty, spatdims..., dw, B)
        simfun = RealDotSimilarity()
        y_ref, _ = circulant_attention(simfun, qw, kw, vw, ws)
        y_fl     = circulant_flash_attention(simfun, qw, kw, vw, ws)
        @test Array(y_fl) ≈ Array(y_ref)  rtol=1e-4 atol=1e-6
    end

    # multi-head: compare against the composed path on manually split heads
    @testset "multi-head [$tag]" begin
        nheads = 2
        simfun = DistanceSimilarity()
        qr, kr, vr = CircAtt.splitheads.((q, k, v), nheads)
        y_ref, _ = circulant_attention(simfun, qr, kr, vr, ws)
        y_mh     = circulant_mh_flash_attention(simfun, q, k, v, ws, nheads)
        @test Array(y_mh) ≈ Array(reshape(y_ref, size(v)...))  rtol=1e-4 atol=1e-6
    end
end

# unsupported configurations raise informative errors
@testset "flash error paths" begin
    q = CUDA.randn(Float32, 8, 8, 4, 2)
    @test_throws ArgumentError circulant_flash_attention(TopKSimilarity(DotSimilarity(), 3), q, q, q, 5)
    @test_throws ArgumentError circulant_flash_attention(SparsemaxSimilarity(DotSimilarity()), q, q, q, 5)

    qc = CUDA.randn(ComplexF32, 8, 8, 4, 2)
    @test_throws ArgumentError circulant_flash_attention(DotSimilarity(), qc, qc, qc, 5)
end
end
