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

        # rrule of the fused core against FiniteDifferences (the public wrapper
        # only adds the τ-scaling broadcast, which Zygote traces)
        @testset "rrule $(typeof(simfun)) [$tag]" begin
            test_rrule(CircAtt._circulant_flash_attention, simfun,
                q ⊢ CUDA.randn(elty, size(q)...),
                k ⊢ CUDA.randn(elty, size(k)...),
                v ⊢ CUDA.randn(elty, size(v)...),
                ws;
                output_tangent=CUDA.randn(elty, size(v)...),
                rtol=1e-3, atol=1e-5, check_inferred=false)
        end
    end

    # Fill cotangent path (sum produces a FillArrays.Fill tangent on the flash
    # side; the reference gradient uses an explicit dense ones cotangent, since
    # the composed path's ⊗ pullback does not accept Fill tangents)
    @testset "Fill cotangent [$tag]" begin
        simfun = RealDotSimilarity()
        y_ref, back = Zygote.pullback((q, k, v) -> first(circulant_attention(simfun, q, k, v, ws)), q, k, v)
        g_ref = back(CUDA.fill(one(elty), size(y_ref)...))
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

    # warp / block / thread kernels: independent implementations of the same
    # fused math must agree in forward, lse, and all gradients
    @testset "kernel modes agree [$tag]" begin
        simfuns_wt = elty <: Complex ?
            (RealDotSimilarity(), DistanceSimilarity(), PIDotSimilarity()) :
            (DotSimilarity(), DistanceSimilarity(), PIDistanceSimilarity())
        for simfun in simfuns_wt
            Δ = CUDA.randn(elty, size(v)...)
            y_t, lse_t = CircAtt._circulant_flash_attention_fwd(simfun, q, k, v, ws; mode=:thread)
            g_t = CircAtt.∇circulant_flash_attention(simfun, Δ, y_t, lse_t, q, k, v, ws; mode=:thread)
            for mode in (:warp, :block)
                y_m, lse_m = CircAtt._circulant_flash_attention_fwd(simfun, q, k, v, ws; mode)
                @test Array(y_m)   ≈ Array(y_t)    rtol=1e-5 atol=1e-7
                @test Array(lse_m) ≈ Array(lse_t)  rtol=1e-5 atol=1e-7

                g_m = CircAtt.∇circulant_flash_attention(simfun, Δ, y_m, lse_m, q, k, v, ws; mode)
                for (gm, gt) in zip(g_m, g_t)
                    @test Array(gm) ≈ Array(gt)  rtol=1e-4 atol=1e-6
                end
            end
        end
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
