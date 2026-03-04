# test/rrule.jl
# Tests for circulant_similarity and circulant_attention rrules.
# Low-level Circulant arithmetic rrules are tested in test/lowlevel_rrules.jl.

@testset "Similarity and Attention" begin 
for elty in TEST_ELTYPES, nspatdims in TEST_SPATDIMS
    N  = 8
    d  = 4
    B  = 2
    ws = 5
    spatdims = ntuple(_ -> N, nspatdims)
    tag = "elty=$elty, nspatdims=$nspatdims"

    x = CUDA.randn(elty, spatdims..., d, B)
    y = CUDA.randn(elty, spatdims..., d, B)
    u = CUDA.randn(elty, spatdims..., d, B)
    v = CUDA.randn(elty, spatdims..., d, B)

    # Reference Circulants for output tangents.
    # nzVals are offset from zero to avoid sign discontinuities in PI variants.
    A = circulant_similarity(RealDotSimilarity(), x, y, ws)
    B_circ = copy(A)
    C_circ = copy(A)
    for circ in (A, B_circ, C_circ)
        circ.data.nzVal .= CUDA.randn(real(elty), size(circ.data.nzVal)...) .+ real(elty)(0.5)
    end

    Ac = circulant_similarity(DotSimilarity(), x, y, ws)
    Bc = copy(Ac)
    for circ in (Ac, Bc)
        circ.data.nzVal .= CUDA.randn(elty, size(circ.data.nzVal)...) .+ elty(0.5)
    end

    # ------------------------------------------------------------------
    # circulant_similarity rrules
    # ------------------------------------------------------------------

    @testset "circulant_similarity RealDotSimilarity [$tag]" begin
        test_rrule(circulant_similarity, RealDotSimilarity(), x ⊢ u, y ⊢ v, ws;
            output_tangent=A, rtol=1e-3, atol=1e-5, check_inferred=false)
    end

    @testset "circulant_similarity DotSimilarity [$tag]" begin
        test_rrule(circulant_similarity, DotSimilarity(), x ⊢ u, y ⊢ v, ws;
            output_tangent=Ac, rtol=1e-3, atol=1e-5, check_inferred=false)
    end

    @testset "circulant_similarity DistanceSimilarity [$tag]" begin
        test_rrule(circulant_similarity, DistanceSimilarity(), x ⊢ u, y ⊢ v, ws;
            output_tangent=A, rtol=1e-3, atol=1e-5, check_inferred=false)
    end

    @testset "circulant_similarity PIDotSimilarity [$tag]" begin
        test_rrule(circulant_similarity, PIDotSimilarity(), x ⊢ u, y ⊢ v, ws;
            output_tangent=A, rtol=1e-3, atol=1e-5, check_inferred=false)
    end

    @testset "circulant_similarity PIDistanceSimilarity [$tag]" begin
        test_rrule(circulant_similarity, PIDistanceSimilarity(), x ⊢ u, y ⊢ v, ws;
            output_tangent=A, rtol=1e-3, atol=1e-5, check_inferred=false)
    end

    # ------------------------------------------------------------------
    # circulant_attention rrule (A ⊗ x)
    # Real Circulant with real features
    # ------------------------------------------------------------------

    @testset "circulant_attention (real sim, real features) [$tag]" begin
        test_rrule(circulant_attention, A ⊢ B_circ, x ⊢ u;
            output_tangent=v, rtol=1e-3, atol=1e-5, check_inferred=false)
    end

    # Complex Circulant only meaningful for ComplexF32
    if elty == ComplexF32
        @testset "circulant_attention (complex sim, complex features) [$tag]" begin
            test_rrule(circulant_attention, Ac ⊢ Bc, x ⊢ u;
                output_tangent=v, rtol=1e-3, atol=1e-5, check_inferred=false)
        end

        @testset "circulant_attention (complex sim, real features) [$tag]" begin
            test_rrule(circulant_attention, Ac ⊢ Bc, real(x) ⊢ real(u);
                output_tangent=v, rtol=1e-3, atol=1e-5, check_inferred=false)
        end

        @testset "circulant_attention (mixed: real sim, complex features) [$tag]" begin
            test_rrule(circulant_attention, A ⊢ Bc, x ⊢ u;
                output_tangent=v, rtol=1e-3, atol=1e-5, check_inferred=false)
        end
    end

    # ------------------------------------------------------------------
    # NNlib.softmax on a Circulant — uses windowview + NNlib rrule, no
    # custom rrule needed. Verified here that the chain composes correctly.
    # ------------------------------------------------------------------

    @testset "softmax (Circulant) [$tag]" begin
        test_rrule(NNlib.softmax, A ⊢ B_circ;
            output_tangent=C_circ, rtol=1e-3, atol=1e-3, check_inferred=false)
    end

    # ------------------------------------------------------------------
    # CuArray .* Circulant — replaces scale; Zygote traces through broadcast.
    # Verified against analytically known gradients.
    # ------------------------------------------------------------------

    @testset "CuArray .* Circulant (per-batch scale) [Zygote] [$tag]" begin
        # α is (1,1,1,B) — scales each batch element independently
        B_size = size(A, ndims(A))
        α = CUDA.randn(real(elty), 1, 1, 1, B_size)
        α_cpu = Array(α)
        nz_A  = Array(A)

        gs = Zygote.gradient(α, A) do a, x
            sum(real.((a .* x)))
        end

        # ∂α[b] = sum of A.nzVal for that batch
        for b in 1:B_size
            @test Array(gs[1])[1,1,1,b] ≈ sum(real.(nz_A[:,b]))  rtol=1e-3
        end
        # ∂A.nzVal[i,b] = α[b]
        for b in 1:B_size
            @test all(≈(real(α_cpu[1,1,1,b]); rtol=1e-3), Array(gs[2].data.nzVal)[:,b])
        end
    end

    @testset "CuArray .* Circulant (scalar expand 1x1x1x1) [Zygote] [$tag]" begin
        α = CUDA.randn(real(elty), 1, 1, 1, 1)
        nz_A = Array(A.data.nzVal)

        gs = Zygote.gradient(α, A) do a, x
            sum(real.((a .* x)))
        end

        # ∂α = sum of all A.nzVal (scalar accumulated over all batch/entries)
        @test Array(gs[1])[1] ≈ sum(real.(nz_A))  rtol=1e-3
        # ∂A.nzVal = α everywhere
        α_val = real(Array(α)[1])
        @test all(≈(α_val; rtol=1e-3), Array(gs[2].data.nzVal))
    end

    # ------------------------------------------------------------------
    # Convex combination: α*A + (1-α)*B where α is a CuArray
    # This is the pattern from grad.jl — exercises CuArray broadcast + add rrules
    # ------------------------------------------------------------------

    @testset "Convex combination α*A + (1-α)*B [Zygote end-to-end] [$tag]" begin
        α = CUDA.rand(real(elty), 1, 1, 1, 1)  # scalar in (0,1)
        gs = Zygote.gradient(α) do a
            C = a .* A + (real(elty)(1) .- a) .* B_circ
            sum(real.(C))
        end
        # ∂α = sum(A.nzVal) - sum(B.nzVal)
        expected = sum(Array(A.data.nzVal)) - sum(Array(B_circ.data.nzVal))
        @test Array(gs[1])[1] ≈ real(expected)  rtol=1e-3
    end
end
end
