# test/rrule.jl
# Tests for circulant_similarity and circulant_attention rrules.

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

    ws1, ws2, ws3 = 3, 5, 7
    A1 = make_circulant(real(elty), spatdims, ws1, B)
    A2 = make_circulant(real(elty), spatdims, ws2, B)
    A3 = make_circulant(real(elty), spatdims, ws3, B)

    @testset "forward: joint normalisation [$tag, nspatdims=$nspatdims]" begin
        S1, S2, S3 = joint_softmax(A1, A2, A3)
        # Each row sums to 1 jointly across all three kernels
        total_nnz = ws1^nspatdims + ws2^nspatdims + ws3^nspatdims  # nnz_per_row for each
        W1, W2, W3 = windowview(S1), windowview(S2), windowview(S3)
        row_sums = sum(W1; dims=1) .+ sum(W2; dims=1) .+ sum(W3; dims=1)
        @test Array(row_sums) ≈ ones(real(elty), 1, size(W1)[2:end]...)  rtol=1e-4
    end

    @testset "gradient: softmax finite and non-nothing [$tag, nspatdims=$nspatdims]" begin
        gs = Zygote.gradient(A1, A2, A3) do a1, a2, a3
            S1 = NNlib.softmax(a1)
            S2 = NNlib.softmax(a2)
            S3 = NNlib.softmax(a3)
            sum(real.(S1)) + sum(real.(S2)) + sum(real.(S3))
        end

        for (i, g) in enumerate(gs)
            @test !isnothing(g) 
            @test all(isfinite.(Array(g.data.nzVal))) 
        end
    end

    @testset "gradient: joint-softmax finite and non-nothing [$tag, nspatdims=$nspatdims]" begin
        gs = Zygote.gradient(A1, A2, A3) do a1, a2, a3
            S1, S2, S3 = joint_softmax(a1, a2, a3)
            sum(real(S1)) + sum(real(S2)) + sum(real(S3))
        end

        for (i, g) in enumerate(gs)
            @test !isnothing(g) 
            @test all(isfinite.(Array(g.data.nzVal))) 
        end
    end

    if elty == Float32
        @testset "joint_softmax rrule [$tag]" begin
            for K in [2,]
                circs = ntuple(_ -> make_circulant(elty, spatdims, rand((3,5,7)), B), K)
                ΔAs   = ntuple(i -> rand_tangent(circs[i]), K)
                ΔYs   = ntuple(i -> rand_tangent(joint_softmax(circs...)[i]), K)
                test_rrule(
                    joint_softmax, map((c, Δ) -> c ⊢ Δ, circs, ΔAs)...;
                    output_tangent=ΔYs, rtol=1e-3, atol=1e-5, check_inferred=false,
                )
            end
        end
    end

end
end
