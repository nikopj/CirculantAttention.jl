# @testset "rrule $elty, nspatdims=$nspatdims" for elty=(Float32, ComplexF32), nspatdims=(1,2) 
@testset "rrule $elty, nspatdims=$nspatdims" for elty=(Float32, ComplexF32), nspatdims=(1,) 
    x = CUDA.randn(elty, ntuple(_->8, nspatdims)..., 4, 2)
    y = CUDA.randn(elty, ntuple(_->8, nspatdims)..., 4, 2)
    u = CUDA.randn(elty, ntuple(_->8, nspatdims)..., 4, 2)
    v = CUDA.randn(elty, ntuple(_->8, nspatdims)..., 4, 2)
    ws = 5
    A = circulant_similarity(RealDotSimilarity(), x, y, ws);
    B = copy(A)
    C = copy(A)
    # we're adding 0.5 here because gradient is unstable at zero (for Phase-Invariant Sims) so this will get rid of some errors due to numerical instability.
    A.data.nzVal = CUDA.randn(real(elty), size(A.data.nzVal)...) .+ real(elty(0.5))
    B.data.nzVal = CUDA.randn(real(elty), size(A.data.nzVal)...) .+ real(elty(0.5))
    C.data.nzVal = CUDA.randn(real(elty), size(A.data.nzVal)...) .+ real(elty(0.5))

    Ac = circulant_similarity(DotSimilarity(), x, y, ws);
    Bc = copy(Ac)
    Ac.data.nzVal = CUDA.randn(elty, size(Ac.data.nzVal)...) .+ elty(0.5)
    Bc.data.nzVal = CUDA.randn(elty, size(Ac.data.nzVal)...) .+ elty(0.5)

    a = CUDA.randn(real(elty), 1, 1, 2, 2)
    b = CUDA.randn(real(elty), 1, 1, 2, 2)
    c = CUDA.randn(real(elty), 1, 1, 1, 1)

    test_rrule(circulant_similarity, RealDotSimilarity(), x ⊢ u, y ⊢ v, ws; output_tangent=A, rtol=1e-3, atol=1e-5, check_inferred=false)
    test_rrule(circulant_similarity, DistanceSimilarity(), x ⊢ u, y ⊢ v, ws; output_tangent=A, rtol=1e-3, atol=1e-5, check_inferred=false)
    test_rrule(circulant_similarity, PIDotSimilarity(), x ⊢ u, y ⊢ v, ws; output_tangent=A, rtol=1e-3, atol=1e-5, check_inferred=false)
    test_rrule(circulant_similarity, PIDistanceSimilarity(), x ⊢ u, y ⊢ v, ws; output_tangent=A, rtol=1e-3, atol=1e-5, check_inferred=false)
    test_rrule(circulant_similarity, CircAtt.DotSimilarity(), x ⊢ u, y ⊢ v, ws; output_tangent=Ac, rtol=1e-3, atol=1e-5, check_inferred=false)

    test_rrule(circulant_attention, A ⊢ B, x ⊢ u; output_tangent=v, rtol=1e-3, atol=1e-5, check_inferred=false)

    if elty == ComplexF32
        test_rrule(circulant_attention, Ac ⊢ Bc, x ⊢ u; output_tangent=v, rtol=1e-3, atol=1e-5, check_inferred=false)
        test_rrule(circulant_attention, Ac ⊢ Bc, real(x) ⊢ u; output_tangent=v, rtol=1e-3, atol=1e-5, check_inferred=false)
        test_rrule(circulant_attention, A ⊢ Bc, x ⊢ u; output_tangent=v, rtol=1e-3, atol=1e-5, check_inferred=false)
    end

    test_rrule(NNlib.softmax, A ⊢ B; output_tangent=C, rtol=1e-3, atol=1e-3, check_inferred=false)
    test_rrule(CirculantAttention.scale, a ⊢ b, A ⊢ C; output_tangent=b*B, rtol=1e-3, atol=1e-4, check_inferred=false)
end
