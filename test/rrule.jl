@testset "rrule nspatdims=$(nspatdims)" for nspatdims=(1,2) 
    x = CUDA.randn(ntuple(_->8, nspatdims)..., 4, 2)
    y = CUDA.randn(ntuple(_->8, nspatdims)..., 4, 2)
    ws = 3
    A = circulant_similarity(DotSimilarity(), x, y, ws);
    B = copy(A)
    C = copy(A)
    A.data.nzVal = CUDA.randn(size(A.data.nzVal)...)
    B.data.nzVal = CUDA.randn(size(A.data.nzVal)...)
    C.data.nzVal = CUDA.randn(size(A.data.nzVal)...)

    a = CUDA.randn(1,1,2,2)
    b = CUDA.randn(1,1,2,2)
    c = CUDA.randn(1,1,1,1)

    test_rrule(circulant_similarity, DotSimilarity(), x, y, ws; output_tangent=A, rtol=1e-3, atol=1e-5, check_inferred=false)
    test_rrule(circulant_similarity, DistanceSimilarity(), x, y, ws; output_tangent=A, rtol=1e-3, atol=1e-5, check_inferred=false)
    test_rrule(circulant_attention, A ⊢ B, x; rtol=1e-3, atol=1e-5, check_inferred=false)
    test_rrule(NNlib.softmax, A ⊢ B; output_tangent=C, rtol=1e-3, atol=1e-3, check_inferred=false)
    test_rrule(CirculantAttention.scale, a, A ⊢ C; output_tangent=b*B, rtol=1e-3, atol=1e-5, check_inferred=false)
    test_rrule(CirculantAttention.scale, c, A ⊢ C; output_tangent=B, rtol=1e-3, atol=1e-5, check_inferred=false)
end
