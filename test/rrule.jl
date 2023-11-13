@testset "rrule nspatdims=$(nspatdims)" for nspatdims=(1,2) 
    x = CUDA.randn(ntuple(_->16, nspatdims)..., 4, 2)
    y = CUDA.randn(ntuple(_->16, nspatdims)..., 4, 2)
    ws = 5
    A = circulant_similarity(DotSimilarity(), x, y, ws);
    A = reshape(A, :, :, :)
    B = copy(A)
    C = copy(A)
    A.data.nzVal = CUDA.randn(size(A.data.nzVal)...)
    B.data.nzVal = CUDA.randn(size(A.data.nzVal)...)
    C.data.nzVal = CUDA.randn(size(A.data.nzVal)...)

    CUDA.@allowscalar test_rrule(circulant_similarity, DotSimilarity(), x, y, ws; output_tangent=A, rtol=1e-3, atol=1e-5, check_inferred=false)
    CUDA.@allowscalar test_rrule(circulant_similarity, DistanceSimilarity(), x, y, ws; output_tangent=A, rtol=1e-3, atol=1e-5, check_inferred=false)
    CUDA.@allowscalar test_rrule(circulant_attention, A ⊢ B, x; rtol=1e-3, atol=1e-5, check_inferred=false)
    CUDA.@allowscalar test_rrule(NNlib.softmax, A ⊢ B; output_tangent=C, rtol=1e-3, atol=1e-3, check_inferred=false)
end
