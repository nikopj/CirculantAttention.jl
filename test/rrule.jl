@testset "$elty" for elty=(Float32, ComplexF32)
    @testset "rrule nspatdims=$(nspatdims)" for nspatdims=(1,2) 
        x = CUDA.randn(elty, ntuple(_->8, nspatdims)..., 4, 2)
        y = CUDA.randn(elty, ntuple(_->8, nspatdims)..., 4, 2)
        u = CUDA.randn(elty, ntuple(_->8, nspatdims)..., 4, 2)
        v = CUDA.randn(elty, ntuple(_->8, nspatdims)..., 4, 2)
        ws = 3
        A = circulant_similarity(DotSimilarity(), x, y, ws);
        B = copy(A)
        C = copy(A)
        A.data.nzVal = CUDA.randn(real(elty), size(A.data.nzVal)...)
        B.data.nzVal = CUDA.randn(real(elty), size(A.data.nzVal)...)
        C.data.nzVal = CUDA.randn(real(elty), size(A.data.nzVal)...)

        a = CUDA.randn(real(elty), 1, 1, 2, 2)
        b = CUDA.randn(real(elty), 1, 1, 2, 2)
        c = CUDA.randn(real(elty), 1, 1, 1, 1)

        test_rrule(circulant_similarity, DotSimilarity(), x ⊢ u, y ⊢ v, ws; output_tangent=A, rtol=1e-3, atol=1e-5, check_inferred=false)
        test_rrule(circulant_similarity, DistanceSimilarity(), x ⊢ u, y ⊢ v, ws; output_tangent=A, rtol=1e-3, atol=1e-5, check_inferred=false)
        test_rrule(circulant_attention, A ⊢ B, x ⊢ u; output_tangent=v, rtol=1e-3, atol=1e-5, check_inferred=false)
        test_rrule(NNlib.softmax, A ⊢ B; output_tangent=C, rtol=1e-3, atol=1e-3, check_inferred=false)
        test_rrule(CirculantAttention.scale, a, A ⊢ C; output_tangent=b*B, rtol=1e-3, atol=1e-4, check_inferred=false)
        test_rrule(CirculantAttention.scale, c, A ⊢ C; output_tangent=B, rtol=1e-3, atol=1e-4, check_inferred=false)
    end
end
