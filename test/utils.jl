function ChainRulesTestUtils.test_approx(actual::CuSparseArrayCSR, expected::CuSparseArrayCSR, msg="", args...; rtol=1e-6, atol=1e-6, kws...)
    @test_msg msg*" rowPtrs do not match." actual.rowPtr ≈ expected.rowPtr rtol=rtol atol=atol
    @test_msg msg*" colVals do not match." actual.colVal ≈ expected.colVal rtol=rtol atol=atol
    @test_msg msg*" nzVals do not match." actual.nzVal ≈ expected.nzVal    rtol=rtol atol=atol
end
function ChainRulesTestUtils.test_approx(actual::Circulant, expected::Circulant, msg="", args...; kws...)
    @test_msg msg*" spatial-sizes do not match." actual.spatial_size == expected.spatial_size
    ChainRulesTestUtils.test_approx(actual.data, expected.data, msg, args...; kws...)
end

FD.to_vec(x::CuVector{<:Real}) = Array(x), cu
FD.to_vec(x::CuArray{<:Real}) = FD.to_vec(vec(x))[1], y->reshape(cu(y), size(x)...)
function FD.to_vec(A::CuSparseArrayCSR)
    x_vec, back = FD.to_vec(A.nzVal)
    function CuSparseArrayCSR_from_vec(x_v)
        v_values = back(x_v)
        out = copy(A)
        out.nzVal = v_values
        return out
    end
    return x_vec, CuSparseArrayCSR_from_vec
end
function FD.to_vec(A::Circulant{T, N, M}) where {T,N,M}
    x_vec, back = FD.to_vec(A.data)
    function Circulant_from_vec(x_v)
        data = back(x_v)
        return Circulant(data, M, A.spatial_size)
    end
    return x_vec, Circulant_from_vec
end
