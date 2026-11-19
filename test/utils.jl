# ============================================================
# Helpers
# ============================================================
 
# selectdim on the last dimension — works regardless of nzVal rank
nz_batch(nzVal, b) = Array(selectdim(nzVal, ndims(nzVal), b))
nz_cpu(A::Circulant) = Array(A.data.nzVal)

# Build a random Circulant with known sparsity pattern.
function make_circulant(elty, spatdims, ws, batch)
    x = CUDA.randn(elty, spatdims..., size_d(elty), batch)
    circulant_similarity(CircAtt.DotSimilarity(), x, x, ws)
end

# Random tangent with same sparsity pattern, random nzVal.
function rand_tangent(A::Circulant{T}) where T
    B = copy(A)
    B.data.nzVal .= CUDA.randn(T, size(A.data.nzVal)...)
    return B
end

function rand_csr_tangent(A::Circulant{T}) where T
    CuSparseArrayCSR(
        copy(A.data.rowPtr), copy(A.data.colVal),
        CUDA.randn(real(T), size(A.data.nzVal)...), size(A.data),
    )
end

size_d(::Type{Float32})    = 4
size_d(::Type{ComplexF32}) = 4

# ============================================================
# CRC and FD Utils
# ============================================================
 
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

function FD.to_vec(A::CuArray{ComplexF32}) 
    xr, xi = reim(A)
    xr_vec, rback = FD.to_vec(xr)
    xi_vec, iback = FD.to_vec(xi)
    x_vec = [xr_vec; xi_vec]
    function complex_from_vec(x_v)
        N = length(x_v) ÷ 2
        xr_vec = x_v[1:N]
        xi_vec = x_v[N+1:end]
        xr = rback(xr_vec)
        xi = iback(xi_vec)
        x = xr +  1im .* xi
        return x
    end
    return x_vec, complex_from_vec
end
