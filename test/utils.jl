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

# Inputs for PIDot/PIDistance gradient checks. Their similarity takes |x⋅y|, which has a
# kink at x⋅y = 0, and finite differences across it are wrong. Entries of magnitude in
# [0.5, 1.5) (real part, for complex) with one random sign per pixel keep every dot product
# at |x⋅y| ≥ d/4 while still drawing both signs.
function pi_safe_rand(elty, dims...)
    R = real(elty)
    x = CUDA.rand(R, dims...) .+ R(0.5)
    elty <: Complex && (x = complex.(x, CUDA.rand(R, dims...)))
    sign.(CUDA.randn(R, dims[1:end-2]..., 1, dims[end])) .* x
end

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

# FiniteDifferences combines its stencil in the eltype of the vectors it perturbs. In
# Float32, every output the perturbation does not touch keeps the same rounding residual,
# which shifts the whole reference j′vp by an amount that grows with the number of outputs
# (the 2D tests have 40× as many as the 1D ones). So the finite differences run on Float64
# vectors around the Float32 functions, and FDM's `factor` tells the step selection that
# the function values are only Float32-accurate (eps(Float32)/eps(Float64) ≈ 5e8).
# Every test_rrule call passes fdm=FDM, or FDM_SMOOTH below.
const FDM = FD.central_fdm(5, 1; max_range=1e-2, factor=1e9)

# max_range caps the step, which FDM keeps small for sparsemax/entmax support boundaries.
# The Float32 rounding of the forward enters the estimate divided by the step, and through
# a softmax over 2D windows it can exceed rtol=1e-3 (flash attention with PIDistance).
# The flash kernels are smooth within ±0.1 of their inputs (for the PI similarities, of
# pi_safe_rand inputs), so their checks may take steps 10× wider.
const FDM_SMOOTH = FD.central_fdm(5, 1; max_range=1e-1, factor=1e9)

FD.to_vec(x::Float32) = [Float64(x)], v -> Float32(only(v))
FD.to_vec(x::CuVector{T}) where {T<:Real} = Float64.(Array(x)), v -> CuArray{T}(v)
FD.to_vec(x::CuArray{T}) where {T<:Real} = first(FD.to_vec(vec(x))), v -> reshape(CuArray{T}(v), size(x))
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
