struct Circulant{T, N, M, S, A<:AbstractArray{T, N}} <: AbstractArray{T, N}
    data::A
    kernel_length::Int
    spatial_size::NTuple{S, Int}

    function Circulant{T}(M::Int, dims::NTuple{N,Int}) where {T,N}
        spatsize, batchsize = dims[1:N-2], dims[N-1:end]
        data = cucirculant(M, spatsize..., T)
        data = repeat(data, 1, 1, batchsize...)
        Circulant(data, M, spatsize)
    end
    function Circulant(M::Int, dims::NTuple{N,Int}) where N
        Circulant{Float32}(M, dims)
    end
    function Circulant(D::A, M::Int, spatsize::NTuple{S,Int}) where {T, N, S, A<:AbstractArray{T,N}}
        new{T, N, M, S, A}(D, M, spatsize)
    end
end
Circulant(A::Circulant) = A
Circulant(a::T, M, spatdims) where T <: Number = a

function Adapt.adapt_structure(to, A::Circulant{T, N, M}) where {T, N, M}
    data = Adapt.adapt_structure(to, A.data)
    Circulant(data, M, A.spatial_size)
end

CUDA.unsafe_free!(A::Circulant) = CUDA.unsafe_free!(A.data)

circulant(M::Int, x::AnyCuArray{T, N}) where {T,N} = Circulant{real(T)}(M, (size(x)[1:N-2]..., 1, size(x,N)))

kernel_length(A::Circulant{T, N, M}) where {T, N, M} = M
spatial_dims(A::Circulant{T, N, M, S}) where {T, N, M, S} = S
spatial_size(A::Circulant) = A.spatial_size

Base.eltype(A::Circulant{T}) where T = T
Base.size(A::Circulant) = size(A.data)
Base.size(A::Circulant, i::Int) = size(A.data, i)
Base.ndims(A::Circulant) = ndims(A.data)
Base.getindex(A::Circulant, idxs...) = Base.getindex(A.data, idxs...)

Base.similar(A::Circulant{T, N, M}) where {T, N, M} = Circulant(similar(A.data), M, spatial_size(A))
Base.copy(A::Circulant{T, N, M}) where {T, N, M} = Circulant(copy(A.data), M, spatial_size(A))

# ------------------------------------------------------------------------------
# windowview: canonical (nnz_per_row, n_rows, batch...) view of nzVal.
# Used throughout to avoid manual reshape arithmetic.
# ------------------------------------------------------------------------------
windowview(A::Circulant) = windowview(A.data)
windowview(A::CuSparseArrayCSR) = reshape(A.nzVal, :, A.dims[1], A.dims[3:end]...)
# Flatten a window-shaped result back to the nzVal storage shape of a reference CSR.
_flatten_window(W, ref::CuSparseArrayCSR) = reshape(W, size(ref.nzVal, 1), size(W)[3:end]...)

function Base.show(io::IOContext, m::MIME"text/plain", A::Circulant{T, N, M}) where {T, N, M}
    print(io, typeof(A), " with kernel-length $M, spatial-size $(spatial_size(A)), and data,\n")
    show(io, m, A.data)
end

function Base.repeat(A::Circulant{T, N, M}, dims::Int...) where {T, N, M}
    data = repeat(A.data, dims...)
    Circulant(data, M, spatial_size(A))
end

function Base.reshape(A::Circulant{T, N, M}, dims::Union{Colon,Int}...) where {T, N, M}
    Circulant(reshape(A.data, dims...), M, spatial_size(A))
end

function _circulant_reshape(A::Circulant{T, N, M}, dims...) where {T, N, M}
    reshape(A, dims...)
end

function Base.cat(As::Circulant{T, N, M}...; dims=3) where {T, N, M}
    Circulant(cat([A.data for A in As]...; dims=dims), M, spatial_size(first(As)))
end

# ------------------------------------------------------------------
# Arithmetic — delegate to nzVal ops, sharing rowPtr/colVal by reference.
# Copies of rowPtr/colVal are only made when the result must own its
# own sparsity structure (i.e. when the inputs may be mutated later).
# For +/-/* the output nzVal is a fresh allocation from the broadcast,
# so we can share the structural arrays safely via copy (they are small
# relative to nzVal for large batches).
# ------------------------------------------------------------------

function _csr_like(src::CuSparseArrayCSR, nzVal::CuArray, sz=size(src))
    CuSparseArrayCSR(copy(src.rowPtr), copy(src.colVal), nzVal, sz)
end

function Base.:(+)(X::Circulant{T, N, M, S, A}, Y::Circulant{T, N, M, S, A}) where {T, N, M, S, A<:CuSparseArrayCSR}
    Circulant(_csr_like(X.data, X.data.nzVal .+ Y.data.nzVal), M, spatial_size(X))
end

function Base.:(-)(X::Circulant{T, N, M, S, A}, Y::Circulant{T, N, M, S, A}) where {T, N, M, S, A<:CuSparseArrayCSR}
    Circulant(_csr_like(X.data, X.data.nzVal .- Y.data.nzVal), M, spatial_size(X))
end

function Base.:(*)(c::Union{Real, Complex}, X::Circulant{T, N, M, S, A}) where {T, N, M, S, A<:CuSparseArrayCSR}
    Circulant(_csr_like(X.data, c .* X.data.nzVal), M, spatial_size(X))
end

Base.:(-)(A::Circulant{T}) where {T} = -T(1) * A
Base.:(*)(A::Circulant, c::Number) = c * A

function LinearAlgebra.dot(X::Circulant{Tx, N, M, S, A}, Y::Circulant{Ty, N, M, S}) where {Tx, Ty, N, M, S, A<:CuSparseArrayCSR}
    dot(X.data.nzVal, Y.data.nzVal)
end
function LinearAlgebra.dot(X::Circulant{T, N, M, S, A}, f::Zygote.FillArrays.Fill) where {T, N, M, S, A<:CuSparseArrayCSR}
    conj(sum(X.data.nzVal)) * f.value
end
function LinearAlgebra.dot(f::Zygote.FillArrays.Fill, X::Circulant{T, N, M, S, A}) where {T, N, M, S, A<:CuSparseArrayCSR}
    sum(X.data.nzVal) * conj(f.value)
end

# ------------------------------------------------------------------
# Equality
# ------------------------------------------------------------------

function Base.:(==)(A::CuSparseArrayCSR, B::CuSparseArrayCSR)
    axes(A) == axes(B)     || return false
    A.nzVal  == B.nzVal    || return false
    A.rowPtr == B.rowPtr   || return false
    A.colVal == B.colVal   || return false
    return true
end

function Base.:(==)(A::Circulant, B::Circulant)
    axes(A)         == axes(B)         || return false
    spatial_size(A) == spatial_size(B) || return false
    kernel_length(A)== kernel_length(B)|| return false
    A.data          == B.data          || return false
    return true
end

# ------------------------------------------------------------------
# Sum helpers for CuSparseArrayCSR
#
# nzVal layout: (nnz_per_row, n_rows, batch...)
# so nzVal dim k corresponds to Circulant dim k+1 for batch dims.
#
# sumdim1: sum over rows   (Circulant dim 1) → dense result
# sumdim2: sum over cols   (Circulant dim 2) → dense result, shape (n_rows, 1, batch...)
# sumdim3: sum over batch  (Circulant dim >2) → sparse result, same sparsity
# sumdim12: sum over rows+cols → scalar-per-batch
# ------------------------------------------------------------------

function sumdim1(A::CuSparseArrayCSR{T}) where T
    # Multiply Aᵀ by a ones vector: result shape (1, n_rows, batch...)
    o = CUDA.ones(T, (size(A,1), 2, prod(size(A)[3:end])))
    # second dim is 2 bc n==1 causes errors
    s = batched_transpose(reshape(A, :, :, :)) ⊠ o
    reshape(s[:,1,:], (1, size(A, 1), size(A)[3:end]...))
end

function sumdim2(A::CuSparseArrayCSR)
    # Sum nzVal along the nnz-per-row axis (dim 1 of nzVal = dim 2 of Circulant)
    V = reshape(A.nzVal, :, size(A, 1), size(A)[3:end]...)
    s = sum(V; dims=1)
    reshape(s, size(A,1), 1, size(A)[3:end]...)
end

function sumdim3(A::CuSparseArrayCSR, i::Int)
    @assert i > 2 "sumdim3 can only sum batch dimensions of a CSR"
    nzVal  = sum(A.nzVal; dims=i-1)
    # Keep only the first slice of the structural arrays along the summed dim
    rowPtr = copy(selectdim(A.rowPtr, i-1, 1:1))
    colVal = copy(selectdim(A.colVal, i-1, 1:1))
    sz     = ntuple(d -> d == i ? 1 : size(A, d), ndims(A))
    CuSparseArrayCSR(rowPtr, colVal, nzVal, sz)
end

function sumdim3(A::CuSparseArrayCSR, dims::Int...)
    @assert all(dims .> 2) "sumdim3 can only sum batch dimensions of a CSR"
    nzVal  = sum(A.nzVal; dims=dims .- 1)
    idx    = ntuple(i -> (i + 1) ∈ dims ? (1:1) : Colon(), ndims(A) - 1)
    rowPtr = copy(getindex(A.rowPtr, idx...))
    colVal = copy(getindex(A.colVal, idx...))
    sz     = ntuple(i -> i ∈ dims ? 1 : size(A, i), ndims(A))
    CuSparseArrayCSR(rowPtr, colVal, nzVal, sz)
end

function sumdim12(A::CuSparseArrayCSR)
    s = sum(A.nzVal; dims=1)
    reshape(s, 1, size(s)...)
end

function Base.sum(A::CuSparseArrayCSR; dims=:)
    if dims == 1
        return sumdim1(A)
    elseif dims == 2
        return sumdim2(A)
    elseif dims == Colon()
        return sum(A.nzVal)
    elseif all(dims .> 2)
        return sumdim3(A, dims...)
    elseif dims == (1, 2)
        return sumdim12(A)
    else
        throw(ErrorException("dims=$dims not implemented for CuSparseArrayCSR."))
    end
end

function Base.sum(A::Circulant; dims=:)
    # Batch dims (>2): result is still a Circulant with the same sparsity pattern
    if dims != Colon() && all(dims .> 2)
        return Circulant(sum(A.data; dims=dims), kernel_length(A), spatial_size(A))
    end
    # dims 1, 2, (1,2), or : all return dense arrays
    return sum(A.data; dims=dims)
end

function scale(c::CuArray{Tc,N}, A::CuSparseArrayCSR{Ta,Ti,N}) where {Tc, Ta, Ti, N}
    @assert size(c, 1) == 1 && size(c, 2) == 1 "Scaling of non-batchdims of CuSparseArrayCSR not implemented"
    nzVal = selectdim(c, 1, 1) .* A.nzVal
    rowPtr = repeat(A.rowPtr, 1, ntuple(i->size(nzVal,i+1) ÷ size(A.rowPtr,i+1),N-2)...)
    colVal = repeat(A.colVal, 1, ntuple(i->size(nzVal,i+1) ÷ size(A.colVal,i+1),N-2)...)
    return CuSparseArrayCSR(rowPtr, colVal, nzVal, (size(A,1), size(A,2), size(nzVal)[2:end]...))
end
scale(c::CuArray{T1,N}, A::Circulant{T2,N,M}) where {T1,T2,N,M} = Circulant(scale(c, A.data), M, spatial_size(A))
scale(c, A) = c .* A

Base.:(*)(c::CuArray, A::Circulant) = scale(c, A)

# function _csr_broadcast(f, A::CuSparseArrayCSR{Ta,Ti,N}, c::CuArray) where {Tc, Ta, Ti, N}
#     @assert size(c, 1) == 1 && size(c, 2) == 1 "Scaling of non-batchdims of CuSparseArrayCSR not implemented"
#     nzVal = f(windowview(A), C)
#     rowPtr = repeat(A.rowPtr, 1, ntuple(i->size(nzVal,i+1) ÷ size(A.rowPtr,i+1),N-2)...)
#     colVal = repeat(A.colVal, 1, ntuple(i->size(nzVal,i+1) ÷ size(A.colVal,i+1),N-2)...)
#     return CuSparseArrayCSR(rowPtr, colVal, nzVal, (size(A,1), size(A,2), size(nzVal)[2:end]...))
# end
