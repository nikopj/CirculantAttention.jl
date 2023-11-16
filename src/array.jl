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
    Circulant(cat([A.data for A in As]...; dims=dims), M, spatial_size(A))
end

function Base.:(+)(X::Circulant{T, N, M, S, A}, Y::Circulant{T, N, M, S, A}) where {T, N, M, S, A<:CuSparseArrayCSR}
    data = CuSparseArrayCSR(copy(X.data.rowPtr), copy(X.data.colVal), X.data.nzVal + Y.data.nzVal, size(X))
    Circulant(data, M, spatial_size(X))
end

function Base.:(-)(X::Circulant{T, N, M, S, A}, Y::Circulant{T, N, M, S, A}) where {T, N, M, S, A<:CuSparseArrayCSR}
    data = CuSparseArrayCSR(copy(X.data.rowPtr), copy(X.data.colVal), X.data.nzVal - Y.data.nzVal, size(X))
    Circulant(data, M, spatial_size(X))
end

function Base.:(*)(c::Number, X::Circulant{T, N, M, S, A}) where {T, N, M, S, A <: CuSparseArrayCSR}
    data = CuSparseArrayCSR(copy(X.data.rowPtr), copy(X.data.colVal), c .* X.data.nzVal, size(X))
    Circulant(data, M, spatial_size(X))
end
Base.:(-)(A::Circulant) = -1*A
Base.:(*)(A::Circulant, c::Number) = c*A

function LinearAlgebra.dot(X::Circulant{T, N, M, S, A}, Y::Circulant{T, N, M, S, A}) where {T, N, M, S, A <: CuSparseArrayCSR}
    return dot(X.data.nzVal, Y.data.nzVal)
end

function Base.:(==)(A::CuSparseArrayCSR, B::CuSparseArrayCSR)
    if axes(A) != axes(B)
        return false
    end
    if A.nzVal != B.nzVal
        return false
    end
    if A.rowPtr != B.rowPtr
        return false
    end
    if A.colVal != B.colVal
        return false
    end
    return true
end

function Base.:(==)(A::Circulant, B::Circulant)
    if axes(A) != axes(B)
        return false
    end
    if spatial_size(A) != spatial_size(B)
        return false
    end
    if kernel_length(A) != kernel_length(B)
        return false
    end
    if A.data != B.data
        return false
    end
    return true
end

function sumdim1(A::CuSparseArrayCSR{T}) where T
    o = CUDA.ones(T, (size(A,1), 2, prod(size(A)[3:end])))
    # second dim is 2 bc n==1 causes errors
    s =  batched_transpose(reshape(A, :, :, :)) ⊠ o
    reshape(s[:,1,:], (1, size(A, 1), size(A)[3:end]...))
end

function sumdim2(A::CuSparseArrayCSR)
    V = reshape(A.nzVal, :, size(A, 1), size(A)[3:end]...)
    s = sum(V; dims=1)
    return reshape(s, size(A,1), 1, size(A)[3:end]...)
end

function Base.sum(A::CuSparseArrayCSR; dims=:)
    if dims == 1
        return sumdim1(A)
    elseif dims == 2
        return sumdim2(A)
    elseif dims == Colon()
        return sum(A.nzVal)
    else
        throw(ErrorException("dims=$dims not implemented."))
    end
end
Base.sum(A::Circulant; dims=:) = sum(A.data; dims=dims)

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

