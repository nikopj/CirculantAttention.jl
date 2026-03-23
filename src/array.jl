# array.jl
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
    Circulant(Adapt.adapt_structure(to, A.data), M, A.spatial_size)
end

function Adapt.adapt_structure(to::CUDA.KernelAdaptor, A::CuSparseArrayCSR{T,Ti,N}) where {T,Ti,N}
    rowPtr = Adapt.adapt(to, A.rowPtr)
    colVal = Adapt.adapt(to, A.colVal)
    nzVal  = Adapt.adapt(to, A.nzVal)
    GPUArrays.GPUSparseDeviceArrayCSR{T, Ti, typeof(rowPtr), typeof(nzVal), N, N-1, 1}(rowPtr, colVal, nzVal, size(A), Ti(length(A.nzVal)))
end

CUDA.unsafe_free!(A::Circulant) = CUDA.unsafe_free!(A.data)

circulant(M::Int, x::AnyCuArray{T, N}) where {T,N} = Circulant{real(T)}(M, (size(x)[1:N-2]..., 1, size(x,N)))

kernel_length(A::Circulant{T, N, M}) where {T, N, M} = M
spatial_dims(A::Circulant{T, N, M, S}) where {T, N, M, S} = S
spatial_size(A::Circulant) = A.spatial_size

# ------------------------------------------------------------------------------
# windowview: canonical (nnz_per_row, n_rows, batch...) view of nzVal.
# Used throughout to avoid manual reshape arithmetic.
# ------------------------------------------------------------------------------
windowview(A::Circulant) = windowview(A.data)
windowview(A::CuSparseArrayCSR) = reshape(A.nzVal, :, A.dims[1], A.dims[3:end]...)

Base.eltype(A::Circulant{T}) where T = T
Base.size(A::Circulant)              = size(A.data)
Base.size(A::Circulant, i::Int)      = size(A.data, i)
Base.ndims(A::Circulant)             = ndims(A.data)
Base.getindex(A::Circulant, idxs...) = Base.getindex(A.data, idxs...)
Base.copy(A::Circulant{T,N,M})    where {T,N,M} = Circulant(copy(A.data), M, spatial_size(A))

Base.similar(A::Circulant{T,N,M}) where {T,N,M} = Circulant(similar(A.data), M, spatial_size(A))
Base.similar(A::Circulant{<:Any,N,M}, ::Type{T}) where {T,N,M} = Circulant(similar(A.data, T), M, spatial_size(A))
Base.similar(A::CuSparseArrayCSR, ::Type{T}) where T = CuSparseArrayCSR(copy(A.rowPtr), copy(A.colVal), similar(A.nzVal, T), size(A))

function Base.show(io::IOContext, m::MIME"text/plain", A::Circulant{T,N,M}) where {T,N,M}
    print(io, typeof(A), " with kernel-length $M, spatial-size $(spatial_size(A)), and data,\n")
    show(io, m, A.data)
end

# useful for showing the results of CircAtt.joint_softmax
function Base.show(io::IO, m::MIME"text/plain", As::Tuple{Vararg{Circulant}})
    print(io, length(As), "-tuple of Circulant matrices:\n")
    for (i, A) in enumerate(As)
        print(io, "  [$i] ")
        show(io, m, A)
        i < length(As) && print(io, "\n")
    end
end

function Base.repeat(A::Circulant{T,N,M}, dims::Int...) where {T,N,M}
    Circulant(repeat(A.data, dims...), M, spatial_size(A))
end

function Base.reshape(A::Circulant{T,N,M}, dims::Union{Colon,Int}...) where {T,N,M}
    Circulant(reshape(A.data, dims...), M, spatial_size(A))
end

function Base.cat(As::Circulant{T,N,M}...; dims=3) where {T,N,M}
    Circulant(cat([A.data for A in As]...; dims=dims), M, spatial_size(first(As)))
end

# ------------------------------------------------------------------
# Arithmetic
#
# Non-broadcasting +/- require identical sizes — use .+ / .- for
# batch-expanding operations.
# Scalar *, unary -, and CuArray * all delegate to broadcast.
# ------------------------------------------------------------------

function _check_same_size(X::Circulant, Y::Circulant)
    size(X) == size(Y) || throw(DimensionMismatch(
        "Circulant sizes $(size(X)) and $(size(Y)) must match for non-broadcasting " *
        "arithmetic. Use .+ / .- for batch-expanding operations."
    ))
end

function Base.:(+)(X::Circulant{Tx,N,M,S,A}, Y::Circulant{Ty,N,M,S,A}) where {Tx,Ty,N,M,S,A<:CuSparseArrayCSR}
    _check_same_size(X, Y)
    X .+ Y
end

function Base.:(-)(X::Circulant{Tx,N,M,S,A}, Y::Circulant{Ty,N,M,S,A}) where {Tx,Ty,N,M,S,A<:CuSparseArrayCSR}
    _check_same_size(X, Y)
    X .- Y
end

Base.:(-)(X::Circulant) = -one(eltype(X)) .* X

# ------------------------------------------------------------------
# Dot product — operates on nzVal directly
# ------------------------------------------------------------------

function LinearAlgebra.dot(X::Circulant{Tx,N,M,S,A}, Y::Circulant{Ty,N,M,S}) where {Tx,Ty,N,M,S,A<:CuSparseArrayCSR}
    dot(X.data.nzVal, Y.data.nzVal)
end

# ------------------------------------------------------------------
# Equality
# ------------------------------------------------------------------

function Base.:(==)(A::CuSparseArrayCSR, B::CuSparseArrayCSR)
    axes(A)  == axes(B)  || return false
    A.nzVal  == B.nzVal  || return false
    A.rowPtr == B.rowPtr || return false
    A.colVal == B.colVal || return false
    return true
end

function Base.:(==)(A::Circulant, B::Circulant)
    axes(A)          == axes(B)          || return false
    spatial_size(A)  == spatial_size(B)  || return false
    kernel_length(A) == kernel_length(B) || return false
    A.data           == B.data           || return false
    return true
end

# ------------------------------------------------------------------
# Sum helpers for CuSparseArrayCSR
#
# nzVal layout: (nnz_per_row, n_rows, batch...)
# Circulant dim k+1 corresponds to nzVal dim k for batch dims (k>1).
#
# sumdim1:  sum over rows   (Circulant dim 1) -> dense
# sumdim2:  sum over cols   (Circulant dim 2) -> dense
# sumdim3:  sum over batch  (Circulant dim >2) -> sparse, same pattern
# sumdim12: sum over rows+cols -> scalar-per-batch
# ------------------------------------------------------------------

function sumdim1(A::CuSparseArrayCSR{T}) where T
    # Aᵀ * ones gives column sums; use dim=2 size 2 to avoid n==1 errors
    o = CUDA.ones(T, (size(A,1), 2, prod(size(A)[3:end])))
    s = batched_transpose(reshape(A, :, :, :)) ⊠ o
    reshape(s[:,1,:], (1, size(A,1), size(A)[3:end]...))
end

function sumdim2(A::CuSparseArrayCSR)
    # windowview dim 1 is nnz_per_row (= kernel entries per row = col dimension)
    s = sum(windowview(A); dims=1)
    reshape(s, size(A,1), 1, size(A)[3:end]...)
end

function sumdim3(A::CuSparseArrayCSR, dims::Int...)
    @assert all(dims .> 2) "sumdim3 only reduces batch dims (dims > 2)"
    nzVal  = sum(A.nzVal; dims=dims .- 1)
    idx    = ntuple(i -> (i + 1) ∈ dims ? (1:1) : Colon(), ndims(A) - 1)
    rowPtr = copy(getindex(A.rowPtr, idx...))
    colVal = copy(getindex(A.colVal, idx...))
    CuSparseArrayCSR(rowPtr, colVal, nzVal, ntuple(i -> i ∈ dims ? 1 : size(A,i), ndims(A)))
end

function sumdim12(A::CuSparseArrayCSR)
    s = sum(A.nzVal; dims=1)
    reshape(s, 1, size(s)...)
end

function Base.sum(A::CuSparseArrayCSR; dims=:)
    dims == 1       && return sumdim1(A)
    dims == 2       && return sumdim2(A)
    dims == (1,2)   && return sumdim12(A)
    dims == Colon() && return sum(A.nzVal)
    all(dims .> 2)  && return sumdim3(A, dims...)
    throw(ErrorException("dims=$dims not implemented for CuSparseArrayCSR."))
end

function Base.sum(A::Circulant; dims=:)
    if dims != Colon() && all(dims .> 2)
        return Circulant(sum(A.data; dims=dims), kernel_length(A), spatial_size(A))
    end
    return sum(A.data; dims=dims)
end
