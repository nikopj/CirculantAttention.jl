abstract type AbstractSimilarity end

@doc raw"""
    DotSimilarity()

Used in `circulant_attention`, `circulant_similarity`, and `circulant_adjacency` to indicate use of
dot-product similarity:

``
    S_{ij} = \mathrm{Real}(q[i]^H k[j]).
``

See also [`DistanceSimilarity`](@ref).
"""
struct DotSimilarity <: AbstractSimilarity end 

@doc raw"""
    DistanceSimilarity()

Used in `circulant_attention`, `circulant_similarity`, and `circulant_adjacency` to indicate use of
distance similarity:

``
    S_{ij} = \frac{1}{2}\mathrm{sum}(\mathrm{abs2}, q[i] - k[j]).
``

See also [`DotSimilarity`](@ref).
"""
struct DistanceSimilarity <: AbstractSimilarity end

function circulant_similarity!(A::Circulant{T, N}, simfun, x, y) where {T, N}
    circulant_similarity!(reshape(A, :, :, :), simfun, x, y)
    return A
end

function circulant_similarity!(
        A::Circulant{Tv, 3, W, S}, 
        simfun::AbstractSimilarity, 
        x::AbstractArray{Tx,N}, 
        y::AbstractArray{Ty,N},
    ) where {Tv, W, S, Tx, Ty, N}

    @assert S == N - 2 "spatial_dims ($(S)) of circulant matrix must match those of data signal ($(N-2))."
    @assert size(A, 3) == size(x, N) "batchdim of circulant matrix ($(size(A,3))) must match that of x, y ($(size(x,N)))."
    maxidx = A.data.nnz 
    args = A, simfun, x, y, maxidx
    kernel = @cuda launch=false circulant_similarity_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(maxidx, config.threads)
    blocks = cld(maxidx, threads)
    kernel(args...; threads=threads, blocks=blocks)
    return A
end

function circulant_similarity_kernel!(
        S::Circulant{Tv, 3, W}, 
        ::DistanceSimilarity, 
        x::AbstractArray{Tx, N}, 
        y, 
        maxidx,
    ) where {Tv, W, Tx, N}

    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds if idx <= maxidx
        B = size(S, 3)
        n, b = CartesianIndices((S.data.nnz ÷ B, B))[idx].I
        spatdims = ntuple(i->size(x, i), N-2)
        C = CartesianIndices(spatdims)
        i, j = cartesian_circulant(n, spatdims, W)
        Ci, Cj = C[i], C[j]
        s = zero(Tv)
        for m=1:size(x, N-1)
            s -= abs2(x[Cj, m, b] - y[Ci, m, b])
        end
        S.data.nzVal[n, b] = s / Tv(2)
    end
    return nothing
end

function circulant_similarity_kernel!(
        S::Circulant{Tv, 3, W}, 
        ::DotSimilarity, 
        x::AbstractArray{Tx,N}, 
        y, 
        maxidx,
    ) where {Tv, W, Tx, N}

    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds if idx <= maxidx
        B = size(S, 3)
        n, b = CartesianIndices((S.data.nnz ÷ B, B))[idx].I
        spatdims = ntuple(i->size(x, i), N-2)
        C = CartesianIndices(spatdims)
        i, j = cartesian_circulant(n, spatdims, W)
        Ci, Cj = C[i], C[j]
        s = zero(Tv)
        for m=1:size(x, N-1)
            s += real(x[Cj, m, b]*conj(y[Ci, m, b]))
        end
        S.data.nzVal[n, b] = s
    end
    return nothing
end

function circulant_softmax!(Y::CuSparseArrayCSR, X::CuSparseArrayCSR=Y)
    V = reshape(X.nzVal, :, X.dims[1], prod(X.dims[3:end]))
    U = reshape(Y.nzVal, size(V))
    NNlib.softmax!(U, V; dims=1)
    return Y
end
circulant_softmax(X::CuSparseArrayCSR) = circulant_softmax!(copy(X), X)

function NNlib.softmax!(A::Circulant{T,N,M}, B::Circulant{T,N,M}=A) where {T,N,M} 
    circulant_softmax!(A.data, B.data)
    return A
end

"""
    NNlib.softmax(A::Circulant)

Row-wise softmax of Circulant matrix `A`. 
"""
function NNlib.softmax(A::Circulant) 
    data = circulant_softmax(A.data)
    return Circulant(data, kernel_length(A), spatial_size(A))
end

"""
    circulant_similarity(simfun::AbstractSimilarity, x, y, W::Int)

Returns Circulant matrix with circulant-sparse data. Each non-zero `S[i,j,b]` is 
populated by `simfun` evaluated at the linearized pixel locations of `x` and `y`, 
i.e. `S[i,j,b] = simfun(x[...,i,b], y[...,j,b], W)` for max(i⃗, j⃗) ≤ W. The non-zero entrie 
locations are determined by the windowsize `W` and number of spatial dimensions in `x` and `y`.

See also [`DotSimilarity`](@ref), [`DistanceSimilarity`](@ref).
"""
function circulant_similarity(simfun::AbstractSimilarity, x::AbstractArray{T,N}, y::AbstractArray{T,N}, W::Integer)::Circulant where {T,N}
    S = Circulant(W, (size(x)[1:N-2]..., 1, size(x, N)))
    circulant_similarity!(S, simfun, x, y)
    return S
end

function circulant_adjacency!(A::Circulant, simfun, x, y)
    circulant_similarity!(A, simfun, x, y)
    NNlib.softmax!(A)
    return A
end

"""
    circulant_adjacency(simfun::AbstractSimilarity, x, y, W::Int)

Equivalent to `(softmax ∘ circulant_similarity)(simfun, x, y, W)`.

See also [`circulant_similarity`](@ref), [`NNlib.softmax`](@ref).
"""
function circulant_adjacency(simfun::AbstractSimilarity, x, y, W::Integer)
    A = circulant_similarity(simfun, x, y, W)
    B = NNlib.softmax(A)
    return B
end

