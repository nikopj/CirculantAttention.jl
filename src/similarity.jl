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

@inline function simval(::DotSimilarity, xv, yv)
    real(xv * conj(yv))
end

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

@inline function simval(::DistanceSimilarity, xv, yv)
    - abs2(xv - yv) * 0.5f0
end

function circulant_similarity!(A::Circulant{T, N}, simfun, x, y) where {T, N}
    circulant_similarity!(reshape(A, :, :, :), simfun, x, y)
    return A
end

function circulant_similarity!(
    A::Circulant{Tv,3,W,S},
    simfun::AbstractSimilarity,
    x::AbstractArray{Tx,N},
    y::AbstractArray{Ty,N},
) where {Tv,W,S,Tx,Ty,N}

    maxidx   = A.data.nnz
    nnzb     = A.data.nnz ÷ Int32(size(A,3))
    M        = Int32(size(x, N-1))
    spatdims = ntuple(i -> Int32(size(x, i)), N-2)
    CartInd  = CartesianIndices(spatdims)
    Wi32 = Int32(W)

    args = (A, simfun, x, y, nnzb, M, spatdims, CartInd, Wi32, maxidx)
    kernel = @cuda launch=false circulant_similarity_kernel!(args...)
    config = launch_configuration(kernel.fun)
    threads = min(maxidx, config.threads)
    blocks  = cld(maxidx, threads)

    kernel(args...; threads=threads, blocks=blocks)
    return A
end

function circulant_similarity_kernel!(
        S::Circulant{Tv,3},
        simfun::AbstractSimilarity,
        x::AbstractArray{Tx,N},
        y,
        nnzb,
        M,
        spatdims,
        CartInd,
        W,
        maxidx,
    ) where {Tv,Tx,N}

    tid    = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    @inbounds while tid<=maxidx
        n = (tid - Int32(1)) % nnzb + Int32(1)
        b = (tid - Int32(1)) ÷ nnzb + Int32(1)

        # spatial indices
        i, j = cartesian_circulant(n, spatdims, W)
        Ci, Cj = CartInd[i], CartInd[j]

        s = zero(Tv)
        for m=Int32(1):M
            s += simval(simfun, x[Cj, m, b], y[Ci, m, b])
        end

        S.data.nzVal[n, b] = s 
        tid += stride
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

