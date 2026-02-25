abstract type AbstractSimilarity end

@inline function simval_dtype(::AbstractSimilarity, Tx, Ty) 
    real(promote_type(Tx, Ty))
end
simval_dtype(sf::AbstractSimilarity, ::AbstractArray{Tx}, ::AbstractArray{Ty}) where {Tx, Ty} = simval_dtype(sf, Tx, Ty)

@doc raw"""
    RealDotSimilarity()

Used in `circulant_attention`, `circulant_similarity`, and `circulant_adjacency` to indicate use of
real dot-product similarity:

``
    S_{ij} = \mathrm{Real}(q[i]^H k[j]).
``

See also [`DistanceSimilarity`](@ref).
"""
struct RealDotSimilarity <: AbstractSimilarity end 

@inline function simval(::RealDotSimilarity, xv, yv)
    real(xv * conj(yv))
end
@inline function simval(::RealDotSimilarity, x::AbstractArray{Tx}, y::AbstractArray{Ty}, M::Int32) where {Tx, Ty}
    Ts = real(promote_type(Tx, Ty))
    s = zero(Ts)
    @inbounds for m=1i32:M
        s += real(x[m] * conj(y[m]))
    end
    return s
end

@doc raw"""
    DistanceSimilarity()

Used in `circulant_attention`, `circulant_similarity`, and `circulant_adjacency` to indicate use of
distance similarity:

``
    S_{ij} = -\frac{1}{2}\mathrm{sum}(\mathrm{abs2}, q[i] - k[j]).
``

See also [`DotSimilarity`](@ref).
"""
struct DistanceSimilarity <: AbstractSimilarity end

@inline function simval(::DistanceSimilarity, xv, yv)
    - abs2(xv - yv) * 0.5f0
end
@inline function simval(::DistanceSimilarity, x::AbstractArray{Tx}, y::AbstractArray{Ty}, M::Int32) where {Tx, Ty}
    Ts = real(promote_type(Tx, Ty))
    s = zero(Ts)
    @inbounds for m=1i32:M
        s -= abs2(x[m] - y[m]) 
    end
    return 0.5f0 * s
end

@doc raw"""
    PIDotSimilarity()

Phase-invariant dot-product similarity:

``
    S_{ij} = \lvert q[i]^H k[j] \rvert.
``

See also [`DotSimilarity`](@ref), [`PIDistanceSimilarity`](@ref).
"""
struct PIDotSimilarity <: AbstractSimilarity end 

@inline function simval(::PIDotSimilarity, x::AbstractArray{Tx}, y::AbstractArray{Ty}, M::Int32) where {Tx, Ty}
    Ts = promote_type(Tx, Ty)
    s = zero(Ts)
    @inbounds for m=1i32:M
        s += x[m]*conj(y[m]) 
    end
    return abs(s)
end


@doc raw"""
    PIDistanceSimilarity()

Phase-invariant distance similarity:

``
    S_{ij} = -\frac{1}{2}(\lVert q[i] \rVert_2^2 - 2 \lvert q[i]^H k[j] \rvert + \lVert k[j] \rVert_2^2).
``

See also [`PIDotSimilarity`](@ref), [`DistanceSimilarity`](@ref).
"""
struct PIDistanceSimilarity <: AbstractSimilarity end 

@inline function simval(::PIDistanceSimilarity, x::AbstractArray{Tx}, y::AbstractArray{Ty}, M::Int32) where {Tx, Ty}
    Ts = promote_type(Tx, Ty)
    s_xx = zero(real(Ts)); s_xy = zero(Ts); s_yy = zero(real(Ts))

    @inbounds for m=1i32:M
        xm = x[m]; ym = y[m]
        s_xx += abs2(xm)
        s_xy += xm * conj(ym)
        s_yy += abs2(ym)
    end

    s = -0.5f0 * s_xx + abs(s_xy) - 0.5f0 * s_yy
    return s
end

@doc raw"""
    DotSimilarity()

Dot-product similarity:

``
    S_{ij} = q[i]^H k[j].
``

See also [`RealDotSimilarity`](@ref).
"""
struct DotSimilarity <: AbstractSimilarity end 

@inline function simval(::DotSimilarity, xv, yv)
    xv * conj(yv)
end
@inline function simval(sf::DotSimilarity, x::AbstractArray{Tx}, y::AbstractArray{Ty}, M::Int32) where {Tx, Ty}
    Ts = promote_type(Tx, Ty)
    s = zero(Ts)
    @inbounds for m=1i32:M
        s += x[m]*conj(y[m]) 
    end
    return s
end
simval_dtype(::DotSimilarity, Tx::Type, Ty::Type) = promote_type(Tx, Ty)

"""
    circulant_similarity(simfun::AbstractSimilarity, x, y, W::Int)

Returns Circulant matrix with circulant-sparse data. Each non-zero `S[i,j,b]` is 
populated by `simfun` evaluated at the linearized pixel locations of `x` and `y`, 
i.e. `S[i,j,b] = simfun(x[...,i,b], y[...,j,b], W)` for max(i⃗, j⃗) ≤ W. The non-zero entrie 
locations are determined by the windowsize `W` and number of spatial dimensions in `x` and `y`.

See also [`DotSimilarity`](@ref), [`DistanceSimilarity`](@ref).
"""
function circulant_similarity(simfun::AbstractSimilarity, x::AbstractArray{T,N}, y::AbstractArray{T,N}, W::Integer)::Circulant where {T,N}
    Tv = simval_dtype(simfun, x, y)
    S = Circulant{Tv}(W, (size(x)[1:N-2]..., 1, size(x, N)))
    circulant_similarity!(S, simfun, x, y)
    return S
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

        xj = @view x[Cj, :, b]
        yi = @view y[Ci, :, b]
        s = simval(simfun, xj, yi, M)

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

