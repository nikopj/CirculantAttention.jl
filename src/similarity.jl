# similarity.jl

abstract type AbstractSimilarity end

@inline function simval_dtype(::AbstractSimilarity, Tx, Ty)
    real(promote_type(Tx, Ty))
end
simval_dtype(sf::AbstractSimilarity, ::AbstractArray{Tx}, ::AbstractArray{Ty}) where {Tx, Ty} = simval_dtype(sf, Tx, Ty)

@doc raw"""
    RealDotSimilarity()

Real dot-product similarity: ``S_{ij} = \mathrm{Real}(q[i]^H k[j])``.
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

Distance similarity: ``S_{ij} = -\frac{1}{2}\mathrm{sum}(\mathrm{abs2}, q[i] - k[j])``.
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

Phase-invariant dot-product similarity: ``S_{ij} = \lvert q[i]^H k[j] \rvert``.
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
``S_{ij} = -\frac{1}{2}(\lVert q[i] \rVert_2^2 - 2 \lvert q[i]^H k[j] \rvert + \lVert k[j] \rVert_2^2)``.
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
    return -0.5f0 * s_xx + abs(s_xy) - 0.5f0 * s_yy
end

@doc raw"""
    DotSimilarity()

Dot-product similarity: ``S_{ij} = q[i]^H k[j]``.
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
    circulant_similarity(simfun, x, y, W)

Returns a Circulant matrix where `S[i,j,b] = simfun(x[...,i,b], y[...,j,b])` for
positions within window `W`. Sparsity pattern is determined by `W` and the spatial
dims of `x` and `y`.
"""
function circulant_similarity(simfun::AbstractSimilarity, x::AbstractArray{Tx,N}, y::AbstractArray{Ty,N}, W::Integer)::Circulant where {Tx, Ty, N}
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

# ------------------------------------------------------------------------------
# Softmax over circulant rows.
#
# circulant_softmax uses windowview to expose nzVal as (nnz_per_row, n_rows, batch...),
# applies NNlib.softmax over dim=1 (the within-row / kernel dim), then wraps the
# result back into a Circulant via the constructor.
#
# This is fully AD-traceable: NNlib.softmax has a ChainRules rrule, and the
# Circulant constructor rrule handles the wrap. No custom softmax rrule needed.
# ------------------------------------------------------------------------------

# function NNlib.softmax(A::Circulant{T,N,M}) where {T,N,M}
#     V = windowview(A)                          # (nnz_per_row, n_rows, batch...)
#     R = NNlib.softmax(V; dims=1)               # softmax over kernel dim
#     # Reconstruct: windowview shape → flat nzVal via constructor
#     # NNlib.softmax returns same shape as V; wrap back using Circulant constructor
#     # which copies rowPtr/colVal from A and uses R as the new nzVal window.
#     data = CuSparseArrayCSR(
#         copy(A.data.rowPtr), copy(A.data.colVal),
#         reshape(R, size(A.data.nzVal)...),
#         size(A)
#     )
#     return Circulant(data, M, spatial_size(A))
# end
# 
# function NNlib.softmax!(A::Circulant{T,N,M}, B::Circulant{T,N,M}=A) where {T,N,M}
#     V = windowview(B)
#     R = NNlib.softmax!(windowview(A), V; dims=1)
#     return A
# end

# Operates in window space; out is pre-allocated as a Circulant by softmax(x).
function NNlib.softmax!(out::Circulant, x::Circulant; dims=1)
    @assert dims==1
    NNlib.softmax!(windowview(out), windowview(x); dims=dims)
    return out
end

# ------------------------------------------------------------------------------
# circulant_adjacency = softmax ∘ circulant_similarity
# Both components are AD-traceable so this composes for free.
# ------------------------------------------------------------------------------

function circulant_adjacency!(A::Circulant, simfun, x, y)
    circulant_similarity!(A, simfun, x, y)
    NNlib.softmax!(A)
    return A
end

"""
    circulant_adjacency(simfun, x, y, W)

Equivalent to `softmax(circulant_similarity(simfun, x, y, W))`.
"""
function circulant_adjacency(simfun::AbstractSimilarity, x, y, W::Integer)
    NNlib.softmax(circulant_similarity(simfun, x, y, W))
end

function joint_softmax(As::Circulant...)
    Ws   = map(windowview, As)
    Wcat = cat(Ws...; dims=1)               # (sum(nnz_per_row), n_rows, ch..., batch)
    Scat = NNlib.softmax(Wcat; dims=1)      # normalize jointly across kernel dim
    sizes = map(w -> size(w, 1), Ws)        # nnz_per_row for each input
    splits = _split_first_dim(Scat, sizes)
    return map((S, A) -> _circ_from_window(S, A), splits, As)
end

# Split array along dim 1 at the given sizes.
function _split_first_dim(W::CuArray, sizes)
    offsets = cumsum((0, sizes...))
    ntuple(length(sizes)) do i
        W[offsets[i]+1:offsets[i+1], ntuple(_->Colon(), ndims(W)-1)...]
    end
end
