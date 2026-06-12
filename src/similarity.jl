# NEW SIMILARITY WITH PERFORMANCE REGRESSION 2x OLD Dot SIMILARITY SPEED
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
@inline function simval(::RealDotSimilarity, x::AbstractArray{Tx}, y::AbstractArray{Ty}, Ci, Cj, b, M::Int32) where {Tx, Ty}
    Ts = real(promote_type(Tx, Ty))
    s = zero(Ts)
    @fastmath @inbounds for m=1i32:M
        s += real(x[Ci, m, b] * conj(y[Cj, m, b]))
    end
    return s
end

@doc raw"""
    DistanceSimilarity()

Distance similarity: ``S_{ij} = -\frac{1}{2}\mathrm{sum}(\mathrm{abs2}, q[i] - k[j])``.
"""
struct DistanceSimilarity <: AbstractSimilarity end

@inline function simval(::DistanceSimilarity, xv, yv)
    - abs2(xv - yv) * typeof(xv)(0.5f0)
end
@inline function simval(::DistanceSimilarity, x::AbstractArray{Tx}, y::AbstractArray{Ty}, Ci, Cj, b, M::Int32) where {Tx, Ty}
    Ts = real(promote_type(Tx, Ty))
    s = zero(Ts)
    @fastmath @inbounds for m=1i32:M
        s -= abs2(x[Ci, m, b] - y[Cj, m, b])
    end
    return Ts(0.5) * s
end

@doc raw"""
    PIDotSimilarity()

Phase-invariant dot-product similarity: ``S_{ij} = \lvert q[i]^H k[j] \rvert``.
"""
struct PIDotSimilarity <: AbstractSimilarity end

@inline function simval(::PIDotSimilarity, x::AbstractArray{Tx}, y::AbstractArray{Ty}, Ci, Cj, b, M::Int32) where {Tx, Ty}
    Ts = promote_type(Tx, Ty)
    s = zero(Ts)
    @fastmath @inbounds for m=1i32:M
        s += x[Ci, m, b]*conj(y[Cj, m, b])
    end
    return abs(s)
end

@doc raw"""
    PIDistanceSimilarity()

Phase-invariant distance similarity:
``S_{ij} = -\frac{1}{2}(\lVert q[i] \rVert_2^2 - 2 \lvert q[i]^H k[j] \rvert + \lVert k[j] \rVert_2^2)``.
"""
struct PIDistanceSimilarity <: AbstractSimilarity end

@inline function simval(::PIDistanceSimilarity, x::AbstractArray{Tx}, y::AbstractArray{Ty}, Ci, Cj, b, M::Int32) where {Tx, Ty}
    Ts = promote_type(Tx, Ty)
    s_xx = zero(real(Ts)); s_xy = zero(Ts); s_yy = zero(real(Ts))
    @fastmath @inbounds for m=1i32:M
        xm = x[Ci, m, b]; ym = y[Cj, m, b]
        s_xx += abs2(xm)
        s_xy += xm * conj(ym)
        s_yy += abs2(ym)
    end
    return -real(Ts)(0.5) * s_xx + abs(s_xy) - real(Ts)(0.5) * s_yy
end

@doc raw"""
    DotSimilarity()

Dot-product similarity: ``S_{ij} = q[i]^H k[j]``.
"""
struct DotSimilarity <: AbstractSimilarity end

@inline function simval(::DotSimilarity, xv, yv)
    xv * conj(yv)
end
@inline function simval(sf::DotSimilarity, x::AbstractArray{Tx}, y::AbstractArray{Ty}, Ci, Cj, b, M::Int32) where {Tx, Ty}
    Ts = promote_type(Tx, Ty)
    s = zero(Ts)
    @fastmath @inbounds for m=1i32:M
        s += x[Ci, m, b]*conj(y[Cj, m, b])
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
    # all-finite kernel (no sentinels, no exp): the fastmath compile flag is
    # safe here and re-enables accumulation-loop reassociation/contraction
    kernel = @cuda launch=false fastmath=true circulant_similarity_kernel!(args...)
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
        # nzVal[n] sits at CSR (row j, col i): x is indexed by the row and y by
        # the column (S[j,i] = simval(x_j, y_i)), matching the rrules in
        # rrules.jl and the docstring S_ij = simfun(q_i, k_j).
        Ci, Cj = CartInd[i], CartInd[j]
        s = simval(simfun, x, y, Cj, Ci, b, M)

        S.data.nzVal[n, b] = s
        tid += stride
    end

    return nothing
end

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
