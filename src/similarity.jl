# similarity.jl

abstract type AbstractSimilarity end

@inline function simval_dtype(::AbstractSimilarity, Tx, Ty)
    real(promote_type(Tx, Ty))
end
simval_dtype(sf::AbstractSimilarity, ::AbstractArray{Tx}, ::AbstractArray{Ty}) where {Tx, Ty} = simval_dtype(sf, Tx, Ty)

# Strided channel reduction shared by every similarity: applies f(acc, xv, yv)
# over the C channels of x[Ci,:,b] and y[Cj,:,b] (arrays are (nrows, C, batch)).
#
# On LLVM ≥ 17 the automatic loop-strength-reduction of the multidim access
# x[Ci,m,b] regressed — it recomputes each element's address with a
# per-iteration `mul.lo.s64` instead of a constant-stride pointer bump (see
# benchmark/TOOLCHAIN_REGRESSION_REPORT.md), ~1.2–1.8× slower. We hand it
# linear indices stepped by `nrows` so the pointer increment is explicit. On
# LLVM 16 the multidim form is already optimal (and the linear form is slower),
# so we keep it. The branch is resolved at compile time by @static.
@inline function _strided_reduce(f::F, init, x, y, Ci, Cj, b, M::Int32) where F
    acc = init
    @static if Base.libllvm_version >= v"17"
        nr  = size(x, 1)
        off = (Int(b) - 1) * nr * size(x, 2)
        xl  = off + Int(Ci)
        yl  = off + Int(Cj)
        @fastmath @inbounds for _ in 1i32:M
            acc = f(acc, x[xl], y[yl])
            xl += nr; yl += nr
        end
    else
        @fastmath @inbounds for m in 1i32:M
            acc = f(acc, x[Ci, m, b], y[Cj, m, b])
        end
    end
    return acc
end

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
    _strided_reduce((s, xv, yv) -> s + real(xv * conj(yv)), zero(Ts), x, y, Ci, Cj, b, M)
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
    s = _strided_reduce((s, xv, yv) -> s - abs2(xv - yv), zero(Ts), x, y, Ci, Cj, b, M)
    return Ts(0.5) * s
end

@doc raw"""
    PIDotSimilarity()

Phase-invariant dot-product similarity: ``S_{ij} = \lvert q[i]^H k[j] \rvert``.
"""
struct PIDotSimilarity <: AbstractSimilarity end

@inline function simval(::PIDotSimilarity, x::AbstractArray{Tx}, y::AbstractArray{Ty}, Ci, Cj, b, M::Int32) where {Tx, Ty}
    Ts = promote_type(Tx, Ty)
    s = _strided_reduce((s, xv, yv) -> s + xv * conj(yv), zero(Ts), x, y, Ci, Cj, b, M)
    return abs(s)
end

@doc raw"""
    PIDistanceSimilarity()

Phase-invariant distance similarity:
``S_{ij} = -\frac{1}{2}(\lVert q[i] \rVert_2^2 - 2 \lvert q[i]^H k[j] \rvert + \lVert k[j] \rVert_2^2)``.
"""
struct PIDistanceSimilarity <: AbstractSimilarity end

@inline function simval(::PIDistanceSimilarity, x::AbstractArray{Tx}, y::AbstractArray{Ty}, Ci, Cj, b, M::Int32) where {Tx, Ty}
    Ts = promote_type(Tx, Ty); R = real(Ts)
    # 3-component accumulator: (Σ|x|², Σ x·conj(y), Σ|y|²)
    a = _strided_reduce(
        (a, xv, yv) -> (a[1] + abs2(xv), a[2] + xv * conj(yv), a[3] + abs2(yv)),
        (zero(R), zero(Ts), zero(R)), x, y, Ci, Cj, b, M)
    return -R(0.5) * a[1] + abs(a[2]) - R(0.5) * a[3]
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
    _strided_reduce((s, xv, yv) -> s + xv * conj(yv), zero(Ts), x, y, Ci, Cj, b, M)
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
    nrows    = Int32(size(A, 1))
    Krow     = nnzb ÷ nrows                # nnz per row (= W^spatial_dims)
    M        = Int32(size(x, N-1))

    # kernel indexes spatial positions linearly: reshape to (nrows, C, batch)
    # so simval's x[i,m,b] addresses (spatial, channel, batch) directly
    xr = reshape(x, :, size(x, N-1), size(x, N))
    yr = reshape(y, :, size(y, N-1), size(y, N))

    args = (A, simfun, xr, yr, nnzb, M, Krow, maxidx)
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
        Krow,
        maxidx,
    ) where {Tv,Tx,N}

    tid    = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    @inbounds while tid<=maxidx
        n = (tid - Int32(1)) % nnzb + Int32(1)
        b = (tid - Int32(1)) ÷ nnzb + Int32(1)

        # The column index i is already materialized in colVal during
        # construction, and the row is j = cld(n, Krow). Reading colVal (one
        # coalesced load) instead of re-deriving (i,j) via cartesian_circulant
        # (~15 integer div/mod ops) is ~1.1–1.5× faster on A100 (the win shrinks
        # as W grows and simval memory traffic dominates). nzVal[n] = S[j,i]:
        # x indexed by row j, y by column i — see rrules.jl / S_ij convention.
        i = S.data.colVal[n, b]
        j = cld(n, Krow)
        s = simval(simfun, x, y, j, i, b, M)

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
