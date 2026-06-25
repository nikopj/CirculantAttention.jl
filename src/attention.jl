# attention.jl
const BatchedCirculant{T,N,M,S} = Union{
    Circulant{T, N, M, S},
    NNlib.BatchedTranspose{T, <:Circulant{T,N,M,S}},
    NNlib.BatchedAdjoint{T, <:Circulant{T,N,M,S}}
   } where {T <: Number, N, M}

@doc raw"""
    y, A = circulant_attention(simfun::AbstractSimilarity, q, k, v, W::Int)

Perform circulant attention on `y=Av`, where A is a row-softmax normalized
circulant-sparse attention matrix (A = rowsoftmax(S)). Each non-zero entry ``S_{ij}``
is generated via the similarity function acting on the channel representations of `q` and `k`
at (linearly indexed) pixels `i`, `j` (``S_{ij} = \mathrm{simfun}(q_i, k_j)``).
Adjacency matrix $A$ is generated internal and returned as the
second argument. Note: q and k are internally scaled by `sqrt(sqrt(channels))` before being passed to `circulant_adjacency`.

See also [`circulant_adjacency`](@ref), [`circulant_similarity`](@ref), [`DotSimilarity`](@ref), [`DistanceSimilarity`](@ref).
"""
function circulant_attention(simfun::AbstractSimilarity, q::AbstractArray{Tq,N}, k::AbstractArray{Tk,N}, v::AbstractArray{Tv,N}, W::Int) where {Tq, Tk, Tv, N}
    # q, k, v may have distinct element types (e.g. complex q/k, real v); the
    # similarity scale is keyed off the query/key channel count.
    τ = sqrt(real(Tk)(size(k, N-1)))
    A = circulant_adjacency(simfun, q ./ sqrt(τ), k ./ sqrt(τ), W)
    # Apply via ⊗, whose rrule keeps ∂A circulant-sparse. Tracing through a raw
    # ⊠ would hit NNlib's generic batched_mul pullback, which materializes a
    # dense N×N×B ∂A (and cannot be converted back to the sparse pattern).
    return A ⊗ v, A
end
circulant_attention(q::AbstractArray, k::AbstractArray, v::AbstractArray, W::Int) = circulant_attention(DotSimilarity(), q, k, v, W)

@doc raw"""
    y, A = circulant_mh_attention(simfun::AbstractSimilarity, q, k, v, W::Int, nheads::Int)

Performs circulant multi-head attention, i.e., performing circulant attetion of
`nheads`-groups separately and concatenating the result along channels. The
number of channels in `q`, `k`, `v` must be divisible by `nheads`.
The returned adjacency matrix `A` will have `size(A, 3) == nheads`.

See also [`circulant_attention`](@ref), [`DotSimilarity`](@ref), [`DistanceSimilarity`](@ref).
"""
function circulant_mh_attention(simfun::AbstractSimilarity, q::AbstractArray{Tq,N}, k::AbstractArray{Tk,N}, v::AbstractArray{Tv,N}, W::Int, nheads::Int) where {Tq, Tk, Tv, N}
    qr, kr, vr = splitheads.((q, k, v), nheads)
    yr, A = circulant_attention(simfun, qr, kr, vr, W)
    return reshape(yr, size(q)...), reshape(A, :, :, nheads, size(q, N))
end
circulant_mh_attention(q::AbstractArray, k::AbstractArray, v::AbstractArray, W::Int, nheads::Int) = circulant_mh_attention(DotSimilarity(), q, k, v, W, nheads)
splitheads(x::AbstractArray{T,N}, nheads) where {T,N} = reshape(x, ntuple(i->size(x,i), N-2)..., size(x, N-1) ÷ nheads, :)

function circulant_mh_adjacency(simfun, x::AbstractArray{T,N}, y, W::Integer, nheads::Int) where {T, N}
    xr, yr = splitheads.((x, y), nheads)
    A = circulant_adjacency(simfun, xr, yr, W)
    return reshape(A, :, :, nheads, size(x, N))
end

function circulant_mh_similarity(simfun, x::AbstractArray{T,N}, y, W::Integer, nheads::Int) where {T, N}
    xr, yr = splitheads.((x, y), nheads)
    A = circulant_similarity(simfun, xr, yr, W)
    return reshape(A, :, :, nheads, size(x, N))
end

function circulant_attention!(y::AbstractArray{Ty, N}, A::Circulant{Ta, Na}, x::AbstractArray{Tx, N}) where {Ty, Ta, Na, Tx, N}
    @assert spatial_dims(A) == N - 2 "spatial-dims ($(spatial_dims(A)) of A must match spatial dims of x ($(N-2))"
    X = reshape(x, :, size(x)[N-1:end]...)
    Y = reshape(y, :, size(y)[N-1:end]...)
    NNlib.batched_mul!(Y, reshape(A, :, :, :), X)
    return y
end

@doc raw"""
    y = circulant_attention(A::Circulant, x::AbstractArray)
    y = A ⊗ x # \otimes

Applies circulant matrix `A` to `x`.
See also [`circulant_adjacency`](@ref).
"""
function circulant_attention(A::Circulant{Ta}, x::AbstractArray{Tx}) where {Ta, Tx}
    y = similar(x, promote_type(Ta, Tx))
    circulant_attention!(y, A, x)
    return y
end

const ⊗ = circulant_attention

function circulant_transposed_attention!(y::AbstractArray{Ty, N}, A::Circulant{Ta, Na}, x::AbstractArray{T, N}) where {Ty, Ta, Na, T, N}
    @assert spatial_dims(A) == N - 2 "spatial-dims ($(spatial_dims(A)) of A must match spatial dims of x ($(N-2))"
    X = reshape(x, :, size(x)[N-1:end]...)
    Y = reshape(y, :, size(y)[N-1:end]...)
    At = NNlib.batched_adjoint(reshape(A, :, :, :))
    NNlib.batched_mul!(Y, At, X)
    return y
end
circulant_transposed_attention(A::Circulant, x) = circulant_transposed_attention!(similar(x), A, x)

@doc raw"""
    y = circulant_mh_attention(A::Circulant, x::AbstractArray)
    y = A ⨷ x # \Otimes

Applies circulant matrix `A` (with channel dimension > 1) to `x`.
See also [`circulant_attention`](@ref), [`circulant_adjacency`](@ref).
"""
function circulant_mh_attention(A::Circulant{Ta, 4}, x::AbstractArray{T, N}) where {Ta, T, N}
    nheads = size(A, 3)
    xr = splitheads(x, nheads)
    yr = A ⊗ xr
    return reshape(yr, size(x)...)
end

const ⨷ = circulant_mh_attention

function circulant_mh_transposed_attention(A::Circulant{Ta, 4}, x::AbstractArray{T, N}) where {Ta, T, N}
    nheads = size(A, 3)
    xr = splitheads(x, nheads)
    yr = circulant_transposed_attention(A, xr)
    return reshape(yr, size(x)...)
end
