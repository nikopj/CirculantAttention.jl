const BatchedCirculant{T,N,M,S} = Union{
    Circulant{T, N, M, S},
    NNlib.BatchedTranspose{T, <:Circulant{T,N,M,S}},
    NNlib.BatchedAdjoint{T, <:Circulant{T,N,M,S}}
   } where {T <: Number, N, M}

function circulant_attention(simfun::AbstractSimilarity, q::T, k::T, v::T, W::Int) where {Tv, N, T<: AbstractArray{Tv,N}}
    τ = sqrt(Tv(size(k, N-1)))
    A = circulant_adjacency(simfun, q ./ sqrt(τ), k ./ sqrt(τ), W)
    V = reshape(v, :, size(v)[N-1:end]...)
    Y = reshape(A, :, :, :) ⊠ V
    return reshape(Y, size(q)...), A
end
circulant_attention(q::T, k::T, v::T, W::Int) where T = circulant_attention(DotSimilarity(), q, k, v, W)

function circulant_mh_attention(simfun::AbstractSimilarity, q::T, k::T, v::T, W::Int, nheads::Int) where {Tv, N, T<: AbstractArray{Tv,N}}
    qr, kr, vr = splitheads.((q, k, v), nheads)
    yr, A = circulant_attention(qr, kr, vr, W)
    return reshape(yr, size(q)...), reshape(A, :, :, nheads, size(q, N))
end
circulant_mh_attention(q::T, k::T, v::T, W::Int, nheads::Int) where T = circulant_attention(DotSimilarity(), q, k, v, W, nheads)
splitheads(x::AbstractArray{T,N}, nheads) where {T,N} = reshape(x, ntuple(i->size(x,i), N-2)..., size(x, N-1) ÷ nheads, :)

function circulant_mh_adjacency(simfun, x::AbstractArray{T,N}, y, W::Integer, nheads::Int) where {T, N}
    xr, yr = splitheads.((x, y), nheads)
    A = circulant_adjacency(simfun, xr, yr, W)
    return reshape(A, :, :, nheads, size(x, N))
end

function circulant_attention!(y::AbstractArray{T, N}, A::Circulant{Ta, Na}, x::AbstractArray{T, N}) where {Ta, Na, T, N}
    @assert spatial_dims(A) == N - 2 "spatial-dims ($(spatial_dims(A)) of A must match spatial dims of x ($(N-2))"
    X = reshape(x, :, size(x)[N-1:end]...)
    Y = reshape(y, :, size(y)[N-1:end]...)
    NNlib.batched_mul!(Y, reshape(A, :, :, :), X)
    return y
end
circulant_attention(A::Circulant, x) = circulant_attention!(similar(x), A, x)

const ⊗ = circulant_attention

function circulant_mh_attention(A::Circulant{Ta, 4}, x::AbstractArray{T, N}) where {Ta, T, N}
    nheads = size(A, 3)
    xr = splitheads(x, nheads)
    yr = A ⊗ xr
    return reshape(yr, size(x)...)
end

const ⨷ = circulant_mh_attention
