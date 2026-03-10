# ==============================================================================
# TopKSimilarity — wraps any similarity, keeps top-k entries per row before
# softmax normalization. K is stored as a hyperparameter.
# ==============================================================================
struct TopKSimilarity{S<:AbstractSimilarity} <: AbstractSimilarity
    sim::S
    k::Int
end

# Top-k mask: true for the k largest values along dim 1 (nnz_per_row).
function _topk_mask(W::CuArray, k::Int)
    k_clamped = min(k, size(W, 1))
    Wsorted   = sort(W; dims=1, rev=true)
    kth       = reshape(selectdim(Wsorted, 1, k_clamped), 1, size(W)[2:end]...)
    return W .>= kth
end

# Combined top-k masking + softmax in window space.
# Masked-out entries are set to -Inf so they contribute 0 after softmax.
function _topk_softmax(W::CuArray, k::Int)
    mask = _topk_mask(W, k)
    Wm   = ifelse.(mask, W, typemin(eltype(W)))
    NNlib.softmax(Wm; dims=1)
end

# Custom rrule: sorting is not differentiable, but the mask is treated as
# a fixed boolean gate. Gradient flows only through unmasked entries.
function CRC.rrule(::typeof(_topk_softmax), W::CuArray, k::Int)
    mask = _topk_mask(W, k)
    Wm   = ifelse.(mask, W, typemin(eltype(W)))
    Y    = NNlib.softmax(Wm; dims=1)
    function topk_softmax_back(ΔY)
        ΔWm = NNlib.∇softmax_data(CRC.unthunk(ΔY), Y; dims=1)
        ∂W  = ifelse.(mask, ΔWm, zero(eltype(ΔWm)))
        return CRC.NoTangent(), ∂W, CRC.NoTangent()
    end
    return Y, topk_softmax_back
end

function circulant_similarity(ts::TopKSimilarity, x::AbstractArray, y::AbstractArray, W::Integer)
    A = circulant_similarity(ts.sim, x, y, W)
    W = windowview(A)
    mask = _topk_mask(W, ts.k)
    Wm = ifelse.(mask, W, typemin(eltype(W)))
    return _circ_from_window(Wm, A)
end

function circulant_adjacency(ts::TopKSimilarity, x::AbstractArray, y::AbstractArray, W::Integer)
    A  = circulant_similarity(ts.sim, x, y, W)
    Yw = _topk_softmax(windowview(A), ts.k)
    return _circ_from_window(Yw, A)
end

function joint_topk_softmax(k::Int, As::Circulant...)
    Ws      = map(windowview, As)
    Wcat    = cat(Ws...; dims=1)
    Scat    = _topk_softmax(Wcat, k)
    sizes   = map(w -> size(w, 1), Ws)
    offsets = cumsum((0, sizes...))
    splits  = ntuple(i -> Scat[offsets[i]+1:offsets[i+1], ntuple(_->Colon(), ndims(Scat)-1)...], length(As))
    return ntuple(i -> _circ_from_window(splits[i], As[i]), length(As))
end

function CRC.rrule(::typeof(joint_topk_softmax), k::Int, As::Circulant...)
    Ws      = map(windowview, As)
    Wcat    = cat(Ws...; dims=1)
    mask    = _topk_mask(Wcat, k)
    Wm      = ifelse.(mask, Wcat, typemin(eltype(Wcat)))
    Scat    = NNlib.softmax(Wm; dims=1)
    sizes   = map(w -> size(w, 1), Ws)
    offsets = cumsum((0, sizes...))
    splits  = ntuple(i -> Scat[offsets[i]+1:offsets[i+1], ntuple(_->Colon(), ndims(Scat)-1)...], length(As))
    results = ntuple(i -> _circ_from_window(splits[i], As[i]), length(As))

    function joint_topk_softmax_back(ΔYs)
        ΔYs   = CRC.unthunk(ΔYs)
        ΔYs   = ntuple(i -> _concretize_tangent(CRC.unthunk(ΔYs[i]), results[i]), length(As))
        ΔScat = cat(ntuple(i -> windowview(ΔYs[i]), length(As))...; dims=1)
        ΔWm   = NNlib.∇softmax_data(ΔScat, Scat; dims=1)
        ΔWcat = ifelse.(mask, ΔWm, zero(eltype(ΔWm)))
        ∂As   = ntuple(length(As)) do i
            _circ_from_window(ΔWcat[offsets[i]+1:offsets[i+1], ntuple(_->Colon(), ndims(ΔWcat)-1)...], As[i])
        end
        return (CRC.NoTangent(), CRC.NoTangent(), ∂As...)
    end
    return results, joint_topk_softmax_back
end
