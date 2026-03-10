# ==============================================================================
# SparsemaxSimilarity
# ==============================================================================

struct SparsemaxSimilarity{S<:AbstractSimilarity} <: AbstractSimilarity
    sim::S
end

function _sparsemax_threshold(W::CuArray)
    # Sort descending along dim 1
    Ws   = sort(W; dims=1, rev=true)
    # Cumulative sum along dim 1
    Wcum = cumsum(Ws; dims=1)
    # Position indices broadcast along dim 1
    kr   = reshape(1:size(W,1), size(W,1), ntuple(_->1, ndims(W)-1)...)
    # Find support: positions where 1 + k*Ws[k] > Wcum[k]
    supp = 1 .+ kr .* Ws .> Wcum
    # k* = number of true entries per column
    kstar = sum(supp; dims=1)
    # Gather Wcum[k*] per column via broadcast mask — no scalar indexing
    τ = (sum(Wcum .* (kr .== kstar); dims=1) .- 1) ./ kstar
    return τ
end

function sparsemax(W::AbstractArray{T}) where T
    τ = _sparsemax_threshold(W)
    max.(W .- τ, zero(T))
end
sparsemax(A::Circulant) = _circ_from_window(sparsemax(windowview(A)), A)

circulant_similarity(ss::SparsemaxSimilarity, x::AbstractArray, y::AbstractArray, W::Integer) = circulant_similarity(ss.sim, x, y, W)

function circulant_adjacency(ss::SparsemaxSimilarity, x, y, W::Integer)
    A  = circulant_similarity(ss.sim, x, y, W)
    Yw = sparsemax(windowview(A))
    _circ_from_window(Yw, A)
end

function joint_sparsemax(As::Circulant...)
    Ws      = map(windowview, As)
    Wcat    = cat(Ws...; dims=1)
    Scat    = sparsemax(Wcat)
    sizes   = map(w -> size(w, 1), Ws)
    offsets = cumsum((0, sizes...))
    splits  = ntuple(i -> Scat[offsets[i]+1:offsets[i+1], ntuple(_->Colon(), ndims(Scat)-1)...], length(As))
    return ntuple(i -> _circ_from_window(splits[i], As[i]), length(As))
end

# ==============================================================================
# _sparsemax rrule
#
# Pullback of sparsemax: gradient flows only through the support S = {i: p_i > 0}.
# ∂L/∂z_i = mask_i * (∂L/∂p_i - mean_{j∈S}(∂L/∂p_j))
#
# τ is treated as a data-dependent constant — sort is not differentiated.
# ==============================================================================

function CRC.rrule(::typeof(sparsemax), W::CuArray)
    τ    = _sparsemax_threshold(W)
    p    = max.(W .- τ, zero(eltype(W)))
    mask = p .> zero(eltype(p))
    kstar = sum(mask; dims=1)
    project_W = CRC.ProjectTo(W)

    function sparsemax_back(Δp)
        Δp   = CRC.unthunk(Δp)
        # Mean of upstream gradient over support
        supp_sum  = sum(ifelse.(mask, Δp, zero(eltype(Δp))); dims=1)
        supp_mean = supp_sum ./ kstar
        # Gradient is zero outside support, mean-subtracted inside
        ∂W = project_W(ifelse.(mask, Δp .- supp_mean, zero(eltype(Δp))))
        return CRC.NoTangent(), ∂W
    end
    return p, sparsemax_back
end

function CRC.rrule(::typeof(joint_sparsemax), As::Circulant...)
    Ws      = map(windowview, As)
    Wcat    = cat(Ws...; dims=1)
    Scat, back_sparse = CRC.rrule(sparsemax, Wcat)
    sizes   = map(w -> size(w, 1), Ws)
    offsets = cumsum((0, sizes...))
    splits  = ntuple(i -> Scat[offsets[i]+1:offsets[i+1], ntuple(_->Colon(), ndims(Scat)-1)...], length(As))
    results = ntuple(i -> _circ_from_window(splits[i], As[i]), length(As))

    function joint_sparsemax_back(ΔYs)
        ΔYs   = CRC.unthunk(ΔYs)
        ΔYs   = ntuple(i -> _concretize_tangent(CRC.unthunk(ΔYs[i]), results[i]), length(As))
        ΔScat = cat(ntuple(i -> windowview(ΔYs[i]), length(As))...; dims=1)
        _, ΔWcat = back_sparse(ΔScat)
        ∂As   = ntuple(length(As)) do i
            _circ_from_window(ΔWcat[offsets[i]+1:offsets[i+1], ntuple(_->Colon(), ndims(ΔWcat)-1)...], As[i])
        end
        return CRC.NoTangent(), ∂As...
    end
    return results, joint_sparsemax_back
end

function CRC.rrule(::typeof(circulant_adjacency), ss::SparsemaxSimilarity, x, y, W::Integer)
    A, sim_back = CRC.rrule(circulant_similarity, ss.sim, x, y, W)
    Ww   = windowview(A)
    p    = sparsemax(Ww)
    Y    = _circ_from_window(p, A)
    mask  = p .> zero(eltype(p))
    kstar = sum(mask; dims=1)

    function sparsemax_adjacency_back(ΔY)
        ΔY  = _concretize_tangent(CRC.unthunk(ΔY), Y)
        Δp  = windowview(ΔY)
        Δp_sum = sum(ifelse.(mask, Δp, zero(eltype(Δp))); dims=1)
        ΔWw = ifelse.(mask, Δp .- Δp_sum ./ max.(kstar, 1), zero(eltype(Δp)))
        ∂sim = sim_back(_circ_from_window(ΔWw, A))
        # sim_back returns (∂circulant_similarity, ∂sim, ∂x, ∂y, ∂W)
        return CRC.NoTangent(), CRC.NoTangent(), ∂sim[3], ∂sim[4], CRC.NoTangent()
    end
    return Y, sparsemax_adjacency_back
end

Zygote.@adjoint function circulant_adjacency(ss::SparsemaxSimilarity, x, y, W::Integer)
    A, A_back = Zygote._pullback(__context__, circulant_similarity, ss.sim, x, y, W)
    Ww    = windowview(A)
    p     = sparsemax(Ww)
    Y     = _circ_from_window(p, A)
    mask  = p .> zero(eltype(p))
    kstar = sum(mask; dims=1)
    function sparsemax_adj_back(ΔY)
        ΔY     = _concretize_tangent(Zygote.unthunk(ΔY), Y)
        Δp     = windowview(ΔY)
        Δp_sum = sum(ifelse.(mask, Δp, zero(eltype(Δp))); dims=1)
        ΔWw    = ifelse.(mask, Δp .- Δp_sum ./ max.(kstar, 1), zero(eltype(Δp)))
        ∂all   = A_back(_circ_from_window(ΔWw, A))
        return (nothing, ∂all[3], ∂all[4], nothing)
    end
    return Y, sparsemax_adj_back
end

CRC.@non_differentiable _sparsemax_threshold(::Any)

