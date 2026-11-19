# ==============================================================================
# EntmaxSimilarity — alpha-entmax normalization
#
# Generalizes softmax (α→1) and sparsemax (α=2).
# For general α ∈ (1,2], threshold τ is found via bisection on the GPU.
#
# Forward:  p_i = max((α-1)*z_i - τ, 0)^(1/(α-1)),  sum(p) = 1
# Backward: implicit function theorem — bisection is not differentiated.
# ==============================================================================

"""
    EntmaxSimilarity(sim, α)

Normalizes the similarities `sim` of each row with α-[`entmax`](@ref) instead of the
softmax: `α → 1` recovers the softmax and `α = 2` sparsemax, and values in between give
attention weights of intermediate sparsity. `α` is a fixed hyperparameter (not
differentiated); for a learnable `α`, apply [`entmax`](@ref) to the output of
[`circulant_similarity`](@ref). Supported by [`circulant_attention`](@ref) and the flash
functions.
"""
struct EntmaxSimilarity{S<:AbstractSimilarity} <: AbstractSimilarity
    sim::S
    α::Float32
end

const _ENTMAX_BISECT_ITERS = 50
const _ENTMAX_EPS          = 1f-6

# ------------------------------------------------------------------------------
# Core forward: threshold via bisection, α is a AbstractArray broadcastable against W
# ------------------------------------------------------------------------------

# α can be a scalar or any broadcastable array — no type annotation required.
# Special-case dispatches for α≈1 and α≈2 only apply for scalar α.

function _entmax_threshold(W::AbstractArray{T}, α) where T
    zα   = (α .- T(1)) .* W
    τ_lo = minimum(zα; dims=1) .- T(1)
    τ_hi = maximum(zα; dims=1)
    for _ in 1:_ENTMAX_BISECT_ITERS
        τ_mid = (τ_lo .+ τ_hi) ./ T(2)
        p_sum = sum(max.(zα .- τ_mid, T(0)) .^ (T(1) ./ (α .- T(1))); dims=1)
        τ_lo  = ifelse.(p_sum .> T(1), τ_mid, τ_lo)
        τ_hi  = ifelse.(p_sum .> T(1), τ_hi,  τ_mid)
    end
    return (τ_lo .+ τ_hi) ./ T(2)
end

"""
    entmax(W, α)
    entmax(A::Circulant, α)

α-entmax (Peters et al., 2019) along the first dimension of `W`, or over the window
entries of each row of `A`: `max.((α - 1) .* W .- τ, 0) .^ (1 / (α - 1))`, with the
threshold `τ` (found by bisection) chosen so that each column sums to one. `α ∈ (1, 2]` is
a number or an array broadcastable against `W` (e.g. one value per head); a scalar `α ≈ 1`
or `α ≈ 2` gives the softmax or [`sparsemax`](@ref). Differentiable in `W` and `α`.
"""
function entmax(W::AbstractArray{T}, α) where T
    if α isa Real
        abs(T(α - 1)) < _ENTMAX_EPS && return NNlib.softmax(W; dims=1)
        abs(T(α - 2)) < _ENTMAX_EPS && return sparsemax(W)
    end
    τ = _entmax_threshold(W, α)
    max.((α .- T(1)) .* W .- τ, T(0)) .^ (T(1) ./ (α .- T(1)))
end

function entmax(A::Circulant, α)
    Yw = entmax(windowview(A), α)
    _circ_from_window(Yw, A)
end

circulant_similarity(ss::EntmaxSimilarity, x::AbstractArray, y::AbstractArray, W::Integer) = circulant_similarity(ss.sim, x, y, W)

function circulant_adjacency(es::EntmaxSimilarity, x, y, W::Integer)
    A  = circulant_similarity(es.sim, x, y, W)
    _circ_from_window(entmax(windowview(A), es.α), A)
end

"""
    joint_entmax(α, A₁, A₂, …)

[`entmax`](@ref) normalized jointly over the window entries of several
[`Circulant`](@ref) matrices, as in [`joint_softmax`](@ref). Differentiable in `α` too.
"""
function joint_entmax(α, As::Circulant...)
    Ws      = map(windowview, As)
    Wcat    = cat(Ws...; dims=1)
    Scat    = entmax(Wcat, α)
    sizes   = map(w -> size(w, 1), Ws)
    offsets = cumsum((0, sizes...))
    splits  = ntuple(i -> Scat[offsets[i]+1:offsets[i+1], ntuple(_->Colon(), ndims(Scat)-1)...], length(As))
    return ntuple(i -> _circ_from_window(splits[i], As[i]), length(As))
end

function _entmax_pullback_z(Δp::AbstractArray, p::AbstractArray, W::AbstractArray, α)
    mask    = p .> sqrt(eps(eltype(p)))
    w       = ifelse.(mask, p .^ (2f0 .- α), zero(eltype(p)))   # p_i^(2-α)
    W_sum   = sum(w; dims=1)
    # Weighted upstream sum: sum_{i∈S} Δp_i * w_i
    wΔp_sum = sum(ifelse.(mask, Δp .* w, zero(eltype(Δp))); dims=1)
    # ∂L/∂z_j = w_j * (Δp_j - wΔp_sum / W_sum)
    ∂W_full = ifelse.(mask, w .* (Δp .- wΔp_sum ./ max.(W_sum, _ENTMAX_EPS)), zero(eltype(Δp)))
    # Sum over any dims that α broadcast over W
    dims = Tuple(filter(d -> size(W, d) < size(∂W_full, d), 1:ndims(∂W_full)))
    isempty(dims) ? ∂W_full : sum(∂W_full; dims=dims)
end

function _entmax_pullback_α(Δp::CuArray, p::CuArray, W::CuArray, α)
    mask    = p .> sqrt(eps(eltype(p)))
    w     = ifelse.(mask, p .^ (2f0 .- α), zero(eltype(p)))   # p_i^(2-α)
    W_sum = sum(w; dims=1)
    logp  = ifelse.(mask, log.(p .+ _ENTMAX_EPS), zero(eltype(p)))

    # ∂τ/∂α from d/dα sum(p_i) = 0:
    # ∂τ/∂α = (sum(w_i * z_i) - sum(p_i * log(p_i))) / W_sum
    dτ_dα = (sum(w .* W; dims=1) .- sum(p .* logp; dims=1)) ./ max.(W_sum, _ENTMAX_EPS)

    # ∂p_i/∂α = [w_i * (z_i - ∂τ/∂α) - p_i * log(p_i)] / (α - 1)
    dp_dα = ifelse.(mask,
                (w .* (W .- dτ_dα) .- p .* logp) ./ (α .- 1f0),
                zero(eltype(p)))

    # Contract with upstream tangent, reduce over all dims except α's own
    ∂α_full = Δp .* dp_dα
    if α isa Real
        return sum(∂α_full)
    else
        # Sum over dims where α has size 1
        dims = Tuple(filter(d -> size(α, d) == 1, 1:ndims(∂α_full)))
        return sum(∂α_full; dims=dims)
    end
end

function CRC.rrule(::typeof(entmax), W::AbstractArray, α)
    p         = entmax(W, α)
    project_W = CRC.ProjectTo(W)
    project_α = CRC.ProjectTo(α)

    function entmax_back(Δp)
        Δp = CRC.unthunk(Δp)
        ∂W = project_W(_entmax_pullback_z(Δp, p, W, α))
        ∂α = project_α(_entmax_pullback_α(Δp, p, W, α))
        return CRC.NoTangent(), ∂W, ∂α
    end
    return p, entmax_back
end

function CRC.rrule(::typeof(joint_entmax), α, As::Circulant...)
    Ws      = map(windowview, As)
    Wcat    = cat(Ws...; dims=1)
    Scat, back_entmax = CRC.rrule(entmax, Wcat, α)
    sizes   = map(w -> size(w, 1), Ws)
    offsets = cumsum((0, sizes...))
    splits  = ntuple(i -> Scat[offsets[i]+1:offsets[i+1], ntuple(_->Colon(), ndims(Scat)-1)...], length(As))
    results = ntuple(i -> _circ_from_window(splits[i], As[i]), length(As))

    function joint_entmax_back(ΔYs)
        ΔYs   = CRC.unthunk(ΔYs)
        ΔYs   = ntuple(i -> _concretize_tangent(CRC.unthunk(ΔYs[i]), results[i]), length(As))
        ΔScat = cat(ntuple(i -> windowview(ΔYs[i]), length(As))...; dims=1)
        _, ΔWcat, ∂α = back_entmax(ΔScat)
        ∂As = ntuple(length(As)) do i
            _circ_from_window(ΔWcat[offsets[i]+1:offsets[i+1], ntuple(_->Colon(), ndims(ΔWcat)-1)...], As[i])
        end
        return CRC.NoTangent(), ∂α, ∂As...
    end
    return results, joint_entmax_back
end

function CRC.rrule(::typeof(circulant_adjacency), es::EntmaxSimilarity, x, y, W::Integer)
    A, sim_back = CRC.rrule(circulant_similarity, es.sim, x, y, W)
    Ww  = windowview(A)
    p   = entmax(Ww, es.α)
    Y   = _circ_from_window(p, A)

    function entmax_adjacency_back(ΔY)
        ΔY  = _concretize_tangent(CRC.unthunk(ΔY), Y)
        ΔWw = _entmax_pullback_z(windowview(ΔY), p, Ww, es.α)
        ∂sim = sim_back(_circ_from_window(ΔWw, A))
        return CRC.NoTangent(), CRC.NoTangent(), ∂sim[3], ∂sim[4], CRC.NoTangent()
    end
    return Y, entmax_adjacency_back
end

Zygote.@adjoint function circulant_adjacency(es::EntmaxSimilarity, x, y, W::Integer)
    A, A_back = Zygote._pullback(__context__, circulant_similarity, es.sim, x, y, W)
    Ww  = windowview(A)
    p   = entmax(Ww, es.α)
    Y   = _circ_from_window(p, A)
    function entmax_adj_back(ΔY)
        ΔY  = _concretize_tangent(Zygote.unthunk(ΔY), Y)
        ΔWw = _entmax_pullback_z(windowview(ΔY), p, Ww, es.α)
        ∂all = A_back(_circ_from_window(ΔWw, A))
        return (nothing, ∂all[3], ∂all[4], nothing)
    end
    return Y, entmax_adj_back
end

CRC.@non_differentiable _entmax_threshold(::Any, ::Any)
