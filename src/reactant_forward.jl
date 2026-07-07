# reactant_forward.jl
#
# Traceable (array-op) circulant flash attention for the Reactant + Enzyme path.
#
# Under Reactant the hand-written CUDA flash kernels lower to an opaque
# `enzymexla.kernel_call` that Enzyme-MLIR cannot differentiate (custom-kernel
# adjoints are not yet supported, and raising fails on the online-softmax loops +
# gather). The Reactant-recommended route is to express the op with native array
# ops (slices/shifts/reductions) and let XLA fuse them. The Reactant extension
# substitutes this implementation for the CUDA one during tracing (via
# @reactant_overlay); normal CUDA / Zygote use is untouched.
#
# Memory: a circulant window is a set of fixed circular spatial SHIFTS, so we do
# K = W^spatialdims `circshift`s and reduce over channels immediately. The only
# per-window state is the score tensor (spatial…, B) × K → O(N·K·B), independent
# of the channel count C (the naive gather would be O(N·K·C)). Result is
# identical to `circulant_flash_attention`: y = rowsoftmax(scale·S)·v.

# Window offsets as spatial shift tuples. Order is irrelevant (softmax + weighted
# sum over the window are permutation-invariant), so this need not match the
# `cartesian_circulant` enumeration — only the neighbour SET must agree, and
# {r+δ mod N : δ ∈ -p:p} is exactly the circulant window.
function _window_offsets(W::Int, Sdim::Int)
    p = (W - 1) ÷ 2
    rng = -p:p
    Sdim == 1 && return [(δ,) for δ in rng]
    return [(δ1, δ2) for δ2 in rng for δ1 in rng]   # W×W box (block-circulant)
end

# circular shift along the spatial dims only (channel/batch fixed). Neighbour at
# spatial index r+δ is circshift(x, -δ) evaluated at r.
@inline function _shift_spatial(x::AbstractArray{T,N}, off::NTuple{S,Int}) where {T,N,S}
    circshift(x, ntuple(i -> i <= S ? -off[i] : 0, N))
end

# per-offset similarity score reduced over the channel dim → (spatial…, B).
# real inputs: real./conj. are identities. cdim = N-1 is the channel dim.
_shift_score(::RealDotSimilarity, q, ksh, cdim) =
    dropdims(sum(real.(q .* conj.(ksh)); dims=cdim); dims=cdim)
function _shift_score(::DistanceSimilarity, q, ksh, cdim)
    R = real(eltype(q))
    (-R(0.5)) .* dropdims(sum(abs2.(q .- ksh); dims=cdim); dims=cdim)
end
_shift_score(::DotSimilarity, q, ksh, cdim) =
    dropdims(sum(q .* conj.(ksh); dims=cdim); dims=cdim)
_shift_score(::PIDotSimilarity, q, ksh, cdim) =
    abs.(dropdims(sum(q .* conj.(ksh); dims=cdim); dims=cdim))
function _shift_score(::PIDistanceSimilarity, q, ksh, cdim)
    R = real(eltype(q))
    nq = dropdims(sum(abs2.(q);          dims=cdim); dims=cdim)
    dt = abs.(dropdims(sum(q .* conj.(ksh); dims=cdim); dims=cdim))
    nk = dropdims(sum(abs2.(ksh);        dims=cdim); dims=cdim)
    (-R(0.5)) .* nq .+ dt .+ (-R(0.5)) .* nk
end

"""
    _circ_flash_attention_shift(simfun, q, k, v, W, scale=true) -> y

Array-op circulant flash attention (kernel-free), numerically equal to
`circulant_flash_attention`. Used on the Reactant+Enzyme path via @reactant_overlay.
Real-valued similarities only.
"""
function _circ_flash_attention_shift(
        simfun::AbstractSimilarity,
        q::AbstractArray{Tq,N}, k::AbstractArray{Tk,N}, v::AbstractArray{Tv,N},
        W::Int, scale::Real=true) where {Tq, Tk, Tv, N}
    Sdim = N - 2
    cdim = N - 1
    offs = _window_offsets(W, Sdim)

    # scores over the window (list of (spatial…, B) arrays)
    Ss = map(offs) do off
        scale .* _shift_score(simfun, q, _shift_spatial(k, off), cdim)
    end

    # numerically-stable softmax over the window list, then weighted sum of the
    # correspondingly-shifted v. No (spatial…, K, …) tensor is materialized.
    m  = reduce((a, b) -> max.(a, b), Ss)
    es = map(s -> exp.(s .- m), Ss)
    l  = reduce(.+, es)

    spat = size(q)[1:Sdim]
    B    = size(q, N)
    return mapreduce(+, zip(es, offs)) do (e, off)
        w = reshape(e ./ l, spat..., 1, B)          # (spatial…, 1, B)
        w .* _shift_spatial(v, off)                 # broadcast over channels
    end
end
