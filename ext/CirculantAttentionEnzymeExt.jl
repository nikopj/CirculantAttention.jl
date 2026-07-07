module CirculantAttentionEnzymeExt

# Enzyme compatibility for CirculantAttention.
#
# The attention operations call hand-written CUDA kernels wrapped in
# ChainRulesCore rrules (src/flash.jl, src/rrules.jl). Enzyme does not use
# ChainRules rules automatically, so here we bridge the existing rrules into
# Enzyme reverse rules with `Enzyme.@import_rrule`. Enzyme then differentiates the
# attention ops by reusing the existing backward kernels — it never has to
# differentiate the CUDA kernels itself.
#
# Phase 1 (this file): the fused flash-attention core `_circulant_flash_attention`
# (arrays in / array out). Both `circulant_flash_attention` and
# `circulant_mh_flash_attention` route their gradients through this rrule, so
# importing it makes the single- and multi-head flash paths Enzyme-differentiable.
# The composed (Circulant-exposing) path is a later phase.

using CirculantAttention
const CA = CirculantAttention
using Enzyme
using CUDA

# _circulant_flash_attention(simfun, q, k, v, W, scale) — the 6-arg rrule that the
# public wrappers call after folding the τ-scale in. (The 5-arg method forwards to
# it, so importing the 6-arg form is sufficient.)
Enzyme.@import_rrule(typeof(CA._circulant_flash_attention),
                     CA.AbstractSimilarity, CuArray, CuArray, CuArray, Int, Real)

# transposed flash core (backs circulant_flash_transposed_attention)
Enzyme.@import_rrule(typeof(CA._circulant_flash_transposed_attention),
                     CA.AbstractSimilarity, CuArray, CuArray, CuArray, Int, Real)

end # module
