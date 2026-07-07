module CirculantAttentionReactantExt

# Reactant compatibility for CirculantAttention.
#
# The attention forward is implemented with hand-written CUDA kernels that
# Reactant lowers to an opaque `enzymexla.kernel_call` — not differentiable by
# Enzyme-MLIR (custom-kernel adjoints are not yet supported upstream). So during
# Reactant tracing we substitute the traceable array-op forward
# `_circ_flash_attention_shift` (src/reactant_forward.jl), which XLA fuses and
# Enzyme-MLIR differentiates natively. Outside Reactant, the fast CUDA kernels and
# their ChainRules rrules are used unchanged.
#
# Phase 1: the single-head flash core `_circulant_flash_attention`. Both
# `circulant_flash_attention` and `circulant_mh_flash_attention` route through it
# (mh only reshapes/splits heads around the core), so this makes the single- and
# multi-head flash paths Reactant+Enzyme differentiable.

using CirculantAttention
const CA = CirculantAttention
using Reactant

# During tracing, replace the CUDA-kernel flash core with the array-op version.
# `scale` is left untyped: `circulant_flash_attention` derives it from the (traced)
# element type, so under Reactant it arrives as a TracedRNumber, not a `Real`.
Reactant.@reactant_overlay @noinline function CA._circulant_flash_attention(
        simfun::CA.AbstractSimilarity, q, k, v, W::Int, scale)
    return CA._circ_flash_attention_shift(simfun, q, k, v, W, scale)
end

end # module
