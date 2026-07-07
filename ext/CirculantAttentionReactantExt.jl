module CirculantAttentionReactantExt

# Reactant + Enzyme autodiff glue for CirculantAttention.
#
# The forward kernels (src/ka_forward.jl) are plain KernelAbstractions kernels
# that Reactant raises into StableHLO. Here we only supply the differentiation:
# an `Enzyme.autodiff` reverse call over the KA forward loss. When this runs
# inside a Reactant `@compile`/`@jit` region, Enzyme-MLIR derives the backward
# from the raised forward — no hand-written backward kernels on this path.
#
# Usage (on a CUDA box):
#   using CirculantAttention, Reactant, Enzyme
#   qr, kr, vr = Reactant.to_rarray.((q, k, v))
#   g = @jit CirculantAttention.reactant_flash_grad(DistanceSimilarity(), qr, kr, vr, W, scale)
#   dq, dk, dv = g

using CirculantAttention
const CA = CirculantAttention
using Reactant
using Enzyme

# Reverse-mode gradient of a scalar loss wrt (q, k, v), with the shape/similarity
# hyperparameters held constant. Shared by every operation's grad entry point.
@inline function _reactant_qkv_grad(loss::F, simfun, q, k, v, W::Int, scale::Real) where F
    dq = Enzyme.make_zero(q)
    dk = Enzyme.make_zero(k)
    dv = Enzyme.make_zero(v)
    Enzyme.autodiff(
        Enzyme.Reverse, loss, Enzyme.Active,
        Enzyme.Const(simfun),
        Enzyme.Duplicated(q, dq),
        Enzyme.Duplicated(k, dk),
        Enzyme.Duplicated(v, dv),
        Enzyme.Const(W),
        Enzyme.Const(scale),
    )
    return dq, dk, dv
end

# Single-head flash attention: ∂/∂(q,k,v) of sum(abs2, flash(q,k,v)).
function CA.reactant_flash_grad(simfun::CA.AbstractSimilarity, q, k, v, W::Int, scale::Real)
    return _reactant_qkv_grad(CA._ka_flash_loss, simfun, q, k, v, W, scale)
end

end # module
