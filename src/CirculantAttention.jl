module CirculantAttention

const CircAtt = CirculantAttention
export CircAtt

using CUDA, CUDA.CUSPARSE, cuDNN
using CUDA: i32
using GPUArrays
import Adapt

using SparseArrays
using LinearAlgebra

using NNlib
import ChainRulesCore as CRC
import Zygote

include("array.jl")
include("broadcast.jl")
include("circulant.jl")
export Circulant, circulant
export kernel_length, spatial_size, spatial_dims, windowview

include("similarity.jl")
export circulant_similarity, circulant_similarity!
export DotSimilarity, RealDotSimilarity, DistanceSimilarity, PIDotSimilarity, PIDistanceSimilarity
export circulant_adjacency, circulant_adjacency!

include("topk.jl")
include("sparsemax.jl")
include("entmax.jl")
export TopKSimilarity
export SparsemaxSimilarity, EntmaxSimilarity
export sparsemax, entmax
export joint_sparsemax, joint_entmax

include("attention.jl")
export joint_softmax
export circulant_attention, circulant_mh_attention, circulant_mh_adjacency, ⊗, ⨷ # \otimes and \Otimes

include("flash.jl")
export circulant_flash_attention, circulant_mh_flash_attention, circulant_flash_joint_attention
export circulant_mh_flash_joint_attention, circulant_mh_flash_guided_joint_attention
export circulant_flash_transposed_attention, circulant_mh_flash_transposed_attention

include("batchedmul.jl")

# Traceable array-op forward for the Reactant + Enzyme path (always loaded — pure
# Julia). The Reactant extension swaps it in during tracing via @reactant_overlay.
include("reactant_forward.jl")

include("rrules.jl")
include("zygote.jl")

end
