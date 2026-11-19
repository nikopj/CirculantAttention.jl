"""
    CirculantAttention

Sliding-window ("circulant-sparse") attention for CUDA arrays: the attention matrix has a
window of odd width around every pixel of a 1-D or 2-D signal, with periodic boundaries.
It is either materialized as a [`Circulant`](@ref) matrix or fused into flash kernels.
Also available as `CircAtt`.
"""
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

# fused (flash) α-entmax / sparsemax attention — extends the flash functions
# for EntmaxSimilarity / SparsemaxSimilarity.
include("flash_entmax.jl")

# fused (flash) JOINT + guided α-entmax / sparsemax over multiple branches.
include("flash_joint_entmax.jl")

include("batchedmul.jl")

include("rrules.jl")
include("zygote.jl")

end
