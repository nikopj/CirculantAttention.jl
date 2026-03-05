module CirculantAttention

const CircAtt = CirculantAttention
export CircAtt

using CUDA, CUDA.CUSPARSE, cuDNN
using CUDA: i32
import Adapt

using SparseArrays
using LinearAlgebra
using KernelAbstractions.Extras: @unroll

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

include("attention.jl")
export joint_softmax
export circulant_attention, circulant_mh_attention, circulant_mh_adjacency, ⊗, ⨷ # \otimes and \Otimes

include("batchedmul.jl")

include("rrules.jl")
include("zygote.jl")

end
