module CirculantAttention

const CircAtt = CirculantAttention
export CircAtt

using CUDA, CUDA.CUSPARSE, cuDNN
import Adapt

using SparseArrays
using LinearAlgebra

using NNlib
import ChainRulesCore as CRC
import Zygote

include("array.jl")
include("circulant.jl")
export Circulant, circulant

include("similarity.jl")
export circulant_similarity, circulant_similarity!, DotSimilarity, DistanceSimilarity
export circulant_adjacency, circulant_adjacency!

include("attention.jl")
export circulant_attention, circulant_mh_attention, circulant_mh_adjacency, ⊗, ⨷ # \otimes and \Otimes

include("batchedmul.jl")

include("rrules.jl")
include("zygote.jl")

end
