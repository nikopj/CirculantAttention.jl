module CirculantAttention

using CUDA, CUDA.CUSPARSE, cuDNN
import Adapt

using SparseArrays
using LinearAlgebra

using NNlib
import ChainRulesCore as CRC

include("array.jl")
include("circulant.jl")
export Circulant, circulant

include("similarity.jl")
export circulant_similarity, circulant_similarity!, DotSimilarity, DistanceSimilarity
export circulant_adjacency, circulant_adjacency!

# include("convex_comb.jl")
# export convex_comb

include("attention.jl")
export circulant_attention, circulant_mh_attention, circulant_mh_adjacency, ⊗, ⨷

include("batchedmul.jl")

include("rrules.jl")

end
