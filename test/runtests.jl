using CirculantAttention
using Test
using Zygote

using Adapt
using LinearAlgebra
using CUDA, CUDA.CUSPARSE
using NNlib
using Statistics

using ChainRulesCore
using ChainRulesTestUtils
using ChainRulesTestUtils: @test_msg
import FiniteDifferences as FD

CUDA.allowscalar(false)
ChainRulesCore.debug_mode() = true

TEST_ELTYPES = (Float32, ComplexF32)
TEST_SPATDIMS = (1,2)
 
include("utils.jl")

@testset "CirculantAttention.jl" begin
    include("array.jl")
    include("rrules.jl")
    include("attention.jl")
    include("grad.jl")
end
