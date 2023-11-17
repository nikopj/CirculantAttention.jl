using CirculantAttention
using Test

using CUDA, CUDA.CUSPARSE
using NNlib

using ChainRulesCore
using ChainRulesTestUtils
using ChainRulesTestUtils: @test_msg
import FiniteDifferences as FD

CUDA.allowscalar(false)
ChainRulesCore.debug_mode() = true

include("utils.jl")

@testset "CirculantAttention.jl" begin
    include("rrule.jl")
    include("grad.jl")
end
