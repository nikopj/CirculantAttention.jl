using CirculantAttention
using Test

using CUDA, CUDA.CUSPARSE
using NNlib

using ChainRulesTestUtils
using ChainRulesTestUtils: @test_msg
import FiniteDifferences as FD

CUDA.allowscalar(false)

include("utils.jl")

@testset "CirculantAttention.jl" begin
    include("rrule.jl")
    include("grad.jl")
end
