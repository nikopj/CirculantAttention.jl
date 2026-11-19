using CirculantAttention
using Test
using Zygote

using Adapt
using LinearAlgebra
using CUDA, CUDA.CUSPARSE
using NNlib
using Random

using ChainRulesCore
using ChainRulesTestUtils
using ChainRulesTestUtils: @test_msg
import FiniteDifferences as FD

const TEST_ELTYPES = (Float32, ComplexF32)
const TEST_SPATDIMS = (1, 2)

if CUDA.functional()
    CUDA.allowscalar(false)
    ChainRulesCore.debug_mode() = true
    include("utils.jl")

    @testset "CirculantAttention.jl" begin
        include("array.jl")
        include("attention.jl")
        include("rrules.jl")
        include("correctness.jl")
        include("flash.jl")
        include("flash_entmax.jl")
        include("grad.jl")
    end
else
    @warn "CUDA is not functional; CirculantAttention needs an NVIDIA GPU, so the tests are skipped."
end
