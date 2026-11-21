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
        for file in ("array.jl", "attention.jl", "rrules.jl", "correctness.jl",
                     "flash.jl", "flash_entmax.jl", "grad.jl")
            # reseed per file, so a file's draws do not depend on the files run before it
            Random.seed!(1234)
            CUDA.seed!(1234)
            include(file)
        end
    end
else
    @warn "CUDA is not functional; CirculantAttention needs an NVIDIA GPU, so the tests are skipped."
end
