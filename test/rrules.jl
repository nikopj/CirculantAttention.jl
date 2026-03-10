@testset "Low-level Circulant rrules" begin

for elty in TEST_ELTYPES, nspatdims in TEST_SPATDIMS
    ws    = 3
    batch = 2
    spatdims = ntuple(_ -> 8, nspatdims)
    tag = "elty=$elty, nspatdims=$nspatdims"

    A  = make_circulant(elty, spatdims, ws, batch)
    B  = make_circulant(elty, spatdims, ws, batch)
    ΔA = rand_tangent(A)
    ΔB = rand_tangent(B)

    nzA = nz_cpu(A)
    nzB = nz_cpu(B)
    nzshape = size(A.data.nzVal)

    # --------------------------------------------------------
    # 1. Constructor rrule
    # --------------------------------------------------------
    @testset "Constructor [$tag]" begin
        test_rrule(
            Circulant, A.data ⊢ rand_csr_tangent(A),
            kernel_length(A), spatial_size(A);
            output_tangent=ΔA, rtol=1e-3, atol=1e-5, check_inferred=false,
        )
    end

    # --------------------------------------------------------
    # 2-5. Arithmetic: traced through CuSparseArrayCSR (no explicit rrules)
    # --------------------------------------------------------

    @testset "Addition [Zygote, $tag]" begin
        gs = Zygote.gradient((a, b) -> sum(real.((a + b))), A, B)
        @test nz_cpu(gs[1]) ≈ ones(real(elty), nzshape...)  rtol=1e-3
        @test nz_cpu(gs[2]) ≈ ones(real(elty), nzshape...)  rtol=1e-3
    end

    @testset "Subtraction [Zygote, $tag]" begin
        gs = Zygote.gradient((a, b) -> sum(real.((a - b))), A, B)
        @test nz_cpu(gs[1]) ≈  ones(real(elty), nzshape...)  rtol=1e-3
        @test nz_cpu(gs[2]) ≈ -ones(real(elty), nzshape...)  rtol=1e-3
    end

    @testset "Negation [Zygote, $tag]" begin
        gs = Zygote.gradient(a -> sum(real.((-a))), A)
        @test nz_cpu(gs[1]) ≈ -ones(real(elty), nzshape...)  rtol=1e-3
    end

    @testset "Scalar multiplication [$tag]" begin
        c  = real(elty)(2.5)
        test_rrule(*, c, A ⊢ ΔA;
            output_tangent=ΔB, rtol=1e-3, atol=1e-5, check_inferred=false)
    end

    # --------------------------------------------------------
    # 6-7. Same-shape broadcast gradients
    # --------------------------------------------------------

    @testset "Broadcast .* same shape [$tag]" begin
        gs = Zygote.gradient((a, b) -> sum(real.((a .* b))), A, B)
        @test nz_cpu(gs[1]) ≈ conj(nzB)  rtol=1e-3
        @test nz_cpu(gs[2]) ≈ conj(nzA)  rtol=1e-3
    end

    @testset "Broadcast .+ same shape [$tag]" begin
        gs = Zygote.gradient((a, b) -> sum(real.((a .+ b))), A, B)
        @test nz_cpu(gs[1]) ≈ ones(real(elty), nzshape...)  rtol=1e-3
        @test nz_cpu(gs[2]) ≈ ones(real(elty), nzshape...)  rtol=1e-3
    end

    # --------------------------------------------------------
    # 8-9. Batch-expanding broadcast: ∂A1 sums over batch dim
    # --------------------------------------------------------

    @testset "Batch-expanding broadcast .+ [$tag]" begin
        A1 = make_circulant(elty, spatdims, ws, 1)
        gs = Zygote.gradient(a1 -> sum(real.((A .+ a1))), A1)
        @test nz_cpu(gs[1]) ≈ fill(real(elty)(batch), size(A1.data.nzVal)...)  rtol=1e-3
    end

    @testset "Batch-expanding broadcast .- [$tag]" begin
        A1 = make_circulant(elty, spatdims, ws, 1)
        gs = Zygote.gradient(a1 -> sum(real.((A .- a1))), A1)
        @test nz_cpu(gs[1]) ≈ fill(real(elty)(-batch), size(A1.data.nzVal)...)  rtol=1e-3
    end

    # --------------------------------------------------------
    # 10. CuArray .* Circulant (replaces scale)
    #     c has shape (1,1,...,batch) — one scalar per batch element.
    #     ∂c[b] = sum of A's nzVals for batch b  (chain rule: ∂(c*nz)/∂c = nz)
    #     ∂A.nzVal[i,b] = c[b]                   (chain rule: ∂(c*nz)/∂nz = c)
    # --------------------------------------------------------
    @testset "CuArray .* Circulant (batch scale) [$tag]" begin
        # c: one scalar per batch, broadcast over all spatial/channel dims
        c     = CUDA.randn(real(elty), ntuple(_ -> 1, ndims(A) - 1)..., batch)
        c_cpu = Array(c)
        gs    = Zygote.gradient((c, a) -> sum(real.(c .* a)), c, A)
        ∂c    = Array(gs[1])       # same shape as c: (1,1,...,batch)
        ∂A_nz = nz_cpu(gs[2])      # same shape as nzVal: (nnz, batch...)

        # nzVal last dim and c last dim both index batch
        nz_last = ndims(∂A_nz)
        c_last  = ndims(∂c)

        for b in 1:batch
            c_b        = real(selectdim(c_cpu, c_last,  b)[])
            nzA_b      = real.(selectdim(nzA,  nz_last, b))
            # ∂c[b] accumulates gradients from every nzVal entry in batch b
            @test selectdim(∂c, c_last, b)[] ≈ sum(nzA_b)  rtol=1e-3
            # every nzVal entry in batch b receives the same gradient c[b]
            @test all(≈(c_b; rtol=1e-3), selectdim(∂A_nz, nz_last, b))
        end
    end

    # --------------------------------------------------------
    # 11. sum (batch dims) rrule
    # --------------------------------------------------------
    @testset "sum (batch dims) [$tag]" begin
        result  = sum(A; dims=ndims(A))
        Δresult = rand_tangent(result)
        test_rrule(
            Base.sum, A ⊢ ΔA;
            fkwargs=(dims=ndims(A),), output_tangent=Δresult,
            rtol=1e-3, atol=1e-5, check_inferred=false,
        )
    end

    # --------------------------------------------------------
    # 12-13. sum/dot: value checks only (nzVals structurally shared)
    # --------------------------------------------------------

    @testset "dot (value) [$tag]" begin
        @test LinearAlgebra.dot(A, B) ≈ dot(vec(nzA), vec(nzB))
    end

    # --------------------------------------------------------
    # 14. repeat rrule
    # --------------------------------------------------------
    @testset "repeat (batch dim) [$tag]" begin
        reps  = ntuple(i -> i == ndims(A) ? 2 : 1, ndims(A))
        Arep  = repeat(A, reps...)
        ΔArep = rand_tangent(Arep)
        test_rrule(
            Base.repeat, A ⊢ ΔA, reps...;
            output_tangent=ΔArep, rtol=1e-3, atol=1e-5, check_inferred=false,
        )
    end

    # --------------------------------------------------------
    # 15. Chain: c*(A+B) - A  →  ∂A = c-1, ∂B = c (uniform)
    # --------------------------------------------------------
    @testset "Chain: c*(A+B) - A [$tag]" begin
        c  = real(elty)(1.5)
        gs = Zygote.gradient((a, b) -> sum(real.(c * (a + b) - a)), A, B)
        @test all(≈(c - 1; atol=1e-4), nz_cpu(gs[1]))
        @test all(≈(c;     atol=1e-4), nz_cpu(gs[2]))
    end

    # --------------------------------------------------------
    # 16. sparsemax, entmax
    # --------------------------------------------------------
    @testset "sparsemax [$tag]" begin
        W = windowview(real(A))
        ΔW = similar(W); CUDA.randn!(ΔW)
        V = similar(W); CUDA.randn!(V)
        test_rrule(sparsemax, W ⊢ ΔW, output_tangent=V,
            rtol=1e-3, atol=1e-3, check_inferred=false,
        )
    end

    @testset "entmax, α scalar [$tag]" begin
        W = windowview(real(A))
        ΔW = similar(W); CUDA.randn!(ΔW)
        V = similar(W); CUDA.randn!(V)
        test_rrule(entmax, W ⊢ ΔW, 1.5f0 ⊢ (1f0 + rand()), output_tangent=V,
            rtol=1e-3, atol=1e-3, check_inferred=false,
        )
    end

    @testset "entmax, α array [$tag]" begin
        α  = 1f0 .+ CUDA.rand(Float32, 1, 1, 2, 1)
        W  = windowview(real(A .* α))
        ΔW = similar(W); CUDA.randn!(ΔW)
        V  = similar(W); CUDA.randn!(V)
        Δα = CUDA.rand(Float32, 1, 1, 2, 1)
        test_rrule(entmax, W ⊢ ΔW, α ⊢ Δα, output_tangent=V,
            rtol=1e-3, atol=1e-3, check_inferred=false,
        )
    end

end  # for elty, nspatdims

end  # @testset
