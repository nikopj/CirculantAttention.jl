# Stretch nzVals away from their per-row mean along the normalisation dim
# (dims=1) so in-/out-of-support entries have a clear margin. sparsemax/entmax
# are piecewise-linear with breakpoints at the support boundary; without a
# margin, FiniteDifferences' finite perturbation flips support membership and
# its reference gradient is wrong at those entries. Matches the conditioning
# used in the joint_sparsemax/joint_entmax tests.
stretch_support!(W) = (W .+= 2 .* (W .- sum(W; dims=1) ./ size(W, 1)); W)

# The opposite conditioning, for the entmax α-gradient. Center each column and
# rescale to a small fixed spread (±0.5) so entmax(α∈(1,2]) is DENSE — every
# entry strictly in-support with clear margin from zero. Stretching (above)
# drives entmax toward one-hot, where ∂p/∂α ≈ 0 (degenerate) AND the marginal
# entry sits on the support boundary, so FiniteDifferences' α-step kinks across
# it and its reference gradient is wrong. In the dense regime both the W- and
# α-gradients are smooth and FD is reliable, at tight tolerance.
function densify!(W)
    m   = sum(W; dims=1) ./ size(W, 1)
    W .-= m
    pk  = maximum(abs.(W); dims=1) .+ 1f-6
    W  .= 0.5f0 .* W ./ pk
    return W
end

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
        # The scalar's cotangent is ⟨A, Δout⟩ — a reduction over all nnz entries.
        # A random output tangent makes this a small, near-cancelled sum that
        # Float32 finite-differencing can't resolve (low-order bits are the whole
        # answer). Seed the output tangent from A itself so ⟨A, Δout⟩ = ‖A‖²:
        # O(nnz), strictly positive, cancellation-free — the FD reference is then
        # reliable without loosening tolerances. (The array cotangent is c·Δout,
        # well-conditioned per element either way.)
        Δout = copy(A)
        test_rrule(*, c, A ⊢ ΔA;
            output_tangent=Δout, rtol=1e-3, atol=1e-5, check_inferred=false)
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
    # 14b. cat rrule (batch dim)
    # --------------------------------------------------------
    @testset "cat (batch dim) [$tag]" begin
        Acat  = cat(A, B; dims=ndims(A))
        ΔAcat = rand_tangent(Acat)
        test_rrule(
            Base.cat, A ⊢ ΔA, B ⊢ ΔB;
            fkwargs=(dims=ndims(A),), output_tangent=ΔAcat,
            rtol=1e-3, atol=1e-5, check_inferred=false,
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
        W = windowview(real(A)); stretch_support!(W)
        ΔW = similar(W); randn!(ΔW)
        V = similar(W); randn!(V)
        test_rrule(sparsemax, W ⊢ ΔW, output_tangent=V,
            rtol=1e-3, atol=1e-3, check_inferred=false,
        )
    end

    @testset "entmax, α scalar [$tag]" begin
        W = windowview(real(A)); densify!(W)
        ΔW = similar(W); randn!(ΔW)
        V = similar(W); randn!(V)
        test_rrule(entmax, W ⊢ ΔW, 1.5f0 ⊢ (1f0 + rand()), output_tangent=V,
            rtol=1e-3, atol=1e-3, check_inferred=false,
        )
    end

    @testset "entmax, α array [$tag]" begin
        α  = 1f0 .+ CUDA.rand(Float32, 1, 1, 2, 1)
        W  = windowview(real(A .* α)); densify!(W)
        ΔW = similar(W); randn!(ΔW)
        V  = similar(W); randn!(V)
        Δα = CUDA.rand(Float32, 1, 1, 2, 1)
        test_rrule(entmax, W ⊢ ΔW, α ⊢ Δα, output_tangent=V,
            rtol=1e-3, atol=1e-3, check_inferred=false,
        )
    end

end  # for elty, nspatdims

end  # @testset
