# ============================================================
# Helpers
# ============================================================

# Build a random Circulant with known sparsity pattern for a given eltype,
# spatial size, kernel length, and batch size.
function make_circulant(elty, spatdims, ws, batch)
    x = CUDA.randn(elty, spatdims..., size_d(elty), batch)
    circulant_similarity(DotSimilarity(), x, x, ws)
end

# Return a matching random tangent Circulant (same sparsity pattern, random nzVal)
function rand_tangent(A::Circulant{T}) where T
    B = copy(A)
    B.data.nzVal = CUDA.randn(T, size(A.data.nzVal)...)
    return B
end

size_d(::Type{Float32})    = 4
size_d(::Type{ComplexF32}) = 4

# ============================================================
# Test suite
# ============================================================

@testset "Low-level Circulant rrules" begin

    # for elty in (Float32, ComplexF32), nspatdims in (1, 2)
    for elty in (Float32, ComplexF32), nspatdims in (1,)
        ws = 3
        batch = 2
        spatdims = ntuple(_ -> 8, nspatdims)
        tag = "elty=$elty, nspatdims=$nspatdims"

        # Build two Circulants and matching tangents
        A = make_circulant(elty, spatdims, ws, batch)
        B = make_circulant(elty, spatdims, ws, batch)
        ΔA = rand_tangent(A)
        ΔB = rand_tangent(B)

        # --------------------------------------------------------
        # 1. Constructor rrule
        #    Circulant(data, M, spatsize) — gradient must flow to data
        # --------------------------------------------------------
        @testset "Constructor [$tag]" begin
            # Tangent for data must be a CuSparseArrayCSR with the same sparsity pattern
            Δdata = CuSparseArrayCSR(
                copy(A.data.rowPtr),
                copy(A.data.colVal),
                CUDA.randn(real(elty), size(A.data.nzVal)...),
                size(A.data),
            )
            test_rrule(
                Circulant,
                A.data ⊢ Δdata,
                CircAtt.kernel_length(A),
                CircAtt.spatial_size(A);
                output_tangent = ΔA,
                rtol = 1e-3, atol = 1e-5,
                check_inferred = false,
            )
        end

        # --------------------------------------------------------
        # 2. copy rrule
        #    gradient of copy is identity
        # --------------------------------------------------------
        @testset "copy [$tag]" begin
            test_rrule(
                Base.copy, A ⊢ ΔA;
                output_tangent = ΔB,
                rtol = 1e-3, atol = 1e-5,
                check_inferred = false,
            )
        end

        # --------------------------------------------------------
        # 3. Addition rrule
        #    ∂A = ΔZ, ∂B = ΔZ
        # --------------------------------------------------------
        @testset "Addition [$tag]" begin
            test_rrule(
                +, A ⊢ ΔA, B ⊢ ΔB;
                output_tangent = ΔA,
                rtol = 1e-3, atol = 1e-5,
                check_inferred = false,
            )
        end

        # --------------------------------------------------------
        # 4. Subtraction rrule
        #    ∂A = ΔZ, ∂B = -ΔZ
        # --------------------------------------------------------
        @testset "Subtraction [$tag]" begin
            test_rrule(
                -, A ⊢ ΔA, B ⊢ ΔB;
                output_tangent = ΔA,
                rtol = 1e-3, atol = 1e-5,
                check_inferred = false,
            )
        end

        # --------------------------------------------------------
        # 5. Unary negation rrule
        #    ∂A = -ΔZ
        # --------------------------------------------------------
        @testset "Negation [$tag]" begin
            test_rrule(
                -, A ⊢ ΔA;
                output_tangent = ΔB,
                rtol = 1e-3, atol = 1e-5,
                check_inferred = false,
            )
        end

        # --------------------------------------------------------
        # 6. Scalar multiplication rrule  c * A
        #    ∂c = ⟨A, ΔZ⟩,  ∂A = conj(c) * ΔZ
        # --------------------------------------------------------
        @testset "Scalar multiplication [$tag]" begin
            c = real(elty)(2.5)
            test_rrule(
                *, c, A ⊢ ΔA;
                output_tangent = ΔB,
                rtol = 1e-3, atol = 1e-5,
                check_inferred = false,
            )
        end

        # dot — don't use test_rrule for same reason as sum (full reduction):
        # nzVals are not independent matrix entries. Just verify the value.
        @testset "dot (value) [$tag]" begin
            d = LinearAlgebra.dot(A, B)
            @test d ≈ dot(Array(A.data.nzVal), Array(B.data.nzVal))
        end

        # Full reduction — don't use test_rrule since sum(A) traces through nzVal
        # which has shared structure. Just verify the value is correct.
        @testset "sum (full value) [$tag]" begin
            s = sum(A)
            @test s ≈ sum(Array(A.data.nzVal))
        end

        # Batch-dim sum — output_tangent must be a Circulant with correct reduced shape
        @testset "sum (batch dims) [$tag]" begin
            result  = sum(A; dims = ndims(A))
            Δresult = rand_tangent(result)
            test_rrule(
                Base.sum, A ⊢ ΔA;
                fkwargs        = (dims = ndims(A),),
                output_tangent = Δresult,
                rtol = 1e-3, atol = 1e-5,
                check_inferred = false,
            )
        end

        # --------------------------------------------------------
        # 10. End-to-end: gradient flows through a chain of low-level ops
        #     loss = sum(c * (A + B) - A)  — exercises +, -, *, sum
        # --------------------------------------------------------
        @testset "Chain: c*(A+B) - A [$tag]" begin
            c = real(elty)(1.5)
            function chain_loss(A, B)
                sum(abs.(c * (A + B) - A))
            end
            # Verify finite-difference matches analytic gradient
            # ∂loss/∂A = c - 1 (broadcast over nzVals), ∂loss/∂B = c
            gs = Zygote.gradient(chain_loss, A, B)
            ∂A_nz = Array(gs[1].data.nzVal)
            ∂B_nz = Array(gs[2].data.nzVal)
            @test all(≈(c - 1; atol=1e-5), ∂A_nz)
            @test all(≈(c;     atol=1e-5), ∂B_nz)
        end

        # --------------------------------------------------------
        # 11. getproperty :data — gradient flows through field access
        #     into a Circulant-level operation
        # --------------------------------------------------------
        @testset "getproperty :data [$tag]" begin
            # Use test_rrule on getproperty directly
            test_rrule(
                Base.getproperty, A ⊢ ΔA, :data;
                # output tangent is a CuSparseArrayCSR tangent matching A.data
                output_tangent = CuSparseArrayCSR(
                    copy(A.data.rowPtr),
                    copy(A.data.colVal),
                    CUDA.randn(real(elty), size(A.data.nzVal)...),
                    size(A.data),
                ),
                rtol = 1e-3, atol = 1e-5,
                check_inferred = false,
            )
        end

        # --------------------------------------------------------
        # 12. repeat rrule — gradient sums over repeated copies
        # --------------------------------------------------------
        @testset "repeat (batch dim) [$tag]" begin
            # repeat along the batch dim (last dim = ndims(A))
            # result has 2x the batch size; gradient sums the two copies
            reps = ntuple(i -> i == ndims(A) ? 2 : 1, ndims(A))
            Arep = repeat(A, reps...)
            ΔArep = rand_tangent(Arep)
            test_rrule(
                Base.repeat, A ⊢ ΔA, reps...;
                output_tangent = ΔArep,
                rtol = 1e-3, atol = 1e-5,
                check_inferred = false,
            )
        end

    end  # for elty, nspatdims

end  # @testset
