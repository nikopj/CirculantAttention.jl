# test/array.jl
 
# ============================================================
# Array and broadcast functionality (no AD)
# ============================================================

@testset "Circulant array and broadcast" begin

for elty in TEST_ELTYPES, nspatdims in TEST_SPATDIMS
        ws    = 3
        batch = 2
        spatdims = ntuple(_ -> 8, nspatdims)
        N   = spatdims[1]
        tag = "elty=$elty, nspatdims=$nspatdims"

        A = make_circulant(elty, spatdims, ws, batch)
        B = make_circulant(elty, spatdims, ws, batch)

        nzA = nz_cpu(A)
        nzB = nz_cpu(B)
        nd  = ndims(A.data.nzVal)   # nzVal rank: 1 + nspatdims (flat rows × batch...)

        # --------------------------------------------------------
        # Basic properties
        # --------------------------------------------------------
        @testset "Properties [$tag]" begin
            @test eltype(A)        == elty
            @test ndims(A)         == 4 
            @test size(A, 1)       == (N^nspatdims)
            @test size(A, 2)       == (N^nspatdims)
            @test size(A)[end]     == batch
            @test kernel_length(A) == ws
            @test spatial_size(A)  == spatdims
            @test spatial_dims(A)  == nspatdims
        end

        # --------------------------------------------------------
        # copy / similar
        # --------------------------------------------------------
        @testset "copy / similar [$tag]" begin
            C = copy(A)
            @test C == A
            @test C.data.nzVal !== A.data.nzVal

            S = similar(A)
            @test size(S)          == size(A)
            @test kernel_length(S) == kernel_length(A)
            @test spatial_size(S)  == spatial_size(A)
        end

        # --------------------------------------------------------
        # Equality
        # --------------------------------------------------------
        @testset "Equality [$tag]" begin
            @test A == A
            @test !(A == B)
            C = copy(A)
            C.data.nzVal[1:1] .+= 1
            @test !(A == C)
        end

        # --------------------------------------------------------
        # repeat
        # --------------------------------------------------------
        @testset "repeat [$tag]" begin
            reps = ntuple(i -> i == ndims(A) ? 3 : 1, ndims(A))
            R    = repeat(A, reps...)
            nzR  = nz_cpu(R)

            @test size(R)[end]     == batch * 3
            @test kernel_length(R) == kernel_length(A)

            for b in 1:batch
                @test nz_batch(nzR, b)       ≈ nz_batch(nzA, b)
                @test nz_batch(nzR, b+batch) ≈ nz_batch(nzA, b)
            end
        end

        # --------------------------------------------------------
        # cat
        # --------------------------------------------------------
        @testset "cat [$tag]" begin
            C   = cat(A, B; dims=ndims(A))
            nzC = nz_cpu(C)

            @test size(C)[end] == 2batch
            for b in 1:batch
                @test nz_batch(nzC, b)       ≈ nz_batch(nzA, b)
                @test nz_batch(nzC, b+batch) ≈ nz_batch(nzB, b)
            end
        end

        # --------------------------------------------------------
        # Arithmetic: +, -, unary -, scalar *
        # --------------------------------------------------------
        @testset "Arithmetic [$tag]" begin
            @test nz_cpu(A + B) ≈ nzA .+ nzB
            @test nz_cpu(A - B) ≈ nzA .- nzB

            c = real(elty)(2.5)
            @test nz_cpu(c * A) ≈ c .* nzA
            @test nz_cpu(A * c) ≈ c .* nzA
            @test nz_cpu(-A)    ≈ .-nzA
        end

        # --------------------------------------------------------
        # Non-broadcasting arithmetic enforces size equality
        # --------------------------------------------------------
        @testset "Size mismatch throws [$tag]" begin
            A1 = make_circulant(elty, spatdims, ws, 1)
            @test_throws DimensionMismatch A + A1
            @test_throws DimensionMismatch A - A1
            @test_nowarn A .+ A1
            @test_nowarn A .- A1
        end

        # --------------------------------------------------------
        # Broadcast: same-shape
        # --------------------------------------------------------
        @testset "Broadcast same shape [$tag]" begin
            @test size(A .+ B) == size(A)
            @test nz_cpu(A .+ B) ≈ nzA .+ nzB
            @test nz_cpu(A .- B) ≈ nzA .- nzB
            @test nz_cpu(A .* B) ≈ nzA .* nzB

            c = real(elty)(3.0)
            @test nz_cpu(A .* c) ≈ c .* nzA
            @test nz_cpu(c .* A) ≈ c .* nzA
        end

        # --------------------------------------------------------
        # Broadcast: batch-expanding
        # --------------------------------------------------------
        @testset "Batch-expanding broadcast [$tag]" begin
            A1   = make_circulant(elty, spatdims, ws, 1)
            nzA1 = nz_cpu(A1)

            for (C, f) in ((A .+ A1, +), (A .- A1, -), (A .* A1, *))
                nzC = nz_cpu(C)
                @test size(C) == size(A)
                for b in 1:batch
                    @test nz_batch(nzC, b) ≈ f.(nz_batch(nzA, b), nz_batch(nzA1, 1))
                end
            end

            # Commutativity of .+
            @test nz_cpu(A .+ A1) ≈ nz_cpu(A1 .+ A)
        end

        # --------------------------------------------------------
        # CuArray .* Circulant (replaces scale)
        # --------------------------------------------------------
        @testset "CuArray .* Circulant [$tag]" begin
            c     = CUDA.randn(real(elty), ntuple(_ -> 1, ndims(A) - 1)..., batch)
            c_cpu = Array(c)
            nzC   = nz_cpu(c .* A)

            @test size(c .* A) == size(A)
            for b in 1:batch
                c_b = real(selectdim(c_cpu, ndims(c_cpu), b)[])
                @test nz_batch(nzC, b) ≈ c_b .* nz_batch(nzA, b)
            end
            @test nz_cpu(A .* c) ≈ nzC  # commutative
        end

        # --------------------------------------------------------
        # In-place broadcast
        # --------------------------------------------------------
        @testset "In-place broadcast [$tag]" begin
            C = copy(A); C .= A .+ B
            @test nz_cpu(C) ≈ nzA .+ nzB

            c = real(elty)(2.0)
            D = copy(A); D .= c .* A
            @test nz_cpu(D) ≈ c .* nzA
        end

        # --------------------------------------------------------
        # windowview shape and write-through
        # --------------------------------------------------------
        @testset "windowview [$tag]" begin
            A_local = copy(A)
            W           = windowview(A_local)
            nnz_per_row = size(A_local.data.nzVal, 1) ÷ size(A_local, 1)
            @test size(W) == (nnz_per_row, size(A_local, 1), size(A_local)[3:end]...)

            W .= zero(elty)
            @test all(iszero, Array(A_local.data.nzVal))
        end

        # --------------------------------------------------------
        # sum
        # --------------------------------------------------------
        @testset "sum [$tag]" begin
            @test sum(A) ≈ sum(nzA)

            S = sum(A; dims=ndims(A))
            @test S isa Circulant
            @test size(S)[end] == 1
            @test kernel_length(S) == kernel_length(A)
            @test nz_cpu(S) ≈ sum(nzA; dims=nd)

            @test sum(A; dims=1) isa CuArray
            @test size(sum(A; dims=1), 1) == 1
            @test sum(A; dims=2) isa CuArray
            @test size(sum(A; dims=2), 2) == 1
        end

        # --------------------------------------------------------
        # dot
        # --------------------------------------------------------
        @testset "dot [$tag]" begin
            @test LinearAlgebra.dot(A, B) ≈ dot(vec(nzA), vec(nzB))
        end

end  # for elty, nspatdims

end  # @testset
