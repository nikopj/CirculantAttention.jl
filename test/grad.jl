using Zygote
using NNlib

@testset "grad" begin
    nheads = 2
    windowsize = 3

    function dnn(x, Θ, simfun)
        wx, wk, wq, wv, wz, α = Θ
        xx= NNlib.conv(x, wx) # (10, 10, 8, 2)
        k = NNlib.conv(x, wk) # (10, 10, 8, 2)
        q = NNlib.conv(x, wq) # (10, 10, 8, 2)
        v = NNlib.conv(x, wv) # (10, 10, 8, 2)

        # (100, 100, 1, 2)
        A = circulant_adjacency(simfun, k, q, windowsize) 
        B = circulant_adjacency(simfun, xx, xx, windowsize) 
        C = α*A + (1f0-α)*B

        z = C ⊗ v             # (10, 10, 8, 2)
        z = NNlib.conv(z, wz) # (8, 8, 4, 2)
        return z
    end

    function dnn_mh(x, Θ, simfun)
        wx, wk, wq, wv, wz, α = Θ
        xx= NNlib.conv(x, wx) # (10, 10, 8, 2)
        k = NNlib.conv(x, wk) # (10, 10, 8, 2)
        q = NNlib.conv(x, wq) # (10, 10, 8, 2)
        v = NNlib.conv(x, wv) # (10, 10, 8, 2)

        # (100, 100, nheads, 2)
        A = circulant_mh_adjacency(simfun, k, q, windowsize, nheads) 
        B = circulant_mh_adjacency(simfun, xx, xx, windowsize, nheads) 
        C = α*A + (1f0-α)*B

        z = C ⨷ v             # (10, 10, 8, 2)
        z = NNlib.conv(z, wz) # (8, 8, 4, 2)
        return z
    end

    @testset "nspatdims=$N" for N=1:2
        @testset "simfun=$simfun" for simfun=(DotSimilarity(), DistanceSimilarity())
            x = CUDA.randn(ntuple(_->12,N)..., 4, 2)
            y = CUDA.randn(ntuple(_->8,N)..., 4, 2)
            ks = ntuple(_->3,N)
            Wx = CUDA.randn(ks..., 4, 8)
            Wk = CUDA.randn(ks..., 4, 8)
            Wq = CUDA.randn(ks..., 4, 8)
            Wv = CUDA.randn(ks..., 4, 8)
            Wz = CUDA.randn(ks..., 8, 4)
            ps = (Wx, Wk, Wq, Wv, Wz, 0.5f0)

            @testset "attention" begin
                z = dnn(x, ps, simfun)
                @test size(z) == (ntuple(_->8,N)..., 4, 2)

                val, gs = withgradient(ps) do Θ
                    z = dnn(x, Θ, simfun)
                    sum(abs2, y - z) / length(y)
                end

                @test !any(isnothing.(gs))
            end

            @testset "multhead attention" begin
                z = dnn_mh(x, ps, simfun)
                @test size(z) == (ntuple(_->8,N)..., 4, 2)

                val, gs = withgradient(ps) do Θ
                    z = dnn_mh(x, Θ, simfun)
                    sum(abs2, y - z) / length(y)
                end

                @test !any(isnothing.(gs))
            end
        end
    end
end
