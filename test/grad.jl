# test/grad.jl
#
# End-to-end gradient tests through full DNN forward passes.
# Verifies that gradients are non-nothing and finite for all parameters.
# These tests exercise the full AD chain: conv → circulant_similarity →
# arithmetic (scalar *, CuArray .*, +) → circulant_attention → conv.

const _nheads    = 2
const _windowsize = 3

# Single-head attention DNN
function dnn(x, Θ, simfun)
    wx, wk, wq, wv, wz, α = Θ
    xx = NNlib.conv(x, wx)   # (N, N, 8, B)
    k  = NNlib.conv(x, wk)
    q  = NNlib.conv(x, wq)
    v  = NNlib.conv(x, wv)

    A = circulant_adjacency(simfun, k, q, _windowsize)
    B = circulant_adjacency(simfun, xx, xx, _windowsize)
    # α is CuArray (1,1,1,1): exercises CuArray .* Circulant broadcast on both sides
    C = α .* A + (1f0 .- α) .* B

    z = C ⊗ v
    return NNlib.conv(z, wz)
end

# Multi-head attention DNN
function dnn_mh(x, Θ, simfun)
    wx, wk, wq, wv, wz, α = Θ
    xx = NNlib.conv(x, wx)
    k  = NNlib.conv(x, wk)
    q  = NNlib.conv(x, wq)
    v  = NNlib.conv(x, wv)

    A = circulant_mh_adjacency(simfun, k, q, _windowsize, _nheads)
    B = circulant_mh_adjacency(simfun, xx, xx, _windowsize, _nheads)
    # α is CuArray (1,1,nheads,1): exercises batch-expanding CuArray .* Circulant
    C = α .* A + (1f0 .- α) .* B

    z = C ⨷ v
    return NNlib.conv(z, wz)
end

function make_weights(elty, N, ks)
    (
        CUDA.randn(elty, ks..., 4, 8),   # wx
        CUDA.randn(elty, ks..., 4, 8),   # wk
        CUDA.randn(elty, ks..., 4, 8),   # wq
        CUDA.randn(elty, ks..., 4, 8),   # wv
        CUDA.randn(elty, ks..., 8, 4),   # wz
    )
end

@testset "DNN-Style Gradients" begin 
for elty in TEST_ELTYPES, nspatdims in TEST_SPATDIMS, 
    simfun in (RealDotSimilarity(), DistanceSimilarity(), PIDotSimilarity(), PIDistanceSimilarity())

    tag = "elty=$elty, nspatdims=$nspatdims, simfun=$simfun"

    x = CUDA.randn(elty, ntuple(_->12, nspatdims)..., 4, 2)
    y = CUDA.randn(elty, ntuple(_->8,  nspatdims)..., 4, 2)
    ks = ntuple(_->3, nspatdims)
    weights = make_weights(elty, nspatdims, ks)

    # ------------------------------------------------------------------
    # Single-head attention
    # α is (1,1,1,1) — scalar expansion over all dims
    # ------------------------------------------------------------------
    @testset "single-head attention [$tag]" begin
        α  = 0.8f0 * CUDA.ones(real(elty), 1, 1, 1, 1)
        ps = (weights..., α)

        z = dnn(x, ps, simfun)
        @test size(z) == (ntuple(_->8+2-3+1, nspatdims)..., 4, 2)

        val, gs = Zygote.withgradient(ps) do Θ
            sum(abs2, y - dnn(x, Θ, simfun)) / length(y)
        end
        gs = gs[1]

        @test isfinite(val)
        @test !any(isnothing.(gs))
        for (i, g) in enumerate(gs)
            @test all(isfinite.(Array(g))) # "weight $i has non-finite gradients"
        end
    end

    # ------------------------------------------------------------------
    # Multi-head attention
    # α is (1,1,nheads,1) — batch-expanding along channel/head dim
    # ------------------------------------------------------------------
    @testset "multi-head attention [$tag]" begin
        α  = 0.8f0 * CUDA.ones(real(elty), 1, 1, _nheads, 1)
        ps = (weights..., α)

        z = dnn_mh(x, ps, simfun)
        @test size(z) == (ntuple(_->8+2-3+1, nspatdims)..., 4, 2)

        val, gs = Zygote.withgradient(ps) do Θ
            sum(abs2, y - dnn_mh(x, Θ, simfun)) / length(y)
        end
        gs = gs[1]

        @test isfinite(val)
        @test !any(isnothing.(gs))
        for (i, g) in enumerate(gs)
            @test all(isfinite.(Array(g))) # "weight $i has non-finite gradients"
        end
    end

    # ------------------------------------------------------------------
    # Isolated adjacency gradient — verifies the similarity→attention
    # gradient chain without the outer conv layers
    # ------------------------------------------------------------------
    @testset "adjacency gradient (single-head) [$tag]" begin
        z1 = CUDA.randn(elty, ntuple(_->12, nspatdims)..., 4, 2)
        z2 = CUDA.randn(elty, ntuple(_->12, nspatdims)..., 4, 2)
        wx = first(weights)

        val, gs = Zygote.withgradient(wx) do w
            x = NNlib.conv(z1, w)
            y = NNlib.conv(z2, w)
            A = circulant_adjacency(simfun, x, y, _windowsize)
            sum(abs2, (A ⊗ x))
        end

        @test isfinite(val)
        @test !isnothing(gs[1])
        @test all(isfinite.(Array(gs[1])))
    end

    @testset "adjacency gradient (multi-head, CuArray batch scale) [$tag]" begin
        z1 = CUDA.randn(elty, ntuple(_->12, nspatdims)..., 4, 2)
        z2 = CUDA.randn(elty, ntuple(_->12, nspatdims)..., 4, 2)
        wx = first(weights)
        θ  = 0.8f0 * CUDA.ones(real(elty), 1, 1, _nheads, 1)

        val, gs = Zygote.withgradient(wx, θ) do w, scale
            x = NNlib.conv(z1, w)
            y = NNlib.conv(z2, w)
            A = circulant_mh_adjacency(simfun, x, y, _windowsize, _nheads)
            # scale is (1,1,nheads,1): batch-expanding CuArray .* Circulant
            sum(abs2, (scale .* A) ⨷ x)
        end

        @test isfinite(val)
        @test !isnothing(gs[1]) && all(isfinite.(Array(gs[1])))
        @test !isnothing(gs[2]) && all(isfinite.(Array(gs[2])))
    end

    d, batch = 4, 2
    x = CUDA.randn(elty, ntuple(_->8, nspatdims)..., d, batch)
    y = CUDA.randn(elty, ntuple(_->8, nspatdims)..., d, batch)
    z = CUDA.randn(elty, ntuple(_->8, nspatdims)..., d, batch)

    W1 = 3
    W2 = 5

    function loss(x, y, z)
        A1 = circulant_similarity(simfun, x, y, W1)
        A2 = circulant_similarity(simfun, x, z, W2)
        S1, S2 = joint_softmax(real(A1), real(A2))
        # Weighted aggregation with both attention maps
        sum(abs2, (S1 ⊗ y) .+ (S2 ⊗ z))
    end

    val, gs = Zygote.withgradient(loss, x, y, z)

    @testset "joint_softmax loss [$tag]" begin
        @test isfinite(val)
        for (i, g) in enumerate(gs)
            @test !isnothing(g) 
            @test all(isfinite.(Array(g))) 
        end
    end

end
end
