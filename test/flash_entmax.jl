# test/flash_entmax.jl
# Fused flash α-entmax / sparsemax attention (AdaSplash-based) vs the composed
# pipeline (circulant_adjacency ∘ EntmaxSimilarity → ⊗). The two paths must agree
# in forward values and gradients; the three kernel variants (thread/warp/block)
# must agree with each other.

# Reference: entmax adjacency with the same 1/√C similarity scaling the flash
# forward folds in (q,k scaled by 1/√τ, τ=√C, as circulant_attention does).
function _entmax_ref(sim, α, q, k, v, ws)
    C = size(q, ndims(q) - 1)
    τ = sqrt(real(eltype(k))(C))
    A = circulant_adjacency(EntmaxSimilarity(sim, Float32(α)), q ./ sqrt(τ), k ./ sqrt(τ), ws)
    return A ⊗ v
end
function _sparsemax_ref(sim, q, k, v, ws)
    C = size(q, ndims(q) - 1)
    τ = sqrt(real(eltype(k))(C))
    A = circulant_adjacency(SparsemaxSimilarity(sim), q ./ sqrt(τ), k ./ sqrt(τ), ws)
    return A ⊗ v
end

# Composed JOINT reference: per-branch scaled similarities, one joint entmax /
# sparsemax over the concatenated windows, then apply per branch (returns a tuple).
function _joint_entmax_ref(sim, α, q, k, v, Ws)
    C = size(q, ndims(q) - 1); τ = sqrt(real(eltype(k))(C))
    Ss = map(W -> circulant_similarity(sim, q ./ sqrt(τ), k ./ sqrt(τ), W), Ws)
    As = joint_entmax(Float32(α), Ss...)
    return map(A -> A ⊗ v, As)
end
function _joint_sparsemax_ref(sim, q, k, v, Ws)
    C = size(q, ndims(q) - 1); τ = sqrt(real(eltype(k))(C))
    Ss = map(W -> circulant_similarity(sim, q ./ sqrt(τ), k ./ sqrt(τ), W), Ws)
    As = joint_sparsemax(Ss...)
    return map(A -> A ⊗ v, As)
end

@testset "Flash Entmax Attention" begin
for elty in TEST_ELTYPES, nspatdims in TEST_SPATDIMS
    N  = 8
    d  = 4
    B  = 2
    ws = 5
    spatdims = ntuple(_ -> N, nspatdims)
    tag = "elty=$elty, nspatdims=$nspatdims"

    q = CUDA.randn(elty, spatdims..., d, B)
    k = CUDA.randn(elty, spatdims..., d, B)
    v = CUDA.randn(elty, spatdims..., d, B)

    # inner similarities that yield real-valued scores (entmax needs real scores)
    simfuns = elty <: Complex ?
        (RealDotSimilarity(), DistanceSimilarity()) :
        (DotSimilarity(), RealDotSimilarity(), DistanceSimilarity())

    for sim in simfuns, α in (1.5f0, 2.0f0)
        es = EntmaxSimilarity(sim, α)

        @testset "forward $(typeof(sim)) α=$α [$tag]" begin
            y_ref = _entmax_ref(sim, α, q, k, v, ws)
            y_fl  = circulant_flash_attention(es, q, k, v, ws)
            @test eltype(y_fl) == eltype(y_ref)
            @test Array(y_fl) ≈ Array(y_ref)  rtol=1e-3 atol=1e-5
        end

        # α-entmax gradients are discontinuous at the support boundary: the flash
        # Halley τ and the composed path's bisection τ differ by ~1e-7, so for a
        # window entry straddling the boundary the two paths make opposite
        # in/out-of-support choices and that entry's gradient legitimately differs
        # (the entmax analog of the sparsemax subgradient jump). This is isolated
        # to boundary entries; a real bug perturbs the whole array. Tolerance is
        # loosened accordingly (forward, which is continuous, stays tight above).
        @testset "gradient $(typeof(sim)) α=$α [$tag]" begin
            g_ref = Zygote.gradient((q, k, v) -> sum(abs2, _entmax_ref(sim, α, q, k, v, ws)), q, k, v)
            g_fl  = Zygote.gradient((q, k, v) -> sum(abs2, circulant_flash_attention(es, q, k, v, ws)), q, k, v)
            for (gr, gf) in zip(g_ref, g_fl)
                @test Array(gf) ≈ Array(gr)  rtol=1e-2 atol=1e-3
            end
        end

        # thread / warp / block: independent implementations of the same fused
        # math must agree in forward (y, τ, ỹ) and all gradients
        @testset "kernel modes agree $(typeof(sim)) α=$α [$tag]" begin
            Δ = CUDA.randn(elty, size(v)...)
            scale = inv(sqrt(real(elty)(d)))
            y_t, τ_t, yt_t = CircAtt._circulant_flash_entmax_fwd(sim, α, q, k, v, ws, scale; mode=:thread)
            g_t = CircAtt.∇circulant_flash_entmax(sim, α, Δ, y_t, τ_t, yt_t, q, k, v, ws, scale; mode=:thread)
            # the three modes differ only in reduction order; the Halley solver
            # can amplify that to a few ×1e-4 in τ (hence y), so a tight-but-not-
            # machine tolerance still catches any real divergence.
            for mode in (:warp, :block)
                y_m, τ_m, yt_m = CircAtt._circulant_flash_entmax_fwd(sim, α, q, k, v, ws, scale; mode)
                @test Array(y_m)  ≈ Array(y_t)   rtol=1e-3 atol=1e-5
                @test Array(τ_m)  ≈ Array(τ_t)   rtol=1e-3 atol=1e-5
                @test Array(yt_m) ≈ Array(yt_t)  rtol=1e-3 atol=1e-5

                g_m = CircAtt.∇circulant_flash_entmax(sim, α, Δ, y_m, τ_m, yt_m, q, k, v, ws, scale; mode)
                for (gm, gt) in zip(g_m, g_t)
                    @test Array(gm) ≈ Array(gt)  rtol=1e-3 atol=1e-5
                end
            end
        end
    end

    # sparsemax path == α=2 entmax path == non-flash sparsemax adjacency
    @testset "sparsemax path [$tag]" begin
        sim = DistanceSimilarity()
        y_ref  = _sparsemax_ref(sim, q, k, v, ws)
        y_flss = circulant_flash_attention(SparsemaxSimilarity(sim), q, k, v, ws)
        y_fle2 = circulant_flash_attention(EntmaxSimilarity(sim, 2.0f0), q, k, v, ws)
        @test Array(y_flss) ≈ Array(y_ref)   rtol=1e-3 atol=1e-5
        @test Array(y_flss) ≈ Array(y_fle2)  rtol=1e-5 atol=1e-7

        # sparsemax gradient is likewise subgradient-discontinuous at the support
        # boundary (see the entmax gradient note above) — loosened tolerance.
        g_ref = Zygote.gradient((q, k, v) -> sum(abs2, _sparsemax_ref(sim, q, k, v, ws)), q, k, v)
        g_fl  = Zygote.gradient((q, k, v) -> sum(abs2, circulant_flash_attention(SparsemaxSimilarity(sim), q, k, v, ws)), q, k, v)
        for (gr, gf) in zip(g_ref, g_fl)
            @test Array(gf) ≈ Array(gr)  rtol=1e-2 atol=1e-3
        end
    end

    # α ≈ 1 dispatches to (and equals) the softmax flash
    @testset "alpha≈1 == softmax flash [$tag]" begin
        sim = elty <: Complex ? RealDotSimilarity() : DotSimilarity()
        y_soft = circulant_flash_attention(sim, q, k, v, ws)
        y_ent  = circulant_flash_attention(EntmaxSimilarity(sim, 1.0f0), q, k, v, ws)
        @test Array(y_ent) ≈ Array(y_soft)  rtol=1e-5 atol=1e-7
    end

    # multi-head vs the composed path on manually split heads
    @testset "multi-head [$tag]" begin
        nheads = 2
        sim = DistanceSimilarity()
        qr, kr, vr = CircAtt.splitheads.((q, k, v), nheads)
        y_ref = _entmax_ref(sim, 1.5f0, qr, kr, vr, ws)
        y_mh  = circulant_mh_flash_attention(EntmaxSimilarity(sim, 1.5f0), q, k, v, ws, nheads)
        @test Array(y_mh) ≈ Array(reshape(y_ref, size(v)...))  rtol=1e-3 atol=1e-5
    end

    # ----- JOINT (multi-branch) entmax / sparsemax over windows (3, 5) -----
    Wsj = (3, 5)
    @testset "joint forward/gradient [$tag]" begin
        sim = elty <: Complex ? DistanceSimilarity() : DotSimilarity()
        cases = ((EntmaxSimilarity(sim, 1.5f0), (q,k,v) -> _joint_entmax_ref(sim, 1.5f0, q, k, v, Wsj)),
                 (SparsemaxSimilarity(sim),     (q,k,v) -> _joint_sparsemax_ref(sim, q, k, v, Wsj)))
        for (mk, ref) in cases
            y1r, y2r = ref(q, k, v)
            y1, y2 = circulant_flash_joint_attention(mk, q, k, v, Wsj)
            @test Array(y1) ≈ Array(y1r)  rtol=1e-3 atol=1e-5
            @test Array(y2) ≈ Array(y2r)  rtol=1e-3 atol=1e-5

            loss(f) = ((ya, yb) = f; sum(abs2, ya) + sum(abs2, yb))
            g_ref = Zygote.gradient((q,k,v) -> loss(ref(q,k,v)), q, k, v)
            g_fl  = Zygote.gradient((q,k,v) -> loss(circulant_flash_joint_attention(mk, q, k, v, Wsj)), q, k, v)
            for (gr, gf) in zip(g_ref, g_fl)   # entmax boundary subgradient → loose tol
                @test Array(gf) ≈ Array(gr)  rtol=1e-2 atol=1e-3
            end
        end
    end

    # thread / warp / block agree for the joint forward (ys, τ, ỹs) and gradients
    @testset "joint modes agree [$tag]" begin
        sim = DistanceSimilarity(); α = 1.5f0; M = length(Wsj)
        sims = ntuple(_ -> sim, M); qs = ntuple(_ -> q, M); ks = ntuple(_ -> k, M); vs = ntuple(_ -> v, M)
        scales = ntuple(_ -> inv(sqrt(real(elty)(d))), M)
        yt, τt, ytt = CircAtt._circulant_flash_joint_entmax_fwd(sims, α, qs, ks, vs, Wsj, scales; mode=:thread)
        Δs = ntuple(_ -> CUDA.randn(elty, size(v)...), M)
        gt = CircAtt.∇circulant_flash_joint_entmax(sims, α, Δs, yt, τt, ytt, qs, ks, vs, Wsj, scales; mode=:thread)
        for mode in (:warp, :block)
            ym, τm, ytm = CircAtt._circulant_flash_joint_entmax_fwd(sims, α, qs, ks, vs, Wsj, scales; mode)
            @test Array(τm) ≈ Array(τt)  rtol=1e-3 atol=1e-5
            for m in 1:M
                @test Array(ym[m])  ≈ Array(yt[m])   rtol=1e-3 atol=1e-5
                @test Array(ytm[m]) ≈ Array(ytt[m])  rtol=1e-3 atol=1e-5
            end
            gm = CircAtt.∇circulant_flash_joint_entmax(sims, α, Δs, ym, τm, ytm, qs, ks, vs, Wsj, scales; mode)
            for (gmt, gtt) in zip(gm, gt), (a, b) in zip(gmt, gtt)
                @test Array(a) ≈ Array(b)  rtol=1e-2 atol=1e-3
            end
        end
    end

    # guided joint = mh joint over (self, guide slices), guide outputs summed
    @testset "guided joint [$tag]" begin
        G = 2; nheads = 2; sim = DistanceSimilarity(); α = 1.5f0; Wz = Wg = ws
        qz = CUDA.randn(elty, spatdims..., d, B)
        kz = CUDA.randn(elty, spatdims..., d, B)
        vz = CUDA.randn(elty, spatdims..., d, B)
        kg = CUDA.randn(elty, spatdims..., d, G*B)   # guide-fastest batch
        vg = CUDA.randn(elty, spatdims..., d, G*B)
        gsl(x5, g) = x5[ntuple(_ -> Colon(), nspatdims)..., :, g, :]
        kgr = reshape(kg, spatdims..., d, G, B); vgr = reshape(vg, spatdims..., d, G, B)
        sfs = ntuple(_ -> EntmaxSimilarity(sim, α), G+1)
        qs = (qz, ntuple(_ -> qz, G)...); ks = (kz, ntuple(g -> gsl(kgr, g), G)...)
        vs = (vz, ntuple(g -> gsl(vgr, g), G)...); Ws = (Wz, ntuple(_ -> Wg, G)...)
        ysr = circulant_mh_flash_joint_attention(sfs, qs, ks, vs, Ws, nheads)
        rz, rg = ysr[1], reduce(+, ysr[2:end])
        ξz, ξg = circulant_mh_flash_guided_joint_attention(EntmaxSimilarity(sim, α), qz, kz, vz, Wz, kg, vg, Wg, G, nheads)
        @test Array(ξz) ≈ Array(rz)  rtol=1e-4 atol=1e-6
        @test Array(ξg) ≈ Array(rg)  rtol=1e-4 atol=1e-6
    end

    # transposed Γᵀx (used by the multigrid ∂g): flash entmax/sparsemax vs the
    # materialized entmax-adjacency transpose, the adjoint identity against the
    # flash forward, and gradients vs the composed path.
    @testset "transposed [$tag]" begin
        sim = elty <: Complex ? DistanceSimilarity() : DotSimilarity()
        scale = inv(sqrt(real(elty)(d)))
        x = CUDA.randn(elty, size(q)...)
        entref(mk, q, k, x) = CircAtt.circulant_transposed_attention(
            circulant_adjacency(mk, q .* sqrt(scale), k .* sqrt(scale), ws), x)
        for mk in (EntmaxSimilarity(sim, 1.5f0), SparsemaxSimilarity(sim))
            yt_ref = entref(mk, q, k, x)
            yt_fl  = circulant_flash_transposed_attention(mk, q, k, x, ws)
            @test Array(yt_fl) ≈ Array(yt_ref)  rtol=1e-3 atol=1e-5

            # adjoint identity ⟨Γv, x⟩ = ⟨v, Γᵀx⟩ against the flash forward
            u  = CUDA.randn(elty, size(q)...)
            Γu = circulant_flash_attention(mk, q, k, u, ws)
            @test sum(conj.(u) .* yt_fl) ≈ sum(conj.(Γu) .* x)  rtol=1e-3 atol=1e-5

            # gradient vs composed (entmax boundary subgradient → loose tol)
            g_ref = Zygote.gradient((q,k,x) -> sum(abs2, entref(mk, q, k, x)), q, k, x)
            g_fl  = Zygote.gradient((q,k,x) -> sum(abs2, circulant_flash_transposed_attention(mk, q, k, x, ws)), q, k, x)
            for (gr, gf) in zip(g_ref, g_fl)
                @test Array(gf) ≈ Array(gr)  rtol=1e-2 atol=1e-3
            end
        end
    end
end

# complex inner similarity (complex scores) is rejected — softmax over complex is
# undefined, and so is entmax's threshold.
@testset "flash entmax error paths" begin
    qc = CUDA.randn(ComplexF32, 8, 8, 4, 2)
    @test_throws ArgumentError circulant_flash_attention(EntmaxSimilarity(DotSimilarity(), 1.5f0), qc, qc, qc, 5)
    @test_throws ArgumentError circulant_flash_attention(SparsemaxSimilarity(DotSimilarity()), qc, qc, qc, 5)
end
end
