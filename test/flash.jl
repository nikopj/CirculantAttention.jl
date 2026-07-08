# test/flash.jl
# Fused flash attention vs the composed pipeline (similarity → softmax → ⊠).
# The two paths must agree in forward values and gradients; gradients share the
# same underlying rrules (the flash pullback is checkpointed through the
# composed path), so only forward roundoff separates them.

@testset "Flash Attention" begin
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

    simfuns = elty <: Complex ?
        (RealDotSimilarity(), DistanceSimilarity(), PIDotSimilarity(), PIDistanceSimilarity()) :
        (DotSimilarity(), RealDotSimilarity(), DistanceSimilarity(), PIDotSimilarity(), PIDistanceSimilarity())

    for simfun in simfuns
        @testset "forward $(typeof(simfun)) [$tag]" begin
            y_ref, _ = circulant_attention(simfun, q, k, v, ws)
            y_fl     = circulant_flash_attention(simfun, q, k, v, ws)
            @test eltype(y_fl) == eltype(y_ref)
            @test Array(y_fl) ≈ Array(y_ref)  rtol=1e-4 atol=1e-6
        end

        @testset "gradient $(typeof(simfun)) [$tag]" begin
            g_ref = Zygote.gradient((q, k, v) -> sum(abs2, first(circulant_attention(simfun, q, k, v, ws))), q, k, v)
            g_fl  = Zygote.gradient((q, k, v) -> sum(abs2, circulant_flash_attention(simfun, q, k, v, ws)), q, k, v)
            for (gr, gf) in zip(g_ref, g_fl)
                @test Array(gf) ≈ Array(gr)  rtol=1e-3 atol=1e-5
            end
        end

        # rrule of the fused core against FiniteDifferences (the public wrapper
        # only adds the τ-scaling broadcast, which Zygote traces)
        @testset "rrule $(typeof(simfun)) [$tag]" begin
            test_rrule(CircAtt._circulant_flash_attention, simfun,
                q ⊢ CUDA.randn(elty, size(q)...),
                k ⊢ CUDA.randn(elty, size(k)...),
                v ⊢ CUDA.randn(elty, size(v)...),
                ws;
                output_tangent=CUDA.randn(elty, size(v)...),
                rtol=1e-3, atol=1e-5, check_inferred=false)
        end
    end

    # Fill cotangent path (sum produces a FillArrays.Fill tangent on the flash
    # side; the reference gradient uses an explicit dense ones cotangent, since
    # the composed path's ⊗ pullback does not accept Fill tangents)
    @testset "Fill cotangent [$tag]" begin
        simfun = RealDotSimilarity()
        y_ref, back = Zygote.pullback((q, k, v) -> first(circulant_attention(simfun, q, k, v, ws)), q, k, v)
        g_ref = back(CUDA.fill(one(elty), size(y_ref)...))
        g_fl  = Zygote.gradient((q, k, v) -> sum(real(circulant_flash_attention(simfun, q, k, v, ws))), q, k, v)
        for (gr, gf) in zip(g_ref, g_fl)
            @test Array(gf) ≈ Array(gr)  rtol=1e-3 atol=1e-5
        end
    end

    # default simfun is DotSimilarity (real eltypes only)
    if elty <: Real
        @testset "default simfun [$tag]" begin
            y_ref, _ = circulant_attention(DotSimilarity(), q, k, v, ws)
            @test Array(circulant_flash_attention(q, k, v, ws)) ≈ Array(y_ref)  rtol=1e-4 atol=1e-6
        end
    end

    # channel counts spanning multiple register chunks, incl. partial chunks
    @testset "wide channels [$tag]" begin
        dw = 40
        qw = CUDA.randn(elty, spatdims..., dw, B)
        kw = CUDA.randn(elty, spatdims..., dw, B)
        vw = CUDA.randn(elty, spatdims..., dw, B)
        simfun = RealDotSimilarity()
        y_ref, _ = circulant_attention(simfun, qw, kw, vw, ws)
        y_fl     = circulant_flash_attention(simfun, qw, kw, vw, ws)
        @test Array(y_fl) ≈ Array(y_ref)  rtol=1e-4 atol=1e-6
    end

    # warp / block / thread kernels: independent implementations of the same
    # fused math must agree in forward, lse, and all gradients
    @testset "kernel modes agree [$tag]" begin
        simfuns_wt = elty <: Complex ?
            (RealDotSimilarity(), DistanceSimilarity(), PIDotSimilarity()) :
            (DotSimilarity(), DistanceSimilarity(), PIDistanceSimilarity())
        for simfun in simfuns_wt
            Δ = CUDA.randn(elty, size(v)...)
            y_t, lse_t = CircAtt._circulant_flash_attention_fwd(simfun, q, k, v, ws; mode=:thread)
            g_t = CircAtt.∇circulant_flash_attention(simfun, Δ, y_t, lse_t, q, k, v, ws; mode=:thread)
            for mode in (:warp, :block)
                y_m, lse_m = CircAtt._circulant_flash_attention_fwd(simfun, q, k, v, ws; mode)
                @test Array(y_m)   ≈ Array(y_t)    rtol=1e-5 atol=1e-7
                @test Array(lse_m) ≈ Array(lse_t)  rtol=1e-5 atol=1e-7

                g_m = CircAtt.∇circulant_flash_attention(simfun, Δ, y_m, lse_m, q, k, v, ws; mode)
                for (gm, gt) in zip(g_m, g_t)
                    @test Array(gm) ≈ Array(gt)  rtol=1e-4 atol=1e-6
                end
            end
        end
    end

    # joint-softmax attention: flash decomposition vs circulant_similarity +
    # joint_softmax + ⊗ (forward and gradients)
    @testset "joint attention [$tag]" begin
        simfun = DistanceSimilarity()
        Wsj = (3, 5)
        τ = sqrt(elty(d))
        joint_ref(q, k, v) = begin
            S1 = circulant_similarity(simfun, q ./ sqrt(τ), k ./ sqrt(τ), Wsj[1])
            S2 = circulant_similarity(simfun, q ./ sqrt(τ), k ./ sqrt(τ), Wsj[2])
            A1, A2 = joint_softmax(S1, S2)
            (A1 ⊗ v, A2 ⊗ v)
        end
        y1_ref, y2_ref = joint_ref(q, k, v)
        y1, y2 = circulant_flash_joint_attention(simfun, q, k, v, Wsj)
        @test Array(y1) ≈ Array(y1_ref)  rtol=1e-4 atol=1e-6
        @test Array(y2) ≈ Array(y2_ref)  rtol=1e-4 atol=1e-6

        g_ref = Zygote.gradient((q, k, v) -> begin
            ya, yb = joint_ref(q, k, v)
            sum(abs2, ya) + sum(abs2, yb)
        end, q, k, v)
        g_fl = Zygote.gradient((q, k, v) -> begin
            ya, yb = circulant_flash_joint_attention(simfun, q, k, v, Wsj)
            sum(abs2, ya) + sum(abs2, yb)
        end, q, k, v)
        for (gr, gf) in zip(g_ref, g_fl)
            @test Array(gf) ≈ Array(gr)  rtol=1e-3 atol=1e-5
        end
    end

    # rrule of the (y, lse) primitive — exercises the lse cotangent path
    @testset "rrule lse output [$tag]" begin
        simfun = RealDotSimilarity()
        nrows = N^nspatdims
        test_rrule(CircAtt._circulant_flash_attention_lse, simfun,
            q ⊢ CUDA.randn(elty, size(q)...),
            k ⊢ CUDA.randn(elty, size(k)...),
            v ⊢ CUDA.randn(elty, size(v)...),
            ws;
            output_tangent=(CUDA.randn(elty, size(v)...), CUDA.randn(real(elty), nrows, B)),
            rtol=1e-3, atol=1e-5, check_inferred=false)
    end

    # multi-head: compare against the composed path on manually split heads
    @testset "multi-head [$tag]" begin
        nheads = 2
        simfun = DistanceSimilarity()
        qr, kr, vr = CircAtt.splitheads.((q, k, v), nheads)
        y_ref, _ = circulant_attention(simfun, qr, kr, vr, ws)
        y_mh     = circulant_mh_flash_attention(simfun, q, k, v, ws, nheads)
        @test Array(y_mh) ≈ Array(reshape(y_ref, size(v)...))  rtol=1e-4 atol=1e-6
    end

    # transposed flash Γᵀx: validated against the materialized adjacency built
    # with the same 1/√C scaling the flash forward uses, and via the adjoint
    # identity ⟨Γv, x⟩ = ⟨v, Γᵀx⟩.
    for simfun in simfuns
        @testset "transposed $(typeof(simfun)) [$tag]" begin
            scale = inv(sqrt(real(elty)(d)))
            A  = circulant_adjacency(simfun, q .* sqrt(scale), k .* sqrt(scale), ws)
            x  = CUDA.randn(elty, size(q)...)
            yt_ref = CircAtt.circulant_transposed_attention(A, x)
            yt_fl  = circulant_flash_transposed_attention(simfun, q, k, x, ws)
            @test Array(yt_fl) ≈ Array(yt_ref)  rtol=1e-4 atol=1e-6

            # adjoint property against the flash forward
            v  = CUDA.randn(elty, size(q)...)
            Γv = circulant_flash_attention(simfun, q, k, v, ws)
            @test sum(conj.(v) .* yt_fl) ≈ sum(conj.(Γv) .* x)  rtol=1e-4 atol=1e-6
        end

        @testset "transposed rrule $(typeof(simfun)) [$tag]" begin
            test_rrule(CircAtt._circulant_flash_transposed_attention, simfun,
                q ⊢ CUDA.randn(elty, size(q)...),
                k ⊢ CUDA.randn(elty, size(k)...),
                CUDA.randn(elty, size(q)...) ⊢ CUDA.randn(elty, size(q)...),
                ws;
                output_tangent=CUDA.randn(elty, size(q)...),
                rtol=1e-3, atol=1e-5, check_inferred=false)
        end
    end

    # multi-head transposed against split-head reference
    @testset "multi-head transposed [$tag]" begin
        nheads = 2
        simfun = DistanceSimilarity()
        x  = CUDA.randn(elty, size(q)...)
        qr, kr, xr = CircAtt.splitheads.((q, k, x), nheads)
        yt_ref = circulant_flash_transposed_attention(simfun, qr, kr, xr, ws)
        yt_mh  = circulant_mh_flash_transposed_attention(simfun, q, k, x, ws, nheads)
        @test Array(yt_mh) ≈ Array(reshape(yt_ref, size(x)...))  rtol=1e-4 atol=1e-6
    end

    # batched guided-joint flash: 1 self + G guides (guides batched in one flash
    # call) must match the tuple-based circulant_mh_flash_joint_attention over
    # branches (self, g₁, …, g_G), in both forward and gradient.
    @testset "guided joint flash [$tag]" begin
        G = 3; nheads = 2
        simfun = DistanceSimilarity()
        Wz, Wg = ws, ws
        nsd = nspatdims
        gslice(x5, g) = x5[ntuple(_->Colon(), nsd)..., :, g, :]

        qz = CUDA.randn(elty, spatdims..., d, B)
        kz = CUDA.randn(elty, spatdims..., d, B)
        vz = CUDA.randn(elty, spatdims..., d, B)
        kg = CUDA.randn(elty, spatdims..., d, G*B)   # guide-fastest layout
        vg = CUDA.randn(elty, spatdims..., d, G*B)

        # tuple reference (built from the same kg/vg via guide slices)
        ref = (qz, kz, vz, kg, vg) -> begin
            kgr = reshape(kg, spatdims..., d, G, B)
            vgr = reshape(vg, spatdims..., d, G, B)
            qs = (qz, ntuple(_->qz, G)...)
            ks = (kz, ntuple(g->gslice(kgr, g), G)...)
            vs = (vz, ntuple(g->gslice(vgr, g), G)...)
            Ws = (Wz, ntuple(_->Wg, G)...)
            ys = circulant_mh_flash_joint_attention(simfun, qs, ks, vs, Ws, nheads)
            return ys[1], reduce(+, ys[2:end])
        end
        bat = (qz, kz, vz, kg, vg) ->
            circulant_mh_flash_guided_joint_attention(simfun, qz, kz, vz, Wz, kg, vg, Wg, G, nheads)

        ξz, ξg = bat(qz, kz, vz, kg, vg)
        rz, rg = ref(qz, kz, vz, kg, vg)
        @test Array(ξz) ≈ Array(rz)  rtol=1e-4 atol=1e-6
        @test Array(ξg) ≈ Array(rg)  rtol=1e-4 atol=1e-6

        loss(f, args...) = ((a, b) = f(args...); sum(abs2, a) + sum(abs2, b))
        gb = Zygote.gradient((args...) -> loss(bat, args...), qz, kz, vz, kg, vg)
        gr = Zygote.gradient((args...) -> loss(ref, args...), qz, kz, vz, kg, vg)
        for (b, r) in zip(gb, gr)
            @test Array(b) ≈ Array(r)  rtol=1e-3 atol=1e-5
        end
    end
end

# q/k may be complex while v is real (e.g. an energy map abs2.(·)); the
# attention methods must accept distinct q/k/v element types and agree with the
# composed path.
@testset "mixed eltypes (complex q,k; real v)" begin
    N, d, B, ws = 8, 4, 2, 5
    for nsd in TEST_SPATDIMS
        sp = ntuple(_ -> N, nsd)
        q = CUDA.randn(ComplexF32, sp..., d, B)
        k = CUDA.randn(ComplexF32, sp..., d, B)
        v = CUDA.randn(Float32,    sp..., d, B)
        for simfun in (RealDotSimilarity(), DistanceSimilarity(), PIDotSimilarity(), PIDistanceSimilarity())
            yr, _ = circulant_attention(simfun, q, k, v, ws)
            yf    = circulant_flash_attention(simfun, q, k, v, ws)
            @test eltype(yf) == eltype(yr)
            @test Array(yf) ≈ Array(yr)  rtol=1e-4 atol=1e-6

            ymr, _ = circulant_mh_attention(simfun, q, k, v, ws, 2)
            ymf    = circulant_mh_flash_attention(simfun, q, k, v, ws, 2)
            @test Array(ymf) ≈ Array(reshape(ymr, size(v)...))  rtol=1e-4 atol=1e-6

            # transposed: complex q,k applied to a real value
            xt = circulant_flash_transposed_attention(simfun, q, k, v, ws)
            @test eltype(xt) == eltype(v)
        end
    end
end

# Large 2-D windows exercise the sub-warp widths the ws=5 tests never reach:
# _flash_warp_dims picks WS=16 for ws=9 (K=81) and WS=32 for ws=13 (K=169),
# whereas every other test only hits WS=8. Validate the warp/block kernels
# against the CPU-tested thread kernel (fwd, lse, and all gradients).
@testset "sub-warp widths (large 2D windows)" begin
    N, d, B = 20, 8, 2
    for ws in (9, 13)
        q = CUDA.randn(Float32, N, N, d, B)
        k = CUDA.randn(Float32, N, N, d, B)
        v = CUDA.randn(Float32, N, N, d, B)
        Δ = CUDA.randn(Float32, N, N, d, B)
        for sim in (DotSimilarity(), DistanceSimilarity())
            WS, NE = CircAtt._flash_warp_dims(Int32(ws^2))
            @testset "$(nameof(typeof(sim))) ws=$ws (WS=$WS)" begin
                yt, lt = CircAtt._circulant_flash_attention_fwd(sim, q, k, v, ws; mode=:thread)
                gt = CircAtt.∇circulant_flash_attention(sim, Δ, yt, lt, q, k, v, ws; mode=:thread)
                for m in (:warp, :block)
                    ym, lm = CircAtt._circulant_flash_attention_fwd(sim, q, k, v, ws; mode=m)
                    @test Array(ym) ≈ Array(yt)  rtol=1e-4 atol=1e-6
                    @test Array(lm) ≈ Array(lt)  rtol=1e-4 atol=1e-6
                    gm = CircAtt.∇circulant_flash_attention(sim, Δ, ym, lm, q, k, v, ws; mode=m)
                    for (a, b) in zip(gm, gt)
                        @test Array(a) ≈ Array(b)  rtol=1e-4 atol=1e-6
                    end
                end
            end
        end
    end
end

# unsupported configurations raise informative errors
@testset "flash error paths" begin
    q = CUDA.randn(Float32, 8, 8, 4, 2)
    # TopK renormalizes over the full window and remains unfusable. (Sparsemax
    # and Entmax are now fused — see test/flash_entmax.jl.)
    @test_throws ArgumentError circulant_flash_attention(TopKSimilarity(DotSimilarity(), 3), q, q, q, 5)

    qc = CUDA.randn(ComplexF32, 8, 8, 4, 2)
    @test_throws ArgumentError circulant_flash_attention(DotSimilarity(), qc, qc, qc, 5)
end
end
