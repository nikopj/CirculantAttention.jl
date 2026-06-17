# test/correctness.jl
#
# Verifies the row/column convention of circulant attention:
#
#   S[i, j] = simfun(q_i, k_j)        (i = row, j = column)
#
# i.e. ROW i holds the comparison of QUERY pixel q_i against every KEY pixel
# k_j inside its sliding window, and y = A·v with A = rowsoftmax(S) so that
# y_i = Σ_j A[i,j] v_j. Orientation is proven by recomputing the similarities
# on the CPU from q and k, and by checking the SWAPPED convention
# S[i,j] = sim(q_j, k_i) does NOT match (q, k are distinct, so the matrix is
# non-symmetric and the orientation is observable).

# CPU reference similarity functions (mirror src/similarity.jl).
ref_sim(::DotSimilarity, qi, kj)      = sum(qi .* conj.(kj))
ref_sim(::RealDotSimilarity, qi, kj)  = real(sum(qi .* conj.(kj)))
ref_sim(::DistanceSimilarity, qi, kj) = -sum(abs2, qi .- kj) / 2

# Densify a Circulant's stored (i, j, value) triples on the CPU using its own
# sparsity pattern (windowview + colVal) — no reimplementation of the circulant
# boundary indexing. Returns (Wv, Cv) :: (Krow, Nrows, B) of stored values and
# their matching column indices.
function stored_entries(S::Circulant)
    Wv = Array(windowview(S))                           # (Krow, Nrows, B)
    Krow, Nrows, B = size(Wv, 1), size(Wv, 2), size(Wv, 3)
    Cv = Array(reshape(S.data.colVal, Krow, Nrows, B))  # matching column idx
    return Wv, Cv
end

function similarity_orientation_errs(simfun, q, k, W)
    N     = ndims(q)
    C, B  = size(q, N-1), size(q, N)
    Nrows = prod(size(q)[1:N-2])

    S = circulant_similarity(simfun, q, k, W)
    Wv, Cv = stored_entries(S)
    Krow = size(Wv, 1)

    qr = reshape(Array(q), Nrows, C, B)
    kr = reshape(Array(k), Nrows, C, B)

    err_correct = 0.0   # S[i,j] vs sim(q_i, k_j)  — expected to match
    err_swapped = 0.0   # S[i,j] vs sim(q_j, k_i)  — expected to DIFFER
    for b in 1:B, i in 1:Nrows, t in 1:Krow
        j   = Int(Cv[t, i, b])
        got = Wv[t, i, b]
        err_correct = max(err_correct, abs(got - ref_sim(simfun, qr[i,:,b], kr[j,:,b])))
        err_swapped = max(err_swapped, abs(got - ref_sim(simfun, qr[j,:,b], kr[i,:,b])))
    end
    return err_correct, err_swapped
end

function attention_output_errs(simfun, q, k, v, W)
    Tv, N = eltype(q), ndims(q)
    C, B  = size(q, N-1), size(q, N)
    Nrows = prod(size(q)[1:N-2])

    y, A = circulant_attention(simfun, q, k, v, W)

    # circulant_attention scales q and k by 1/sqrt(τ), τ = sqrt(C), each.
    τ  = sqrt(Tv(C))
    sc = 1 / sqrt(τ)
    qr = reshape(Array(q), Nrows, C, B) .* sc
    kr = reshape(Array(k), Nrows, C, B) .* sc
    vr = reshape(Array(v), Nrows, C, B)

    # Reconstruct A = rowsoftmax(S) and y = A·v on the CPU from A's own pattern.
    _, Cv = stored_entries(A)
    A_ref = Array(windowview(A))                    # (Krow, Nrows, B)
    y_ref = zeros(Tv, Nrows, C, B)
    A_err = 0.0
    for b in 1:B, i in 1:Nrows
        cols   = Int.(Cv[:, i, b])
        logits = [ref_sim(simfun, qr[i,:,b], kr[j,:,b]) for j in cols]
        wts    = NNlib.softmax(logits)              # row-softmax over the window
        A_err  = max(A_err, maximum(abs.(A_ref[:, i, b] .- wts)))
        for (t, j) in enumerate(cols)
            y_ref[i, :, b] .+= wts[t] .* vr[j, :, b]
        end
    end
    y_err = maximum(abs.(Array(y) .- y_ref))
    return A_err, y_err
end

@testset "Attention orientation: S[i,j] = sim(q_i, k_j)" begin
for elty in TEST_ELTYPES, nspatdims in TEST_SPATDIMS
    C, B = 8, 2
    W    = 3
    spatdims = ntuple(_ -> 6, nspatdims)
    sims = (DotSimilarity(), RealDotSimilarity(), DistanceSimilarity())

    for simfun in sims
        tag = "$(nameof(typeof(simfun))) [elty=$elty, nspatdims=$nspatdims, W=$W]"

        # distinct q, k so the matrix is non-symmetric and orientation shows.
        q = CUDA.randn(elty, spatdims..., C, B)
        k = CUDA.randn(elty, spatdims..., C, B)
        v = CUDA.randn(elty, spatdims..., C, B)

        @testset "similarity convention [$tag]" begin
            ec, es = similarity_orientation_errs(simfun, q, k, W)
            @test ec < 1e-3      # correct convention matches
            @test es > 1e-2      # swapped convention is clearly wrong
        end

        @testset "attention output y = rowsoftmax(S)·v [$tag]" begin
            ae, ye = attention_output_errs(simfun, q, k, v, W)
            @test ae < 1e-3
            @test ye < 1e-3
        end
    end
end
end
