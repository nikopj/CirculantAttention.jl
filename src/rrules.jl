function CRC.rrule(::typeof(circulant_similarity), ::DistanceSimilarity, x::AbstractArray{T,N}, y::AbstractArray{T,N}, W) where {T,N}
    function dist_sim_pullback(dS)
        d = size(x, N-1)
        X = reshape(x, prod(size(x)[1:N-2]), d, :)
        Y = reshape(y, prod(size(y)[1:N-2]), d, :)
        ΔS  = reshape(CRC.unthunk(dS), :, :, :)
        ΔSᵀ = batched_transpose(ΔS)
        ∂x = CRC.@thunk reshape((ΔS ⊠ Y)  - (sum(ΔS; dims=2) .* X), size(x)...)
        ∂y = CRC.@thunk reshape((ΔSᵀ ⊠ X) - (reshape(sum(ΔS; dims=1), :, 1, size(ΔS,3)) .* Y), size(y)...)
        return (CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y, CRC.NoTangent())
    end
    S = circulant_similarity(DistanceSimilarity(), x, y, W)
    return S, dist_sim_pullback
end

function CRC.rrule(::typeof(circulant_similarity), ::DotSimilarity, x::AbstractArray{T,N}, y::AbstractArray{T,N}, W) where {T,N}
    function dot_sim_pullback(dS)
        d = size(x, N-1)
        X = reshape(x, prod(size(x)[1:N-2]), d, :)
        Y = reshape(y, prod(size(y)[1:N-2]), d, :)
        ΔS  = reshape(CRC.unthunk(dS), :, :, :)
        ∂x = CRC.@thunk reshape(ΔS ⊠ Y, size(x)...)
        ∂y = CRC.@thunk reshape(batched_transpose(ΔS) ⊠ X, size(y)...)
        return (CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y, CRC.NoTangent())
    end
    S = circulant_similarity(DotSimilarity(), x, y, W)
    return S, dot_sim_pullback
end

function CRC.rrule(::typeof(NNlib.softmax), X::Circulant{T, N, M}) where {T, N, M}
    Y = NNlib.softmax(X)
    function softmax_pullback(∂Y) 
        ΔY = CRC.unthunk(∂Y)
        dUr = reshape(ΔY.data.nzVal, :, size(ΔY,1), prod(size(ΔY)[3:end]))
        Ur  = reshape(Y.data.nzVal, :, size(X,1), prod(size(X)[3:end]))
        dVr = NNlib.∇softmax_data(dUr, Ur; dims=1)
        dV = reshape(dVr, size(ΔY.data.nzVal)...)
        data = CuSparseArrayCSR(X.data.rowPtr, X.data.colVal, dV, size(X))
        ΔZ = Circulant(data, kernel_length(X), spatial_size(X))
        return (CRC.NoTangent(), ΔZ)
    end
    return Y, softmax_pullback
end

function CRC.rrule(::typeof(circulant_attention), A::Circulant{T, N, M}, b::AbstractArray{Tb,Nb}) where {T, N, M, Tb, Nb}
    function circulant_attention_pullback(dc)
        Δc = CRC.unthunk(dc)
        ∂A = CRC.@thunk reshape(circulant_similarity(DotSimilarity(), Δc, b, M), size(A)...)
        ∂b = CRC.@thunk begin
            ΔC = reshape(Δc, :, size(Δc)[Nb-1:end]...)
            ∂B = NNlib.batched_transpose(reshape(A, :,:,:)) ⊠ ΔC
            reshape(∂B, size(b)...)
        end
        return (CRC.NoTangent(), ∂A, ∂b)
    end
    return A ⊗ b, circulant_attention_pullback
end
