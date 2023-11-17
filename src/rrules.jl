function CRC.rrule(::typeof(circulant_similarity), ::DistanceSimilarity, x::AbstractArray{T,N}, y::AbstractArray{T,N}, W) where {T,N}
    project_x = CRC.ProjectTo(x)
    project_y = CRC.ProjectTo(y)
    function dist_sim_pullback(dS)
        d = size(x, N-1)
        X = reshape(x, prod(size(x)[1:N-2]), d, :)
        Y = reshape(y, prod(size(y)[1:N-2]), d, :)
        ΔS  = reshape(CRC.unthunk(dS), :, :, :)
        ΔSᵀ = batched_transpose(ΔS)
        ∂x = CRC.@thunk project_x(reshape((ΔS ⊠ Y)  - (sum(ΔS; dims=2) .* X), size(x)...))
        ∂y = CRC.@thunk project_y(reshape((ΔSᵀ ⊠ X) - (reshape(sum(ΔS; dims=1), :, 1, size(ΔS,3)) .* Y), size(y)...))
        return (CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y, CRC.NoTangent())
    end
    S = circulant_similarity(DistanceSimilarity(), x, y, W)
    return S, dist_sim_pullback
end

function CRC.rrule(::typeof(circulant_similarity), ::DotSimilarity, x::AbstractArray{T,N}, y::AbstractArray{T,N}, W) where {T,N}
    project_x = CRC.ProjectTo(x)
    project_y = CRC.ProjectTo(y)
    function dot_sim_pullback(dS)
        d = size(x, N-1)
        X = reshape(x, prod(size(x)[1:N-2]), d, :)
        Y = reshape(y, prod(size(y)[1:N-2]), d, :)
        ΔS  = reshape(CRC.unthunk(dS), :, :, :)
        ∂x = CRC.@thunk project_x(reshape(ΔS ⊠ Y, size(x)...))
        ∂y = CRC.@thunk project_y(reshape(batched_transpose(ΔS) ⊠ X, size(y)...))
        return (CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y, CRC.NoTangent())
    end
    S = circulant_similarity(DotSimilarity(), x, y, W)
    return S, dot_sim_pullback
end

function CRC.rrule(::typeof(_circulant_reshape), X::Circulant{T,N,M}, dims...) where {T,N,M}
    project_X = CRC.ProjectTo(X)
    function reshape_pullback(dX) 
        return (CRC.NoTangent(), project_X(reshape(dX, size(X)...)), map(_->CRC.NoTangent(), dims)...)
    end
    return reshape(X, dims...), reshape_pullback
end

function CRC.rrule(::typeof(NNlib.softmax), X::Circulant{T, N, M}) where {T, N, M}
    Y = NNlib.softmax(X)
    project_X = CRC.ProjectTo(X)
    function softmax_pullback(∂Y) 
        ΔY = CRC.unthunk(∂Y)
        dUr = reshape(ΔY.data.nzVal, :, size(ΔY,1), prod(size(ΔY)[3:end]))
        Ur  = reshape(Y.data.nzVal, :, size(X,1), prod(size(X)[3:end]))
        dVr = NNlib.∇softmax_data(dUr, Ur; dims=1)
        dV = reshape(dVr, size(ΔY.data.nzVal)...)
        data = CuSparseArrayCSR(X.data.rowPtr, X.data.colVal, dV, size(X))
        ΔZ = Circulant(data, kernel_length(X), spatial_size(X))
        return (CRC.NoTangent(), project_X(ΔZ))
    end
    return Y, softmax_pullback
end

function CRC.rrule(::typeof(circulant_attention), A::Circulant{T, N, M}, b::AbstractArray{Tb,Nb}) where {T, N, M, Tb, Nb}
    project_A = CRC.ProjectTo(A)
    project_b = CRC.ProjectTo(b)
    function circulant_attention_pullback(dc)
        Δc = CRC.unthunk(dc)
        ∂A = CRC.@thunk project_A(reshape(circulant_similarity(DotSimilarity(), Δc, b, M), size(A)...))
        ∂b = CRC.@thunk begin
            ΔC = reshape(Δc, :, size(Δc)[Nb-1:end]...)
            ∂B = NNlib.batched_transpose(reshape(A, :,:,:)) ⊠ ΔC
            project_b(reshape(∂B, size(b)...))
        end
        return (CRC.NoTangent(), ∂A, ∂b)
    end
    return A ⊗ b, circulant_attention_pullback
end

function LinearAlgebra.dot(f::Zygote.FillArrays.Fill{T,N}, A::Circulant{T,N}) where {T,N}
    return f.value * sum(A)
end

_scale_collectnz(X::Circulant) = X.data.nzVal
_scale_collectnz(X::Zygote.FillArrays.Fill) = X.value

function CRC.rrule(::typeof(scale), c::AbstractArray{T,N}, A::Circulant{T,N}) where {T,N}
    project_c = CRC.ProjectTo(c)
    project_A = CRC.ProjectTo(A)
    function scale_pullback(dB)
        ΔB = CRC.unthunk(dB)
        ∂c = CRC.@thunk begin
            sumdims = filter(d -> size(c,d)==1, 2:ndims(c))
            s = sum(_scale_collectnz(ΔB) .* A.data.nzVal; dims=sumdims .- 1)
            project_c(reshape(s, size(c)...))
        end
        ∂A = CRC.@thunk begin
            sumdims = filter(d -> size(A,d)==1, 1:ndims(A))
            project_A(sum(scale(c, ΔB); dims=sumdims))
        end
        return (CRC.NoTangent(), ∂c, ∂A)
    end
    return scale(c, A), scale_pullback
end

