function convex_comb!(C::CuSparseArrayCSR{T}, α::CuVector, A::CuSparseArrayCSR, B::CuSparseArrayCSR) where T
    @. C.nzVal = α*A.nzVal + (one(T)-α)*B.nzVal
    return C
end
function convex_comb!(α, C::T, A::T, B::T) where {Tv, N, M, T <: Circulant{Tv, N, M}}
    convex_comb!(α, C.data, A.data, B.data)
    return C
end
convex_comb(α, A, B) = convex_comb!(copy(A), α, A, B)

# this is a special function for the convex combination stuff
function CRC.rrule(::typeof(convex_comb), α::CuVector{Tv}, A::T, B::T) where {Tv, N, M, S, T <: Circulant{Tv, N, M, S, <:CuSparseArrayCSR}}
    @assert length(α) == 1
    function convex_comb_pullback(dC::Circulant)
        ΔC = CRC.unthunk(dC)
        ∂α = CRC.@thunk begin
            tmp = similar(α)
            AB = A - B
            a = sum(ΔC.data.nzVal .* AB.data.nzVal)
            CUDA.@allowscalar tmp[1] = a
            tmp
        end
        ∂A = CRC.@thunk begin
            tmp = copy(ΔC)
            @. tmp.data.nzVal *= α
            tmp
        end
        ∂B = CRC.@thunk begin
            tmp = copy(ΔC)
            @. tmp.data.nzVal *= one(Tv) - α
            tmp
        end
        return (CRC.NoTangent(), ∂α, ∂A, ∂B)
    end
    return convex_comb(α, A, B), convex_comb_pullback
end
