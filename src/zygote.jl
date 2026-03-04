function Zygote.accum(x::T, y::T) where T <: Circulant 
    x === nothing ? y : 
    y === nothing ? x :
    x + y
end

Zygote.accum(x::T, y::T, z::T...) where T <: Circulant = Zygote.accum(Zygote.accum(x, y), z...)

# General rrule for _circ_copy(f, c::CuArray, X::Circulant) and reverse.
# Computes gradients entirely in nzVal/window space — never materializes dense N×N.
function CRC.rrule(::typeof(_circ_copy), f, c::CuArray, X::Circulant{T,N,M}) where {T,N,M}
    project_X = CRC.ProjectTo(X)
    project_c = CRC.ProjectTo(c)
    # Differentiate f.(c, windowview(X)) as a plain CuArray broadcast
    Wc = c  # c already broadcastable in window space
    WX = windowview(X)
    W, back_w = Zygote._pullback(Zygote.__context__, (c, w) -> f.(c, w), Wc, WX)
    result = _circ_from_window(W, X)
    function cuarray_circ_back(Δ)
        Δ  = _concretize_tangent(CRC.unthunk(Δ), result)
        Δw = windowview(Δ)   # upstream tangent in window space — a plain CuArray
        ∂Wc, ∂WX = back_w(Δw)
        # ∂c: sum ∂Wc over nzVal dims where c was size-1, reshape to size(c)
        nz_nd = ndims(WX)
        dims  = Tuple(filter(1:nz_nd) do d
            circ_d = d + 1
            circ_d > ndims(c) || size(c, circ_d) == 1
        end)
        ∂c = project_c(reshape(sum(∂Wc; dims=dims), size(c)))
        # ∂X: wrap ∂WX (CuArray in window space) back into a Circulant
        ∂X = project_X(_circ_from_window(∂WX, X))
        return CRC.NoTangent(), CRC.NoTangent(), ∂c, ∂X
    end
    return result, cuarray_circ_back
end

function CRC.rrule(::typeof(_circ_copy), f, X::Circulant{T,N,M}, c::CuArray) where {T,N,M}
    project_X = CRC.ProjectTo(X)
    project_c = CRC.ProjectTo(c)
    WX = windowview(X)
    W, back_w = Zygote._pullback(Zygote.__context__, (w, c) -> f.(w, c), WX, c)
    result = _circ_from_window(W, X)
    function circ_cuarray_back(Δ)
        Δ  = _concretize_tangent(CRC.unthunk(Δ), result)
        Δw = windowview(Δ)
        ∂WX, ∂Wc = back_w(Δw)
        nz_nd = ndims(WX)
        dims  = Tuple(filter(1:nz_nd) do d
            circ_d = d + 1
            circ_d > ndims(c) || size(c, circ_d) == 1
        end)
        ∂c = project_c(reshape(sum(∂Wc; dims=dims), size(c)))
        ∂X = project_X(_circ_from_window(∂WX, X))
        return CRC.NoTangent(), CRC.NoTangent(), ∂X, ∂c
    end
    return result, circ_cuarray_back
end
