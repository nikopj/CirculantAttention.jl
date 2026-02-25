# Core helper: apply a function elementwise to the nzVal of two CSR arrays
# with the same sparsity pattern
function _csr_ewise(f, X::CuSparseArrayCSR, Y::CuSparseArrayCSR)
    CuSparseArrayCSR(copy(X.rowPtr), copy(X.colVal), f(X.nzVal, Y.nzVal), size(X))
end

function _csr_ewise(f, X::CuSparseArrayCSR)
    CuSparseArrayCSR(copy(X.rowPtr), copy(X.colVal), f(X.nzVal), size(X))
end

# Broadcast style — tell Julia's broadcast machinery to treat Circulant as a scalar-like
# container so it doesn't try to iterate over elements
struct CirculantStyle <: Broadcast.AbstractArrayStyle{Any} end
CirculantStyle(::Val{N}) where N = CirculantStyle()

Base.BroadcastStyle(::Type{<:Circulant}) = CirculantStyle()
Base.BroadcastStyle(::CirculantStyle, ::Broadcast.AbstractArrayStyle) = CirculantStyle()

# Materialize: intercept broadcast and dispatch on the function + argument types
function Base.copy(bc::Broadcast.Broadcasted{CirculantStyle})
    _circulant_broadcast(bc.f, bc.args...)
end

# Two Circulant arrays: apply f elementwise to nzVals
function _circulant_broadcast(f, X::Circulant{T,N,M,S,A}, Y::Circulant{T,N,M,S,A}) where {T,N,M,S,A<:CuSparseArrayCSR}
    data = _csr_ewise((x, y) -> f.(x, y), X.data, Y.data)
    Circulant(data, M, spatial_size(X))
end

# Unary: e.g. sign.(A)
function _circulant_broadcast(f, X::Circulant{T,N,M,S,A}) where {T,N,M,S,A<:CuSparseArrayCSR}
    data = _csr_ewise(x -> f.(x), X.data)
    Circulant(data, M, spatial_size(X))
end

# Scalar * Circulant broadcast: e.g. c .* A where c is a Number
function _circulant_broadcast(f, c::Number, X::Circulant{T,N,M,S,A}) where {T,N,M,S,A<:CuSparseArrayCSR}
    data = _csr_ewise(x -> f.(c, x), X.data)
    Circulant(data, M, spatial_size(X))
end
function _circulant_broadcast(f, X::Circulant{T,N,M,S,A}, c::Number) where {T,N,M,S,A<:CuSparseArrayCSR}
    data = _csr_ewise(x -> f.(x, c), X.data)
    Circulant(data, M, spatial_size(X))
end

# CuArray scalar broadcast (e.g. per-batch scaling): delegate to scale when f === *
# otherwise apply elementwise to nzVal
function _circulant_broadcast(f, c::CuArray, X::Circulant{T,N,M,S,A}) where {T,N,M,S,A<:CuSparseArrayCSR}
    if f === (*)
        return scale(c, X)
    end
    # For other functions, fall through to nzVal broadcast
    data = _csr_ewise(x -> f.(c, x), X.data)   # NOTE: only valid if c broadcasts over nzVal shape
    Circulant(data, M, spatial_size(X))
end
