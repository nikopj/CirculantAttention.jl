# ==============================================================================
# Broadcast infrastructure for CuSparseArrayCSR and Circulant
#
# Design:
#   CuSparseCSRStyle  — handles CuSparseArrayCSR operands
#   CirculantStyle    — unwraps args to .data, delegates to _csr_broadcast,
#                       rewraps result — no separate Circulant dispatch layer
#
# Priority: CirculantStyle > CuSparseCSRStyle > everything else.
#
# nzVal layout:      (nnz_per_row * n_rows, batch...)  — flat storage
# windowview layout: (nnz_per_row, n_rows, batch...)   — logical shape for broadcasting
#
# All binary out-of-place ops go through _csr_binary:
#   1. Lift both operands into window space
#   2. Broadcast f. — Julia handles all size-1 dim expansion
#   3. Flatten result back to nzVal shape
#   4. Repeat rowPtr/colVal to match expanded output batch shape
#
# In-place ops assign through windowview(dest) — since windowview is a
# same-layout reshape, assignments pass through directly to nzVal.
# ==============================================================================
# ------------------------------------------------------------------------------
# _csr_unary / _csr_binary — the two core primitives
# ------------------------------------------------------------------------------

function _csr_unary(f, X::CuSparseArrayCSR)
    CuSparseArrayCSR(copy(X.rowPtr), copy(X.colVal), f(X.nzVal), size(X))
end

function _csr_binary(f, X::CuSparseArrayCSR, Yw)
    W      = f.(windowview(X), Yw)
    nzVal  = reshape(W, size(X.nzVal, 1), size(W)[3:end]...)
    reps   = ntuple(i -> size(nzVal, i+1) ÷ size(X.rowPtr, i+1), ndims(X) - 2)
    CuSparseArrayCSR(repeat(X.rowPtr, 1, reps...), repeat(X.colVal, 1, reps...),
                     nzVal, (size(X, 1), size(X, 2), size(nzVal)[2:end]...))
end

# ------------------------------------------------------------------------------
# Broadcast styles
# ------------------------------------------------------------------------------

struct CuSparseCSRStyle <: Broadcast.AbstractArrayStyle{Any} end
CuSparseCSRStyle(::Val{N}) where N = CuSparseCSRStyle()

Base.BroadcastStyle(::Type{<:CuSparseArrayCSR})                          = CuSparseCSRStyle()
Base.BroadcastStyle(::CuSparseCSRStyle, ::Broadcast.AbstractArrayStyle)   = CuSparseCSRStyle()
Base.BroadcastStyle(::CuSparseCSRStyle, ::Broadcast.DefaultArrayStyle{0}) = CuSparseCSRStyle()
Base.BroadcastStyle(::CuSparseCSRStyle, ::Broadcast.DefaultArrayStyle)    = CuSparseCSRStyle()

struct CirculantStyle <: Broadcast.AbstractArrayStyle{Any} end
CirculantStyle(::Val{N}) where N = CirculantStyle()

Base.BroadcastStyle(::Type{<:Circulant})                                  = CirculantStyle()
Base.BroadcastStyle(::CirculantStyle, ::Broadcast.AbstractArrayStyle)     = CirculantStyle()
Base.BroadcastStyle(::CirculantStyle, ::Broadcast.DefaultArrayStyle{0})   = CirculantStyle()
Base.BroadcastStyle(::CirculantStyle, ::Broadcast.DefaultArrayStyle)      = CirculantStyle()

# Cross-style — defined after both structs
Base.BroadcastStyle(::CirculantStyle, ::CuSparseCSRStyle) = CirculantStyle()
Base.BroadcastStyle(::CuSparseCSRStyle, ::CirculantStyle) = CirculantStyle()

# ------------------------------------------------------------------------------
# Arg realization — materialize nested Broadcasted nodes, unwrap RefValue.
# _unwrap strips the Circulant wrapper so args are always CuSparseArrayCSR
# (or Number/CuArray) by the time _csr_broadcast sees them.
# ------------------------------------------------------------------------------

_realize(x,                                            ::Val) = x
_realize(ref::Base.RefValue,                           ::Val) = ref[]
_realize(bc::Broadcast.Broadcasted{CuSparseCSRStyle},  ::Val) = copy(bc)
_realize(bc::Broadcast.Broadcasted{CirculantStyle},    ::Val) = copy(bc)

_unwrap(x::Circulant) = x.data
_unwrap(x)            = x

# Find the first Circulant in realized args — used to recover M/spatial_size for rewrapping
_first_circulant(x::Circulant, rest...) = x
_first_circulant(_, rest...)            = _first_circulant(rest...)

# ------------------------------------------------------------------------------
# CuSparseCSRStyle — materialize / copyto!
# ------------------------------------------------------------------------------

function Base.copy(bc::Broadcast.Broadcasted{CuSparseCSRStyle})
    _csr_broadcast(bc.f, map(a -> _realize(a, Val(:csr)), bc.args)...)
end

function Base.copyto!(dest::CuSparseArrayCSR, bc::Broadcast.Broadcasted{CuSparseCSRStyle})
    _csr_broadcast!(bc.f, dest, map(a -> _realize(a, Val(:csr)), bc.args)...)
    return dest
end

function Base.copyto!(dest::CuSparseArrayCSR, ::Broadcast.Broadcasted)
    throw(ArgumentError("In-place broadcast into CuSparseArrayCSR requires matching sparsity patterns."))
end

# ------------------------------------------------------------------------------
# CirculantStyle — realize, find ref for rewrapping, unwrap to .data, delegate
# ------------------------------------------------------------------------------

function Base.copy(bc::Broadcast.Broadcasted{CirculantStyle})
    realized = map(a -> _realize(a, Val(:circ)), bc.args)
    ref      = _first_circulant(realized...)
    Circulant(_csr_broadcast(bc.f, map(_unwrap, realized)...), kernel_length(ref), spatial_size(ref))
end

function Base.copyto!(dest::Circulant, bc::Broadcast.Broadcasted{CirculantStyle})
    _csr_broadcast!(bc.f, dest.data, map(a -> _unwrap(_realize(a, Val(:circ))), bc.args)...)
    return dest
end

function Base.copyto!(dest::Circulant, ::Broadcast.Broadcasted)
    throw(ArgumentError("In-place broadcast into Circulant requires matching sparsity patterns."))
end

# ------------------------------------------------------------------------------
# _csr_broadcast — out-of-place dispatch to _csr_unary / _csr_binary
# ------------------------------------------------------------------------------

_csr_broadcast(f, X::CuSparseArrayCSR)                         = _csr_unary(nz -> f.(nz), X)
_csr_broadcast(f, c::Number, X::CuSparseArrayCSR)              = _csr_unary(nz -> f.(c, nz), X)
_csr_broadcast(f, X::CuSparseArrayCSR, c::Number)              = _csr_unary(nz -> f.(nz, c), X)
_csr_broadcast(f, X::CuSparseArrayCSR, Y::CuSparseArrayCSR)    = _csr_binary((xw, yw) -> f.(xw, yw), X, windowview(Y))
_csr_broadcast(f, c::CuArray, X::CuSparseArrayCSR)             = _csr_binary((xw, cw) -> f.(cw, xw), X, c)
_csr_broadcast(f, X::CuSparseArrayCSR, c::CuArray)             = _csr_binary((xw, cw) -> f.(xw, cw), X, c)
_csr_broadcast(::typeof(|>), X::CuSparseArrayCSR, f::Function) = _csr_broadcast(f, X)

# Catch-all: concretize unknown array-like (e.g. Zygote Fill) then redispatch
_csr_broadcast(f, other, X::CuSparseArrayCSR) = _csr_broadcast(f, _concretize_tangent(other, X), X)
_csr_broadcast(f, X::CuSparseArrayCSR, other) = _csr_broadcast(f, X, _concretize_tangent(other, X))

# ------------------------------------------------------------------------------
# _csr_broadcast! — in-place via windowview(dest)
# windowview is a same-layout reshape so assignments pass through to nzVal.
# ------------------------------------------------------------------------------

_csr_broadcast!(f, dest::CuSparseArrayCSR, X::CuSparseArrayCSR) =
    (windowview(dest) .= f.(windowview(X)))
_csr_broadcast!(f, dest::CuSparseArrayCSR, c::Number, X::CuSparseArrayCSR) =
    (windowview(dest) .= f.(c, windowview(X)))
_csr_broadcast!(f, dest::CuSparseArrayCSR, X::CuSparseArrayCSR, c::Number) =
    (windowview(dest) .= f.(windowview(X), c))
_csr_broadcast!(f, dest::CuSparseArrayCSR, X::CuSparseArrayCSR, Y::CuSparseArrayCSR) =
    (windowview(dest) .= f.(windowview(X), windowview(Y)))
_csr_broadcast!(f, dest::CuSparseArrayCSR, c::CuArray, X::CuSparseArrayCSR) =
    (windowview(dest) .= f.(c, windowview(X)))
_csr_broadcast!(f, dest::CuSparseArrayCSR, X::CuSparseArrayCSR, c::CuArray) =
    (windowview(dest) .= f.(windowview(X), c))

# ==============================================================================
# _concretize_tangent — convert any tangent type into a concrete CSR or Circulant.
# Handles: concrete instances (identity), CRC Tangent{} structs (unwrap nzVal),
# FillArrays.Fill (broadcast scalar), and raw arrays/scalars.
# Used by catch-all broadcast dispatch.
# ==============================================================================

# CuSparseArrayCSR
_concretize_tangent(Δ::CuSparseArrayCSR, _::CuSparseArrayCSR) = Δ
function _concretize_tangent(Δ::CRC.Tangent, ref::CuSparseArrayCSR)
    _concretize_tangent(CRC.unthunk(Δ.nzVal), ref)
end
function _concretize_tangent(Δ::Zygote.FillArrays.Fill, ref::CuSparseArrayCSR{T}) where T
    CuSparseArrayCSR(copy(ref.rowPtr), copy(ref.colVal),
                     CUDA.fill(T(Δ.value), size(ref.nzVal)...), size(ref))
end
function _concretize_tangent(Δ, ref::CuSparseArrayCSR{T}) where T
    nzVal = similar(ref.nzVal); nzVal .= Δ
    CuSparseArrayCSR(copy(ref.rowPtr), copy(ref.colVal), nzVal, size(ref))
end

# Circulant — delegate CSR-level concretization, then rewrap
_concretize_tangent(Δ::Circulant, _::Circulant) = Δ
function _concretize_tangent(Δ::CRC.Tangent, ref::Circulant{T,N,M}) where {T,N,M}
    Circulant(_concretize_tangent(CRC.unthunk(Δ.data), ref.data), M, spatial_size(ref))
end
function _concretize_tangent(Δ, ref::Circulant{T,N,M}) where {T,N,M}
    Circulant(_concretize_tangent(Δ, ref.data), M, spatial_size(ref))
end
