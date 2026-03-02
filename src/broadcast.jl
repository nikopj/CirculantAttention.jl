# ==============================================================================
# Broadcast infrastructure for CuSparseArrayCSR and Circulant
#
# Design:
#   CuSparseCSRStyle  — handles bare CuSparseArrayCSR operands
#   CirculantStyle    — handles Circulant operands (wraps CSR)
#
# Priority: CirculantStyle > CuSparseCSRStyle > everything else.
#
# nzVal layout:      (nnz_per_row * n_rows, batch...)  — flat storage
# windowview layout: (nnz_per_row, n_rows, batch...)   — logical window shape
#
# CuArray broadcast (batch-expanding):
#   c has shape (1, 1, batch_c...) matching the Circulant/CSR array shape.
#   We broadcast c directly against windowview(X), which has shape
#   (nnz_per_row, n_rows, batch_x...). Julia's broadcast handles size-1 dims
#   in c naturally — no selectdim or reshape of c needed.
#   The result is reshaped back to flat nzVal for storage.
#
# In-place note: windowview returns a reshaped view with shape
#   (nnz_per_row, n_rows, batch...), NOT the flat nzVal shape. Assigning
#   to windowview(dest) does NOT write to dest.nzVal. In-place methods must
#   always assign to dest.nzVal directly, reshaping the rhs if needed.
# ==============================================================================

# ------------------------------------------------------------------------------
# Core helpers
# _csr_ewise:  same-shape CSR operands, no batch expansion
# _csr_expand: c is a CuArray with shape (1,1,batch_c...) — broadcast directly
#              against windowview(X); rowPtr/colVal repeated to match output
# ------------------------------------------------------------------------------

function _csr_ewise(f, X::CuSparseArrayCSR)
    CuSparseArrayCSR(copy(X.rowPtr), copy(X.colVal), f(X.nzVal), size(X))
end

function _csr_ewise(f, X::CuSparseArrayCSR, Y::CuSparseArrayCSR)
    CuSparseArrayCSR(copy(X.rowPtr), copy(X.colVal), f(X.nzVal, Y.nzVal), size(X))
end

function _csr_expand(f, c::CuArray, X::CuSparseArrayCSR)
    @assert size(c, 1) == 1 && size(c, 2) == 1 """
        CuArray argument to batch-expanding broadcast must have size 1 in \
        dims 1 and 2 (the sparse matrix dims). Got size $(size(c)[1:2]).
    """
    # c:           (1, 1, batch_c...)
    # windowview:  (nnz_per_row, n_rows, batch_x...)
    # broadcast:   (nnz_per_row, n_rows, batch_out...)  — Julia handles size-1 dims
    Xw = windowview(X)
    W     = f.(c, Xw)
    nzVal = _flatten_window(W, X)

    reps   = ntuple(i -> size(nzVal, i+1) ÷ size(X.rowPtr, i+1), ndims(X) - 2)
    rowPtr = repeat(X.rowPtr, 1, reps...)
    colVal = repeat(X.colVal, 1, reps...)

    out_size = (size(X, 1), size(X, 2), size(nzVal)[2:end]...)
    CuSparseArrayCSR(rowPtr, colVal, nzVal, out_size)
end

# ------------------------------------------------------------------------------
# CuSparseCSRStyle
# ------------------------------------------------------------------------------

struct CuSparseCSRStyle <: Broadcast.AbstractArrayStyle{Any} end
CuSparseCSRStyle(::Val{N}) where N = CuSparseCSRStyle()

Base.BroadcastStyle(::Type{<:CuSparseArrayCSR})                           = CuSparseCSRStyle()
Base.BroadcastStyle(::CuSparseCSRStyle, ::Broadcast.AbstractArrayStyle)   = CuSparseCSRStyle()
Base.BroadcastStyle(::CuSparseCSRStyle, ::Broadcast.DefaultArrayStyle{0}) = CuSparseCSRStyle()
Base.BroadcastStyle(::CuSparseCSRStyle, ::Broadcast.DefaultArrayStyle)    = CuSparseCSRStyle()

# ------------------------------------------------------------------------------
# CirculantStyle
# ------------------------------------------------------------------------------

struct CirculantStyle <: Broadcast.AbstractArrayStyle{Any} end
CirculantStyle(::Val{N}) where N = CirculantStyle()

Base.BroadcastStyle(::Type{<:Circulant})                                = CirculantStyle()
Base.BroadcastStyle(::CirculantStyle, ::Broadcast.AbstractArrayStyle)   = CirculantStyle()
Base.BroadcastStyle(::CirculantStyle, ::Broadcast.DefaultArrayStyle{0}) = CirculantStyle()
Base.BroadcastStyle(::CirculantStyle, ::Broadcast.DefaultArrayStyle)    = CirculantStyle()

# Cross-style resolution — after both structs are defined
Base.BroadcastStyle(::CirculantStyle, ::CuSparseCSRStyle) = CirculantStyle()
Base.BroadcastStyle(::CuSparseCSRStyle, ::CirculantStyle) = CirculantStyle()

# ==============================================================================
# CuSparseCSRStyle — materialize / copyto! / dispatch
# ==============================================================================

function Base.copy(bc::Broadcast.Broadcasted{CuSparseCSRStyle})
    args = map(_realize_csr_arg, bc.args)
    _csr_broadcast(bc.f, args...)
end

function Base.copyto!(dest::CuSparseArrayCSR, bc::Broadcast.Broadcasted{CuSparseCSRStyle})
    args = map(_realize_csr_arg, bc.args)
    _csr_broadcast!(bc.f, dest, args...)
    return dest
end

function Base.copyto!(dest::CuSparseArrayCSR, bc::Broadcast.Broadcasted)
    throw(ArgumentError(
        "In-place broadcast into CuSparseArrayCSR requires all arguments to share " *
        "the same sparsity pattern. Cannot change rowPtr/colVal in-place."
    ))
end

_realize_csr_arg(x)                                           = x
_realize_csr_arg(x::CuSparseArrayCSR)                         = x
_realize_csr_arg(bc::Broadcast.Broadcasted{CuSparseCSRStyle}) = copy(bc)
_realize_csr_arg(ref::Base.RefValue)                          = ref[]

# --- _csr_broadcast (out-of-place) ---

_csr_broadcast(f, X::CuSparseArrayCSR)                         = _csr_ewise(x -> f.(x), X)
_csr_broadcast(f, X::CuSparseArrayCSR, Y::CuSparseArrayCSR)    = _csr_ewise((x, y) -> f.(x, y), X, Y)
_csr_broadcast(f, c::Number, X::CuSparseArrayCSR)              = _csr_ewise(x -> f.(c, x), X)
_csr_broadcast(f, X::CuSparseArrayCSR, c::Number)              = _csr_ewise(x -> f.(x, c), X)
_csr_broadcast(f, c::CuArray, X::CuSparseArrayCSR)             = _csr_expand(f, c, X)
_csr_broadcast(f, X::CuSparseArrayCSR, c::CuArray)             = _csr_expand((c, xw) -> f.(xw, c), c, X)
_csr_broadcast(::typeof(|>), X::CuSparseArrayCSR, f::Function) = _csr_broadcast(f, X)

# Catch-all: concretize unknown array-like before redispatching
_csr_broadcast(f, other, X::CuSparseArrayCSR) = _csr_broadcast(f, _fill_to_csr(other, X), X)
_csr_broadcast(f, X::CuSparseArrayCSR, other) = _csr_broadcast(f, X, _fill_to_csr(other, X))

# --- _csr_broadcast! (in-place) ---
# Must assign to dest.nzVal (flat), not to windowview(dest).
# For CuArray ops, broadcast against windowview(X) for correct shape alignment,
# then flatten the result before assigning.

_csr_broadcast!(f, dest::CuSparseArrayCSR, X::CuSparseArrayCSR) =
    (dest.nzVal .= f.(X.nzVal))

_csr_broadcast!(f, dest::CuSparseArrayCSR, X::CuSparseArrayCSR, Y::CuSparseArrayCSR) =
    (dest.nzVal .= f.(X.nzVal, Y.nzVal))

_csr_broadcast!(f, dest::CuSparseArrayCSR, c::Number, X::CuSparseArrayCSR) =
    (dest.nzVal .= f.(c, X.nzVal))

_csr_broadcast!(f, dest::CuSparseArrayCSR, X::CuSparseArrayCSR, c::Number) =
    (dest.nzVal .= f.(X.nzVal, c))

function _csr_broadcast!(f, dest::CuSparseArrayCSR, c::CuArray, X::CuSparseArrayCSR)
    # Broadcast c (1,1,batch...) against windowview(X) (nnz,rows,batch...),
    # then flatten back to nzVal shape before assigning.
    dest.nzVal .= _flatten_window(f.(c, windowview(X)), dest)
end

function _csr_broadcast!(f, dest::CuSparseArrayCSR, X::CuSparseArrayCSR, c::CuArray)
    dest.nzVal .= _flatten_window(f.(windowview(X), c), dest)
end

# ==============================================================================
# CirculantStyle — materialize / copyto! / dispatch
# ==============================================================================

function Base.copy(bc::Broadcast.Broadcasted{CirculantStyle})
    args = map(_realize_arg, bc.args)
    _circulant_broadcast(bc.f, args...)
end

function Base.copyto!(dest::Circulant, bc::Broadcast.Broadcasted{CirculantStyle})
    _circulant_broadcast!(bc.f, dest, bc.args...)
    return dest
end

function Base.copyto!(dest::Circulant, bc::Broadcast.Broadcasted)
    throw(ArgumentError(
        "In-place broadcast into Circulant requires all arguments to be Circulant " *
        "or scalar; cannot change sparsity pattern in-place."
    ))
end

_realize_arg(x)                                         = x
_realize_arg(x::Circulant)                              = x
_realize_arg(bc::Broadcast.Broadcasted{CirculantStyle}) = copy(bc)
_realize_arg(ref::Base.RefValue)                        = ref[]

# --- _circulant_broadcast (out-of-place) ---

function _circulant_broadcast(f, X::Circulant{Tx,N,M,S}, Y::Circulant{Ty,N,M,S}) where {Tx,Ty,N,M,S}
    Circulant(_csr_ewise((x, y) -> f.(x, y), X.data, Y.data), M, spatial_size(X))
end
function _circulant_broadcast(f, X::Circulant{T,N,M,S}) where {T,N,M,S}
    Circulant(_csr_ewise(x -> f.(x), X.data), M, spatial_size(X))
end
function _circulant_broadcast(f, c::Number, X::Circulant{T,N,M,S}) where {T,N,M,S}
    Circulant(_csr_ewise(x -> f.(c, x), X.data), M, spatial_size(X))
end
function _circulant_broadcast(f, X::Circulant{T,N,M,S}, c::Number) where {T,N,M,S}
    Circulant(_csr_ewise(x -> f.(x, c), X.data), M, spatial_size(X))
end

# CuArray: broadcast c (1,1,batch...) against windowview for all binary ops
function _circulant_broadcast(f, c::CuArray, X::Circulant{T,N,M}) where {T,N,M}
    Circulant(_csr_expand(f, c, X.data), M, spatial_size(X))
end
function _circulant_broadcast(f, X::Circulant{T,N,M}, c::CuArray) where {T,N,M}
    Circulant(_csr_expand((c, xw) -> f.(xw, c), c, X.data), M, spatial_size(X))
end

_circulant_broadcast(::typeof(|>), X::Circulant, f::Function) = _circulant_broadcast(f, X)

# Catch-all: concretize unknown array-like (e.g. Zygote Fill) before redispatching
_circulant_broadcast(f, other, X::Circulant) = _circulant_broadcast(f, _concretize_tangent(other, X), X)
_circulant_broadcast(f, X::Circulant, other) = _circulant_broadcast(f, X, _concretize_tangent(other, X))

# --- _circulant_broadcast! (in-place) ---
# Must assign to dest.data.nzVal (flat). For CuArray ops, broadcast against
# windowview(X) then flatten before assigning to dest.data.nzVal.

function _circulant_broadcast!(f, dest::Circulant, X::Circulant, Y::Circulant)
    dest.data.nzVal .= f.(X.data.nzVal, Y.data.nzVal)
end
function _circulant_broadcast!(f, dest::Circulant, X::Circulant)
    dest.data.nzVal .= f.(X.data.nzVal)
end
function _circulant_broadcast!(f, dest::Circulant, c::Number, X::Circulant)
    dest.data.nzVal .= f.(c, X.data.nzVal)
end
function _circulant_broadcast!(f, dest::Circulant, X::Circulant, c::Number)
    dest.data.nzVal .= f.(X.data.nzVal, c)
end
function _circulant_broadcast!(f, dest::Circulant, c::CuArray, X::Circulant)
    dest.data.nzVal .= _flatten_window(f.(c, windowview(X)), dest.data)
end
function _circulant_broadcast!(f, dest::Circulant, X::Circulant, c::CuArray)
    dest.data.nzVal .= _flatten_window(f.(windowview(X), c), dest.data)
end

# ==============================================================================
# Utilities
# ==============================================================================

# Concretize Fill/uniform tangents into a Circulant (used by catch-all dispatch
# and rrules.jl pullbacks).
_concretize_tangent(Δ::Circulant, _ref) = Δ
function _concretize_tangent(Δ::Zygote.FillArrays.Fill, ref::Circulant{T,N,M}) where {T,N,M}
    nzVal = CUDA.fill(T(Δ.value), size(ref.data.nzVal)...)
    data  = CuSparseArrayCSR(copy(ref.data.rowPtr), copy(ref.data.colVal), nzVal, size(ref))
    Circulant(data, M, spatial_size(ref))
end
function _concretize_tangent(Δ, ref::Circulant{T,N,M}) where {T,N,M}
    nzVal = similar(ref.data.nzVal)
    nzVal .= Δ
    data  = CuSparseArrayCSR(copy(ref.data.rowPtr), copy(ref.data.colVal), nzVal, size(ref))
    Circulant(data, M, spatial_size(ref))
end

# Concretize Fill into a CuSparseArrayCSR (used by CSR-level catch-all).
function _fill_to_csr(Δ::Zygote.FillArrays.Fill, ref::CuSparseArrayCSR{T}) where T
    nzVal = CUDA.fill(T(Δ.value), size(ref.nzVal)...)
    CuSparseArrayCSR(copy(ref.rowPtr), copy(ref.colVal), nzVal, size(ref))
end
function _fill_to_csr(Δ, ref::CuSparseArrayCSR{T}) where T
    nzVal = similar(ref.nzVal)
    nzVal .= Δ
    CuSparseArrayCSR(copy(ref.rowPtr), copy(ref.colVal), nzVal, size(ref))
end
