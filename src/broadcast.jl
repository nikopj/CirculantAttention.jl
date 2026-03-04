# ==============================================================================
# Broadcast infrastructure for CuSparseArrayCSR and Circulant
#
# Design:
#   CuSparseCSRStyle — handles CuSparseArrayCSR operands (unchanged)
#   CirculantStyle   — overrides `copy` with a body Zygote can trace:
#                        windowview (reshape) → f.(CuArray) → reshape →
#                        CuSparseArrayCSR (rrule) → Circulant (rrule)
#
# We override `copy` rather than `broadcasted` because Zygote's broadcast AD
# expects `broadcasted` to return a lazy Broadcasted object; overriding it to
# return a concrete Circulant breaks Zygote's pullback construction. By putting
# the differentiable logic in `copy`, Zygote traces through the body directly.
#
# Dispatch is handled by `_circ_copy(f, args...)` which matches on arg types.
#
# Batch expansion: when operands differ in batch size, the smaller one's
# rowPtr/colVal are repeated via `repeat` (which has an explicit rrule).
# The nzVal expansion is handled naturally by broadcasting in window space.
#
# In-place `copyto!` is not on the AD hot path.
#
# nzVal layout:      (nnz_per_row * n_rows, batch...)
# windowview layout: (nnz_per_row, n_rows, batch...)
# ==============================================================================
 
# ------------------------------------------------------------------------------
# Broadcast styles
# ------------------------------------------------------------------------------

struct CuSparseCSRStyle <: Broadcast.AbstractArrayStyle{Any} end
CuSparseCSRStyle(::Val{N}) where N = CuSparseCSRStyle()

Base.BroadcastStyle(::Type{<:CuSparseArrayCSR})                          = CuSparseCSRStyle()
Base.BroadcastStyle(::CuSparseCSRStyle, ::Broadcast.AbstractArrayStyle)   = CuSparseCSRStyle()
Base.BroadcastStyle(::CuSparseCSRStyle, ::Broadcast.DefaultArrayStyle{0}) = CuSparseCSRStyle()
Base.BroadcastStyle(::CuSparseCSRStyle, ::Broadcast.DefaultArrayStyle)    = CuSparseCSRStyle()

struct CirculantStyle <: Broadcast.BroadcastStyle end
CirculantStyle(::Val{N}) where N = CirculantStyle()

Base.BroadcastStyle(::Type{<:Circulant})                                  = CirculantStyle()
Base.BroadcastStyle(::CirculantStyle, ::Broadcast.AbstractArrayStyle)     = CirculantStyle()
Base.BroadcastStyle(::CirculantStyle, ::Broadcast.DefaultArrayStyle{0})   = CirculantStyle()
Base.BroadcastStyle(::CirculantStyle, ::Broadcast.DefaultArrayStyle)      = CirculantStyle()
Base.BroadcastStyle(::CirculantStyle, ::CuSparseCSRStyle)                 = CirculantStyle()
Base.BroadcastStyle(::CuSparseCSRStyle, ::CirculantStyle)                 = CirculantStyle()

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

# Repeat rowPtr/colVal along batch dims to match an expanded nzVal shape.
# reps[i] = size(nzVal, i+1) ÷ size(rowPtr, i+1) for each batch dim.
function _expand_structure(X::Circulant, nzVal::CuArray)
    reps = ntuple(i -> size(nzVal, i+1) ÷ size(X.data.rowPtr, i+1), ndims(X) - 2)
    rowPtr = repeat(X.data.rowPtr, 1, reps...)
    colVal = repeat(X.data.colVal, 1, reps...)
    return rowPtr, colVal
end

# Build a Circulant from a window-space result W, using X as the structure reference.
# Handles both same-size and batch-expanded cases.
function _circ_from_window(W, X::Circulant)
    nzVal  = reshape(W, size(X.data.nzVal, 1), size(W)[3:end]...)
    rowPtr, colVal = _expand_structure(X, nzVal)
    sz     = (size(X, 1), size(X, 2), size(nzVal)[2:end]...)
    data   = CuSparseArrayCSR(rowPtr, colVal, nzVal, sz)
    Circulant(data, kernel_length(X), spatial_size(X))
end

function CRC.rrule(::typeof(_circ_from_window), W, X::Circulant)
    result = _circ_from_window(W, X)
    function _circ_from_window_back(Δ)
        Δ  = _concretize_tangent(CRC.unthunk(Δ), result)
        # ∂W: un-reshape nzVal back to window shape
        ∂W = reshape(Δ.data.nzVal, size(W))
        # ∂X: structure (rowPtr/colVal/sz) is not differentiated
        return CRC.NoTangent(), ∂W, CRC.ZeroTangent()
    end
    return result, _circ_from_window_back
end

function CRC.rrule(::typeof(windowview), X::Circulant)
    W = windowview(X)
    function windowview_back(∂W)
        ∂W = CRC.unthunk(∂W)
        # ∂W is in window space; wrap back into a Circulant via _circ_from_window
        ∂X = _circ_from_window(∂W, X)
        return CRC.NoTangent(), ∂X
    end
    return W, windowview_back
end

# ------------------------------------------------------------------------------
# CirculantStyle — copy + Zygote adjoint
#
# Zygote intercepts at `broadcasted` level (before copy) via its generic
# @adjoint broadcasted(::AbstractArrayStyle, f, args...) which calls
# unbroadcast(arg, Δ) for each arg. unbroadcast on a Circulant calls
# sum(::Circulant; dims=...) with bogus sentinel dims — unavoidable without
# intercepting at the broadcasted level ourselves.
#
# Solution: define a Zygote @adjoint for broadcasted(::CirculantStyle, ...)
# that computes gradients entirely in window space, bypassing unbroadcast.
# The forward pass calls copy → _circ_copy → windowview → f.(CuArray) →
# _circ_from_window, same as before. The adjoint re-broadcasts the upstream
# tangent against each primal arg in window space and wraps back.
# ------------------------------------------------------------------------------

function Base.copy(bc::Broadcast.Broadcasted{CirculantStyle})
    args = map(bc.args) do a
        a isa Broadcast.Broadcasted ? copy(a) : a
    end
    _circ_copy(bc.f, args...)
end

# Zygote adjoint for CirculantStyle broadcast — handles all f uniformly.
# Gradients are computed in window (nzVal) space to avoid unbroadcast on Circulant.
# Lift all args to window space (CuArray), compute forward + backward entirely
# in window space using Zygote's existing CuArray broadcast AD, then wrap
# Circulant results back. This bypasses unbroadcast(::CuArray, ::Circulant).
function _window_args(args)
    map(args) do a
        a isa Circulant ? windowview(a) : a
    end
end

Zygote.@adjoint function Base.Broadcast.materialize(bc::Broadcast.Broadcasted{CirculantStyle})
    args = map(bc.args) do a
        a isa Broadcast.Broadcasted ? Base.Broadcast.materialize(a) : a
    end
    f = bc.f
    wargs = _window_args(args)
    W, back_w = Zygote._pullback(Zygote.__context__, Base.Broadcast.broadcasted, f, wargs...)
    Wmat, back_mat = Zygote._pullback(Zygote.__context__, Base.Broadcast.materialize, W)
    ref = args[findfirst(a -> a isa Circulant, args)]
    result = _circ_from_window(Wmat, ref)

    function materialize_circulant_back(Δ)
        Δ = _concretize_tangent(Zygote.unthunk(Δ), result)
        Δw = windowview(Δ)
        # Backprop through materialize then through broadcasted in window space
        ∂W     = back_mat(Δw)[2]         # ∂Wmat → ∂W (Broadcasted tangent)
        ∂wargs = back_w(∂W)              # (nothing, ∂f, ∂warg1, ∂warg2, ...)
        # Convert window-space arg gradients back to Circulant/CuArray gradients
        ∂args = map(args, ∂wargs[3:end]) do arg, ∂warg
            if arg isa Circulant
                ∂warg === nothing ? nothing : _circ_from_window(∂warg, arg)
            else
                ∂warg  # CuArray or Number gradient, already correct shape
            end
        end
        return nothing, nothing, ∂args...
    end
    return result, materialize_circulant_back
end

# Unary
function _circ_copy(f, X::Circulant)
    _circ_from_window(f.(windowview(X)), X)
end

# Circulant OP Circulant
function _circ_copy(f, X::Circulant, Y::Circulant)
    ref = size(X)[end] >= size(Y)[end] ? X : Y
    _circ_from_window(f.(windowview(X), windowview(Y)), ref)
end

# Scalar OP Circulant / Circulant OP Scalar
_circ_copy(f, c::Number,   X::Circulant) = _circ_from_window(f.(c, windowview(X)), X)
_circ_copy(f, X::Circulant, c::Number)   = _circ_from_window(f.(windowview(X), c), X)

# CuArray OP Circulant / Circulant OP CuArray
_circ_copy(f, c::CuArray,   X::Circulant) = _circ_from_window(f.(c, windowview(X)), X)
_circ_copy(f, X::Circulant, c::CuArray)   = _circ_from_window(f.(windowview(X), c), X)

# Pipe: X .|> f  — Julia wraps f in a RefValue as a broadcast scalar
_circ_copy(::typeof(|>), X::Circulant, f::Base.RefValue) = _circ_from_window(windowview(X) .|> f, X)

# AbstractArray OP Circulant / Circulant OP AbstractArray
# Handles Fill, dense tangents, and other array-likes from Zygote pullbacks.
# The incoming array is in Circulant space (shape N×N×ch×batch), but windowview
# has shape (nnz_per_row×N×ch×batch) — they can't broadcast directly.
# For Fill (uniform scalar) we extract the value and re-fill to window shape.
# For other arrays we route through _concretize_tangent to get a Circulant,
# then dispatch to the Circulant×Circulant case.
function _circ_copy(f, other::Zygote.Zygote.FillArrays.AbstractFill, X::Circulant)
    W = f.(Zygote.Zygote.FillArrays.getindex_value(other), windowview(X))
    _circ_from_window(W, X)
end
function _circ_copy(f, X::Circulant, other::Zygote.Zygote.FillArrays.AbstractFill)
    W = f.(windowview(X), Zygote.Zygote.FillArrays.getindex_value(other))
    _circ_from_window(W, X)
end
function _circ_copy(f, other::AbstractArray, X::Circulant)
    _circ_copy(f, _concretize_tangent(other, X), X)
end
function _circ_copy(f, X::Circulant, other::AbstractArray)
    _circ_copy(f, X, _concretize_tangent(other, X))
end

# Catch-all — error clearly rather than stack overflow
function _circ_copy(f, args...)
    throw(ArgumentError("Unsupported CirculantStyle broadcast: f=$(f), arg types=$(typeof.(args))"))
end

# ------------------------------------------------------------------------------
# CuSparseCSRStyle — unchanged, not on Circulant AD path
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

_csr_broadcast(f, X::CuSparseArrayCSR)                         = _csr_unary(nz -> f.(nz), X)
_csr_broadcast(f, c::Number, X::CuSparseArrayCSR)              = _csr_unary(nz -> f.(c, nz), X)
_csr_broadcast(f, X::CuSparseArrayCSR, c::Number)              = _csr_unary(nz -> f.(nz, c), X)
_csr_broadcast(f, X::CuSparseArrayCSR, Y::CuSparseArrayCSR)    = _csr_binary((xw, yw) -> f.(xw, yw), X, windowview(Y))
_csr_broadcast(f, c::CuArray, X::CuSparseArrayCSR)             = _csr_binary((xw, cw) -> f.(cw, xw), X, c)
_csr_broadcast(f, X::CuSparseArrayCSR, c::CuArray)             = _csr_binary((xw, cw) -> f.(xw, cw), X, c)
_csr_broadcast(f, other, X::CuSparseArrayCSR)                  = _csr_broadcast(f, _concretize_tangent(other, X), X)
_csr_broadcast(f, X::CuSparseArrayCSR, other)                  = _csr_broadcast(f, X, _concretize_tangent(other, X))

function Base.copy(bc::Broadcast.Broadcasted{CuSparseCSRStyle})
    args = map(bc.args) do a
        a isa Broadcast.Broadcasted ? copy(a) : a
    end
    _csr_broadcast(bc.f, args...)
end

function Base.copyto!(dest::CuSparseArrayCSR, bc::Broadcast.Broadcasted{CuSparseCSRStyle})
    args = map(bc.args) do a
        a isa Broadcast.Broadcasted ? copy(a) : a
    end
    _csr_broadcast!(bc.f, dest, args...)
    return dest
end

function Base.copyto!(dest::CuSparseArrayCSR, ::Broadcast.Broadcasted)
    throw(ArgumentError("In-place broadcast into CuSparseArrayCSR requires matching sparsity patterns."))
end

# ------------------------------------------------------------------------------
# In-place CirculantStyle — not on AD path, keep as-is
# ------------------------------------------------------------------------------

function Base.copyto!(dest::Circulant, bc::Broadcast.Broadcasted{CirculantStyle})
    # Realize any nested Broadcasted nodes
    args = map(bc.args) do a
        a isa Broadcast.Broadcasted ? copy(a) : a
    end
    f    = bc.f
    dest_w = windowview(dest)
    if length(args) == 1
        x = args[1]
        dest_w .= f.(x isa Circulant ? windowview(x) : x)
    elseif length(args) == 2
        x, y = args
        xw = x isa Circulant ? windowview(x) : x
        yw = y isa Circulant ? windowview(y) : y
        dest_w .= f.(xw, yw)
    else
        throw(ArgumentError("In-place Circulant broadcast only supports unary and binary ops."))
    end
    return dest
end

function Base.copyto!(dest::Circulant, ::Broadcast.Broadcasted)
    throw(ArgumentError("In-place broadcast into Circulant requires matching sparsity patterns."))
end

# ------------------------------------------------------------------------------
# _csr_broadcast! — in-place for CuSparseCSRStyle
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
# _concretize_tangent — used by copyto! and any remaining pullbacks that
# receive Fill/Tangent tangents and need a concrete CSR or Circulant.
# ==============================================================================

_concretize_tangent(Δ::CuSparseArrayCSR, _::CuSparseArrayCSR) = Δ
function _concretize_tangent(Δ::CRC.Tangent, ref::CuSparseArrayCSR)
    _concretize_tangent(CRC.unthunk(Δ.nzVal), ref)
end
function _concretize_tangent(Δ::Zygote.FillArrays.AbstractFill, ref::CuSparseArrayCSR{T}) where T
    CuSparseArrayCSR(copy(ref.rowPtr), copy(ref.colVal),
                     CUDA.fill(T(Zygote.FillArrays.getindex_value(Δ)), size(ref.nzVal)...), size(ref))
end
function _concretize_tangent(Δ, ref::CuSparseArrayCSR{T}) where T
    nzVal = similar(ref.nzVal); nzVal .= Δ
    CuSparseArrayCSR(copy(ref.rowPtr), copy(ref.colVal), nzVal, size(ref))
end

_concretize_tangent(Δ::Circulant, _::Circulant) = Δ
function _concretize_tangent(Δ::CRC.Tangent, ref::Circulant{T,N,M}) where {T,N,M}
    Circulant(_concretize_tangent(CRC.unthunk(Δ.data), ref.data), M, spatial_size(ref))
end
function _concretize_tangent(Δ, ref::Circulant{T,N,M}) where {T,N,M}
    Circulant(_concretize_tangent(Δ, ref.data), M, spatial_size(ref))
end

# ==============================================================================
# Zygote.unbroadcast overloads for Circulant tangents
#
# Zygote's generic broadcast AD calls unbroadcast(primal_arg, Δ) to reduce
# the upstream tangent back to the shape of each primal argument. When Δ is
# a Circulant, the default implementation calls sum(Δ; dims=...) with sentinel
# dims that our sum doesn't handle. We intercept both cases and operate
# directly on nzVal (a plain CuArray) to avoid touching CuSparseArrayCSR sum.
# ==============================================================================

# CuArray primal, Circulant tangent — ∂c needs spatial dims summed away.
# nzVal has shape (nnz_flat, ch..., batch) while c has shape (1,1,...,batch).
# Sum nzVal over all dims where c is size-1 relative to nzVal, then reshape.
function Zygote.unbroadcast(x::CuArray, x̄::Circulant)
    nz   = x̄.data.nzVal
    nd   = ndims(nz)
    # dims of nzVal where x (broadcast in Circulant space) was effectively size-1
    # x has ndims(x) dims; any nzVal dim beyond that is also size-1 in x
    dims = Tuple(filter(1:nd) do d
        # nzVal dim d corresponds to Circulant dim d+1
        # (nzVal has spatial rows flattened into dim 1, so batch dims shift by 1)
        circ_d = d + 1
        circ_d > ndims(x) || size(x, circ_d) == 1
    end)
    reduced = isempty(dims) ? nz : sum(nz; dims=dims)
    # reduced has same ndims as nzVal but size-1 on summed dims;
    # reshape to size(x) by dropping the extra leading dims
    reshape(reduced, size(x))
end

# Circulant primal, Circulant tangent — batch-expanding broadcast case.
# Sum x̄.nzVal over dims where x.nzVal was size-1.
function Zygote.unbroadcast(x::Circulant, x̄::Circulant)
    size(x) == size(x̄) && return Zygote._project(x, x̄)
    nz   = x̄.data.nzVal
    nd   = ndims(nz)
    dims = Tuple(filter(1:nd) do d
        size(x.data.nzVal, d) == 1
    end)
    reduced = isempty(dims) ? nz : sum(nz; dims=dims)
    data = CuSparseArrayCSR(x.data.rowPtr, x.data.colVal, reduced, size(x.data))
    Zygote._project(x, Circulant(data, kernel_length(x), spatial_size(x)))
end

