# ==============================================================================
# Broadcast infrastructure for CuSparseArrayCSR and Circulant
#
# nzVal layout:      (nnz_per_row * n_rows, ch..., batch...)
# windowview layout: (nnz_per_row, n_rows, ch..., batch...)
#
# AD strategy for CirculantStyle:
#   CRC rrule for broadcasted(::CirculantStyle, f, args...).
#   Forward: lift Circulant args → windowview CuArrays, run f.(wargs...) as a
#   plain CuArray broadcast, wrap result back into a Circulant.
#   Pullback: runs entirely in window (CuArray) space via Zygote's CuArray AD.
#   Zygote never calls sum() or unbroadcast() on a Circulant.
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

struct CirculantStyle <: Broadcast.AbstractArrayStyle{Any} end
CirculantStyle(::Val{N}) where N = CirculantStyle()

Base.BroadcastStyle(::Type{<:Circulant})                                  = CirculantStyle()
Base.BroadcastStyle(::CirculantStyle, ::Broadcast.AbstractArrayStyle)     = CirculantStyle()
Base.BroadcastStyle(::CirculantStyle, ::Broadcast.DefaultArrayStyle{0})   = CirculantStyle()
Base.BroadcastStyle(::CirculantStyle, ::Broadcast.DefaultArrayStyle)      = CirculantStyle()
Base.BroadcastStyle(::CirculantStyle, ::CuSparseCSRStyle)                 = CirculantStyle()
Base.BroadcastStyle(::CuSparseCSRStyle, ::CirculantStyle)                 = CirculantStyle()

# ------------------------------------------------------------------------------
# Core helpers
# ------------------------------------------------------------------------------

# Repeat rowPtr/colVal along batch dims to match an expanded nzVal.
function _expand_structure(X::Circulant, nzVal::CuArray)
    reps   = ntuple(i -> size(nzVal, i+1) ÷ size(X.data.rowPtr, i+1), ndims(X) - 2)
    rowPtr = repeat(X.data.rowPtr, 1, reps...)
    colVal = repeat(X.data.colVal, 1, reps...)
    return rowPtr, colVal
end

# Wrap a window-space CuArray back into a Circulant using X as structure reference.
function _circ_from_window(W::CuArray, X::Circulant)
    nzVal          = reshape(W, size(X.data.nzVal, 1), size(W)[3:end]...)
    rowPtr, colVal = _expand_structure(X, nzVal)
    sz             = (size(X, 1), size(X, 2), size(nzVal)[2:end]...)
    Circulant(CuSparseArrayCSR(rowPtr, colVal, nzVal, sz), kernel_length(X), spatial_size(X))
end

# Lift args to window space: Circulant → windowview, everything else passthrough.
_to_window(a::Circulant) = windowview(a)
_to_window(a)            = a

# Largest-batch Circulant in args — used as structure reference.
function _ref_circulant(args)
    circs = filter(a -> a isa Circulant, collect(args))
    isempty(circs) && error("No Circulant in broadcast args")
    argmax(c -> size(c)[end], circs)
end

# ------------------------------------------------------------------------------
# CirculantStyle forward broadcast
# ------------------------------------------------------------------------------

# Lift a single broadcast arg to window space.
# Zygote.FillArrays.Fill in Circulant space (N×N×...) can't broadcast against
# windowview (nnz×N×...) — extract scalar value instead.
function _to_window_arg(a::Zygote.FillArrays.AbstractFill, ref::Circulant)
    # If shape matches windowview, pass through; otherwise use scalar value
    w = windowview(ref)
    if size(a) == size(w)
        a
    else
        Zygote.FillArrays.getindex_value(a)
    end
end
_to_window_arg(a::Circulant, _) = windowview(a)
_to_window_arg(a, _)            = a

function Base.copy(bc::Broadcast.Broadcasted{CirculantStyle})
    args  = map(bc.args) do a
        a isa Broadcast.Broadcasted ? copy(a) : a
    end
    ref   = _ref_circulant(args)
    wargs = map(a -> _to_window_arg(a, ref), args)
    W     = bc.f.(wargs...)
    _circ_from_window(W, ref)
end

# ------------------------------------------------------------------------------
# CirculantStyle AD — CRC rrule for broadcasted
#
# Zygote intercepts at broadcasted() level before copy() is ever called.
# We register a CRC rrule so the pullback runs in window (CuArray) space,
# delegating to Zygote's existing CuArray broadcast AD for all f.
# ------------------------------------------------------------------------------

# Use Zygote.@adjoint so the pullback is registered in Zygote's adjoint system
# (not CRC), matching how Zygote intercepts broadcasted() calls.
Zygote.@adjoint function Broadcast.broadcasted(::CirculantStyle, f, args...)
    ref   = _ref_circulant(args)
    wargs = map(a -> _to_window_arg(a, ref), args)

    # Forward + backward in window (CuArray) space via Zygote's CuArray AD.
    # This avoids Zygote ever calling unbroadcast() or sum() on a Circulant.
    W, back     = Zygote._pullback(__context__, Broadcast.broadcasted, f, wargs...)
    Wmat, bmat  = Zygote._pullback(__context__, Broadcast.materialize, W)
    result      = _circ_from_window(Wmat, ref)

    function broadcasted_circ_back(Δ)
        Δ  = _concretize_tangent(Zygote.unthunk(Δ), result)
        Δw = windowview(Δ)                          # upstream as plain CuArray
        ∂W     = bmat(Δw)[2]                        # push through materialize
        dback  = back(∂W)                           # push through broadcasted
        ∂wargs = Base.tail(Base.tail(dback))        # drop (∂broadcasted_fn, ∂f)
        ∂args  = map(args, ∂wargs) do arg, ∂w
            ∂w = Zygote.unthunk(∂w)
            ∂w === nothing && return nothing
            arg isa Circulant ? _circ_from_window(∂w, arg) : ∂w
        end
        return nothing, nothing, ∂args...
    end
    return result, broadcasted_circ_back
end

# Override Zygote's generic AbstractArrayStyle adjoint for CirculantStyle.
# Without this, Zygote's @adjoint broadcasted(::AbstractArrayStyle, ...) fires
# first (CirculantStyle <: AbstractArrayStyle) and calls unbroadcast on Circulant.


# ------------------------------------------------------------------------------
# In-place CirculantStyle — not on AD path
# ------------------------------------------------------------------------------

function Base.copyto!(dest::Circulant, bc::Broadcast.Broadcasted{CirculantStyle})
    args = map(bc.args) do a
        a isa Broadcast.Broadcasted ? copy(a) : a
    end
    windowview(dest) .= bc.f.(map(_to_window, args)...)
    return dest
end

function Base.copyto!(dest::Circulant, ::Broadcast.Broadcasted)
    throw(ArgumentError("In-place broadcast into Circulant requires matching sparsity patterns."))
end

# ------------------------------------------------------------------------------
# CuSparseCSRStyle — not on Circulant AD path
# ------------------------------------------------------------------------------

function _csr_unary(f, X::CuSparseArrayCSR)
    CuSparseArrayCSR(copy(X.rowPtr), copy(X.colVal), f(X.nzVal), size(X))
end

function _csr_binary(f, X::CuSparseArrayCSR, Yw)
    W     = f.(windowview(X), Yw)
    nzVal = reshape(W, size(X.nzVal, 1), size(W)[3:end]...)
    reps  = ntuple(i -> size(nzVal, i+1) ÷ size(X.rowPtr, i+1), ndims(X) - 2)
    CuSparseArrayCSR(repeat(X.rowPtr, 1, reps...), repeat(X.colVal, 1, reps...),
                     nzVal, (size(X, 1), size(X, 2), size(nzVal)[2:end]...))
end

_csr_broadcast(f, X::CuSparseArrayCSR)                      = _csr_unary(nz -> f.(nz), X)
_csr_broadcast(f, c::Number, X::CuSparseArrayCSR)           = _csr_unary(nz -> f.(c, nz), X)
_csr_broadcast(f, X::CuSparseArrayCSR, c::Number)           = _csr_unary(nz -> f.(nz, c), X)
_csr_broadcast(f, X::CuSparseArrayCSR, Y::CuSparseArrayCSR) = _csr_binary((xw, yw) -> f.(xw, yw), X, windowview(Y))
_csr_broadcast(f, c::CuArray, X::CuSparseArrayCSR)          = _csr_binary((xw, cw) -> f.(cw, xw), X, c)
_csr_broadcast(f, X::CuSparseArrayCSR, c::CuArray)          = _csr_binary((xw, cw) -> f.(xw, cw), X, c)
_csr_broadcast(f, other, X::CuSparseArrayCSR)               = _csr_broadcast(f, _concretize_tangent(other, X), X)
_csr_broadcast(f, X::CuSparseArrayCSR, other)               = _csr_broadcast(f, X, _concretize_tangent(other, X))

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
# _concretize_tangent — convert Fill/Tangent/generic tangents to concrete types
# ==============================================================================

_concretize_tangent(Δ::CuSparseArrayCSR, _::CuSparseArrayCSR) = Δ
_concretize_tangent(Δ::CRC.Tangent, ref::CuSparseArrayCSR) =
    _concretize_tangent(CRC.unthunk(Δ.nzVal), ref)
_concretize_tangent(Δ::Base.ReshapedArray{<:Any,<:Any,<:CuSparseArrayCSR}, ref::CuSparseArrayCSR) =
    _concretize_tangent(parent(Δ), ref)
function _concretize_tangent(Δ::Zygote.FillArrays.AbstractFill, ref::CuSparseArrayCSR{T}) where T
    nzVal = CUDA.fill(T(Zygote.FillArrays.getindex_value(Δ)), size(ref.nzVal)...)
    CuSparseArrayCSR(copy(ref.rowPtr), copy(ref.colVal), nzVal, size(ref))
end
function _concretize_tangent(Δ, ref::CuSparseArrayCSR{T}) where T
    nzVal = similar(ref.nzVal); nzVal .= Δ
    CuSparseArrayCSR(copy(ref.rowPtr), copy(ref.colVal), nzVal, size(ref))
end

_concretize_tangent(Δ::Circulant, _::Circulant) = Δ
_concretize_tangent(Δ::CRC.Tangent, ref::Circulant{T,N,M}) where {T,N,M} =
    Circulant(_concretize_tangent(CRC.unthunk(Δ.data), ref.data), M, spatial_size(ref))
function _concretize_tangent(Δ::Zygote.FillArrays.AbstractFill, ref::Circulant{T,N,M}) where {T,N,M}
    Circulant(_concretize_tangent(Δ, ref.data), M, spatial_size(ref))
end
# ReshapedArray wrapping a Circulant — unwrap and recurse
_concretize_tangent(Δ::Base.ReshapedArray{<:Any,<:Any,<:Circulant}, ref::Circulant) =
    _concretize_tangent(parent(Δ), ref)
_concretize_tangent(Δ, ref::Circulant{T,N,M}) where {T,N,M} =
    Circulant(_concretize_tangent(Δ, ref.data), M, spatial_size(ref))

# ==============================================================================
# Zygote @adjoints for Circulant broadcast — intercept before Zygote's generic
# Numeric×Numeric adjoints, which materialise dense N×N tangents we can't use.
#
# These fire for the user-level broadcasted(f, args...) call (no style arg),
# which is more specific than broadcasted(::typeof(*), x::Numeric, y::Numeric).
# Gradients are computed entirely in window (CuArray) space.
# ==============================================================================

function _circ_broadcasted_back(f, args, result, Δ, __context__)
    Δ  = _concretize_tangent(Zygote.unthunk(Δ), result)
    Δw = windowview(Δ)
    wargs = map(a -> _to_window_arg(a, result), args)
    _, back = Zygote._pullback(__context__, (wa...) -> f.(wa...), wargs...)
    ∂wargs = back(Δw)   # (∂f_closure, ∂warg1, ∂warg2, ...)
    ∂args = map(args, Base.tail(∂wargs)) do arg, ∂w
        ∂w = Zygote.unthunk(∂w)
        ∂w === nothing && return nothing
        arg isa Circulant ? _circ_from_window(∂w, arg) : ∂w
    end
    return (nothing, ∂args...)
end

Zygote.@adjoint function Broadcast.broadcasted(f, A::Circulant)
    result = f.(A)
    back(Δ) = _circ_broadcasted_back(f, (A,), result, Δ, __context__)
    return result, back
end

# Tiebreakers for Zygote's specific unary adjoints (real, imag, conj, abs2,
# tanh, identity) — Circulant <: Numeric so those methods are equally specific.
for _f in (:real, :imag, :conj, :abs2, :tanh, :identity)
    @eval Zygote.@adjoint function Broadcast.broadcasted(::typeof($_f), A::Circulant)
        result = $_f.(A)
        back(Δ) = _circ_broadcasted_back($_f, (A,), result, Δ, __context__)
        return result, back
    end
end

Zygote.@adjoint function Broadcast.broadcasted(f, A::Circulant, B::Circulant)
    result = f.(A, B)
    back(Δ) = _circ_broadcasted_back(f, (A, B), result, Δ, __context__)
    return result, back
end

Zygote.@adjoint function Broadcast.broadcasted(f, c::CuArray, A::Circulant)
    result = f.(c, A)
    back(Δ) = _circ_broadcasted_back(f, (c, A), result, Δ, __context__)
    return result, back
end

Zygote.@adjoint function Broadcast.broadcasted(f, A::Circulant, c::CuArray)
    result = f.(A, c)
    back(Δ) = _circ_broadcasted_back(f, (A, c), result, Δ, __context__)
    return result, back
end

Zygote.@adjoint function Broadcast.broadcasted(f, c::Number, A::Circulant)
    result = f.(c, A)
    back(Δ) = _circ_broadcasted_back(f, (c, A), result, Δ, __context__)
    return result, back
end

Zygote.@adjoint function Broadcast.broadcasted(f, A::Circulant, c::Number)
    result = f.(A, c)
    back(Δ) = _circ_broadcasted_back(f, (A, c), result, Δ, __context__)
    return result, back
end

# Tiebreakers for Zygote's specific binary adjoints with Number/AbstractArray
for _f in (:+, :-, :*, :/)
    @eval Zygote.@adjoint function Broadcast.broadcasted(::typeof($_f), c::Number, A::Circulant)
        result = $_f.(c, A)
        back(Δ) = _circ_broadcasted_back($_f, (c, A), result, Δ, __context__)
        return result, back
    end
    @eval Zygote.@adjoint function Broadcast.broadcasted(::typeof($_f), A::Circulant, c::Number)
        result = $_f.(A, c)
        back(Δ) = _circ_broadcasted_back($_f, (A, c), result, Δ, __context__)
        return result, back
    end
    @eval Zygote.@adjoint function Broadcast.broadcasted(::typeof($_f), A::Circulant, B::Circulant)
        result = $_f.(A, B)
        back(Δ) = _circ_broadcasted_back($_f, (A, B), result, Δ, __context__)
        return result, back
    end
    @eval Zygote.@adjoint function Broadcast.broadcasted(::typeof($_f), c::CuArray, A::Circulant)
        result = $_f.(c, A)
        back(Δ) = _circ_broadcasted_back($_f, (c, A), result, Δ, __context__)
        return result, back
    end
    @eval Zygote.@adjoint function Broadcast.broadcasted(::typeof($_f), A::Circulant, c::CuArray)
        result = $_f.(A, c)
        back(Δ) = _circ_broadcasted_back($_f, (A, c), result, Δ, __context__)
        return result, back
    end
end

# ==============================================================================
# Zygote.unbroadcast overloads
#
# Zygote's generic broadcasted adjoint calls unbroadcast(arg, Δ) for each arg.
# When Δ is a Circulant it calls sum(Circulant; dims=...) with sentinel dims
# that our sum doesn't support. We intercept the two failing cases and reduce
# directly on nzVal (a plain CuArray).
#
# nzVal dim d corresponds to Circulant dim d+1 (spatial rows are flattened into
# dim 1 of nzVal, so all other dims shift by 1).
# ==============================================================================

# CuArray primal, Circulant tangent: ∂c by summing nzVal over structural dims.
function Zygote.unbroadcast(x::CuArray, x̄::Circulant)
    nz    = x̄.data.nzVal
    nd_nz = ndims(nz)
    nd_x  = ndims(x)
    # nzVal dim d ↔ x dim (nd_x - nd_nz + d), aligned from the right (batch last)
    dims = Tuple(filter(1:nd_nz) do d
        xd = nd_x - nd_nz + d
        xd < 1 || size(x, xd) == 1
    end)
    reduced = isempty(dims) ? nz : sum(nz; dims=dims)
    reshape(reduced, size(x))
end

# Circulant primal, CuArray tangent: x̄ is in Circulant space (e.g. result of
# Δ .* conj.(c) where c::CuArray). x̄ has the shape of the broadcast result,
# not nzVal shape. Extract the batch-dim values and broadcast to nzVal shape.
function Zygote.unbroadcast(x::Circulant, x̄::CuArray)
    nzVal  = similar(x.data.nzVal)
    nd_nz  = ndims(nzVal)
    nd_xbar = ndims(x̄)
    # x̄ is in Circulant space (ndims = nd_nz+1) or broadcast-result space.
    # Reshape to be broadcastable against nzVal by dropping the leading spatial
    # dim (nzVal dim 1 = nnz_per_row*N rows; Circulant dims 1,2 = N×N matrix).
    # Align from the right: batch dims must match.
    x̄_aligned = if nd_xbar == nd_nz + 1
        # Circulant space → extract batch tail, reshape to (1,...,1,batch...)
        ntuple_ones = ntuple(_ -> 1, nd_nz - 1)
        reshape(x̄, ntuple_ones..., size(x̄)[end])
    else
        x̄  # already compatible shape
    end
    nzVal .= x̄_aligned
    data = CuSparseArrayCSR(x.data.rowPtr, x.data.colVal, nzVal, size(x.data))
    Zygote._project(x, Circulant(data, kernel_length(x), spatial_size(x)))
end

# Circulant primal, CuArray tangent (e.g. Δ .* conj.(c) where c::CuArray).
# x̄ is in Circulant space (N×N×ch×batch); broadcast it down to nzVal space
# by aligning dims from the right and letting Julia broadcast the structural dims.
function Zygote.unbroadcast(x::Circulant, x̄::CuArray)
    nzVal  = similar(x.data.nzVal)
    nd_nz  = ndims(nzVal)
    nd_x̄   = ndims(x̄)
    # Align x̄ dims to nzVal dims from the right (batch dim last in both).
    # Any nzVal dim that has no corresponding x̄ dim gets size 1.
    new_shape = ntuple(nd_nz) do d
        xd = nd_x̄ - nd_nz + d
        xd < 1 ? 1 : size(x̄, xd)
    end
    nzVal .= reshape(x̄, new_shape)   # broadcasts x̄ scalar/batch values across nzVal
    data = CuSparseArrayCSR(x.data.rowPtr, x.data.colVal, nzVal, size(x.data))
    Zygote._project(x, Circulant(data, kernel_length(x), spatial_size(x)))
end

# Circulant primal, Circulant tangent: ∂A by summing nzVal over size-1 dims.
function Zygote.unbroadcast(x::Circulant, x̄::Circulant)
    size(x) == size(x̄) && return Zygote._project(x, x̄)
    nz   = x̄.data.nzVal
    dims = Tuple(filter(1:ndims(nz)) do d
        size(x.data.nzVal, d) == 1
    end)
    reduced = isempty(dims) ? nz : sum(nz; dims=dims)
    data = CuSparseArrayCSR(x.data.rowPtr, x.data.colVal, reduced, size(x.data))
    Zygote._project(x, Circulant(data, kernel_length(x), spatial_size(x)))
end
