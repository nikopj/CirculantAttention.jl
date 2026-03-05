# rrules.jl
#
# Design principle: only write rrules for operations that are opaque to Zygote/CRC.
#
#   1. CUDA kernels (@cuda calls) — opaque to Zygote.
#      Affected: circulant_similarity (all variants).
#
#   2. struct construction — `new` is opaque to Zygote. We register explicit
#      constructor rrules for both CuSparseArrayCSR and Circulant so that the
#      full chain nzVal ops → CuSparseArrayCSR → Circulant flows through CRC
#      rather than Zygote's ad-hoc struct tracing.
#
#      CuSparseArrayCSR pullback: ∂nzVal passes through; rowPtr/colVal/sz are
#        structural (NoTangent).
#      Circulant pullback: ∂data passes through; M/spatsize are NoTangent.
#
#   3. repeat and sum — manipulate rowPtr/colVal structurally; need explicit
#      pullbacks to sum over repeated/reduced batch dims.
#
# Everything else (broadcast, softmax, attention, copy, unary -, +, -)
# is handled by Zygote tracing through the registered constructors.
# Exception: scalar * needs an explicit rrule because the generic ChainRules
# rrule for * computes ∂c via dot(ΔZ, A) which triggers scalar GPU indexing.

# ==============================================================================
# ProjectTo{CuSparseArrayCSR}
# ==============================================================================

function CRC.ProjectTo(A::CuSparseArrayCSR{T}) where T
    CRC.ProjectTo{CuSparseArrayCSR}(;
        element  = CRC.ProjectTo(zero(T)),
        axes     = axes(A.nzVal),
        rowPtr   = A.rowPtr,
        colVal   = A.colVal,
        sz       = size(A),
    )
end

function (project::CRC.ProjectTo{CuSparseArrayCSR})(dx::CuSparseArrayCSR)
    nzVal = project.element.(dx.nzVal)
    CuSparseArrayCSR(copy(project.rowPtr), copy(project.colVal), nzVal, project.sz)
end

function (project::CRC.ProjectTo{CuSparseArrayCSR})(dx::CRC.Tangent)
    project(CRC.unthunk(dx.nzVal))
end

function (project::CRC.ProjectTo{CuSparseArrayCSR})(dx::CRC.AbstractZero)
    return dx
end

# Zygote.Zygote.FillArrays.Fill — broadcast scalar tangent across the nzVal shape.
# project.element is a ProjectTo{T} — call it on the scalar to get a concrete T.
function (project::CRC.ProjectTo{CuSparseArrayCSR})(dx::Zygote.Zygote.FillArrays.AbstractFill)
    val   = project.element(Zygote.Zygote.FillArrays.getindex_value(dx))
    nzVal = CUDA.fill(val, map(length, project.axes)...)
    CuSparseArrayCSR(copy(project.rowPtr), copy(project.colVal), nzVal, project.sz)
end

function (project::CRC.ProjectTo{CuSparseArrayCSR})(dx::CuArray)
    # CuArray tangent — must be nzVal-shaped; wrap with reference structure
    nzVal = project.element.(dx)
    CuSparseArrayCSR(copy(project.rowPtr), copy(project.colVal), nzVal, project.sz)
end

# ==============================================================================
# CuSparseArrayCSR constructor rrule
#
# rowPtr, colVal, sz are structural — NoTangent.
# Only nzVal carries gradient information.
# ==============================================================================

function CRC.rrule(::Type{CuSparseArrayCSR}, rowPtr, colVal, nzVal::V, sz) where {V<:CuArray}
    Y = CuSparseArrayCSR(rowPtr, colVal, nzVal, sz)
    function CuSparseArrayCSR_pullback(ΔY)
        ΔY = CRC.unthunk(ΔY)
        ∂nzVal = if ΔY isa CuSparseArrayCSR
            ΔY.nzVal
        elseif ΔY isa CRC.Tangent
            CRC.unthunk(ΔY.nzVal)
        else
            error("Unexpected tangent type for CuSparseArrayCSR constructor: $(typeof(ΔY))")
        end
        return CRC.NoTangent(), CRC.NoTangent(), CRC.NoTangent(), ∂nzVal, CRC.NoTangent()
    end
    return Y, CuSparseArrayCSR_pullback
end

# ==============================================================================
# ProjectTo{Circulant}
# ==============================================================================

function CRC.ProjectTo(A::Circulant{T, N, M}) where {T, N, M}
    CRC.ProjectTo{Circulant}(;
        kernel_length = M,
        spatial_size  = spatial_size(A),
        data          = CRC.ProjectTo(A.data),   # CSR projector carries rowPtr/colVal/sz/element
    )
end

# Concrete Circulant tangent
function (project::CRC.ProjectTo{Circulant})(dx::Circulant)
    Circulant(project.data(dx.data), project.kernel_length, project.spatial_size)
end

# Structural tangent Tangent{Circulant} — unwrap .data and recurse
function (project::CRC.ProjectTo{Circulant})(dx::CRC.Tangent)
    project(CRC.unthunk(dx.data))
end

# Zero tangent — pass through
function (project::CRC.ProjectTo{Circulant})(dx::CRC.AbstractZero)
    return dx
end

# Zygote.Zygote.FillArrays.Fill — uniform scalar tangent (e.g. from sum(A) backprop).
# Delegate to embedded CSR projector which handles Fill → nzVal construction.
function (project::CRC.ProjectTo{Circulant})(dx::Zygote.Zygote.FillArrays.AbstractFill)
    Circulant(project.data(dx), project.kernel_length, project.spatial_size)
end

# Dense array — route through ProjectTo{CuSparseArrayCSR} which handles
# the conversion via its own AbstractArray handler.
function (project::CRC.ProjectTo{Circulant})(dx::AbstractArray)
    Circulant(project.data(dx), project.kernel_length, project.spatial_size)
end

# ==============================================================================
# Circulant constructor rrule
#
# `new` with CUDA memory is opaque to Zygote. Pullback extracts ∂data from
# whatever tangent type arrives (Circulant or Tangent{Circulant}).
# ==============================================================================

function CRC.rrule(::Type{Circulant}, data::A, M::Int, spatsize::NTuple{S,Int}) where {T, N, S, A<:AbstractArray{T,N}}
    C = Circulant(data, M, spatsize)
    function Circulant_pullback(ΔC)
        ΔC = CRC.unthunk(ΔC)
        ∂data = if ΔC isa Circulant
            ΔC.data
        elseif ΔC isa CRC.Tangent
            CRC.unthunk(ΔC.data)
        elseif ΔC isa Zygote.FillArrays.AbstractFill
            _concretize_tangent(ΔC, C).data
        else
            error("Unexpected tangent type for Circulant constructor: $(typeof(ΔC))")
        end
        return CRC.NoTangent(), ∂data, CRC.NoTangent(), CRC.NoTangent()
    end
    return C, Circulant_pullback
end

# ==============================================================================
# scalar * rrule
#
# The generic ChainRules rrule for * computes ∂c via dot(ΔZ, A), which falls
# through to LinearAlgebra's generic dot and triggers scalar GPU indexing.
# We override it to compute ∂c entirely on-device via nzVal dot product.
# ==============================================================================

function CRC.rrule(::typeof(*), c::Union{Real,Complex}, X::Circulant{T,N,M,S,A}) where {T,N,M,S,A<:CuSparseArrayCSR}
    project_X = CRC.ProjectTo(X)
    project_c = CRC.ProjectTo(c)
    function smul_pullback(ΔZ)
        Δ  = _concretize_tangent(CRC.unthunk(ΔZ), X)
        ∂c = CRC.@thunk project_c(real(sum(conj(X.data.nzVal) .* Δ.data.nzVal)))
        ∂X = CRC.@thunk project_X(conj(c) * Δ)
        return CRC.NoTangent(), ∂c, ∂X
    end
    return c * X, smul_pullback
end

# ==============================================================================
# repeat rrule
#
# rowPtr/colVal repetition is opaque; pullback sums nzVal over repeated copies.
# ==============================================================================

function CRC.rrule(::typeof(Base.repeat), A::Circulant{T,N,M}, dims::Int...) where {T,N,M}
    project_A   = CRC.ProjectTo(A)
    project_csr = CRC.ProjectTo(A.data)
    result = repeat(A, dims...)
    function repeat_pullback(ΔB)
        ΔB    = CRC.unthunk(ΔB)
        # Extract nzVal from whatever tangent type arrives
        nzVal = ΔB isa Circulant      ? ΔB.data.nzVal :
                ΔB isa CRC.Tangent    ? CRC.unthunk(ΔB.data).nzVal :
                error("Unexpected tangent type in repeat_pullback: $(typeof(ΔB))")

        for (circ_dim, n) in enumerate(dims)
            circ_dim > 2 || continue  # dims 1,2 are sparse structure — not repeatable
            n > 1        || continue
            nzval_dim = circ_dim - 1  # Circulant dim k → nzVal dim k-1
            orig  = size(A.data.nzVal, nzval_dim)
            pre   = size(nzVal)[1:nzval_dim-1]
            post  = size(nzVal)[nzval_dim+1:end]
            nzVal = sum(reshape(nzVal, pre..., orig, n, post...); dims=nzval_dim+1)
            nzVal = reshape(nzVal, pre..., orig, post...)
        end

        ∂data = project_csr(CuSparseArrayCSR(copy(A.data.rowPtr), copy(A.data.colVal), nzVal, size(A)))
        return CRC.NoTangent(), project_A(Circulant(∂data, M, spatial_size(A))), map(_ -> CRC.NoTangent(), dims)...
    end
    return result, repeat_pullback
end

# ==============================================================================
# sum rrule (batch dims only)
#
# Full/spatial reductions return dense arrays — let Zygote trace through nzVal.
# Batch-dim reduction preserves Circulant structure; rowPtr/colVal selection
# via getindex is opaque, so we need this explicit rule.
# ==============================================================================

function CRC.rrule(::typeof(Base.sum), A::Circulant{T,N,M}; dims=:) where {T,N,M}
    @assert dims != Colon() && all(collect(dims) .> 2) """
        sum rrule for Circulant only supports batch dims (dims > 2).
        For full reduction use sum(A.data.nzVal) directly.
    """
    project_A   = CRC.ProjectTo(A)
    project_csr = CRC.ProjectTo(A.data)
    result = sum(A; dims=dims)
    function sum_pullback(Δ)
        Δ = CRC.unthunk(Δ)
        # Extract nzVal (size-1 in summed dims) from whatever tangent type arrives
        δ_nzVal = Δ isa Circulant   ? Δ.data.nzVal :
                  Δ isa CRC.Tangent ? CRC.unthunk(Δ.data).nzVal :
                  error("Unexpected tangent type in sum_pullback: $(typeof(Δ))")
        # Broadcast size-1 summed dims back to full nzVal shape
        ∂nzVal = δ_nzVal .+ CUDA.zeros(T, size(A.data.nzVal)...)
        ∂data  = project_csr(CuSparseArrayCSR(copy(A.data.rowPtr), copy(A.data.colVal), ∂nzVal, size(A)))
        return CRC.NoTangent(), project_A(Circulant(∂data, M, spatial_size(A)))
    end
    return result, sum_pullback
end

# ==============================================================================
# circulant_similarity rrules
#
# All variants call @cuda kernels — Zygote cannot trace through.
# Pullbacks use ⊠ (NNlib.batched_mul) which is fully differentiable.
# PI variants chain through sign(Z) where Z is the DotSimilarity result.
#
# _sim_reshape: flatten spatial dims → (n_spatial, d, batch) for batched_mul.
# ==============================================================================

_sim_reshape(x::AbstractArray{T,N}) where {T,N} =
    reshape(x, prod(size(x)[1:N-2]), size(x, N-1), size(x, N))

function CRC.rrule(::typeof(circulant_similarity), ::DotSimilarity, x::AbstractArray{T,N}, y::AbstractArray{T,N}, W) where {T,N}
    project_x = CRC.ProjectTo(x)
    project_y = CRC.ProjectTo(y)
    S = circulant_similarity(DotSimilarity(), x, y, W)
    function dot_sim_pullback(dS)
        X  = _sim_reshape(x)
        Y  = _sim_reshape(y)
        ΔS = reshape(CRC.unthunk(dS), :, :, :)
        ∂x = CRC.@thunk project_x(reshape(ΔS ⊠ Y, size(x)...))
        ∂y = CRC.@thunk project_y(reshape(batched_adjoint(ΔS) ⊠ X, size(y)...))
        return (CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y, CRC.NoTangent())
    end
    return S, dot_sim_pullback
end

function CRC.rrule(::typeof(circulant_similarity), ::RealDotSimilarity, x::AbstractArray{T,N}, y::AbstractArray{T,N}, W) where {T,N}
    project_x = CRC.ProjectTo(x)
    project_y = CRC.ProjectTo(y)
    S = circulant_similarity(RealDotSimilarity(), x, y, W)
    function realdot_sim_pullback(dS)
        X  = _sim_reshape(x)
        Y  = _sim_reshape(y)
        ΔS = reshape(CRC.unthunk(dS), :, :, :)
        ∂x = CRC.@thunk project_x(reshape(ΔS ⊠ Y, size(x)...))
        ∂y = CRC.@thunk project_y(reshape(batched_adjoint(ΔS) ⊠ X, size(y)...))
        return (CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y, CRC.NoTangent())
    end
    return S, realdot_sim_pullback
end

function CRC.rrule(::typeof(circulant_similarity), ::DistanceSimilarity, x::AbstractArray{T,N}, y::AbstractArray{T,N}, W) where {T,N}
    project_x = CRC.ProjectTo(x)
    project_y = CRC.ProjectTo(y)
    S = circulant_similarity(DistanceSimilarity(), x, y, W)
    function dist_sim_pullback(dS)
        X   = _sim_reshape(x)
        Y   = _sim_reshape(y)
        ΔS  = reshape(CRC.unthunk(dS), :, :, :)
        ΔSᴴ = batched_adjoint(ΔS)
        ∂x  = CRC.@thunk project_x(reshape((ΔS ⊠ Y) - (sum(ΔS; dims=2) .* X), size(x)...))
        ∂y  = CRC.@thunk project_y(reshape((ΔSᴴ ⊠ X) - (reshape(sum(ΔS; dims=1), :, 1, size(ΔS,3)) .* Y), size(y)...))
        return (CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y, CRC.NoTangent())
    end
    return S, dist_sim_pullback
end

function CRC.rrule(::typeof(circulant_similarity), ::PIDotSimilarity, x::AbstractArray{T,N}, y::AbstractArray{T,N}, W) where {T,N}
    project_x = CRC.ProjectTo(x)
    project_y = CRC.ProjectTo(y)
    Z = circulant_similarity(DotSimilarity(), x, y, W)
    function pidot_sim_pullback(dS)
        X  = _sim_reshape(x)
        Y  = _sim_reshape(y)
        ΔS = reshape(CRC.unthunk(dS), :, :, :)
        ΔZ = sign.(reshape(Z, :, :, :)) .* ΔS
        ∂x = CRC.@thunk project_x(reshape(ΔZ ⊠ Y, size(x)...))
        ∂y = CRC.@thunk project_y(reshape(batched_adjoint(ΔZ) ⊠ X, size(y)...))
        return (CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y, CRC.NoTangent())
    end
    return abs.(Z), pidot_sim_pullback
end

function CRC.rrule(::typeof(circulant_similarity), ::PIDistanceSimilarity, x::AbstractArray{T,N}, y::AbstractArray{T,N}, W) where {T,N}
    project_x = CRC.ProjectTo(x)
    project_y = CRC.ProjectTo(y)
    S = circulant_similarity(PIDistanceSimilarity(), x, y, W)
    function pidist_sim_pullback(dS)
        X   = _sim_reshape(x)
        Y   = _sim_reshape(y)
        ΔS  = reshape(CRC.unthunk(dS), :, :, :)
        Z   = reshape(circulant_similarity(DotSimilarity(), x, y, W), :, :, :)
        ΔZ  = sign.(Z) .* ΔS
        ΔZᴴ = batched_adjoint(ΔZ)
        ∂x  = CRC.@thunk project_x(reshape((ΔZ ⊠ Y) - (sum(ΔS; dims=2) .* X), size(x)...))
        ∂y  = CRC.@thunk project_y(reshape((ΔZᴴ ⊠ X) - (reshape(sum(ΔS; dims=1), :, 1, size(ΔS,3)) .* Y), size(y)...))
        return (CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y, CRC.NoTangent())
    end
    return S, pidist_sim_pullback
end

function CRC.rrule(::typeof(circulant_attention), A::Circulant{T, N, M}, b::AbstractArray{Tb,Nb}) where {T, N, M, Tb, Nb}
    project_A = CRC.ProjectTo(A)
    project_b = CRC.ProjectTo(b)
    function circulant_attention_pullback(dc)
        Δc = CRC.unthunk(dc)
        ∂A = CRC.@thunk project_A(reshape(circulant_similarity(DotSimilarity(), Δc, b, M), size(A)...))
        ∂b = CRC.@thunk begin
            ΔC = reshape(Δc, :, size(Δc)[Nb-1:end]...)
            ∂B = NNlib.batched_adjoint(reshape(A, :,:,:)) ⊠ ΔC
            project_b(reshape(∂B, size(b)...))
        end
        return (CRC.NoTangent(), ∂A, ∂b)
    end
    return A ⊗ b, circulant_attention_pullback
end

function CRC.rrule(::typeof(circulant_transposed_attention), A::Circulant{T, N, M}, b::AbstractArray{Tb,Nb}) where {T, N, M, Tb, Nb}
    project_A = CRC.ProjectTo(A)
    project_b = CRC.ProjectTo(b)
    function circulant_transposed_attention_pullback(dc)
        Δc = CRC.unthunk(dc)
        ∂A = CRC.@thunk project_A(reshape(circulant_similarity(DotSimilarity(), b, Δc, M), size(A)...))
        ∂b = CRC.@thunk begin
            ΔC = reshape(Δc, :, size(Δc)[Nb-1:end]...)
            ∂B = reshape(A, :,:,:) ⊠ ΔC
            project_b(reshape(∂B, size(b)...))
        end
        return (CRC.NoTangent(), ∂A, ∂b)
    end
    return circulant_transposed_attention(A, b), circulant_transposed_attention_pullback
end

# function CRC.rrule(::typeof(reshape), X::Circulant, dims::Union{Int,Colon}...)
#     Y = reshape(X, dims...)
#     function reshape_circulant_back(ΔY)
#         raw = CRC.unthunk(ΔY)
#         raw = raw isa Base.ReshapedArray ? parent(raw) : raw
#         ∂X  = raw isa Circulant ? raw : _concretize_tangent(raw, X)
#         return (CRC.NoTangent(), ∂X, ntuple(_->CRC.NoTangent(), length(dims))...)
#     end
#     return Y, reshape_circulant_back
# end

Zygote.@adjoint function Base.reshape(X::Circulant, dims::Union{Int,Colon}...)
    Y = reshape(X, dims...)
    function reshape_circulant_back(ΔY)
        raw = CRC.unthunk(ΔY)
        raw = raw isa Base.ReshapedArray ? parent(raw) : raw
        raw = raw isa Circulant ? raw : _concretize_tangent(raw, Y)
        ∂nzVal = reshape(raw.data.nzVal, size(X.data.nzVal))
        ∂data  = CuSparseArrayCSR(copy(X.data.rowPtr), copy(X.data.colVal),
                                   ∂nzVal, size(X.data))
        ∂X = Circulant(∂data, kernel_length(X), spatial_size(X))
        return (∂X, ntuple(_ -> nothing, length(dims))...)
    end
    return Y, reshape_circulant_back
end

# ==============================================================================
# softmax rrule for Circulant
#
# NNlib's softmax falls back to scalar indexing on Circulant. We intercept and
# run softmax in window (nzVal) space, then wrap the result back.
# The pullback follows NNlib's own softmax pullback formula:
#   ∂x = Y ⊙ (∂Y - sum(∂Y ⊙ Y; dims=dims))
# where Y = softmax(X) and ⊙ is elementwise multiply.
# ==============================================================================
function CRC.rrule(::typeof(NNlib.softmax), X::Circulant; dims=1)
    Y = NNlib.softmax(X) 
    function softmax_circulant_back(ΔY)
        ΔY  = _concretize_tangent(CRC.unthunk(ΔY), Y)
        ΔYw = windowview(ΔY)
        ∂Xw = NNlib.∇softmax_data(ΔYw, windowview(Y); dims)
        return CRC.NoTangent(), _circ_from_window(∂Xw, X)
    end
    return Y, softmax_circulant_back
end

function CRC.rrule(::typeof(joint_softmax), As::Circulant...)
    Ws      = map(windowview, As)
    Wcat    = cat(Ws...; dims=1)
    Scat    = NNlib.softmax(Wcat; dims=1)
    sizes   = map(w -> size(w, 1), Ws)
    offsets = cumsum((0, sizes...))
    splits  = ntuple(i -> Scat[offsets[i]+1:offsets[i+1], ntuple(_->Colon(), ndims(Scat)-1)...], length(As))
    results = ntuple(i -> _circ_from_window(splits[i], As[i]), length(As))

    function joint_softmax_back(ΔYs)
        ΔYs  = map(CRC.unthunk, ΔYs)
        ΔYs  = ntuple(i -> _concretize_tangent(ΔYs[i], results[i]), length(As))
        ΔWs  = map(windowview, ΔYs)
        ΔScat = cat(ΔWs...; dims=1)
        ΔWcat = NNlib.∇softmax_data(ΔScat, Scat; dims=1)
        ∂As  = ntuple(i -> _circ_from_window(ΔWcat[offsets[i]+1:offsets[i+1], ntuple(_->Colon(), ndims(ΔWcat)-1)...], As[i]), length(As))
        return CRC.NoTangent(), ∂As...
    end
    return results, joint_softmax_back
end

# ==============================================================================
# @non_differentiable: integer/structural metadata
# ==============================================================================

CRC.@non_differentiable kernel_length(::Any)
CRC.@non_differentiable spatial_dims(::Any)
CRC.@non_differentiable spatial_size(::Any)
CRC.@non_differentiable Base.ndims(::Circulant)
CRC.@non_differentiable Base.axes(::Circulant)
