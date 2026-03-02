function CRC.rrule(::typeof(circulant_similarity), ::DotSimilarity, x::AbstractArray{T,N}, y::AbstractArray{T,N}, W) where {T,N}
    project_x = CRC.ProjectTo(x)
    project_y = CRC.ProjectTo(y)
    function dot_sim_pullback(dS)
        d = size(x, N-1)
        X = reshape(x, prod(size(x)[1:N-2]), d, :)
        Y = reshape(y, prod(size(y)[1:N-2]), d, :)
        ΔS  = reshape(CRC.unthunk(dS), :, :, :)
        ∂x = CRC.@thunk project_x(reshape(ΔS ⊠ Y, size(x)...))
        ∂y = CRC.@thunk project_y(reshape(batched_adjoint(ΔS) ⊠ X, size(y)...))
        return (CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y, CRC.NoTangent())
    end
    S = circulant_similarity(DotSimilarity(), x, y, W)
    return S, dot_sim_pullback
end

function CRC.rrule(::typeof(circulant_similarity), ::RealDotSimilarity, x::AbstractArray{T,N}, y::AbstractArray{T,N}, W) where {T,N}
    project_x = CRC.ProjectTo(x)
    project_y = CRC.ProjectTo(y)
    function realdot_sim_pullback(dS)
        d = size(x, N-1)
        X = reshape(x, prod(size(x)[1:N-2]), d, :)
        Y = reshape(y, prod(size(y)[1:N-2]), d, :)
        ΔS  = reshape(CRC.unthunk(dS), :, :, :)
        ∂x = CRC.@thunk project_x(reshape(ΔS ⊠ Y, size(x)...))
        ∂y = CRC.@thunk project_y(reshape(batched_adjoint(ΔS) ⊠ X, size(y)...))
        return (CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y, CRC.NoTangent())
    end
    S = circulant_similarity(RealDotSimilarity(), x, y, W)
    return S, realdot_sim_pullback
end

function CRC.rrule(::typeof(circulant_similarity), ::DistanceSimilarity, x::AbstractArray{T,N}, y::AbstractArray{T,N}, W) where {T,N}
    project_x = CRC.ProjectTo(x)
    project_y = CRC.ProjectTo(y)
    function dist_sim_pullback(dS)
        d = size(x, N-1)
        X = reshape(x, prod(size(x)[1:N-2]), d, :)
        Y = reshape(y, prod(size(y)[1:N-2]), d, :)
        ΔS  = reshape(CRC.unthunk(dS), :, :, :)
        ΔSᴴ = batched_adjoint(ΔS)
        ∂x = CRC.@thunk project_x(reshape((ΔS ⊠ Y)  - (sum(ΔS; dims=2) .* X), size(x)...))
        ∂y = CRC.@thunk project_y(reshape((ΔSᴴ ⊠ X) - (reshape(sum(ΔS; dims=1), :, 1, size(ΔS,3)) .* Y), size(y)...))
        return (CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y, CRC.NoTangent())
    end
    S = circulant_similarity(DistanceSimilarity(), x, y, W)
    return S, dist_sim_pullback
end

function CRC.rrule(::typeof(circulant_similarity), ::PIDotSimilarity, x::AbstractArray{T,N}, y::AbstractArray{T,N}, W) where {T,N}
    project_x = CRC.ProjectTo(x)
    project_y = CRC.ProjectTo(y)
    S = circulant_similarity(DotSimilarity(), x, y, W)
    function pidot_sim_pullback(dS)
        d = size(x, N-1)
        X = reshape(x, prod(size(x)[1:N-2]), d, :)
        Y = reshape(y, prod(size(y)[1:N-2]), d, :)
        ΔS  = reshape(CRC.unthunk(dS), :, :, :)
        Sr = reshape(S, :, :, :)
        ΔZ = sign.(Sr) .* ΔS
        ∂x = CRC.@thunk project_x(reshape(ΔZ ⊠ Y, size(x)...))
        ∂y = CRC.@thunk project_y(reshape(batched_adjoint(ΔZ) ⊠ X, size(y)...))
        return (CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y, CRC.NoTangent())
    end
    return abs.(S), pidot_sim_pullback
end

function CRC.rrule(::typeof(circulant_similarity), ::PIDistanceSimilarity, x::AbstractArray{T,N}, y::AbstractArray{T,N}, W) where {T,N}
    project_x = CRC.ProjectTo(x)
    project_y = CRC.ProjectTo(y)
    function pidist_sim_pullback(dS)
        d = size(x, N-1)
        X = reshape(x, prod(size(x)[1:N-2]), d, :)
        Y = reshape(y, prod(size(y)[1:N-2]), d, :)
        ΔS  = reshape(CRC.unthunk(dS), :, :, :)
        Z = circulant_similarity(DotSimilarity(), x, y, W)
        Z = reshape(Z, :, :, :)
        ΔZ = sign.(Z) .* ΔS
        ΔZᴴ = batched_adjoint(ΔZ)
        ∂x = CRC.@thunk project_x(reshape((ΔZ ⊠ Y)  - (sum(ΔS; dims=2) .* X), size(x)...))
        ∂y = CRC.@thunk project_y(reshape((ΔZᴴ ⊠ X) - (reshape(sum(ΔS; dims=1), :, 1, size(ΔS,3)) .* Y), size(y)...))
        return (CRC.NoTangent(), CRC.NoTangent(), ∂x, ∂y, CRC.NoTangent())
    end
    S = circulant_similarity(PIDistanceSimilarity(), x, y, W)
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

function CRC.ProjectTo(A::Circulant{T, N, M}) where {T, N, M}
    CRC.ProjectTo{Circulant}(; element=CRC.ProjectTo(zero(T)), kernel_length=M, spatial_size=spatial_size(A), axes=axes(A))
end

function (project::CRC.ProjectTo{Circulant})(dx::Circulant{T, N, M, S, A}) where {T, N, M, S, A<:CuSparseArrayCSR}
    # Project nzVal elementwise, preserving sparsity structure
    nzVal = project.element.(dx.data.nzVal)
    data = CuSparseArrayCSR(copy(dx.data.rowPtr), copy(dx.data.colVal), nzVal, size(dx))
    Circulant(data, project.kernel_length, project.spatial_size)
end

# Handle the case where dx comes in as a dense tangent (e.g. from ZeroTangent or thunks)
function (project::CRC.ProjectTo{Circulant})(dx::CRC.AbstractZero)
    return dx
end

# ============================================================
# Tangent type: structural tangent for Circulant
# Allows AD to decompose Circulant into its fields
# ============================================================

function CRC.rrule(::Type{<:Circulant}, data::A, M::Int, spatsize::NTuple{S,Int}) where {T, N, S, A<:AbstractArray{T,N}}
    C = Circulant(data, M, spatsize)
    function Circulant_pullback(ΔC)
        ΔC = CRC.unthunk(ΔC)
        ∂data = if ΔC isa Circulant
            ΔC.data  # already a CuSparseArrayCSR
        elseif ΔC isa CRC.Tangent
            CRC.unthunk(ΔC.data)  # unwrap the structural tangent field
        else
            error("Unexpected tangent type for Circulant: $(typeof(ΔC))")
        end
        return CRC.NoTangent(), ∂data, CRC.NoTangent(), CRC.NoTangent()
    end
    return C, Circulant_pullback
end

# ============================================================
# ProjectTo — already defined, but add CuArray dense fallback
# in case a Circulant tangent arrives as a dense CuArray
# ============================================================

function (project::CRC.ProjectTo{Circulant})(dx::AbstractArray)
    # Fallback: dx is a dense array (e.g. from generic AD path).
    # Extract the nzVal entries by indexing — this should not happen
    # in the hot path, but provides a safe fallback.
    M   = project.kernel_length
    ss  = project.spatial_size
    # We can't reconstruct the sparse structure from a dense array without
    # the rowPtr/colVal, so error loudly rather than silently giving wrong answer.
    throw(ArgumentError(
        "ProjectTo{Circulant} received a dense $(typeof(dx)) tangent. " *
        "Ensure all operations on Circulant produce Circulant tangents."
    ))
end

# ============================================================
# Structural tangent support: handle Tangent{Circulant}
# arriving in pullbacks (e.g. from nested AD)
# ============================================================

function (project::CRC.ProjectTo{Circulant})(dx::CRC.Tangent)
    # dx.data is the tangent for the .data field
    return project(dx.data)
end

function (project::CRC.ProjectTo{Circulant})(dx::Circulant)
    # Already a Circulant — just project the nzVal element type
    nzVal = project.element.(dx.data.nzVal)
    data  = CuSparseArrayCSR(copy(dx.data.rowPtr), copy(dx.data.colVal), nzVal, size(dx))
    Circulant(data, project.kernel_length, project.spatial_size)
end

# ============================================================
# rrules for core operations that Zygote can't handle:
# field access, arithmetic, reshape, repeat, sum, copy
# ============================================================

# --- getproperty / field access ---
# Zygote would try to scalar-index through getfield; we block that
# by providing structural tangent rules.

function CRC.rrule(::typeof(Base.getproperty), A::Circulant{T,N,M}, f::Symbol) where {T,N,M}
    val = getproperty(A, f)
    function getproperty_pullback(Δ)
        # Return a structural Tangent with only the accessed field populated
        ∂A = if f == :data
            CRC.Tangent{Circulant}(; data=CRC.unthunk(Δ))
        else
            CRC.ZeroTangent()
        end
        return CRC.NoTangent(), ∂A, CRC.NoTangent()
    end
    return val, getproperty_pullback
end

function CRC.rrule(::typeof(Base.repeat), A::Circulant{T,N,M}, dims::Int...) where {T,N,M}
    project_A = CRC.ProjectTo(A)
    result = repeat(A, dims...)
    function repeat_pullback(ΔB)
        Δ = _concretize_tangent(CRC.unthunk(ΔB), result)
        nzVal = Δ.data.nzVal

        for (circ_dim, n) in enumerate(dims)
            circ_dim > 2 || continue   # dims 1,2 are sparse structure — not repeatable
            n > 1        || continue   # nothing to sum if not repeated

            nzval_dim = circ_dim - 1   # Circulant dim k → nzVal dim k-1
            orig = size(A.data.nzVal, nzval_dim)

            # Split (orig*n) → (orig, n), sum over the n axis, drop it
            pre  = size(nzVal)[1:nzval_dim-1]
            post = size(nzVal)[nzval_dim+1:end]
            nzVal = sum(reshape(nzVal, pre..., orig, n, post...); dims = nzval_dim + 1)
            nzVal = reshape(nzVal, pre..., orig, post...)
        end

        ∂data = CuSparseArrayCSR(copy(A.data.rowPtr), copy(A.data.colVal), nzVal, size(A))
        return CRC.NoTangent(), project_A(Circulant(∂data, M, spatial_size(A))), map(_ -> CRC.NoTangent(), dims)...
    end
    return result, repeat_pullback
end

# --- sum ---
function CRC.rrule(::typeof(Base.sum), A::Circulant{T,N,M}; dims=:) where {T,N,M}
    project_A = CRC.ProjectTo(A)
    result = sum(A; dims=dims)
    function sum_pullback(Δ)
        δ = CRC.unthunk(Δ)
        # Only handle batch-dim reductions (dims > 2) — these preserve the
        # Circulant structure and have a clean nzVal-level interpretation.
        # Full reduction (dims=:) and spatial reductions (dims 1,2) should
        # not be given a custom rrule — let Zygote trace through nzVal directly.
        @assert dims != Colon() && all(collect(dims) .> 2) """
            sum rrule for Circulant only supports batch dims (dims > 2).
            For full reduction use sum(A.data.nzVal) directly.
        """
        δ_circ = _concretize_tangent(δ, result)
        pre    = size(A.data.nzVal)[1:minimum(dims)-2]
        post   = size(A.data.nzVal)[maximum(dims):end]
        # broadcast size-1 tangent nzVal back to full nzVal shape
        ∂nzVal = δ_circ.data.nzVal .+ CUDA.zeros(T, size(A.data.nzVal)...)
        ∂data  = CuSparseArrayCSR(copy(A.data.rowPtr), copy(A.data.colVal), ∂nzVal, size(A))
        return CRC.NoTangent(), project_A(Circulant(∂data, M, spatial_size(A)))
    end
    return result, sum_pullback
end

# ============================================================
# @non_differentiable declarations for purely structural/integer ops
# ============================================================

CRC.@non_differentiable kernel_length(::Any)
CRC.@non_differentiable spatial_dims(::Any)
CRC.@non_differentiable spatial_size(::Any)
CRC.@non_differentiable Base.size(::Circulant)
CRC.@non_differentiable Base.size(::Circulant, ::Int)
CRC.@non_differentiable Base.ndims(::Circulant)
CRC.@non_differentiable Base.axes(::Circulant)
