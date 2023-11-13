const BatchedCuSparse{T} = Union{
    CUSPARSE.CuSparseArrayCSR{T},
    NNlib.BatchedTranspose{T, <:CUSPARSE.CuSparseArrayCSR},
    NNlib.BatchedAdjoint{T, <:CUSPARSE.CuSparseArrayCSR}
   } where {T <: Number}

const BatchedCuDense{T} = Union{
    CuArray{T},
    NNlib.BatchedTranspose{T, <:CuArray},
    NNlib.BatchedAdjoint{T, <:CuArray}
   } where {T <: Number}

function NNlib.batched_mul(A::NNlib.BatchedTranspose{T1, <:Circulant}, B::BatchedCuDense{T2}) where {T1, T2} 
    NNlib.batched_transpose(A.parent.data) ⊠ B
end

function NNlib.batched_mul(A::NNlib.BatchedAdjoint{T1, <:Circulant}, B::BatchedCuDense{T2}) where {T1, T2}
    NNlib.batched_adjoint(A.parent.data) ⊠ B
end

function NNlib.batched_mul(A::Circulant{T1}, B::BatchedCuDense{T2}) where {T1, T2} 
    A.data ⊠ B
end

function NNlib.batched_mul!(C::BatchedCuDense, A::Circulant, B::BatchedCuDense)
    NNlib.batched_mul!(C, A.data, B)
end

function NNlib.batched_mul!(C::BatchedCuDense, A::NNlib.BatchedTranspose{T, <:Circulant}, B::BatchedCuDense) where T
    NNlib.batched_mul!(C, NNlib.batched_transpose(A.parent.data), B)
end

function NNlib.batched_mul!(C::BatchedCuDense, A::NNlib.BatchedAdjoint{T, <:Circulant}, B::BatchedCuDense) where T
    NNlib.batched_mul!(C, NNlib.batched_adjoint(A.parent.data), B)
end


function NNlib.batched_mul(A::BatchedCuSparse{TA}, B::BatchedCuDense{TB}) where {TA, TB}
    C = similar(B, promote_type(TA, TB), 
                (size(A, 1), size(B, 2), [max(size(A, i), size(B, i)) for i=3:max(ndims(A), ndims(B))]...))
    NNlib.batched_mul!(C, A, B)
    return C
end

function NNlib.batched_mul!(
        C::DenseCuArray{T},
        A::CUSPARSE.CuSparseArrayCSR{T},
        B::DenseCuArray{T},
        α::Number=one(T),
        β::Number=zero(T)) where T
    CUSPARSE.bmm!('N', 'N', α, A, B, β, C, 'O')
    return C
end

function NNlib.batched_mul!(
        C::DenseCuArray{T},
        A::CUSPARSE.CuSparseArrayCSR{T},
        B::NNlib.BatchedTranspose{T},
        α::Number=one(T),
        β::Number=zero(T)) where T
    CUSPARSE.bmm!('N', 'T', α, A, B.parent, β, C, 'O')
    return C
end

function NNlib.batched_mul!(
        C::DenseCuArray{T},
        A::NNlib.BatchedTranspose{T, <:CUSPARSE.CuSparseArrayCSR},
        B::DenseCuArray{T},
        α::Number=one(T),
        β::Number=zero(T)) where T
    CUSPARSE.bmm!('T', 'N', α, A.parent, B, β, C, 'O')
    return C
end

function NNlib.batched_mul!(
        C::DenseCuArray{T},
        A::NNlib.BatchedTranspose{T, <:CUSPARSE.CuSparseArrayCSR},
        B::NNlib.BatchedTranspose{T},
        α::Number=one(T),
        β::Number=zero(T)) where T
    CUSPARSE.bmm!('T', 'T', α, A.parent, B.parent, β, C, 'O')
    return C
end

function NNlib.batched_mul!(
        C::DenseCuArray{T},
        A::CUSPARSE.CuSparseArrayCSR{T},
        B::NNlib.BatchedAdjoint{T},
        α::Number=one(T),
        β::Number=zero(T)) where T
    CUSPARSE.bmm!('N', 'C', α, A, B.parent, β, C, 'O')
    return C
end

function NNlib.batched_mul!(
        C::DenseCuArray{T},
        A::NNlib.BatchedAdjoint{T, <:CUSPARSE.CuSparseArrayCSR},
        B::DenseCuArray{T},
        α::Number=one(T),
        β::Number=zero(T)) where T
    CUSPARSE.bmm!('C', 'N', α, A.parent, B, β, C, 'O')
    return C
end

function NNlib.batched_mul!(
        C::DenseCuArray{T},
        A::NNlib.BatchedAdjoint{T, <:CUSPARSE.CuSparseArrayCSR},
        B::NNlib.BatchedAdjoint{T},
        α::Number=one(T),
        β::Number=zero(T)) where T
    CUSPARSE.bmm!('C', 'C', α, A.parent, B.parent, β, C, 'O')
    return C
end
