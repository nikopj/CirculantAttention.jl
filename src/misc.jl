function circulant(N::Int, M::Int, Tv=Float64, Ti=Int64) 
    rowval = (Ti∘first∘cartesian_circulant).(1:N*M, N, M) 
    colptr = 1 .+ M .* collect(0:N) .|> Ti
    return SparseMatrixCSC{Tv, Ti}(N, N, colptr, rowval, ones(Tv, N*M))
end

function circulant((N1, N2)::Tuple{Int, Int}, M::Int, Tv=Float64, Ti=Int64) 
    rowval = (Ti∘first∘cartesian_circulant).(1:N1*N2*M*M, N1, N2, M)
    colptr = 1 .+ M^2 .* collect(0:N1*N2) .|> Ti
    return SparseMatrixCSC{Tv, Ti}(N1*N2, N1*N2, colptr, rowval, ones(Tv, N1*N2*M*M))
end

function circulant_kron((N1, N2)::Tuple{Int, Int}, M::Int, Tv=Float64, Ti=Int64) 
    A = circulant(N2, M, Tv, Ti)
    B = circulant(N1, M, Tv, Ti)
    return kron(A, B)
end

function cucirculant_kron((N1, N2)::Tuple{Int, Int}, M::Int, Tv=Float32, Ti=Int32) 
    A = cucirculant(N2, M, Tv, Ti)
    B = cucirculant(N1, M, Tv, Ti)
    return kron(A, B)
end

