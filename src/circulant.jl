# m: filter index, 
# s: shift
# M: filter-length
circshift_index(m, s, M) = mod(m - 1 - s, M) + 1

function cartesian_circulant(n, N, M)
    # filter size must be odd
    p = (M-1) ÷ 2
    j = cld(n, M) # col num
    m = mod(n-1, M) + 1
    if j <= p
        m = circshift_index(m, j - p - 1, M)
    elseif j > N-p 
        m = circshift_index(m, p - N + j, M)
    end
    i = mod((m-1) + (j-1) - p, N) + 1
    return i, j
end
cartesian_circulant(n, (N,)::Tuple{Int}, M) = cartesian_circulant(n, N, M)

function cartesian_circulant(n, N1, N2, M)
    # filter size must be odd
    p = (M-1) ÷ 2
    j = cld(n, M^2)                            # global col num
    jj = cld(j, N1)                            # block col num
    j0 = mod(j-1, N1) + 1                      # intra block col num
    nn = mod(cld(n, M) - 1, M) + 1 + M*(jj-1)  # block num
    mm = mod(nn-1, M) + 1                      # block filter coeff num
    m0 = n - M*(mm-1) - (j-1)*M^2              # intra block col filter coeff num
    if jj <= p
        mm = circshift_index(mm, jj - p - 1, M)
    elseif jj > N2-p
        mm = circshift_index(mm, p - N2 + jj, M)
    end
    if j0 <= p
        m0 = circshift_index(m0, j0 - p - 1, M)
    elseif j0 > N1-p
        m0 = circshift_index(m0, p - N1 + j0, M)
    end
    ii = mod((mm-1) + (jj-1) - p, N2) + 1              # block row num
    i  = N1*(ii-1) + mod((m0-1) + (j0-1) - p, N1) + 1  # rownum
    return i, j
end
cartesian_circulant(n, (N1, N2)::Tuple{Int, Int}, M) = cartesian_circulant(n, N1, N2, M)

function cucirculant_kernel!(colval::AbstractArray{T}, N, M, maxidx) where T
    n = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds if n <= maxidx
        i = cartesian_circulant(n, N, M)[1]
        colval[n] = T(i)
    end
    return nothing
end

function cucirculant_kernel!(colval::AbstractArray{T}, N1, N2, M, maxidx) where T
    n = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds if n <= maxidx
        i = cartesian_circulant(n, N1, N2, M)[1]
        colval[n] = T(i)
    end
    return nothing
end

function cucirculant(M::Int, N::Int, Tv=Float32, Ti=Int32) 
    @assert M % 2 == 1 "filter size M=$M must be odd."

    colval = CuVector{Ti}(undef, N*M)
    nzval  = CUDA.ones(Tv, N*M)
    rowptr = CuVector{Ti}(undef, N + 1)

    @. rowptr = 0:N
    @. rowptr *= M
    @. rowptr += 1

    maxidx = N*M
    args = colval, N, M, maxidx
    kernel = @cuda launch=false cucirculant_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(maxidx, config.threads)
    blocks = cld(maxidx, threads)
    kernel(args...; threads=threads, blocks=blocks)

    return CuSparseMatrixCSR{Tv, Ti}(rowptr, colval, nzval, (N, N))
end

function cucirculant(M::Int, N1::Int, N2::Int, Tv=Float32, Ti=Int32) 
    @assert M % 2 == 1 "filter size M=$M must be odd."

    colval = CuVector{Ti}(undef, N1*N2*M^2)
    nzval  = CUDA.ones(Tv, N1*N2*M^2)
    rowptr = CuVector{Ti}(undef, N1*N2 + 1)

    @. rowptr = 0:(N1*N2)
    @. rowptr *= M^2
    @. rowptr += 1

    maxidx = N1*N2*M^2
    args = colval, N1, N2, M, maxidx
    kernel = @cuda launch=false cucirculant_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(maxidx, config.threads)
    blocks = cld(maxidx, threads)
    kernel(args...; threads=threads, blocks=blocks)

    return CuSparseMatrixCSR{Tv, Ti}(rowptr, colval, nzval, (N1*N2, N1*N2))
end
