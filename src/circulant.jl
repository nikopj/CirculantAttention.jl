@inline circshift_index(m::Ti, s, M) where Ti = mod(m - Ti(1) - s, M) + Ti(1)

@inline function cartesian_circulant(n::Ti, N, M) where Ti
    # filter size must be odd
    p = (M-Ti(1)) >>> Ti(1)
    j = cld(n, M) # col num
    m = mod(n-Ti(1), M) + Ti(1)
    if j <= p
        m = circshift_index(m, j - p - Ti(1), M)
    elseif j > N-p 
        m = circshift_index(m, p - N + j, M)
    end
    k = (m - Ti(1)) + (j - Ti(1)) - p
    i = mod(k, N) + Ti(1)
    return i, j
end

@inline function cartesian_circulant(n::Ti, N1, N2, M) where Ti
    # filter size must be odd
    p = (M - Ti(1)) >>> Ti(1)
    Msq = M*M
    j = cld(n, Msq)                            # global col num
    jj = cld(j, N1)                            # block col num
    j0 = mod(j - Ti(1), N1) + Ti(1)                      # intra block col num
    nn = mod(cld(n, M) - Ti(1), M) + Ti(1) + M*(jj - Ti(1))  # block num
    mm = mod(nn - Ti(1), M) + Ti(1)                      # block filter coeff num
    m0 = n - M*(mm - Ti(1)) - (j - Ti(1))*Msq            # intra block col filter coeff num
    if jj <= p
        mm = circshift_index(mm, jj - p - Ti(1), M)
    elseif jj > N2-p
        mm = circshift_index(mm, p - N2 + jj, M)
    end
    if j0 <= p
        m0 = circshift_index(m0, j0 - p - Ti(1), M)
    elseif j0 > N1-p
        m0 = circshift_index(m0, p - N1 + j0, M)
    end
    ii = mod((mm - Ti(1)) + (jj - Ti(1)) - p, N2) + Ti(1)              # block row num
    i  = N1*(ii - Ti(1)) + mod((m0 - Ti(1)) + (j0 - Ti(1)) - p, N1) + Ti(1)  # rownum
    return i, j
end

@inline cartesian_circulant(n, spatdims::NTuple{1}, W) = cartesian_circulant(n, spatdims[1], W)
@inline cartesian_circulant(n, spatdims::NTuple{2}, W) = cartesian_circulant(n, spatdims[1], spatdims[2], W)

# ------------------------------------------------------------
# kernels
# ------------------------------------------------------------
function cucirculant_kernel!(colval::AbstractVector{T}, N::Int32, M::Int32, maxidx::Int32) where T
    tid    = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds while tid <= maxidx
        i, _ = cartesian_circulant(tid, N, M)
        colval[tid] = T(i)
        tid += stride
    end
    return nothing
end

function cucirculant_kernel!(colval::AbstractVector{T}, N1::Int32, N2::Int32, M::Int32, maxidx::Int32) where T
    tid    = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    @inbounds while tid <= maxidx
        i, _ = cartesian_circulant(tid, N1, N2, M)
        colval[tid] = T(i)
        tid += stride
    end
    return nothing
end

# ------------------------------------------------------------
# constructors
# ------------------------------------------------------------
function cucirculant(M::Int, N::Int, Tv=Float32, Ti=Int32)
    @assert isodd(M)

    M32, N32 = Int32(M), Int32(N)
    maxidx = N32 * M32

    colval = CuVector{Ti}(undef, maxidx)
    nzval  = CUDA.ones(Tv, maxidx)

    rowptr = CuVector{Ti}(undef, N + 1)
    @. rowptr = Ti(0:N) * Ti(M) + Ti(1)

    args = colval, N32, M32, maxidx
    kernel = @cuda launch=false cucirculant_kernel!(args...)

    cfg = launch_configuration(kernel.fun)
    threads = min(maxidx, cfg.threads)
    blocks  = cld(maxidx, threads)

    kernel(args...; threads=threads, blocks=blocks)
    return CuSparseMatrixCSR{Tv, Ti}(rowptr, colval, nzval, (N, N))
end

function cucirculant(M::Int, N1::Int, N2::Int, Tv=Float32, Ti=Int32)
    @assert isodd(M)

    M32, N1_32, N2_32 = Int32(M), Int32(N1), Int32(N2)
    maxidx = N1_32 * N2_32 * M32 * M32

    colval = CuVector{Ti}(undef, maxidx)

    nzval  = CUDA.ones(Tv, maxidx)

    rowptr = CuVector{Ti}(undef, N1 * N2 + 1)
    @. rowptr = Ti(0:(N1*N2)) * Ti(M*M) + Ti(1)

    args = colval, N1_32, N2_32, M32, maxidx
    kernel = @cuda launch=false cucirculant_kernel!(args...)
    cfg = launch_configuration(kernel.fun)

    threads = min(maxidx, cfg.threads)
    blocks  = cld(maxidx, threads)

    kernel(args...; threads=threads, blocks=blocks)
    return CuSparseMatrixCSR{Tv, Ti}(rowptr, colval, nzval, (N1*N2, N1*N2))
end
