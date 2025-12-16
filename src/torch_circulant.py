import torch

def circulant_indices(M, N, device='cpu'):
    """
    Generates the indices for a circulant-sparse matrix.
    """
    if M % 2 == 0:
        raise ValueError("Filter size M must be odd.")

    p = (M - 1) // 2

    # Row indices
    row_indices = torch.arange(N, device=device).repeat_interleave(M)

    # Column indices
    i_base = torch.arange(N, device=device).view(-1, 1)
    shift = torch.arange(-p, p + 1, device=device).view(1, -1)
    col_indices = ((i_base + shift) % N).flatten()

    indices = torch.stack([row_indices, col_indices])
    return indices

def circulant(M, N, Tv=torch.float32, device='cpu'):
    """
    Creates a sparse circulant matrix.
    """
    indices = circulant_indices(M, N, device=device)
    values = torch.ones(indices.size(1), dtype=Tv, device=device)
    return torch.sparse_coo_tensor(indices, values, (N, N))

if __name__ == '__main__':
    # Example usage
    M = 3
    N = 5
    C = circulant(M, N)
    print(C)
    print(C.to_dense())
