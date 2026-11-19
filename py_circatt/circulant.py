"""Circulant sparse matrix construction and the Circulant container class."""

import math
import torch


class Circulant:
    """Sparse circulant matrix stored in COO-compatible format.

    Stores structural indices (shared across batch) and per-batch values.
    The sparsity pattern is a banded circulant: each row has ``nnz_per_row``
    non-zeros arranged symmetrically around the diagonal with wrapping.

    Attributes:
        row_indices:   (nnz_per_batch,) int64 — row indices, shared across batch
        col_indices:   (nnz_per_batch,) int64 — col indices, shared across batch
        values:        (batch, nnz_per_batch) float — the only differentiable part
        kernel_length: int — window size M (odd)
        spatial_size:  tuple — (N,) for 1D or (N1, N2) for 2D
    """

    def __init__(self, row_indices, col_indices, values, kernel_length, spatial_size):
        self.row_indices = row_indices
        self.col_indices = col_indices
        self.values = values
        self.kernel_length = kernel_length
        self.spatial_size = spatial_size

    @property
    def n_rows(self):
        return math.prod(self.spatial_size)

    @property
    def nnz_per_row(self):
        return self.kernel_length ** len(self.spatial_size)

    @property
    def batch_size(self):
        return self.values.shape[0]

    @property
    def device(self):
        return self.values.device

    @property
    def dtype(self):
        return self.values.dtype

    def windowview(self):
        """Reshape values to ``(batch, n_rows, nnz_per_row)`` for row-wise ops."""
        return self.values.reshape(self.batch_size, self.n_rows, self.nnz_per_row)

    def with_values(self, new_values):
        """Return a new Circulant sharing structure but with different values."""
        return Circulant(
            self.row_indices, self.col_indices, new_values,
            self.kernel_length, self.spatial_size,
        )

    def to_sparse(self):
        """Build a batched sparse COO tensor of shape ``(batch, N, N)``."""
        B = self.batch_size
        nnz = self.values.shape[1]
        N = self.n_rows
        dev = self.device

        batch_idx = torch.arange(B, device=dev).repeat_interleave(nnz)
        row_idx = self.row_indices.repeat(B)
        col_idx = self.col_indices.repeat(B)
        indices = torch.stack([batch_idx, row_idx, col_idx])  # (3, B*nnz)
        vals = self.values.reshape(-1)                         # (B*nnz,)
        return torch.sparse_coo_tensor(indices, vals, (B, N, N)).coalesce()

    def to_dense(self):
        """Materialise as a dense ``(batch, N, N)`` tensor (for testing)."""
        return self.to_sparse().to_dense()


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

def _circulant_indices_1d(N, M, device="cpu"):
    """Compute (row, col) index pairs for a 1-D circulant sparsity pattern.

    Each of the *N* rows has *M* non-zeros placed symmetrically around the
    diagonal with circular wrapping.  Returns 0-based indices.

    Translated from ``src/circulant.jl:6-17`` (Julia 1-based).
    """
    p = (M - 1) // 2
    nnz = N * M
    n = torch.arange(nnz, device=device)

    row = n // M
    m = n % M

    # Branchless boundary shift
    shift = torch.clamp(row - p, max=0) + torch.clamp(row - (N - 1 - p), min=0)
    m_shifted = torch.remainder(m - shift, M)

    col = torch.remainder(m_shifted + row - p, N)
    return row.long(), col.long()


def _circulant_indices_2d(N1, N2, M, device="cpu"):
    """Compute (row, col) index pairs for a 2-D BCCB circulant pattern.

    The matrix is ``(N1*N2) x (N1*N2)`` with ``M*M`` non-zeros per row.
    Returns 0-based indices.

    Translated from ``src/circulant.jl:19-41`` (Julia 1-based).
    """
    p = (M - 1) // 2
    Msq = M * M
    n_rows = N1 * N2
    nnz = n_rows * Msq
    n = torch.arange(nnz, device=device)

    row = n // Msq          # global row (0-based)
    jj = row // N1           # block-row index
    j0 = row % N1            # intra-block row index

    local = n % Msq          # position within the row's Msq non-zeros
    mm = local // M           # block-level filter position
    m0 = local % M            # intra-block filter position

    # Block-level boundary shift
    shift_mm = torch.clamp(jj - p, max=0) + torch.clamp(jj - (N2 - 1 - p), min=0)
    mm = torch.remainder(mm - shift_mm, M)

    # Intra-block boundary shift
    shift_m0 = torch.clamp(j0 - p, max=0) + torch.clamp(j0 - (N1 - 1 - p), min=0)
    m0 = torch.remainder(m0 - shift_m0, M)

    # Column index
    ii = torch.remainder(mm + jj - p, N2)
    i_intra = torch.remainder(m0 + j0 - p, N1)
    col = N1 * ii + i_intra

    return row.long(), col.long()


def build_circulant(M, *spatial_dims, batch_size=1, dtype=torch.float32, device="cpu"):
    """Construct a :class:`Circulant` with all-ones values.

    Args:
        M: Window / kernel length (must be odd).
        *spatial_dims: ``N`` for 1-D or ``(N1, N2)`` for 2-D.
        batch_size: Number of batch elements.
        dtype: Value dtype.
        device: Torch device.
    """
    assert M % 2 == 1, "kernel length M must be odd"
    if len(spatial_dims) == 1:
        row_idx, col_idx = _circulant_indices_1d(spatial_dims[0], M, device)
    elif len(spatial_dims) == 2:
        row_idx, col_idx = _circulant_indices_2d(spatial_dims[0], spatial_dims[1], M, device)
    else:
        raise ValueError("Only 1-D and 2-D spatial dims are supported")

    nnz = row_idx.shape[0]
    values = torch.ones(batch_size, nnz, dtype=dtype, device=device)
    return Circulant(row_idx, col_idx, values, M, tuple(spatial_dims))
