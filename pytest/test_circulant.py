"""Tests for circulant sparsity pattern construction."""

import torch
import pytest
from circulant_attention.circulant import (
    Circulant, build_circulant,
    _circulant_indices_1d, _circulant_indices_2d,
)


# ---------------------------------------------------------------------------
# 1-D tests
# ---------------------------------------------------------------------------

class TestCirculantIndices1D:
    def test_shape(self):
        N, M = 8, 3
        row, col = _circulant_indices_1d(N, M)
        assert row.shape == (N * M,)
        assert col.shape == (N * M,)

    def test_rows_uniform(self):
        """Each row should have exactly M non-zeros."""
        N, M = 8, 5
        row, _ = _circulant_indices_1d(N, M)
        counts = torch.bincount(row, minlength=N)
        assert (counts == M).all()

    def test_symmetric_window(self):
        """Middle rows should have columns centered on the diagonal."""
        N, M = 16, 5
        _, col = _circulant_indices_1d(N, M)
        p = (M - 1) // 2
        # Row 4 (well away from boundaries): expect cols 2,3,4,5,6
        row4_cols = col[4 * M : 5 * M].sort()[0]
        expected = torch.arange(4 - p, 4 + p + 1)
        assert torch.equal(row4_cols, expected)

    def test_wrapping_first_row(self):
        """Row 0 wraps around: for N=8, M=3, expect cols {7, 0, 1}."""
        N, M = 8, 3
        _, col = _circulant_indices_1d(N, M)
        row0_cols = set(col[:M].tolist())
        assert row0_cols == {N - 1, 0, 1}

    def test_wrapping_last_row(self):
        """Last row wraps: for N=8, M=3, expect cols {6, 7, 0}."""
        N, M = 8, 3
        _, col = _circulant_indices_1d(N, M)
        last_cols = set(col[(N - 1) * M : N * M].tolist())
        assert last_cols == {N - 2, N - 1, 0}

    @pytest.mark.parametrize("N,M", [(4, 3), (8, 5), (16, 7), (6, 3)])
    def test_dense_is_circulant(self, N, M):
        """Densified matrix should be a banded circulant."""
        C = build_circulant(M, N, batch_size=1)
        D = C.to_dense().squeeze(0)  # (N, N)
        # Each row should be a circular shift of the first row
        first_row = D[0]
        for i in range(1, N):
            expected = torch.roll(first_row, i)
            assert torch.allclose(D[i], expected), f"Row {i} is not a circular shift"


# ---------------------------------------------------------------------------
# 2-D tests
# ---------------------------------------------------------------------------

class TestCirculantIndices2D:
    def test_shape(self):
        N1, N2, M = 4, 4, 3
        row, col = _circulant_indices_2d(N1, N2, M)
        n_rows = N1 * N2
        nnz_per_row = M * M
        assert row.shape == (n_rows * nnz_per_row,)
        assert col.shape == (n_rows * nnz_per_row,)

    def test_rows_uniform(self):
        N1, N2, M = 4, 4, 3
        row, _ = _circulant_indices_2d(N1, N2, M)
        n_rows = N1 * N2
        nnz_per_row = M * M
        counts = torch.bincount(row, minlength=n_rows)
        assert (counts == nnz_per_row).all()

    def test_2d_window_interior(self):
        """Interior position should have a 2D neighborhood."""
        N1, N2, M = 6, 6, 3
        row, col = _circulant_indices_2d(N1, N2, M)
        p = (M - 1) // 2
        # Pick an interior position: (2, 2) → linear index 2*6+2 = 14 (using row-major N1*ii + i_intra, but our layout is N1*block_row + intra_block)
        # Actually position (j0=2, jj=2) → row = jj*N1 + j0 = 2*6+2 = 14
        target_row = 14
        Msq = M * M
        start = target_row * Msq
        end = start + Msq
        row_cols = col[start:end]
        # Expected: 2D offsets (-1..1) x (-1..1) around (2,2) in the 6x6 grid
        expected_cols = set()
        for di in range(-p, p + 1):
            for dj in range(-p, p + 1):
                r = (2 + di) % N2
                c = (2 + dj) % N1
                expected_cols.add(r * N1 + c)
        assert set(row_cols.tolist()) == expected_cols

    @pytest.mark.parametrize("N1,N2,M", [(4, 4, 3), (3, 5, 3), (4, 4, 5)])
    def test_dense_is_bccb(self, N1, N2, M):
        """Densified 2D circulant should have BCCB structure: each row is
        a 2D circular shift of the first row."""
        C = build_circulant(M, N1, N2, batch_size=1)
        D = C.to_dense().squeeze(0)  # (N1*N2, N1*N2)
        n = N1 * N2
        first_row = D[0]
        first_row_2d = first_row.reshape(N2, N1)
        for idx in range(1, n):
            jj = idx // N1
            j0 = idx % N1
            expected_2d = torch.roll(first_row_2d, shifts=(jj, j0), dims=(0, 1))
            expected = expected_2d.reshape(-1)
            assert torch.allclose(D[idx], expected), f"Row {idx} is not a 2D circular shift"


# ---------------------------------------------------------------------------
# Circulant class tests
# ---------------------------------------------------------------------------

class TestCirculantClass:
    def test_windowview_roundtrip(self):
        C = build_circulant(3, 8, batch_size=2)
        W = C.windowview()
        assert W.shape == (2, 8, 3)
        C2 = C.with_values(W.flatten(1))
        assert torch.equal(C.values, C2.values)

    def test_to_sparse_shape(self):
        C = build_circulant(5, 16, batch_size=3)
        S = C.to_sparse()
        assert S.shape == (3, 16, 16)
        assert S.is_sparse

    def test_to_dense_shape(self):
        C = build_circulant(3, 4, 4, batch_size=2)
        D = C.to_dense()
        assert D.shape == (2, 16, 16)

    def test_values_are_ones(self):
        C = build_circulant(3, 8, batch_size=1)
        assert torch.allclose(C.values, torch.ones_like(C.values))

    def test_properties(self):
        C = build_circulant(5, 8, batch_size=4)
        assert C.n_rows == 8
        assert C.nnz_per_row == 5
        assert C.batch_size == 4
        assert C.kernel_length == 5
        assert C.spatial_size == (8,)

    def test_properties_2d(self):
        C = build_circulant(3, 4, 6, batch_size=2)
        assert C.n_rows == 24
        assert C.nnz_per_row == 9
        assert C.batch_size == 2
        assert C.spatial_size == (4, 6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
