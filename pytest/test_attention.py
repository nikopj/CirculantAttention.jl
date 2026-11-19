"""Tests for attention operations and end-to-end gradients."""

import torch
import pytest
from circulant_attention import (
    build_circulant, circulant_similarity, circulant_softmax,
    circulant_matmul, circulant_attention, circulant_adjacency,
    circulant_mh_attention,
    DotSimilarity, DistanceSimilarity,
)


class TestCirculantSoftmax:
    def test_rows_sum_to_one(self):
        torch.manual_seed(0)
        C = build_circulant(3, 8, batch_size=2)
        # Put random values
        C = C.with_values(torch.randn_like(C.values))
        A = circulant_softmax(C)
        W = A.windowview()   # (B, N, nnz_per_row)
        row_sums = W.sum(dim=-1)  # (B, N)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_values_positive(self):
        C = build_circulant(5, 16, batch_size=1)
        C = C.with_values(torch.randn_like(C.values))
        A = circulant_softmax(C)
        assert (A.values >= 0).all()


class TestCirculantMatmul:
    def test_matches_dense(self):
        torch.manual_seed(1)
        B, N, C, W = 2, 8, 4, 3
        circ = build_circulant(W, N, batch_size=B)
        circ = circ.with_values(torch.randn(B, circ.values.shape[1]))
        x = torch.randn(B, N, C)

        y_sparse = circulant_matmul(circ, x)
        A_dense = circ.to_dense()
        y_dense = torch.bmm(A_dense, x)

        assert torch.allclose(y_sparse, y_dense, atol=1e-5)

    def test_gradcheck(self):
        torch.manual_seed(2)
        B, N, C, W = 2, 6, 3, 3
        circ = build_circulant(W, N, batch_size=B, dtype=torch.float64)
        vals = torch.randn(B, circ.values.shape[1], dtype=torch.float64, requires_grad=True)
        x = torch.randn(B, N, C, dtype=torch.float64, requires_grad=True)

        def func(vals, x):
            return circulant_matmul(circ.with_values(vals), x)

        assert torch.autograd.gradcheck(func, (vals, x), eps=1e-6, atol=1e-4)


class TestCirculantAttention:
    def test_output_shape(self):
        torch.manual_seed(0)
        B, C, N, W = 2, 4, 8, 3
        q = torch.randn(B, C, N)
        k = torch.randn(B, C, N)
        v = torch.randn(B, C, N)
        y, A = circulant_attention(DotSimilarity(), q, k, v, W)
        assert y.shape == (B, C, N)
        assert A.batch_size == B

    def test_output_shape_2d(self):
        torch.manual_seed(0)
        B, C, H, W_spatial = 1, 3, 4, 4
        W = 3
        q = torch.randn(B, C, H, W_spatial)
        k = torch.randn(B, C, H, W_spatial)
        v = torch.randn(B, C, H, W_spatial)
        y, A = circulant_attention(DotSimilarity(), q, k, v, W)
        assert y.shape == (B, C, H, W_spatial)

    def test_finite_grads(self):
        torch.manual_seed(3)
        B, C, N, W = 2, 4, 8, 3
        q = torch.randn(B, C, N, requires_grad=True)
        k = torch.randn(B, C, N, requires_grad=True)
        v = torch.randn(B, C, N, requires_grad=True)
        y, A = circulant_attention(DotSimilarity(), q, k, v, W)
        loss = y.sum()
        loss.backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        assert k.grad is not None and torch.isfinite(k.grad).all()
        assert v.grad is not None and torch.isfinite(v.grad).all()

    def test_finite_grads_distance(self):
        torch.manual_seed(4)
        B, C, N, W = 1, 3, 6, 3
        q = torch.randn(B, C, N, requires_grad=True)
        k = torch.randn(B, C, N, requires_grad=True)
        v = torch.randn(B, C, N, requires_grad=True)
        y, _ = circulant_attention(DistanceSimilarity(), q, k, v, W)
        loss = y.sum()
        loss.backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        assert k.grad is not None and torch.isfinite(k.grad).all()
        assert v.grad is not None and torch.isfinite(v.grad).all()


class TestMultiHead:
    def test_output_shape(self):
        torch.manual_seed(0)
        B, C, N, W, nheads = 2, 8, 16, 3, 4
        q = torch.randn(B, C, N)
        k = torch.randn(B, C, N)
        v = torch.randn(B, C, N)
        y, A = circulant_mh_attention(DotSimilarity(), q, k, v, W, nheads)
        assert y.shape == (B, C, N)
        assert A.batch_size == B * nheads

    def test_finite_grads(self):
        torch.manual_seed(5)
        B, C, N, W, nheads = 2, 4, 8, 3, 2
        q = torch.randn(B, C, N, requires_grad=True)
        k = torch.randn(B, C, N, requires_grad=True)
        v = torch.randn(B, C, N, requires_grad=True)
        y, _ = circulant_mh_attention(DotSimilarity(), q, k, v, W, nheads)
        loss = y.sum()
        loss.backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        assert k.grad is not None and torch.isfinite(k.grad).all()
        assert v.grad is not None and torch.isfinite(v.grad).all()


class TestCirculantAdjacency:
    def test_rows_sum_to_one(self):
        torch.manual_seed(0)
        B, C, N, W = 2, 4, 8, 5
        x = torch.randn(B, C, N)
        y = torch.randn(B, C, N)
        A = circulant_adjacency(DotSimilarity(), x, y, W)
        row_sums = A.windowview().sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
