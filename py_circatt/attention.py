"""Circulant attention operations: softmax, sparse matmul, full pipeline."""

import math
import torch
import torch.nn.functional as F

from .circulant import Circulant, build_circulant
from .similarity import circulant_similarity, DotSimilarity


# ---------------------------------------------------------------------------
# Softmax on Circulant
# ---------------------------------------------------------------------------

def circulant_softmax(A):
    """Row-wise softmax over the non-zero entries of a :class:`Circulant`.

    Operates in windowview space so standard ``F.softmax`` handles the math
    and autograd traces through automatically.
    """
    W = A.windowview()                        # (B, n_rows, nnz_per_row)
    S = F.softmax(W, dim=-1)                  # softmax over window entries
    return A.with_values(S.flatten(1))


# ---------------------------------------------------------------------------
# Sparse-dense matmul with custom backward
# ---------------------------------------------------------------------------

class _CirculantMatmulFn(torch.autograd.Function):
    """y = A @ x  with backward through both A.values and x.

    Translated from src/rrules.jl:330-344.
    """

    @staticmethod
    def forward(ctx, values, row_indices, col_indices, x, n_rows):
        # Build sparse COO and matmul
        B, nnz = values.shape
        C = x.shape[2]
        dev = values.device

        batch_idx = torch.arange(B, device=dev).repeat_interleave(nnz)
        indices = torch.stack([batch_idx, row_indices.repeat(B), col_indices.repeat(B)])
        sparse_A = torch.sparse_coo_tensor(indices, values.reshape(-1), (B, n_rows, n_rows)).coalesce()

        y = torch.bmm(sparse_A, x)   # (B, N, C)

        ctx.save_for_backward(values, x)
        ctx.row_indices = row_indices
        ctx.col_indices = col_indices
        ctx.n_rows = n_rows
        return y

    @staticmethod
    def backward(ctx, grad_y):
        values, x = ctx.saved_tensors
        ri = ctx.row_indices
        ci = ctx.col_indices
        N = ctx.n_rows
        B, nnz = values.shape
        dev = values.device

        # ∂values: for nz at (row, col), grad = sum_c grad_y[b, row, c] * x[b, col, c]
        gy_gathered = grad_y[:, ri, :]   # (B, nnz, C)
        x_gathered = x[:, ci, :]         # (B, nnz, C)
        grad_values = (gy_gathered * x_gathered).sum(-1)   # (B, nnz)

        # ∂x = A^T @ grad_y
        batch_idx = torch.arange(B, device=dev).repeat_interleave(nnz)
        indices_t = torch.stack([batch_idx, ci.repeat(B), ri.repeat(B)])  # transposed
        sparse_At = torch.sparse_coo_tensor(indices_t, values.reshape(-1), (B, N, N)).coalesce()
        grad_x = torch.bmm(sparse_At, grad_y)

        return grad_values, None, None, grad_x, None


def circulant_matmul(A, x):
    """Sparse circulant matrix–vector product ``y = A @ x``.

    Args:
        A: :class:`Circulant` with values ``(B, nnz)``.
        x: ``(B, N, C)`` dense tensor.

    Returns:
        ``(B, N, C)`` dense tensor.
    """
    return _CirculantMatmulFn.apply(
        A.values, A.row_indices, A.col_indices, x, A.n_rows,
    )


# ---------------------------------------------------------------------------
# Adjacency (similarity + softmax)
# ---------------------------------------------------------------------------

def circulant_adjacency(sim, x, y, W):
    """``softmax(circulant_similarity(sim, x, y, W))``.

    Args:
        sim: Similarity instance.
        x, y: ``(B, C, *spatial)`` tensors.
        W: Window size (odd).

    Returns:
        :class:`Circulant` — row-normalised adjacency.
    """
    return circulant_softmax(circulant_similarity(sim, x, y, W))


# ---------------------------------------------------------------------------
# Full attention pipeline
# ---------------------------------------------------------------------------

def circulant_attention(sim, q, k, v, W):
    """Circulant sparse attention: ``y = softmax(S(q, k)) @ v``.

    Inputs are scaled by ``1 / sqrt(sqrt(channels))`` before similarity
    computation, matching the Julia implementation.

    Args:
        sim: Similarity instance (e.g. ``DotSimilarity()``).
        q, k, v: ``(B, C, *spatial)`` tensors.
        W: Window / kernel length (odd).

    Returns:
        y: ``(B, C, *spatial)`` output.
        A: :class:`Circulant` attention matrix (after softmax).
    """
    B, C = q.shape[:2]
    spatial_shape = q.shape[2:]

    # Scale by 1/sqrt(sqrt(C))  — matches Julia's sqrt(τ) where τ = sqrt(C)
    scale = math.pow(C, 0.25)
    q_s = q / scale
    k_s = k / scale

    A = circulant_adjacency(sim, q_s, k_s, W)

    # v: (B, C, *spatial) → (B, N, C)
    v_flat = v.flatten(2).transpose(1, 2)
    y_flat = circulant_matmul(A, v_flat)               # (B, N, C)
    y = y_flat.transpose(1, 2).unflatten(2, spatial_shape)  # (B, C, *spatial)

    return y, A


# ---------------------------------------------------------------------------
# Multi-head variants
# ---------------------------------------------------------------------------

def _splitheads(x, nheads):
    """(B, C, *spatial) → (B*nheads, C//nheads, *spatial)"""
    B, C = x.shape[:2]
    spatial = x.shape[2:]
    assert C % nheads == 0
    return x.reshape(B, nheads, C // nheads, *spatial).reshape(B * nheads, C // nheads, *spatial)


def _mergeheads(x, nheads):
    """(B*nheads, C_head, *spatial) → (B, C, *spatial)"""
    Bnh, C_head = x.shape[:2]
    spatial = x.shape[2:]
    B = Bnh // nheads
    return x.reshape(B, nheads, C_head, *spatial).reshape(B, nheads * C_head, *spatial)


def circulant_mh_attention(sim, q, k, v, W, nheads):
    """Multi-head circulant attention.

    Splits channels into *nheads* groups, runs single-head attention per
    group, and concatenates the results.

    Args:
        sim: Similarity instance.
        q, k, v: ``(B, C, *spatial)`` tensors.  ``C`` must be divisible by *nheads*.
        W: Window size (odd).
        nheads: Number of attention heads.

    Returns:
        y: ``(B, C, *spatial)`` output.
        A: :class:`Circulant` attention matrix with batch dim ``B*nheads``.
    """
    qr = _splitheads(q, nheads)
    kr = _splitheads(k, nheads)
    vr = _splitheads(v, nheads)
    yr, A = circulant_attention(sim, qr, kr, vr, W)
    return _mergeheads(yr, nheads), A


def circulant_mh_adjacency(sim, x, y, W, nheads):
    """Multi-head circulant adjacency (similarity + softmax).

    Returns:
        :class:`Circulant` with batch dim ``B*nheads``.
    """
    xr = _splitheads(x, nheads)
    yr = _splitheads(y, nheads)
    return circulant_adjacency(sim, xr, yr, W)


def circulant_mh_similarity(sim, x, y, W, nheads):
    """Multi-head circulant similarity (no softmax).

    Returns:
        :class:`Circulant` with batch dim ``B*nheads``.
    """
    xr = _splitheads(x, nheads)
    yr = _splitheads(y, nheads)
    return circulant_similarity(sim, xr, yr, W)
