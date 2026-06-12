"""Similarity functions and circulant_similarity with custom autograd."""

import math
import torch
from .circulant import Circulant, build_circulant


# ---------------------------------------------------------------------------
# Similarity types
# ---------------------------------------------------------------------------

class DotSimilarity:
    """Dot-product similarity: S[i,j] = x[i] . y[j]"""
    name = "dot"


class RealDotSimilarity:
    """Real dot-product similarity (identical to DotSimilarity for real tensors)."""
    name = "real_dot"


class DistanceSimilarity:
    """Negative squared-distance similarity: S[i,j] = -0.5 * ||x[i] - y[j]||^2"""
    name = "distance"


class PIDotSimilarity:
    """Phase-invariant dot similarity: S[i,j] = |x[i] . y[j]|"""
    name = "pi_dot"


class PIDistanceSimilarity:
    """Phase-invariant distance: S[i,j] = -0.5*||x||^2 + |x.y| - 0.5*||y||^2"""
    name = "pi_distance"


# ---------------------------------------------------------------------------
# Gather-based forward helpers
# ---------------------------------------------------------------------------

def _compute_similarity_values(sim_name, x, y, row_indices, col_indices):
    """Compute similarity values for every non-zero position via gather.

    Args:
        sim_name: string key ('dot', 'real_dot', 'distance', 'pi_dot', 'pi_distance')
        x: (B, N, C) — indexed at row positions
        y: (B, N, C) — indexed at col positions
        row_indices: (nnz,) int64
        col_indices: (nnz,) int64

    Returns:
        values: (B, nnz)
        extra: dict with auxiliary tensors needed by some backward variants
    """
    xg = x[:, row_indices, :]   # (B, nnz, C)
    yg = y[:, col_indices, :]   # (B, nnz, C)

    extra = {}
    if sim_name in ("dot", "real_dot"):
        values = (xg * yg).sum(-1)
    elif sim_name == "distance":
        values = -0.5 * (xg - yg).pow(2).sum(-1)
    elif sim_name == "pi_dot":
        z = (xg * yg).sum(-1)
        extra["z_values"] = z
        values = z.abs()
    elif sim_name == "pi_distance":
        xx = xg.pow(2).sum(-1)
        xy = (xg * yg).sum(-1)
        yy = yg.pow(2).sum(-1)
        extra["z_values"] = xy      # raw dot for sign in backward
        values = -0.5 * xx + xy.abs() - 0.5 * yy
    else:
        raise ValueError(f"Unknown similarity: {sim_name}")

    return values, extra


def _build_sparse_from_values(row_indices, col_indices, values, N):
    """Construct a batched sparse COO tensor (B, N, N) from per-batch values.

    Args:
        row_indices: (nnz,)
        col_indices: (nnz,)
        values: (B, nnz)
        N: matrix size
    """
    B, nnz = values.shape
    dev = values.device
    batch_idx = torch.arange(B, device=dev).repeat_interleave(nnz)
    indices = torch.stack([batch_idx, row_indices.repeat(B), col_indices.repeat(B)])
    return torch.sparse_coo_tensor(indices, values.reshape(-1), (B, N, N)).coalesce()


# ---------------------------------------------------------------------------
# Custom autograd
# ---------------------------------------------------------------------------

class _CirculantSimilarityFn(torch.autograd.Function):
    """Forward: gather-based similarity.  Backward: sparse-dense bmm.

    Translated from src/rrules.jl:250-328.
    """

    @staticmethod
    def forward(ctx, x, y, row_indices, col_indices, sim_name, spatial_size, kernel_length):
        values, extra = _compute_similarity_values(sim_name, x, y, row_indices, col_indices)

        # Save what backward needs
        ctx.save_for_backward(x, y, extra.get("z_values"))
        ctx.row_indices = row_indices
        ctx.col_indices = col_indices
        ctx.sim_name = sim_name
        ctx.spatial_size = spatial_size
        ctx.kernel_length = kernel_length
        return values

    @staticmethod
    def backward(ctx, grad_values):
        x, y, z_values = ctx.saved_tensors
        sim_name = ctx.sim_name
        ri = ctx.row_indices
        ci = ctx.col_indices
        N = math.prod(ctx.spatial_size)

        if sim_name in ("dot", "real_dot"):
            # ∂x = ΔS @ Y,  ∂y = ΔSᵀ @ X
            dS = _build_sparse_from_values(ri, ci, grad_values, N)
            grad_x = torch.bmm(dS, y)
            grad_y = torch.bmm(dS.transpose(1, 2), x)

        elif sim_name == "distance":
            # ∂x = (ΔS @ Y) - sum_cols(ΔS) * X
            # ∂y = (ΔSᵀ @ X) - sum_rows(ΔS)ᵀ * Y
            dS = _build_sparse_from_values(ri, ci, grad_values, N)
            dS_t = dS.transpose(1, 2)

            dS_dense = dS.to_dense()
            col_sum = dS_dense.sum(dim=2, keepdim=True)   # (B, N, 1)
            row_sum = dS_dense.sum(dim=1, keepdim=True)   # (B, 1, N)

            grad_x = torch.bmm(dS, y) - col_sum * x
            grad_y = torch.bmm(dS_t, x) - row_sum.transpose(1, 2) * y

        elif sim_name == "pi_dot":
            # ΔZ = sign(Z) * ΔS, then dot-similarity grads with ΔZ
            dz_values = z_values.sign() * grad_values
            dZ = _build_sparse_from_values(ri, ci, dz_values, N)
            grad_x = torch.bmm(dZ, y)
            grad_y = torch.bmm(dZ.transpose(1, 2), x)

        elif sim_name == "pi_distance":
            # Recompute z if not saved (Julia recomputes in backward)
            if z_values is None:
                xg = x[:, ri, :]
                yg = y[:, ci, :]
                z_values = (xg * yg).sum(-1)

            dz_values = z_values.sign() * grad_values
            dZ = _build_sparse_from_values(ri, ci, dz_values, N)

            # Distance correction uses ΔS (not ΔZ)
            dS = _build_sparse_from_values(ri, ci, grad_values, N)
            dS_dense = dS.to_dense()
            col_sum = dS_dense.sum(dim=2, keepdim=True)
            row_sum = dS_dense.sum(dim=1, keepdim=True)

            grad_x = torch.bmm(dZ, y) - col_sum * x
            grad_y = torch.bmm(dZ.transpose(1, 2), x) - row_sum.transpose(1, 2) * y
        else:
            raise ValueError(f"Unknown similarity: {sim_name}")

        return grad_x, grad_y, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def circulant_similarity(sim, x, y, W):
    """Compute circulant-sparse similarity between feature maps.

    Args:
        sim: Similarity instance (e.g. ``DotSimilarity()``).
        x: ``(B, C, *spatial)`` — features indexed by matrix row.
        y: ``(B, C, *spatial)`` — features indexed by matrix column.
        W: Window / kernel length (odd int).

    Returns:
        :class:`Circulant` whose values are the similarity scores.
    """
    B = x.shape[0]
    C = x.shape[1]
    spatial_shape = x.shape[2:]
    spatial_size = tuple(spatial_shape)

    # (B, C, *spatial) → (B, N, C)
    x_flat = x.flatten(2).transpose(1, 2)
    y_flat = y.flatten(2).transpose(1, 2)

    # Build structural indices (non-differentiable, could be cached)
    template = build_circulant(W, *spatial_size, batch_size=1, dtype=x.dtype, device=x.device)

    values = _CirculantSimilarityFn.apply(
        x_flat, y_flat,
        template.row_indices, template.col_indices,
        sim.name, spatial_size, W,
    )

    return Circulant(
        template.row_indices, template.col_indices,
        values, W, spatial_size,
    )
