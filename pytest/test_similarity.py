"""Tests for similarity computation and gradients."""

import torch
import pytest
from circulant_attention import (
    build_circulant, circulant_similarity,
    DotSimilarity, RealDotSimilarity, DistanceSimilarity,
    PIDotSimilarity, PIDistanceSimilarity,
)


def _dense_similarity(sim_name, x, y):
    """Brute-force dense similarity matrix (B, N, N) for reference."""
    B, N, C = x.shape
    S = torch.zeros(B, N, N, dtype=x.dtype, device=x.device)
    for i in range(N):
        for j in range(N):
            xi = x[:, i, :]
            yj = y[:, j, :]
            if sim_name in ("dot", "real_dot"):
                S[:, i, j] = (xi * yj).sum(-1)
            elif sim_name == "distance":
                S[:, i, j] = -0.5 * (xi - yj).pow(2).sum(-1)
            elif sim_name == "pi_dot":
                S[:, i, j] = (xi * yj).sum(-1).abs()
            elif sim_name == "pi_distance":
                xx = xi.pow(2).sum(-1)
                xy = (xi * yj).sum(-1)
                yy = yj.pow(2).sum(-1)
                S[:, i, j] = -0.5 * xx + xy.abs() - 0.5 * yy
    return S


SIM_TYPES = [
    DotSimilarity(),
    RealDotSimilarity(),
    DistanceSimilarity(),
    PIDotSimilarity(),
    PIDistanceSimilarity(),
]
SIM_IDS = ["dot", "real_dot", "distance", "pi_dot", "pi_distance"]


class TestSimilarityValues:
    """Verify that sparse similarity matches the dense reference."""

    @pytest.mark.parametrize("sim", SIM_TYPES, ids=SIM_IDS)
    def test_1d_matches_dense(self, sim):
        torch.manual_seed(42)
        B, N, C, W = 2, 8, 4, 5
        x = torch.randn(B, C, N)
        y = torch.randn(B, C, N)

        S = circulant_similarity(sim, x, y, W)
        S_dense = S.to_dense()  # (B, N, N)

        # Build reference
        x_flat = x.flatten(2).transpose(1, 2)
        y_flat = y.flatten(2).transpose(1, 2)
        ref = _dense_similarity(sim.name, x_flat, y_flat)

        # At non-zero positions the values should match
        mask = S_dense != 0
        assert torch.allclose(S_dense[mask], ref[mask], atol=1e-5)

    @pytest.mark.parametrize("sim", SIM_TYPES[:3], ids=SIM_IDS[:3])
    def test_2d_matches_dense(self, sim):
        torch.manual_seed(0)
        B, C, N1, N2, W = 1, 3, 4, 4, 3
        x = torch.randn(B, C, N1, N2)
        y = torch.randn(B, C, N1, N2)

        S = circulant_similarity(sim, x, y, W)
        S_dense = S.to_dense()

        x_flat = x.flatten(2).transpose(1, 2)
        y_flat = y.flatten(2).transpose(1, 2)
        ref = _dense_similarity(sim.name, x_flat, y_flat)

        mask = S_dense != 0
        assert torch.allclose(S_dense[mask], ref[mask], atol=1e-5)


class TestSimilarityGrads:
    """Gradient correctness via torch.autograd.gradcheck."""

    @pytest.mark.parametrize("sim", SIM_TYPES, ids=SIM_IDS)
    def test_gradcheck_1d(self, sim):
        torch.manual_seed(1)
        B, N, C, W = 2, 6, 3, 3
        x = torch.randn(B, C, N, dtype=torch.float64, requires_grad=True)
        y = torch.randn(B, C, N, dtype=torch.float64, requires_grad=True)

        def func(x, y):
            S = circulant_similarity(sim, x, y, W)
            return S.values

        assert torch.autograd.gradcheck(func, (x, y), eps=1e-6, atol=1e-4)

    @pytest.mark.parametrize("sim", SIM_TYPES[:3], ids=SIM_IDS[:3])
    def test_gradcheck_2d(self, sim):
        torch.manual_seed(2)
        B, C, N1, N2, W = 1, 2, 4, 4, 3
        x = torch.randn(B, C, N1, N2, dtype=torch.float64, requires_grad=True)
        y = torch.randn(B, C, N1, N2, dtype=torch.float64, requires_grad=True)

        def func(x, y):
            S = circulant_similarity(sim, x, y, W)
            return S.values

        assert torch.autograd.gradcheck(func, (x, y), eps=1e-6, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
