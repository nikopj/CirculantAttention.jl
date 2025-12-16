import torch
import torch.nn as nn
from torch_circulant import circulant
from torch_similarity import DotSimilarity, DistanceSimilarity
class CirculantAttention(nn.Module):
    """
    Circulant Attention Module.

    Args:
        window_size (int): The size of the sliding window. Must be odd.
        similarity (nn.Module): The similarity function to use.
    """
    def __init__(self, window_size, similarity=DotSimilarity()):
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd.")
        self.window_size = window_size
        self.similarity = similarity

    def forward(self, q, k, v):
        """
        Forward pass for Circulant Attention.

        Args:
            q (torch.Tensor): Query tensor of shape (B, C, H, W) or (B, C, N).
            k (torch.Tensor): Key tensor of shape (B, C, H, W) or (B, C, N).
            v (torch.Tensor): Value tensor of shape (B, C, H, W) or (B, C, N).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
            torch.Tensor: Attention matrix of shape (B, N, N).
        """

        shape = q.shape
        is_2d = q.dim() == 4

        if is_2d:
            b, c, h, w = shape
            n = h * w
            q = q.view(b, c, n)
            k = k.view(b, c, n)
            v = v.view(b, c, n)
        else:
            b, c, n = shape

        # Permute to (B, N, C)
        q_perm = q.permute(0, 2, 1)
        k_perm = k.permute(0, 2, 1)
        v_perm = v.permute(0, 2, 1)

        # Scaling factor
        scale = torch.sqrt(torch.tensor(c, dtype=q.dtype, device=q.device))

        # Calculate similarity scores
        S = self.similarity(q_perm / scale, k_perm)

        # Create circulant mask
        mask = circulant(self.window_size, n, device=q.device) # (N, N)

        # Apply mask and compute attention
        S_masked = torch.where(mask.to_dense().bool(), S, -1e9)
        A = torch.softmax(S_masked, dim=-1) # (B, N, N)

        # Apply attention to v
        y = torch.bmm(A, v_perm) # (B, N, C)

        # Permute back to original shape
        y = y.permute(0, 2, 1) # (B, C, N)

        if is_2d:
            y = y.view(b, c, h, w)

        return y, A

if __name__ == '__main__':
    # Test for 1D signal
    print("Testing 1D...")
    b, c, n = 2, 4, 16
    ws = 5

    q = torch.randn(b, c, n)
    k = torch.randn(b, c, n)
    v = torch.randn(b, c, n)

    attn = CirculantAttention(window_size=ws)
    y, A = attn(q, k, v)

    print("Input shape:", q.shape)
    print("Output shape:", y.shape)
    print("Attention matrix shape:", A.shape)

    # Test for 2D signal
    print("\nTesting 2D...")
    b, c, h, w = 2, 4, 8, 8

    q = torch.randn(b, c, h, w)
    k = torch.randn(b, c, h, w)
    v = torch.randn(b, c, h, w)

    attn = CirculantAttention(window_size=ws)
    y, A = attn(q, k, v)

    print("Input shape:", q.shape)
    print("Output shape:", y.shape)
    print("Attention matrix shape:", A.shape)
    print("A nnz:", (A > 0).sum())

    # Test with DistanceSimilarity
    print("\nTesting with DistanceSimilarity...")
    attn = CirculantAttention(window_size=ws, similarity=DistanceSimilarity())
    y, A = attn(q, k, v)

    print("Input shape:", q.shape)
    print("Output shape:", y.shape)
    print("Attention matrix shape:", A.shape)
    print("A nnz:", (A > 0).sum())
