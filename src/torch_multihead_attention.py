import torch
import torch.nn as nn
from torch_attention import CirculantAttention
from torch_similarity import DotSimilarity, DistanceSimilarity

class MultiHeadCirculantAttention(nn.Module):
    """
    Multi-Head Circulant Attention Module.

    Args:
        window_size (int): The size of the sliding window. Must be odd.
        num_heads (int): The number of attention heads.
        embedding_dim (int): The embedding dimension of the input.
        similarity (nn.Module): The similarity function to use.
    """
    def __init__(self, window_size, num_heads, embedding_dim, similarity=DotSimilarity()):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads

        assert self.head_dim * num_heads == self.embedding_dim, "embedding_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        self.attention = CirculantAttention(window_size, similarity)

        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        """
        Forward pass for Multi-Head Circulant Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """

        b, n, c = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, N, D)
        k = k.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = q.flatten(0, 1) # (B*H, N, D)
        k = k.flatten(0, 1)
        v = v.flatten(0, 1)

        # Permute to (B*H, D, N) for CirculantAttention
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)

        y, _ = self.attention(q, k, v) # (B*H, D, N)

        y = y.view(b, self.num_heads, self.head_dim, n).permute(0, 3, 1, 2).reshape(b, n, c)

        return self.out_proj(y)

if __name__ == '__main__':
    # Test for Multi-Head Circulant Attention
    print("Testing Multi-Head Circulant Attention...")
    b, n, c = 2, 16, 32
    ws = 5
    num_heads = 4

    x = torch.randn(b, n, c)

    attn = MultiHeadCirculantAttention(window_size=ws, num_heads=num_heads, embedding_dim=c)
    y = attn(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
