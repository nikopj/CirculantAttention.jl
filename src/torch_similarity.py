import torch

class DotSimilarity(torch.nn.Module):
    """
    Computes similarity as the dot product of q and k.
    """
    def forward(self, q, k):
        return torch.bmm(q, k.transpose(1, 2))

class DistanceSimilarity(torch.nn.Module):
    """
    Computes similarity as the negative L2 distance between q and k.
    """
    def forward(self, q, k):
        # Broadcasting to compute pairwise distances
        return -torch.sum((q.unsqueeze(2) - k.unsqueeze(1))**2, dim=-1)
