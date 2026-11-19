"""CirculantAttention — PyTorch implementation of circulant sparse attention."""

from .circulant import Circulant, build_circulant
from .similarity import (
    DotSimilarity,
    RealDotSimilarity,
    DistanceSimilarity,
    PIDotSimilarity,
    PIDistanceSimilarity,
    circulant_similarity,
)
from .attention import (
    circulant_attention,
    circulant_adjacency,
    circulant_softmax,
    circulant_matmul,
    circulant_mh_attention,
    circulant_mh_adjacency,
    circulant_mh_similarity,
)
