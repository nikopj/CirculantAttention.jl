```@meta
CurrentModule = CirculantAttention
```

# CirculantAttention

Circulant-sparse attention (CircAtt) performs query-key-value attention with a sliding-window mask,

```math
Y = \mathrm{rowsoftmax}(M \circ S / \sqrt{\tau})\, V, \qquad S_{ij} = s(q_i, k_j),
```

where ``s`` is a similarity between the channel vectors of queries and keys, ``\tau`` is
the number of channels, and ``M`` is a mask with a circulant (1D signals) or
block-circulant with circulant blocks (2D signals) sparsity pattern: every pixel attends to
the pixels in a window of odd width centred on it, with circular boundary extension.

The package is CUDA only. Attention matrices are either materialized as sparse
[`Circulant`](@ref) arrays or fused away by the flash functions, in the forward and the
backward pass. Everything is differentiable with ChainRules/Zygote, including for
complex-valued tensors. Window sizes must be odd.

## Quickstart

```julia
using CUDA, NNlib, CirculantAttention

H, W, C, B = 128, 96, 64, 2
ws, nheads = 15, 4
q, k, v = CUDA.randn(H, W, C, B), CUDA.randn(H, W, C, B), CUDA.randn(H, W, C, B)

y, A = circulant_attention(DistanceSimilarity(), q, k, v, ws)  # output and adjacency
size(A)  # (H*W, H*W, 1, B)

y, A = circulant_mh_attention(DotSimilarity(), q, k, v, ws, nheads)
size(A)  # (H*W, H*W, nheads, B)
```

[`circulant_attention`](@ref) scales `q` and `k` by `τ^(-1/4)` each, computes the
[`circulant_similarity`](@ref), normalizes each row, and applies the result with `⊗`:
```julia
t = sqrt(size(q, 3))
S = circulant_similarity(DistanceSimilarity(), q ./ sqrt(t), k ./ sqrt(t), ws)
P = NNlib.softmax(S)
P ⊗ v ≈ first(circulant_attention(DistanceSimilarity(), q, k, v, ws))  # true
```

## Similarities

| similarity                       | ``S_{ij}``                                                              |
|:---------------------------------|:------------------------------------------------------------------------|
| [`DotSimilarity`](@ref)          | ``q_i^H k_j``                                                           |
| [`RealDotSimilarity`](@ref)      | ``\mathrm{Re}(q_i^H k_j)``                                              |
| [`DistanceSimilarity`](@ref)     | ``-\frac{1}{2}\lVert q_i - k_j \rVert_2^2``                             |
| [`PIDotSimilarity`](@ref)        | ``\lvert q_i^H k_j \rvert``                                             |
| [`PIDistanceSimilarity`](@ref)   | ``-\frac{1}{2}(\lVert q_i \rVert_2^2 - 2 \lvert q_i^H k_j \rvert + \lVert k_j \rVert_2^2)`` |

## Normalizations

Wrapping a similarity replaces the row softmax by [`sparsemax`](@ref)
([`SparsemaxSimilarity`](@ref)), α-[`entmax`](@ref) ([`EntmaxSimilarity`](@ref)), or a
softmax over the `k` largest entries ([`TopKSimilarity`](@ref)):
```julia
y, A = circulant_attention(EntmaxSimilarity(DistanceSimilarity(), 1.5f0), q, k, v, ws)
```

[`joint_softmax`](@ref), [`joint_sparsemax`](@ref), and [`joint_entmax`](@ref) normalize
each row over the union of the windows of several matrices, e.g. a signal attending to
itself and to a guide signal `g` with values `vg`:
```julia
S1 = circulant_similarity(DotSimilarity(), q, k, 5)
S2 = circulant_similarity(DotSimilarity(), q, g, 9)
P1, P2 = joint_softmax(S1, S2)   # rows of P1 and P2 together sum to one
y = P1 ⊗ v + P2 ⊗ vg
```

## Flash attention

The flash functions fuse similarity, normalization, and application into single kernels
with an online softmax, so the circulant-sparse matrix is never materialized and memory
traffic drops from ``O(NW^d)`` to ``O(NC)``. Their backward passes recompute the
attention weights from the logsumexp saved in the forward pass.

| function                                                | computes                                                          |
|:--------------------------------------------------------|:------------------------------------------------------------------|
| [`circulant_flash_attention`](@ref)                     | [`circulant_attention`](@ref) without the attention matrix         |
| [`circulant_mh_flash_attention`](@ref)                  | multi-head version                                                |
| [`circulant_flash_joint_attention`](@ref)               | several branches under a joint normalization                      |
| [`circulant_mh_flash_joint_attention`](@ref)            | multi-head version                                                |
| [`circulant_mh_flash_guided_joint_attention`](@ref)     | a self branch plus guide branches sharing its query                |
| [`circulant_flash_transposed_attention`](@ref)          | the transposed apply ``\Gamma^\top x``                             |
| [`circulant_mh_flash_transposed_attention`](@ref)       | multi-head version                                                |

They accept the real-valued similarities ([`DotSimilarity`](@ref) on real inputs only)
and, through fused α-entmax kernels following AdaSplash, [`SparsemaxSimilarity`](@ref)
and [`EntmaxSimilarity`](@ref). [`TopKSimilarity`](@ref) cannot be fused.

## Circulant matrices

[`Circulant`](@ref) matrices of size `(HW, HW, nheads, B)` support `+`, `-`, broadcasting
over their values (e.g. `A + m .* B` with `m` of size `(1, 1, nheads, B)`),
`sum(A; dims)`, `cat(A, B; dims=3)`, and multi-head application `A ⨷ x`.
[`windowview`](@ref) exposes their values as a dense `(ws^d, HW, nheads, B)` array.

## Citing

Circulant attention was introduced in *GroupCDL: Interpretable Denoising and Compressed
Sensing MRI via Learned Group-Sparsity and Circulant Attention*, IEEE Transactions on
Computational Imaging 2025 ([doi:10.1109/TCI.2025.3539021](https://doi.org/10.1109/TCI.2025.3539021),
[arXiv:2407.18967](https://arxiv.org/abs/2407.18967)). The guided joint attention and the
phase-invariant similarities are used in *LGGS: Longitudinal MRI Reconstruction via Guided
Group Sparsity* ([arXiv:2610.22643](https://arxiv.org/abs/2610.22643)). BibTeX entries
are in the [README](https://github.com/nikopj/CirculantAttention.jl#publications).
