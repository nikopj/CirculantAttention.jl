```@meta
CurrentModule = CirculantAttention
```

# API

```@docs
CirculantAttention
```

## Attention

```@docs
circulant_attention
circulant_mh_attention
circulant_similarity
circulant_similarity!
circulant_adjacency
circulant_adjacency!
circulant_mh_adjacency
```

## Flash attention

```@docs
circulant_flash_attention
circulant_mh_flash_attention
circulant_flash_joint_attention
circulant_mh_flash_joint_attention
circulant_mh_flash_guided_joint_attention
circulant_flash_transposed_attention
circulant_mh_flash_transposed_attention
```

## Similarities

```@docs
DotSimilarity
RealDotSimilarity
DistanceSimilarity
PIDotSimilarity
PIDistanceSimilarity
TopKSimilarity
SparsemaxSimilarity
EntmaxSimilarity
```

## Normalizations

```@docs
sparsemax
entmax
joint_softmax
joint_sparsemax
joint_entmax
```

## Circulant matrices

```@docs
Circulant
circulant
windowview
kernel_length
spatial_dims
spatial_size
```
