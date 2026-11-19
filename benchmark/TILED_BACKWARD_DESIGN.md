# Tiled circulant flash backward — design note

Status: **design only** (no implementation). Target: a new **`:tiled`** backward
mode alongside the existing `:thread` / `:warp` / `:block`, opt-in, promoted into
auto-selection only for the (window, tensor) regimes where it is measured faster.
The existing kernels stay as the correctness oracle and the default everywhere else.

## Why (from the profiling pass)

The backward kernel (softmax, W=5, 128×128×64×2) is **1020 µs, 3.8× the forward**,
and we established by elimination it is **compute/instruction-bound**, not memory- or
occupancy-bound:
- occupancy sweep: forcing 62.5–100% occupancy only slowed it (spills) → not latency-bound;
- DRAM traffic (working set ≈ 33 MB fits the 40 MB L2): ~59 MB → ~58 GB/s → **~3% of peak** → not DRAM-bound;
- L2 logical traffic ~872 MB → ~855 GB/s → **~17% of L2** → not L2-bandwidth-bound.

The dominant cost is **redundant compute**: the two sweeps each recompute *every*
pair's similarity, `g = Re⟨Δ,v⟩`, `P`, and `ds`. Concretely, pair `(r,i)` is
computed by row-`r`'s ∂q sweep **and** by column-`i`'s ∂k/∂v sweep — the expensive
`C`- and `Cv`-wide reductions run **twice per pair**, and so does `P = exp(scale·s−L)`.
Computing each pair once and using it for both directions ≈ **halves the reduction FLOPs
AND the transcendental (`exp`) count** (the per-channel ∂q/∂k/∂v accumulations are
unchanged, so expect ~1.3–1.7× on the whole backward, not a full 2×).

## Production-window confirmation (W=35, complex, 2026-07)

The W=5 finding was re-checked at the GroupCDL single-grid window (`block` bwd, W=35,
128²·C64·B4, ComplexF32/PIDistance = **98 ms**). ncu was unavailable (cluster
`ERR_NVGPUCTRPERM` — GPU counters are admin-restricted), but a counter-free roofline
from the measured time + known array sizes is unambiguous:

- **DRAM:** working set ≈ 176 MB streamed once ⇒ 1.9 GB/s = **0.10 % of the 1935 GB/s
  peak**. It would take ~1025× re-reads of the whole working set to be DRAM-bound ⇒
  **not memory-bound.**
- **FP32 compute:** ≈ 1.7 TFLOP/s = **8.6 % of the 19.5 TFLOP/s peak** ⇒ far from the
  throughput roofline.
- **Transcendentals:** ≈ **80 M `exp()`** on the SFU — the real issue-rate limiter.

So at the production window the kernel is **latency/issue-bound** (serialized dependent
reductions + SFU `exp`), the *same* regime as W=5. Recompute-once halves *both* the
reductions and the `exp` count — the two roofline-identified bottlenecks — so the lever
holds at W=35, not just the small-window cases the original note measured.

## Core scheme — "compute `ds` once, apply twice" (no atomics)

A block owns a **core tile** of `T` output positions plus a **window-radius halo**
`p = (W-1)/2` on each side (`loaded = core + halo`). Three phases, all reading q/k/v
from global/L2 (which has plenty of headroom — no need to stage inputs in shared):

1. **Compute + stash `ds`.** Threads sweep the pairs `(r,i)` with `r ∈ loaded`,
   `i ∈ window(r)` and compute `s_{ri}` (one `C`-reduction), `P_{ri}=exp(scale·s−L_r)`,
   `g_{ri}` (one `Cv`-reduction), `ds_{ri}=P·(g−δ_r)` — **once per pair** — and store
   `(ds, P)` in **shared memory**, indexed by the circulant `(r,κ)` layout. `δ_r` is the
   dense host reduction already computed (`δ = Re⟨Δ,y⟩`).
2. **∂q.** Threads own core rows `r`; read `ds_{r,·}` from shared over `window(r)`;
   accumulate `∂q[r] += scale·ds·(α k_i + β q_r)`.
3. **∂k, ∂v.** Threads own core columns `a`; read `ds_{r,a}`, `P_{r,a}` from shared
   for `r ∈ window(a)` (a transpose access, located via `cartesian_circulant` exactly
   as the current sweep-2 does); accumulate `∂k[a] += scale·ds·(conj(α) q_r + β k_a)`,
   `∂v[a] += P·Δ_r`.

Each **core** position's ∂q/∂k/∂v is produced by exactly one block ⇒ **written once,
no atomics**. Redundancy is confined to **boundary pairs** (a pair with an endpoint in
another block's core is recomputed by that neighbor) — fraction ≈ perimeter/area of the
tile, small for large tiles. Interior pairs' reductions run once (the win).

Key property: the shared stash is **independent of `C`** — it scales with *window pairs*,
not channels. Its width per pair is similarity-dependent (**CPU-validated**,
`benchmark/tiled_bwd_validate.jl`):
- **Dot / RealDot / Distance:** `(ds, P)` — 2 floats/pair (α ≡ 1, no need to stash it).
- **PIDot / PIDistance (GroupCDL's choice):** `(ds, P, α)` — **4 floats/pair**. `α = sign(Σ
  qₘ k̄ₘ)` is a per-pair *complex* factor that both ∂q and ∂k need, and recomputing it
  requires the expensive C-reduction we are trying to avoid — so it must be stashed.

⇒ the capacity table below is for the 2-float (α≡1) case; **double every stash figure for
PIDistance-complex.** e.g. the 2-D W=5 8×8-tile case is ~58 KB (not 29 KB) and therefore
does *not* fit the 48 KB default — window-tiling or a smaller core tile is needed even at
W=5 for the GroupCDL similarity.

## The binding constraint: shared capacity vs window size

Shared stash size ≈ `loaded_positions × W^d × 2 floats`. This is fine for small/moderate
windows and murder for large ones:

| case | tile | stash |
|---|---|---|
| 1-D, W=5, T=64 | 64+halo | ~1.5 KB |
| 2-D, W=5, 8×8 tile | (8+4)² px × 25 | ~29 KB (fits 48 KB default) |
| 2-D, W=15, 8×8 tile | (8+14)² px × 225 | **~870 KB (does not fit)** |

So the clean no-atomics scheme is viable for **small-to-moderate windows** and forces
tiny tiles (→ high boundary redundancy) for large ones. This is exactly why we expect
**`:tiled` to win only in a sub-region of (window, size) space**, not universally —
which is the whole reason it's a separate, benchmark-gated mode.

**GroupCDL regime (with the 4-float PIDistance stash, 2-D):** the shared-stash form fits
only tiny windows — W=3 (16×16 core), W=5 (~7×7 core), and is **infeasible by W=9** (K=81
needs ≤37 loaded positions but the halo alone is 8 wide → no positive core). So **neither**
GroupCDL window (MG W=9, single-grid W=35) fits the shared-stash kernel. Recompute-once is
still the right lever there (roofline: issue-bound), but the *write mechanism* must be
atomic-scatter or window-tiling — the shared-stash is reserved for the toy-window regime /
1-D. Concretely:
- **W=9 and W=35 (both GroupCDL windows):** stash ≫ 48 KB ⇒ realize recompute-once via
  one of:

Options for the large-window regime (defer until the small-window case is proven):
- **Window tiling:** process `window(r)` in chunks, streaming the stash — bounds shared
  at the cost of re-reading q/k across chunks (cheap; L2 has headroom). This is the
  standard block-sparse primitive (FlashAttention block-sparse mode / Longformer banded
  matmul / mit-han-lab Block-Sparse-Attention all tile the window this way).
- **Atomic scatter** — **IMPLEMENTED as `mode=:scatter`** (thread-per-row; ∂q single-writer
  so it accumulates in place, ∂k/∂v `atomicAdd`-scattered; complex targets reinterpreted to
  real for the atomics). Compute each pair once, window-size-agnostic (fits W=35). This is the
  mirror of FlashAttention's dQ-atomic problem: because we parallelize over *queries*, it is
  ∂k/∂v (not ∂q) that is shared across blocks. Schedule + reinterpret-indexing CPU-validated
  bit-identical to the two-sweep bwd (`benchmark/tiled_bwd_validate.jl`); GPU correctness gate
  added to the modes-agree testset. **Open question the benchmark answers:** does removing the
  2× recompute pay for the `~K`-way `atomicAdd` contention on ∂k/∂v? Note it's thread-granular,
  so it won't beat the `:block` kernel outright — the clean read is `:scatter` vs the two-sweep
  `:thread` (same parallelism, isolates atomics-vs-redundancy). If atomics win there, a
  block-level scatter is the follow-up; if not, window-tiling is the path.

## SOTA cross-check (flash-attention literature, 2026-07)

A survey of FA-1/FA-2/FA-4, the Triton reference, and block-sparse kernels confirms this
note's approach is the state of the art, and adds three refinements:
- **We already match SOTA on:** recompute S,P from the stored logsumexp (never store P);
  precompute `δ = rowsum(Δ∘y)` separately (FA's `_bwd_preprocess`); fused `ds = P·(g−δ)`;
  and no-atomics via output partitioning (FA-2's two-kernel split has the same goal).
  FA's "reduce non-matmul FLOPs" is about *tensor-core* kernels; ours is all-scalar FP32,
  so its scalar analogue is precisely recompute-once (fewer reduction + `exp` ops).
- **Transposed stash layout (from FA-4/CUTLASS):** store `(ds,P)` in the layout the ∂k/∂v
  phase reads it (the `cartesian_circulant` transpose), so phase-3 access is bank-conflict
  free — directly retires the "shared-bank conflicts in the ∂k phase" open risk below.
- **Register pressure = combinatorial over ~10 operands** (q,k,v,O,L,dO,dv,dk,dq,ds), each
  cache-vs-spill: this is the lens for our warp→block crossover. Triton's d=128 win came
  from *casting `ds` to the narrow input dtype* to relieve exactly this — cheap to try.
- **Two-kernel split vs single-kernel two-sweep:** separate ∂q and ∂k/∂v kernels each hold
  fewer live accumulators (better occupancy) but re-read inputs; worth A/B-ing against the
  shared-stash single kernel once the stash version is correct.

## Where we expect wins / losses

- **Win:** compute-bound regimes — moderate windows, images that fit L2, high channel
  counts (reductions dominate, and halving them matters most).
- **Neutral/lose:** small windows (little redundant compute to reclaim; tile overhead
  dominates), or images larger than L2 (backward becomes DRAM-bound → tiling can't help),
  or large windows (shared capacity forces tiny tiles / atomics).

## Rollout (matches "prove before promote")

1. Implement **`:tiled`** for the **backward only** (forward is memory-bound — leave it).
   Add `:tiled` to `_flash_mode`'s explicit options; it is **never** chosen by `:auto`
   initially.
2. **CPU-validate:** the tiled schedule is generic over `AbstractArray`, so the
   3-phase thread-equivalent runs on CPU — validate `dq/dk/dv` vs the existing
   CPU-validated `_flash_entmax_bwd_row!` path / finite differences, as we did for the
   entmax kernels.
3. **Modes-agree GPU test:** `:tiled` must match `:thread` for a grid of window sizes
   (incl. the WS=16/32-triggering large 2-D windows) — reuse the harness from
   `test/flash.jl`'s new "sub-warp widths" testset.
4. **Benchmark sweep:** time `:tiled` vs `:warp`/`:block` across `(window ∈ {5,9,15},
   image ∈ {64², 128², 256²}, C ∈ {32,64,128})`. Produce a win/lose map.
5. **Promote conditionally:** teach `_flash_mode(:auto, …)` to pick `:tiled` **only** in
   the measured-faster region; everything else keeps the current kernel. If it never
   wins cleanly, it stays an opt-in `mode=:tiled` and the default is untouched.

## Open risks (call these out before committing kernel time)

- **Shared-bank conflicts / stash indexing** in the transpose (∂k) phase — the circulant
  `(r,κ)`→`(r,a)` mapping must land conflict-free or it stalls.
- **Boundary redundancy** eating the interior savings at the tile sizes shared capacity
  actually allows — the small-window 29 KB case has a healthy tile, but it's tight.
- **2-D BCCB halo** indexing is fiddlier than 1-D; get the 1-D `:tiled` correct + measured
  first, then generalize.
- Payoff is **~1.3–1.7×** on the backward *in its best regime*, i.e. maybe ~1.2–1.4× on
  end-to-end training step — worth it, but not transformative. Scope accordingly.

## Measured results (2026-07, A100, W=35 complex PIDistance 128²·C64·B4)

| mode | bwd kernel | note |
|---|---|---|
| `:thread` (two-sweep, ⌈C/CH⌉-recompute) | 1080 ms | baseline |
| `:scatter` (thread-level recompute-once, atomic ∂k/∂v) | **354 ms** | **3.0× over thread** — atomics win vs redundancy at thread occupancy |
| `:block` (two-sweep, shared-staged) | 96.9 ms | production |
| `:bscatter` (block-level recompute-once, atomic ∂k/∂v) | **100.4 ms** | **≈ block (no win)** |

**Conclusion — `:block` is near the structural floor; recompute-once has no headroom there.**
The 3× that `:scatter` won over `:thread` was from removing the thread kernel's `⌈C/CH⌉`×2
similarity recompute — but `:block` *already* removes `⌈C/CH⌉` via shared staging. The only
redundancy left for `:bscatter` is the cross-sweep 2×, which is a *small fraction* of the
block kernel's cost (dominated by the inherent channel×window ∂q/∂k/∂v accumulation), and
halving it is offset by **block-level atomic contention** (~K concurrent writers per output
element at high occupancy — far worse than at thread occupancy). So the atomic-scatter write
mechanism does not scale to block parallelism, exactly as the "measure before trusting"
warning above anticipated. Remaining levers are NOT kernel micro-opts: modeling (Mh/nheads/
window — `benchmark/group_hparam_sweep.jl`) or a matmul/tensor-core reformulation. `:scatter`
and `:bscatter` remain as validated, non-default modes (`:scatter` is a 3× `:thread` fallback).
Window-tiling (below) is the only untried recompute-once mechanism, but the atomic result caps
the expected upside — the block kernel's cost is accumulation-bound, not recompute-bound.

## Recommendation

Prototype the **1-D, small/moderate-window, no-atomics** variant first (smallest,
CPU-validatable, cleanest correctness story). Measure it against `:warp` on a
window/size sweep. Only if it shows a clear win in some regime do we (a) generalize to
2-D and (b) tackle the large-window flavor. This keeps the working code untouched and
spends kernel-writing effort only after the first measurement confirms the compute-bound
inference holds up in practice.
