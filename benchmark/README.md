# Benchmarks and profiling

All scripts need an NVIDIA GPU and run from the repository root, e.g.
`julia --project benchmark/benchmark.jl`.

Benchmark suite:

- `benchmark.jl` times the materialized and fused (flash) paths, forward and gradient, with
  distance similarity and softmax/entmax/sparsemax normalization (single-head, multi-head,
  joint, and guided), over window sizes 5–45, and appends the results to
  `benchmark_results.csv`.
- `plot.jl` draws the `*.png` figures from `benchmark_results.csv`.
- `benchmark_matvec.jl`, `matvec.jl`: circulant matrix-vector product kernels.

Profiling and diagnostics of the flash kernels:

- `kernel_info.jl`: registers, shared memory, and occupancy of the forward/backward kernels.
- `profile_flash.jl`: isolates one kernel launch for Nsight Compute (`ncu`).
- `nsys_flash.jl`: per-kernel timing table of the forward and backward passes.
- `guided_profile.jl`: breakdown of the guided joint attention backward.
- `bwd_occupancy_sweep.jl`: backward kernel speed versus occupancy.
- `check_warp_ws.jl`: correctness of the warp/block kernels at every sub-warp width.
- `tiled_bwd_validate.jl`, `TILED_BACKWARD_DESIGN.md`: CPU oracle and design note for a
  recompute-once (tiled) backward, not yet implemented.
- `diagnose_similarity.jl`, `toolchain_regression_mwe.jl`, `run_regression_matrix.sh`:
  the similarity-kernel slowdown between Julia 1.11/LLVM 16 and Julia 1.12/LLVM 18, worked
  around in `src/similarity.jl`.
