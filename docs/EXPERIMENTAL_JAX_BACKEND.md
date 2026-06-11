# Experimental JAX Backend Backup

GROND's production Bloch backend is SciPy. Keep `--compute-backend scipy` as the default and treat all JAX paths as experimental until a future JAX/Metal implementation can run the required complex eigensolvers correctly and faster than SciPy.

## Current Status

The backend options are:

- `scipy`: production reference and default.
- `jax_cpu`: experimental, mathematically matches SciPy on benchmark rows, but is slower in the current row-by-row kernel.
- `jax_cuda`: experimental CUDA path. It uses JAX complex128 and `pmap` over visible CUDA devices for same-size Bloch matrix batches. It was correct on an internal three-GPU RTX 2080 Ti test, but slower than optimized SciPy CPU parallelism for the tested N128/Q20 workload.
- `jax_cpu_padded`, `jax_cuda_padded`, and the other `*_padded` variants: experimental bucketed-shape paths motivated by Takanori Nakane's suggestion to keep matrices the same size where possible. They pad matrices to small size buckets rather than padding every row to the worst-case basis size.
- `jax_cpu_real_expm`, `jax_mps_real_expm`, and `jax_mps_real_expm_padded`: guarded experimental real-embedded matrix-exponential paths. They rewrite the complex propagation matrix as a real block matrix and evaluate a fixed-order scaling/squaring Taylor exponential, which avoids asking Apple MPS for unsupported complex eigensolvers. The MPS route is limited to float32 by current JAX/MLX behavior, so it is an agreement and future-proofing probe rather than a production backend.
- `jax`, `jax_auto`, `jax_mps`: experimental. On the tested Apple Silicon/JAX-MPS environment, MPS saw the GPU but failed the required complex `eigh`/`eig` operations, so the ordinary complex-eigensolver MPS backend is not usable for GROND.

GROND intentionally raises a backend error if a JAX backend cannot run the complex Hermitian and non-Hermitian eigensolvers. It must not silently fall back to SciPy, because that would make timing comparisons meaningless.

## Benchmark Snapshot

N20/Q10, 200 selected benchmark rows, no absorption, one process:

| backend | wall time | row rate | max relative `dyn_raw` difference from SciPy |
| --- | ---: | ---: | ---: |
| SciPy | 3.13 s | 82.08 rows/s | reference |
| JAX CPU | 5.62 s | 41.11 rows/s | 1.96e-13 |

Bounded high-level smoke:

`GROND auto --full-fast --compute-backend jax_cpu` completed successfully with a bounded 200-row smoke in 487.24 s. This confirmed that the high-level workflow can route the backend through transform scouting, nuisance gates, absorption, structure update, final rows, forensics, and publication tables.

## Reactivation Criteria

Before promoting any JAX/Metal path, rerun a real agreement benchmark and require:

- Complex Hermitian `eigh` works on the intended device.
- Complex non-Hermitian `eig` works on the intended device.
- `dyn_raw`, `born_raw`, basis size, and row metadata match SciPy within numerical tolerance.
- GROND row/MR `Robs`, `Rall`, `wRall`, `MRobs`, `MRall`, and `MwRall` match SciPy.
- Corrected HKL files and maps match SciPy to acceptable crystallographic tolerance.
- End-to-end wall time beats SciPy on realistic row counts, not just a synthetic eigensolver microbenchmark.

The likely useful future path is batched same-size Bloch matrices, not simply replacing each small per-row SciPy eigensolver call with a separate JAX call.

## Apple MPS Real-Expm Snapshot

This backend exists because Apple MPS currently exposes useful matrix multiplication but not the complex eigensolver path GROND needs for the reference Bloch calculation. GROND therefore embeds a complex Hamiltonian `H = A + iB` into the real matrix

```text
[[Re(iH), -Im(iH)],
 [Im(iH),  Re(iH)]]
```

with the `2*pi*thickness` factor applied, then evaluates the first propagated column with a fixed Taylor polynomial plus scaling and squaring. This is mathematically close to `scipy.linalg.expm` on small Hermitian test cases, but it is not the same numerical method as the SciPy eigensolver reference and on MPS it runs in float32.

On the local 370 benchmark with N24/Q10 and 64 rows, `jax_mps_real_expm` matched the displayed residuals but took 2.07 s wall time versus 2.02 s for dense SciPy and 1.19 s for `scipy_bethe`. On a harder N108/Q20, 8-row check, it matched row intensities to roughly `2e-4` maximum relative `dyn_raw` difference but again lost on wall time. It should therefore remain opt-in and guarded. Promotion would require float64 support or a substantially faster same-size batched implementation that beats SciPy on full 370-style runs.

## Bucketed Matrix Snapshot

The padded JAX path was tested after the vectorized Hamiltonian work. Padding every row to the largest matrix size was not useful on CPU, but small buckets were. On a production-like N108/Q20 Bethe-Schur matrix set, exact JAX grouped 45 separate matrix sizes; `jax_cpu_padded` with bucket size 8 reduced that to 7 compiled shapes. In the hot loop this changed the JAX timing from about 1.51 s to 1.35 s for the benchmark batch, and a 32-row end-to-end smoke matched SciPy with maximum `Idyn` difference at numerical noise. This is a real implementation win for the experimental JAX path, but it is still not enough to replace SciPy as the default until whole-workflow timings improve.

## CUDA Snapshot

Hardware and software probe:

- 3 x NVIDIA GeForce RTX 2080 Ti, 11 GB each.
- Python 3.13.13, JAX 0.10.1, jaxlib 0.10.1.
- JAX default backend `gpu`; devices `cuda:0`, `cuda:1`, `cuda:2`.
- Complex128 `eig` and `eigh` both work.

The N128/Q20 direct Bloch benchmark used 4,816 selected rows, `l,-k,h`, no absorption, `thickness=1200 A`, `thickness_grad_frac=1.0`, `thickness_points=5`, `sg_window=0.004`, and `d_min=0.70 A`.

| backend | workers/devices | wall time | row rate | status |
| --- | ---: | ---: | ---: | --- |
| SciPy, vectorized Hamiltonian | 1 CPU worker | 122.17 s | 39.4 rows/s | reference |
| SciPy, vectorized Hamiltonian | 10 CPU workers | 15.03 s | 320.4 rows/s | production winner |
| JAX CUDA batched `pmap` | 3 GPUs | 121.45 s | 39.7 rows/s | correct but not promoted |

The JAX full-run R factors matched the SciPy full-run endpoint to displayed precision. On the bounded 96-row N128/Q20 agreement check, JAX matched SciPy with maximum relative `dyn_raw` difference `2.04e-13`, identical `born_raw`, identical basis sizes, and maximum relative `dyn_scaled` difference `2.10e-13`.

The largest performance win in this test was not CUDA; it was vectorizing Hamiltonian construction and diagonalizing only once per profile point before applying thickness quadrature. That changed the 96-row N128/Q20 SciPy single-worker wall time from 27.90 s to 4.10 s with identical row output. Keep SciPy CPU parallelism as the default until a future accelerator moves more of row setup, basis selection, Hamiltonian construction, and same-size batching onto the accelerator.
