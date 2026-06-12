# Full 370 GROND Backend Benchmark Summary

Date: 2026-06-09

Input data:

- `INTEGRATE.HKL`: `<TYROSINE_370_XDS_NO_PRF>/INTEGRATE.HKL`
- model: `<TYROSINE_370_XDS_NO_PRF>/temp_a.res`
- rows: 5094
- rocking samples: Q20
- thickness points: 7
- absorption mode: `midgley_like`, `absorption_frac=0.05`

`N128` in the normal GROND support contract means a beam cap of 128, not that every Hamiltonian is 128 by 128. For this 370 dataset with `max_support_sg=0.005`, the actual basis sizes were median 44 and max 81. The literal forced-128 benchmark used `max_support_sg=0.05` so every row reached a 128 by 128 Hamiltonian.

## Normal GROND Support, N128 Cap

| backend | machine | workers | wall time | basis median/max | note |
|---|---:|---:|---:|---:|---|
| SciPy dense eig/eigh | laptop | 14 | 17.395 s | 44 / 81 | production-valid |
| SciPy dense eig/eigh | cc02 | 64 | 5.792 s | 44 / 81 | production-valid, best measured |
| SciPy Bethe/Schur, strong=48 | laptop | 14 | 17.261 s | 44 / 81 | no real speed gain because most matrices are already near or below 48 beams |
| SciPy Bethe/Schur, strong=48 | cc02 | 64 | 5.821 s | 44 / 81 | essentially tied with dense SciPy |
| JAX CUDA padded eig/eigh | cc02 | 1 batch engine | 418.921 s | 44 / 81 | not viable for this workload |
| MLX Metal real-expm | laptop | GPU batch | 31.301 s | 44 / 81 | parity p95 relative error 1.76e-4, but max relative error fails strict cutoff on near-zero rows and total time loses to SciPy CPU |

## Literal Forced 128x128 Matrices

| backend | machine | workers | wall time | basis median/max | note |
|---|---:|---:|---:|---:|---|
| SciPy dense eig/eigh | laptop | 10 | 234.338 s | 128 / 128 | production-valid but expensive |
| SciPy dense eig/eigh | cc02 | 64 | 37.261 s | 128 / 128 | production-valid |
| SciPy Bethe/Schur, strong=48 | laptop | 10 | 52.542 s | 128 / 128 | strong speedup over dense forced-128 |
| SciPy Bethe/Schur, strong=48 | cc02 | 64 | 9.100 s | 128 / 128 | fastest forced-128 result |
| MLX Metal real-expm | laptop | GPU batch 64 | 163.204 s | 128 / 128 | faster than local dense SciPy, slower than local Bethe and much slower than cc02 Bethe |

## Main Conclusions

For the real 370 full simulation under the normal GROND support contract, cc02 SciPy CPU with about 64 workers is the best measured production path. It is about 3.0 times faster than the laptop, not 10 times faster.

The Bethe/Schur strong/weak split is not useful for the normal 370 support contract because most matrices are already small. It becomes very useful when the calculation is artificially forced to full 128 by 128 matrices.

MLX Metal real-expm is real and can run on Apple GPU cores, but it is not the winning full-data path. It loses to local SciPy CPU under normal support and loses badly to cc02 Bethe/Schur for forced full matrices.

JAX CUDA padded eig/eigh on cc02 is not viable for this workload in the current implementation. The full normal-support run took 418.921 seconds.

The best practical defaults suggested by these benchmarks are:

- normal production: SciPy dense, adaptive support, Q20, cc02 with about 64 workers;
- forced high-N diagnostic: SciPy Bethe/Schur, strong=48, cc02 with about 64 workers;
- laptop fallback: SciPy dense for normal support, SciPy Bethe/Schur for forced high-N diagnostics;
- do not use JAX CUDA or MLX as default production paths yet.
