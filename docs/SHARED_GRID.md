# Shared orientation grid (large unit cells)

The per-row engine builds one Bloch Hamiltonian for every profile point of every observation: rows × points ×
mosaic tilts matrices, each N × N. For a large cell the matrices are big (SU-102: N ≈ 750 coupled beams at
|Sg| ≤ 0.01 Å⁻¹, 3,000 at 0.04) and the same orientation is solved again for every reflection excited near it.

The shared grid solves one Hamiltonian per orientation of a uniform scan-angle grid instead (one grid per transverse
mosaic tilt). From one eigendecomposition it takes the couplings V[t, j] conj(V[0, j]) of every target t whose rocking
window covers that orientation, evaluates their thickness tables, and maps the grid onto each row's own profile points
and weights with not-a-knot cubic-spline cardinal weights. Profile points, Jacobian weights, mosaic tilts, windows,
Born terms and the thickness evaluation are the per-row engine's own.

Module: `src/grond/bloch/shared_grid.py`. Hooks: `XDSBlochPort.predict_rows` (row predictions in every
`native_row_regularize` stage) and `legacy/engine/response_stage.compute_response_table` (nested selection and the
per-cycle thickness updates with `--nested-response-tables`).

## Switching it on in `grond auto` / `grond run`

```
grond auto ... --shared-grid auto [--shared-grid-gpus 0,1,2] [--basis-centring-coset on]
```

- `--shared-grid off|auto|force` sets `GROND_SHARED_GRID` once, at the start, for every stage of the run. Every stage
  inherits it: screens, nested selection tables, gates, feedback passes, thickness updates and the final recompute.
  Left unset, the environment's value is kept. The default is `off`, so a run without the option is unchanged.
- `auto` decides per prediction pass and per response table, from the numbers of reflections and beams (next
  section). Small passes, unsupported physics and small capped bases stay per reflection automatically, and the
  reason is printed (`Bloch evaluator for N rows: ...`).
- `--shared-grid-gpus` lists CUDA devices for prediction passes. They are used only when the grid basis has at least
  2,000 beams; below that, CPU cores are faster.
- `--basis-centring-coset on` drops the beams of other lattice-centring cosets, which are exactly decoupled. It is
  exact and a large saving for centred lattices (SU-102, R: N falls by 3).
- The settings in force are printed and recorded in `run_config.json` (`bloch_evaluator`).
- Prediction caches written while the grid could be used carry these settings in their key. Such a cache never
  serves a per-reflection request, and the reverse. With `off` the keys are unchanged.

## Settings (environment)

| variable | values | meaning |
|---|---|---|
| `GROND_SHARED_GRID` | `off` (default), `auto`, `force` | `auto` uses the grid when the configuration is supported and the cost model says it is at least 3x cheaper (`choose_path`); `force` whenever supported |
| `GROND_SHARED_GRID_PER_DEGREE` | default 100 | grid density for row predictions |
| `GROND_SHARED_GRID_TABLE_PER_DEGREE` | default as above | grid density for response tables (selection only) |
| `GROND_SHARED_GRID_PRECISION` | `double` (default), `single` | CPU grid solves for predictions |
| `GROND_SHARED_GRID_TABLE_PRECISION` | default as above | CPU grid solves for response tables |
| `GROND_SHARED_GRID_GPUS` | e.g. `0,1,2` | CUDA devices for prediction passes (JAX); empty = CPU |
| `GROND_SHARED_GRID_WORKERS_PER_GPU` | default 4 | GPU worker processes per device |
| `GROND_SHARED_GRID_GPU_PRECISION` | `single` (default), `double` | GPU eigensolver precision |
| `GROND_SHARED_GRID_CPU_WITH_GPU` | `0` (default) | also run CPU workers next to the GPUs |
| `GROND_SHARED_GRID_GPU_MIN_N` | default 2000 | use the GPUs only when a sampled grid matrix has at least this N (smaller matrices solve faster on CPU cores, in cache) |
| `GROND_SHARED_GRID_MIN_ROWS` | default 200 | smaller requests always run per row |
| `GROND_UG_BUILD_WORKERS` | default 1 | processes for the dense U_g cube build (fork platforms) |
| `GROND_BASIS_CENTRING_COSET` | `1` | keep only the direct beam's lattice-centring coset in the beam pool (exact) |

## Physics choices (what differs from the per-row engine)

- **Basis rule.** A grid Hamiltonian holds the beams with |Sg| ≤ `max_support_sg` at the grid orientation (ranked and
  capped by `n_basis` exactly as `select_basis`), plus every active target. The per-row engine uses the same rule at
  each row's centre for the whole rocking window. The two agree as the basis converges (SU-102: 1.3% median apart at
  |Sg| ≤ 0.01, 0.4% at 0.04). At an unconverged cutoff neither is right; check convergence in the cutoff.
- **Active targets** are always in a grid basis. When the beam pool stops short of the data resolution
  (`topup_d_min` above the rows' d), a target outside the pool is an extra beam the per-row engine leaves out; keep
  `topup_d_min` at or below the data resolution (the standard settings do).
- **Dose damage** is evaluated at the grid angle's frame, not at the row's centre frame.
- **Interpolation.** The grid value at each profile point is a cubic spline through the grid. Measured on SU-102 with a
  fixed basis: p95 0.26% at 100 per degree, 0.05% at 150, 0.03% at 200. With the grid's own basis at cutoffs ≥ 0.02 it
  reproduces per-row values with the same rule to median 0.05-0.08%, p95 0.24-0.34% at 100 per degree.
- **Precision.** CPU grid solves default to float64: one float32 MRRR (?stemr) failure was measured in 16,888 SU-102
  rows (128% on that row; float64 agrees to 1e-11 in either beam order). GPU solves use cuSOLVER divide and conquer;
  float32 agrees with float64 to median 2e-5, max 1.3e-4 per target (SU-102, N 1,500-3,050).

## Supported

Target-Sg profiles; mosaicity `none` or `transverse_target`; no target-Sg jitter; no absorption; the table thickness
evaluator for predictions (response tables always); scipy backends; XDS/DIALS rotation scans (not PETS zone files);
centrosymmetric (real) and non-centrosymmetric (complex Hermitian) models, both hands. Anything else falls back to the
per-row engine with a printed reason.

## When to use which path (the chooser)

`choose_path` compares the expected eigensolve work: per row, rows × points × tilts × N_row^2.5 (N capped by
`n_basis`); on the grid, span × density × tilts × (N_grid^2.5 × 2 for float64 + active targets × 126 N_grid for the
thickness tables), with N sampled at a few row centres. Rules of thumb behind it:

- **Large cells, many rows per degree, converged bases** (SU-102: 159 rows per degree, 400 active targets per grid
  point): the grid wins by 100-300x in eigensolves. Use it, with the coset pool when the lattice is centred.
- **Small cells, small capped bases, wide windows** (tyrosine: 64-beam cap, union 133): the grid gains little (1.4x) or
  loses; stay per row (the chooser does).
- **Few rows** (screens, gates, hand checks on subsets): per row.
- **Large N on CPUs is memory-bandwidth bound.** At N ≈ 3,000 (SU-102, |Sg| ≤ 0.04) 120 concurrent float64
  tridiagonalisations ran at 2.3 solves/s for the whole 128-core EPYC server (float32: 8.6/s); three RTX 2080 Ti ran
  6.2/s (float64) and 23/s (float32). Use GPUs for prediction passes at large N.
- **Lanczos from the direct beam** (Krylov depth m instead of a full eigendecomposition) was tested on the same
  matrices: m = 20-100 gave 35-260% median errors in the thickness-averaged intensity; m = 800 (2.5x faster than the
  eigensolver) still had p95 1-2%, because the gamma thickness tail to 1 µm needs a Krylov depth close to N. Not used.
- **Fewer rocking points after the thickness average** (Gauss-Legendre 7-30 nodes) was tested per row on SU-102:
  29-212% maximum errors; the thickness-averaged rocking curve is not smooth for a gamma shape of 0.35. The uniform
  rule stays; on the grid, profile points cost only cardinal weights anyway.

## Measured in production runs (SU-102, cc02: 128-core EPYC, 3x RTX 2080 Ti)

Both runs were full `auto` runs, with both arms (full pore and empty pore) side by side on one server. Settings:
nested selection over mosaic and dose, Q240, coset pool, `--shared-grid auto`, GPUs for prediction passes.

| | \|Sg\| ≤ 0.04 (N ≈ 3,000) | \|Sg\| ≤ 0.01 (N ≈ 800) |
|---|---|---|
| nested selection (20 variants) | 2.3 h | about 2 h |
| prediction pass (16,888 rows, 3 tilts, 32,600 grid points) | 72 min on the GPUs | minutes on CPU |
| whole run | 6.5 h | see the run records |
| chooser estimate (tables / passes) | 37x / 163x cheaper | 37-38x / grid |

The published per-row SU-102 runs (\|Sg\| ≤ 0.01, 1,000-beam cap, Q60, no mosaic) took 54 h.

Fe(II) (12 crystals, N ≈ 900 at 0.06): nested-selection tables were estimated 96x cheaper on the grid. Tyrosine
(64-beam cap, about 1.4x): stays per reflection.

## Diagnostics

Each pass prints the chosen path and reason, then `shared grid: <rows>, <tilts>, <grid points> at <density>/deg,
|Sg| <= ..., n_basis cap ..., <precision> ...` and a closing line with solves, N median/max and CPU split. Response
tables built on the grid carry `evaluator: {shared_grid: true, ...}` in their identity, so they never mix with per-row
tables in a cache.
