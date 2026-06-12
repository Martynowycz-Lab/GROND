# Artifact Requirements

Every manuscript number needs a source. This file defines the minimum artifacts each workflow must produce before it can support a paper claim.

## Input Row Contract

For every dataset, preserve the raw integration route and the GROND row contract.

| Artifact | Required content |
|---|---|
| Integration source | XDS or DIALS folder, integration settings, summation/profile policy, gain/offset notes, and resolution limit used for GROND. |
| Starting model | Starting `.res` or `.cif`, source of solution, space group, cell, formula, and whether hydrogens were present. |
| Row table | Signed Miller index, row id, intensity, sigma, frame or scan coordinate, detector coordinate if available, row flags, and unique reflection key. |
| Geometry metadata | Wavelength, beam vector, goniometer axis, detector panel metadata, orientation matrix or scan-point matrices, and cell. |
| Preflight report | Row count, unique count, dmin, completeness if available, I/sigma distribution, sigma policy, transform or polarity decision, and rejected-row counts. |

## GROND Auto Or Regularization Run

The regularization run is the source of Bloch and Born row predictions. It must be inspectable independent of the final refinement endpoint.

| Artifact | Required content |
|---|---|
| `summary.json` | Run config, input paths, N/Q, backend, lambda policy, thickness, absorption or jitter policy, scale contract, output endpoints, and timing. |
| Row prediction table | Row id, hkl, Iobs, sigma, Idyn, Iborn, q, scale group, split label, corrected intensity, corrected sigma, d spacing, and diagnostics. |
| Native row metrics | R1all, wR2all, CC, work/free versions, Born controls, shell metrics, and intensity-bin metrics. |
| Endpoint exports | Corrected HKL, endpoint model, refinement log, map inputs, and table-ready R values. |
| Command log | Exact CLI invocation, package version, git commit, environment, CPU/GPU/backend, worker count, and wall time. |

## LS1 Validation

`grond validate-ls1` is the reviewer-facing audit for whether the row correction matters.

| Artifact | Required content |
|---|---|
| `LS1_VALIDATION_REPORT.md` | Updated-q cycles, selected cycle, native row metrics, endpoint metrics, controls, native polish result, and interpretation. |
| `ls1_validation_summary.json` | Machine-readable config and metrics. |
| Updated-q cycles | Native check from the next q recomputation where available, endpoint R values, GooF, and coordinate/ADP movement. |
| Frozen-q control | Initial q held fixed through endpoint loop. |
| q=1 control | Born-only or kinematic correction branch. |
| Alternate-polarity control | Opposite polarity or mirrored correction branch when meaningful. |
| Shuffled-q controls | Global logq shuffle, resolution-shell logq shuffle, and shell-plus-scan logq shuffle. |
| Native polish | Matched production physics, sampled parameter classes, objective movement, model shifts, and accepted/rejected status. |

## Endpoint Calibration

`grond calibrate-endpoint` is the publication-facing endpoint chooser. It should run after the row table and q corrections are fixed.

| Artifact | Required content |
|---|---|
| `REPORT.md` | Selected publication endpoint, low-R diagnostic endpoint, candidate table, gate rejections, and runtime. |
| `summary.json` | Machine-readable endpoint selection and candidate metrics. |
| Scale-candidate table | Scale family, scale parameter count, detector terms, R values, GooF, held-out row metrics, and sigma scale. |
| Promoted endpoint folder | Fixed `WGHT 0 0` endpoint files, HKL, RES/CIF, LST, maps, and final table-ready metrics. |
| Low-R diagnostic folder | Lower-R branch retained for audit, with rejection reason. |
| Lambda or sensitivity sweep | Stability check showing the promoted endpoint class does not depend on one hand-tuned lambda value. |

## Multi-Dataset Runs

Multi-dataset results must keep dataset-level nuisance and sigma behavior visible.

| Artifact | Required content |
|---|---|
| Per-dataset summaries | Thickness, jitter, scale contract, sigma policy, row count, native metrics, and endpoint contribution. |
| Dataset policy | Include, downweight, completeness-only, or exclude, with reason. |
| Combined row table | Dataset id per row, corrected intensity, corrected sigma, row q, dataset weight, split label, and merge key. |
| Merge report | Interdataset scale, robust row scatter, merged sigma policy, rejected rows/reflections, completeness, and resolution promotion. |
| Combined endpoint | Corrected merged HKL, endpoint model, maps, Table 1 metrics, and validation controls. |

## DIALS Route

DIALS support must be documented as an explicit row contract, not inferred from scalar endpoint statistics.

| Artifact | Required content |
|---|---|
| Source files | `integrated.refl`, `integrated.expt`, DIALS version, import gain, integration settings, and scaling/error-model choices if used. |
| Bridge report | Exported columns, metadata reconstruction, frame/scan handling, row flags, sigma policy, and unsupported columns. |
| Sigma calibration report | Raw DIALS sigma behavior, any grouped variance correction, held-out residual check, and final sigma policy. |
| Endpoint comparison | DIALS route versus XDS route on common metrics, with a clear note if DIALS is supported but not best for that dataset. |

## Maps And Chemistry

Map claims must distinguish conventional comparator maps from GROND-conditioned maps.

| Artifact | Required content |
|---|---|
| Conventional comparator | Kinematic endpoint, EXTI branch if used as comparator, maps, R values, and GooF. |
| GROND endpoint maps | Promoted endpoint maps, difference-map sigma, peak/hole, and map scripts. |
| H/no-H or omit branch | Starting model, omitted atoms, whether q was recomputed from omitted model, endpoint metrics, peak heights, and controls. |
| Structural stability | Coordinate RMSD, ADP changes, NPD warnings, restraint policy, and final chemistry checks. |

## Runtime And Benchmarking

Timing claims must be tied to hardware and backend.

| Artifact | Required content |
|---|---|
| Timing log | Step name, wall time, CPU/GPU, worker count, backend, N/Q, cache state, and row count. |
| Backend note | SciPy, scipy_bethe, cached row plan, JAX, MLX, or other backend. |
| Scaling curve | Runtime versus rows, N, Q, dataset count, and endpoint calibration. |
