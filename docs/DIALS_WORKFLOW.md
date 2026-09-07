# DIALS Workflow For GROND

GROND can use DIALS as a first-class integration front end through the pair of files written by `dials.integrate`: `integrated.refl` and `integrated.expt`. The reflection table contains the signed `miller_index`, summation intensity and variance, predicted and observed detector/frame positions, partiality, LP correction, flags, and related row metadata. The experiment list contains the beam, detector, goniometer, scan, crystal model, and scan-varying orientation matrices. GROND needs both files.

The recommended DIALS path is to process the data normally through indexing and regularization, solve the structure by a conventional DIALS/XDS/SHELX route, and then give GROND the unmerged DIALS integration output. Use a chemically sensible SHELX `RES` or `INS` with explicit electron factors; a CIF input to `GROND run` must embed the complete SHELX instruction block. Do not give GROND only a merged MTZ, merged SHELX HKL, or a `dials.export xds_ascii` file when the original `.refl` and `.expt` files are available. The exported HKL-style files are useful for conventional crystallography, but they do not carry the full scan-varying row geometry GROND uses for Bloch rocking curves.

## Integration

DIALS normally writes summation intensities and, when profile fitting is enabled, profile-fitted intensities in the same reflection table. GROND uses `intensity.sum.value` and `intensity.sum.variance` by default, so profile fitting does not have to be disabled for GROND to find summation rows. If you want a cleaner and smaller GROND input table, you can integrate with profile fitting disabled:

```bash
dials.integrate updated.expt updated.refl integration.profile.fitting=False prediction.d_min=0.70
```

If the detector gain is known, put it on the detector model during import, for example `dials.import /path/to/images_*.cbf panel.gain=<gain>`, and then integrate that experiment file. For rotation data this is the correct place for the gain; do not rely on `integration.summation.detector_gain` as the routine GROND workflow. GROND keeps DIALS' detector-model variance by default. A second gain multiplier can double-count a correction that DIALS has already applied, so `detector_gain`, `auto`, and an explicit `--dials-sigma-scale` are diagnostic choices rather than production defaults.

DIALS reflection tables also carry a `bbox` for each integrated row. GROND stores the z-range from that box in the generated `INTEGRATE.HKL` compatibility file as `ALF0=z_min`, `BET0=z_max`, and `MAXC=z_width`. This gives GROND a cheap record of the DIALS integration support without requiring debug shoebox files. It is not promoted by default as the Bloch rocking width because the integration support is deliberately broad; use `--sg-window-mode dials_bbox` only as an explicit diagnostic.

For forensic work, DIALS can write the full diagnostic shoeboxes:

```bash
dials.integrate refined.expt refined.refl \
  integration.profile.fitting=False \
  prediction.d_min=0.70 \
  debug.output=True \
  debug.delete_shoeboxes=False
```

Those files can be large because they preserve pixel-level shoebox data for each integration block. They are useful for auditing frame profiles and rejected pixels, but they are not required for ordinary GROND auto runs.

The important point is that the GROND input remains a row-preserving reflection table and matching experiment list. It must not be merged or symmetry-averaged before the Bloch calculation. The default path uses `integrated.refl` and `integrated.expt`. A `scaled.refl` table still contains the raw summation columns, so a deliberate alternative is `--dials-row-selection scaled` (or `--row-selection scaled` with `import-dials`): this keeps `intensity.sum.value` and its variance while restricting the rows to observations accepted by `dials.scale`. That is useful after a documented scaling rejection such as delta-CC1/2, and is distinct from `--dials-intensity-mode scale`, which changes the intensity values themselves. The selected row contract is recorded in `dials_bridge_summary.json`; requesting scaled selection from a table with no scaled rows fails closed.

When the reflection table contains DIALS `inverse_scale_factor` values, the
bridge preserves them and their reported variance in
`DIALS_INVERSE_SCALE_CONTROL.npz`, with source row indices, signed HKLs,
calculated centroids, and panel IDs. GROND does not apply these values to the
summation target automatically. They are a fixed prior or comparison for the
detector-scale audit, not a license to silently exchange the DIALS scaling
model for a flexible GROND detector surface.

DIALS does not need to be integrated in P1 just to preserve signed rows. The reflection table stores the signed `miller_index` assigned to each observation. GROND keeps that exact signed row identity for simulation and performs transform alignment against the solved structure. The DIALS crystal and laboratory frames are proper, right-handed rotations. GROND therefore keeps automatic setting selection to determinant-positive transforms, ranks them with a coordinate-hand-invariant matched-Born work-row target, and fixes the selected transform before comparing dynamical hands. If you are testing unusual indexing or space-group alternatives, keep separate DIALS output directories so each GROND run has a clear row contract.

## One-Command Use

If the Python running GROND can also import DIALS and dxtbx, no extra path is needed. On many workstations and clusters DIALS lives in its own Python, so pass that executable explicitly:

```bash
GROND auto \
  --config /path/to/grond.config.json \
  --dials-refl /data/crystal/integrated.refl \
  --dials-expt /data/crystal/integrated.expt \
  --dials-python /path/to/dials/python \
  --structure /data/crystal/solved_model.res \
  --baseline-hkl /data/crystal/solved_model.hkl \
  --out-dir grond_dials_auto
```

During startup GROND writes a prepared XDS-like row directory under `grond_dials_auto/prepared/dials_<dataset_name>/`. That directory contains `INTEGRATE.HKL`, `DIALS_A_AT_SCAN_POINTS.txt`, `dials_bridge_summary.json`, and a short bridge README. The generated `INTEGRATE.HKL` is a compatibility contract for GROND, not a native XDS result.

One bridge contains exactly one DIALS experiment. If a `.refl/.expt` pair holds
several experiments, split it first and import each output separately:

```bash
dials.split_experiments combined.expt combined.refl

GROND import-dials --refl split_0.refl --expt split_0.expt \
  --dials-python /absolute/path/to/dials.python --dataset-name crystal_0 \
  --out-dir bridges/crystal_0
GROND import-dials --refl split_1.refl --expt split_1.expt \
  --dials-python /absolute/path/to/dials.python --dataset-name crystal_1 \
  --out-dir bridges/crystal_1

GROND prep --structure model.res --baseline-hkl model.hkl \
  --dataset crystal_0=bridges/crystal_0,dmin=0.70,min_peak=90 \
  --dataset crystal_1=bridges/crystal_1,dmin=0.70,min_peak=90 \
  --out prep.json
GROND auto --config grond.config.json --prep prep.json --out-dir grond_dials_multi
```

The numeric suffix is the experiment-list order written by DIALS. Keep the
split files and each bridge summary so that experiment UUID and row provenance
remain auditable.

You can also make the converted directory once and inspect it before running:

```bash
GROND import-dials \
  --refl /data/crystal/integrated.refl \
  --expt /data/crystal/integrated.expt \
  --dials-python /path/to/dials/python \
  --structure /data/crystal/solved_model.res \
  --baseline-hkl /data/crystal/solved_model.hkl \
  --dataset-name crystal_a \
  --out-dir dials_grond_bridge

GROND auto \
  --config /path/to/grond.config.json \
  --xds-dir dials_grond_bridge \
  --structure /data/crystal/solved_model.res \
  --baseline-hkl /data/crystal/solved_model.hkl \
  --out-dir grond_from_bridge
```

## Intensity Mapping

The default bridge mode is `sum_complete`. It computes the GROND row intensity as

```text
correction = lp / qe
IOBS_GROND = intensity.sum.value * correction / partiality
SIGMA_GROND = sqrt(intensity.sum.variance) * |correction / partiality|
PEAK_GROND = 100 * partiality
RLP_GROND = correction
```

This makes GROND's normal `--min-peak 90` selection equivalent to DIALS partiality `>= 0.90`, while leaving weaker partial rows available for diagnostics if you lower `--min-peak`. The `recorded_proxy()` value in the GROND engine maps back to the raw DIALS summation intensity for this bridge mode. That is useful for audits because it keeps the raw summation observation recoverable while the default target remains a complete-intensity estimate.

The default sigma policy is `--dials-sigma-mode raw`. This preserves the detector-model variance attached to the integrated reflection. The empirical `auto` mode estimates excess scatter from symmetry-related observations, but that scatter can also contain real orientation-dependent dynamical signal, partiality, dose, or scale variation; use it only as a controlled diagnostic. `--dials-sigma-mode detector_gain` and `--dials-sigma-scale <factor>` are likewise explicit tests. The chosen policy and before/after `I/sigma` distributions are recorded in `dials_bridge_summary.json`.

Do not graft `intensity.scale.variance` onto `intensity.sum.value`. The scaled variance belongs to the intensity estimator selected by `dials.scale`, which may be profile rather than summation intensity, and the experiment JSON does not retain a reliable intensity-choice label. If a clean `dials.scale` run was explicitly forced to summation intensity and its fitted basic error model is effectively a uniform multiplier, that multiplier can be replayed as the diagnostic `--dials-sigma-scale`. This changes normalized residuals and GooF, but it cannot change unweighted row `Rall`, correlation, or the row-specific Bloch correction. A nonuniform DIALS error model should be consumed together with its matching scaled intensity branch, not silently spliced into the default summation contract.

Two diagnostic alternatives exist. `--dials-intensity-mode sum_raw` uses the raw summation intensity without `(lp/qe)/partiality` completion. `--dials-intensity-mode scale` uses `intensity.scale.value` from `scaled.refl`, and should be treated as a conventional comparison rather than the default Bloch row target.

When comparing XDS and DIALS on the same frames, first record the accepted-row contract. This example retains DIALS-scaled rows while using completed summation intensities:

```bash
GROND import-dials \
  --refl scaled.refl \
  --expt scaled.expt \
  --dials-python /path/to/dials.python \
  --intensity-mode sum_complete \
  --row-selection scaled \
  --out-dir dials_scaled_accepted_sum_bridge
```

This is not automatically the promoted production mode. It is a row-contract choice that should match the stated data-processing protocol. For an intensity-mapping diagnostic, retain the same rows and use `--intensity-mode sum_raw` instead. On some datasets the raw DIALS summation rows have an `I` and `I/sigma` distribution much closer to XDS no-profile rows, while the rows completed by `(lp/qe)/partiality` give better GROND row residuals. That means the DIALS/XDS disagreement is not only a global intensity scale question. It can also come from the refined cell, scan-varying orientation model, row filtering, partiality definitions, and which signed observations each program actually integrated.

## Geometry And Sg

DIALS reflection tables may contain `zeta`, but GROND does not use it as excitation error. GROND recomputes the excitation error from the experiment geometry:

```text
g_h(alpha) = S R(axis_datum, alpha) F A(alpha) h
Sg(alpha) = (k^2 - |s0 + g_h(alpha)|^2) / (2k)
```

Here `F` and `S` are the DIALS fixed and setting rotations, and `A(alpha)` is obtained by linearly interpolating the crystal scan-point A matrices. DIALS reflection z coordinates are zero-based array indices, so the bridge preserves those raw values and records `scan.get_array_range()[0]` as `STARTING_FRAME`. This matters for cropped scans whose first image number is not one. The bridge records the resulting check in `dials_bridge_summary.json` under `sg_validation`. A scan-varying setting rotation is currently rejected with a clear error because silently treating it as constant would give the wrong excitation geometry; scan-varying crystal A matrices remain supported.

## Practical Checks

After import, inspect `dials_bridge_summary.json`. A healthy conversion should have nonzero `rows_written`, a sensible number of rows with `partiality >= 0.90`, an explicit sigma policy, and `sg_validation.median_abs_sg_inv_A` close to zero for the predicted row positions. GROND refuses the conversion when the 95th percentile of `|Sg|` exceeds `0.005 A^-1`, because such rows indicate mismatched or unreliable imported geometry. This strict import-geometry gate is separate from the wider `0.01 A^-1` beam-support cutoff used by new Bloch calculations. A failed conversion leaves a diagnostic summary but does not write a new `INTEGRATE.HKL`. Production DIALS imports drop rows flagged `dont_integrate` and rows with severe bad-pixel, overloaded, powder-ring, bad-shoebox, failed-background, or failed-summation flags. Use `--dials-keep-dont-integrate` or `--dials-keep-bad-flags` only for diagnostics.

The DIALS bridge is newer than the XDS route and should be tested against known structures before being used as the only production path. A known structure used for sign calibration must be applied only after its proper operator and matched hand result have been frozen; it must not choose the operator and must be excluded from the validation count. Use a separate calibration record for each acquisition/import setup. The current final-elastic `auto` hand artifact is a conditional result and is not yet accepted by the legacy parity-calibration replay, so preserve the external known-chemistry assignment rather than deriving a named configuration from residual sign alone. If XDS and DIALS disagree strongly on the same raw frames, compare the row counts, partiality selection, DIALS flag policy, sigma policy, resolution limit, proper-transform decision, conditional hand decision, and `Sg` validation before interpreting the model-update statistics.

Be especially careful when comparing fitted thickness values across integration programs. GROND thickness is a nuisance parameter fitted to a row contract, not a direct microscope thickness measurement. If XDS and DIALS use different cells, rotation axes, scan-varying orientation models, row filters, or intensity completion conventions, their fitted thickness values can differ even when both final structures are chemically reasonable. Treat a cross-pipeline thickness mismatch as a prompt to audit the row contract before treating either number as absolute.
