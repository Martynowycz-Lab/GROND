# DIALS Workflow For GROND

GROND can use DIALS as a first-class integration front end through the pair of files written by `dials.integrate`: `integrated.refl` and `integrated.expt`. The reflection table contains the signed `miller_index`, summation intensity and variance, predicted and observed detector/frame positions, partiality, LP correction, flags, and related row metadata. The experiment list contains the beam, detector, goniometer, scan, crystal model, and scan-varying orientation matrices. GROND needs both files.

The recommended DIALS path is to process the data normally through indexing and regularization, solve the structure from any conventional DIALS/XDS/SHELX route that gives a chemically sensible `RES` or `CIF`, and then give GROND the unmerged DIALS integration output. Do not give GROND only a merged MTZ, merged SHELX HKL, or a `dials.export xds_ascii` file when the original `.refl` and `.expt` files are available. The exported HKL-style files are useful for conventional crystallography, but they do not carry the full scan-varying row geometry GROND uses for Bloch rocking curves.

## Integration

DIALS normally writes summation intensities and, when profile fitting is enabled, profile-fitted intensities in the same reflection table. GROND uses `intensity.sum.value` and `intensity.sum.variance` by default, so profile fitting does not have to be disabled for GROND to find summation rows. If you want a cleaner and smaller GROND input table, you can integrate with profile fitting disabled:

```bash
dials.integrate updated.expt updated.refl integration.profile.fitting=False prediction.d_min=0.70
```

If the detector gain is known, pass it explicitly during DIALS integration, for example `integration.summation.detector_gain=<gain>`. DIALS applies this parameter to summation variances, and GROND records whether the bridge later had to apply the equivalent sigma scaling because the integration output still looked over-precise. For the 370 electron-diffraction CBF test, the detector model carried gain 204, so omitting the integration parameter made DIALS `I/sigma` values much too optimistic for row-level weighted regularization.

The important point is that the GROND input remains the row-preserving integrated reflection table and matching experiment list. It should not be scaled, merged, or symmetry-averaged before the Bloch calculation. If you also run `dials.scale`, keep the scaled files for conventional comparisons, but use `integrated.refl` and `integrated.expt` for the default GROND path unless you deliberately choose the diagnostic `--dials-intensity-mode scale`.

DIALS does not need to be integrated in P1 just to preserve signed rows. The reflection table stores the signed `miller_index` assigned to each observation. GROND keeps that exact signed row identity for simulation and still performs the ordinary transform scouts against the solved structure. If you are testing unusual indexing or space-group alternatives, keep separate DIALS output directories so each GROND run has a clear row contract.

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
IOBS_GROND = intensity.sum.value * lp / partiality
SIGMA_GROND = sqrt(intensity.sum.variance) * |lp / partiality| * sigma_multiplier
PEAK_GROND = 100 * partiality
RLP_GROND = lp
```

This makes GROND's normal `--min-peak 90` selection equivalent to DIALS partiality `>= 0.90`, while leaving weaker partial rows available for diagnostics if you lower `--min-peak`. The `recorded_proxy()` value in the GROND engine maps back to the raw DIALS summation intensity for this bridge mode. That is useful for audits because it keeps the raw summation observation recoverable while the default target remains a complete-intensity estimate.

The default sigma policy is `--dials-sigma-mode auto`. DIALS stores a detector gain on the detector model, but the summation variance path is controlled by the `integration.summation.detector_gain` integration parameter. Electron-diffraction CBF data can therefore carry summation variances that are too optimistic if the detector gain was not applied during integration. In auto mode, GROND checks the detector gain and the high-partiality `I/sigma` distribution; when the rows look over-precise, it multiplies only the bridge sigma values by `sqrt(detector_gain)` and records the decision in `dials_bridge_summary.json`. Use `--dials-sigma-mode raw` to reproduce the raw DIALS variances, `--dials-sigma-mode detector_gain` to force detector-gain scaling, or `--dials-sigma-scale <factor>` for a deliberate diagnostic multiplier.

Two diagnostic alternatives exist. `--dials-intensity-mode sum_raw` uses the raw summation intensity without LP/partiality completion. `--dials-intensity-mode scale` uses `intensity.scale.value` from `scaled.refl`, and should be treated as a conventional comparison rather than the default Bloch row target.

## Geometry And Sg

DIALS reflection tables may contain `zeta`, but GROND does not use it as excitation error. GROND recomputes the excitation error from the experiment geometry:

```text
g_h(alpha) = R(axis, alpha) A(alpha) h
Sg(alpha) = (k^2 - |s0 + g_h(alpha)|^2) / (2k)
```

Here `A(alpha)` is obtained by linearly interpolating the DIALS scan-point A matrices and then applying the goniometer rotation. This scan-varying orientation is essential. On the 370 DIALS test data, using the static crystal A matrix produced millithangstrom-scale excitation-error differences, while the scan-varying bridge reproduced predicted-row `Sg` values near zero. The bridge records this check in `dials_bridge_summary.json` under `sg_validation`.

## Practical Checks

After import, inspect `dials_bridge_summary.json`. A healthy conversion should have nonzero `rows_written`, a sensible number of rows with `partiality >= 0.90`, `sigma_model` values that explain whether detector gain was applied, and `sg_validation.median_abs_sg_inv_A` close to zero for the predicted row positions. Production DIALS imports drop rows flagged `dont_integrate` and rows with severe bad-pixel, overloaded, powder-ring, bad-shoebox, failed-background, or failed-summation flags. Use `--dials-keep-dont-integrate` or `--dials-keep-bad-flags` only for diagnostics.

The DIALS bridge is new compared with the XDS route and should be tested against known structures before being used as the only production path. If XDS and DIALS disagree strongly on the same raw frames, compare the row counts, partiality selection, DIALS flag policy, detector-gain sigma policy, resolution limit, transform scout result, polarity decision, and `Sg` validation before interpreting the model-update statistics.
