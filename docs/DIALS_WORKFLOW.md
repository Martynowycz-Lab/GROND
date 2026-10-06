# DIALS workflow

GROND reads the two files written by `dials.integrate`: `integrated.refl` and `integrated.expt`. The reflection table holds the signed `miller_index`, the summation intensity and variance, predicted and observed positions, partiality, LP correction and flags. The experiment list holds the beam, detector, goniometer, scan and the scan-varying crystal model. GROND needs both.

Process the data as usual through indexing and refinement, solve the structure by your usual route, and give GROND the unmerged DIALS integration together with a SHELX `RES` or `INS` that has numeric electron SFAC lines (or a CIF that embeds the SHELX instructions). Do not give GROND only a merged MTZ, a merged HKL or a `dials.export` `xds_ascii` file when the `.refl` and `.expt` files exist: those do not carry the scan-varying geometry of each observation.

## Integration

DIALS writes summation intensities, and profile-fitted intensities when profile fitting is on, in the same table. GROND uses `intensity.sum.value` and `intensity.sum.variance`, so profile fitting does not have to be turned off. For a smaller summation-only table:

```bash
dials.integrate refined.expt refined.refl integration.profile.fitting=False prediction.d_min=0.70
```

If the detector gain is known, put it on the detector model at import, for example `dials.import /path/to/images_*.cbf panel.gain=<gain>`, and integrate that experiment. Do not use `integration.summation.detector_gain` for rotation data. GROND keeps the DIALS variance by default (`--dials-sigma-mode raw`); a second gain factor would count the gain twice, so `detector_gain`, `auto` and `--dials-sigma-scale` are for diagnostics.

DIALS also records a `bbox` for each observation. GROND stores its frame range in the converted `INTEGRATE.HKL` as `ALF0` (start), `BET0` (end) and `MAXC` (width). With the default `--sg-window-mode auto`, GROND uses this range as the rocking window of each observation, capped at 0.03 Å⁻¹ (`--sg-window-max`); XDS observations use a fixed 0.010 Å⁻¹ window.

The input must stay unmerged. The default uses `integrated.refl` and `integrated.expt`. A `scaled.refl` still holds the summation columns, so `--dials-row-selection scaled` (`--row-selection scaled` for `import-dials`) keeps `intensity.sum.value` and its variance but only the observations accepted by `dials.scale`, for example after a ΔCC½ rejection. This is different from `--dials-intensity-mode scale`, which uses the scaled intensities themselves. The choice is recorded in `dials_bridge_summary.json`; asking for scaled observations from a table without any fails.

When the table contains DIALS `inverse_scale_factor` values, the import writes them with their variance to `DIALS_INVERSE_SCALE_CONTROL.npz` for reference. GROND does not apply them.

DIALS does not need to integrate in P1. GROND keeps the signed `miller_index` of each observation and finds the transform to the model setting itself, using only proper (determinant +1) transforms, before the hand is compared.

## One crystal

If the Python that runs GROND can also import DIALS, no extra option is needed. Usually DIALS has its own Python; pass it:

```bash
grond auto \
  --config /path/to/grond.config.json \
  --dials-refl /data/crystal/integrated.refl \
  --dials-expt /data/crystal/integrated.expt \
  --dials-python /path/to/dials.python \
  --structure /data/crystal/solved_model.res \
  --baseline-hkl /data/crystal/solved_model.hkl \
  --out-dir grond_dials_auto
```

At startup GROND writes a converted directory under `grond_dials_auto/prepared/dials_<dataset_name>/` with `INTEGRATE.HKL`, `DIALS_A_AT_SCAN_POINTS.txt`, `dials_bridge_summary.json` and `DIALS_BRIDGE_README.md`. The `INTEGRATE.HKL` there is GROND's own format, not an XDS result.

You can also convert once and inspect the result first:

```bash
grond import-dials \
  --refl /data/crystal/integrated.refl \
  --expt /data/crystal/integrated.expt \
  --dials-python /path/to/dials.python \
  --dataset-name crystal_a \
  --out-dir dials_grond_bridge

grond auto \
  --config /path/to/grond.config.json \
  --xds-dir dials_grond_bridge \
  --structure /data/crystal/solved_model.res \
  --baseline-hkl /data/crystal/solved_model.hkl \
  --out-dir grond_from_bridge
```

## Several experiments

One converted directory holds one DIALS experiment. Split a multi-experiment pair first and convert each part:

```bash
dials.split_experiments combined.expt combined.refl

grond import-dials --refl split_0.refl --expt split_0.expt \
  --dials-python /path/to/dials.python --dataset-name crystal_0 --out-dir bridges/crystal_0
grond import-dials --refl split_1.refl --expt split_1.expt \
  --dials-python /path/to/dials.python --dataset-name crystal_1 --out-dir bridges/crystal_1

grond prep --structure model.res --baseline-hkl model.hkl \
  --dataset crystal_0=bridges/crystal_0,dmin=0.70,min_peak=90 \
  --dataset crystal_1=bridges/crystal_1,dmin=0.70,min_peak=90 \
  --out prep.json
grond auto --config grond.config.json --prep prep.json --out-dir grond_dials_multi
```

The number in `split_N` is the experiment order written by DIALS. Keep the split files and each `dials_bridge_summary.json` so the experiment identity stays traceable.

## Intensities and uncertainties

The default `--dials-intensity-mode sum_complete` writes

```text
correction  = lp / qe
IOBS_GROND  = intensity.sum.value * correction / partiality
SIGMA_GROND = sqrt(intensity.sum.variance) * |correction / partiality|
PEAK_GROND  = 100 * partiality
RLP_GROND   = correction
```

so the default `--min-peak 90` keeps observations with DIALS partiality ≥ 0.90, and the summation intensity is recovered as `IOBS × (PEAK/100) / RLP`. `sum_raw` uses the summation intensity without the `(lp/qe)/partiality` completion, and `scale` uses `intensity.scale.value` from `scaled.refl`; both are comparisons, not the default target.

`--dials-sigma-mode raw` (default) keeps the DIALS variance. `auto` estimates extra scatter from symmetry-related observations, but that scatter also contains real dynamical, partiality, dose and scale effects. `detector_gain` and `--dials-sigma-scale` are explicit tests. The mode and the I/σ distributions before and after are recorded in `dials_bridge_summary.json`.

Do not combine `intensity.scale.variance` with `intensity.sum.value`. The scaled variance belongs to the intensity estimator chosen by `dials.scale`, which may be the profile-fitted one.

To compare XDS and DIALS on the same frames, first fix which observations are kept. This example keeps the observations accepted by `dials.scale` but uses completed summation intensities:

```bash
grond import-dials \
  --refl scaled.refl \
  --expt scaled.expt \
  --dials-python /path/to/dials.python \
  --intensity-mode sum_complete \
  --row-selection scaled \
  --out-dir dials_scaled_accepted_sum_bridge
```

Differences between the two programs can come from the refined cell, the scan-varying orientation model, observation filtering, partiality definitions and which observations each program integrated, not only from an overall scale.

## Geometry

GROND does not use the DIALS `zeta` column as the excitation error. It recalculates the excitation error from the experiment geometry:

```text
g_h(alpha) = S R(axis_datum, alpha) F A(alpha) h
Sg(alpha)  = (k^2 - |s0 + g_h(alpha)|^2) / (2k)
```

`F` and `S` are the DIALS fixed and setting rotations, and `A(alpha)` is interpolated linearly between the scan-point A matrices. DIALS z coordinates are zero-based array indices; the import keeps them and writes `scan.get_array_range()[0]` as `STARTING_FRAME`, so cropped scans need no renumbering. A scan-varying setting rotation is rejected; scan-varying crystal A matrices are supported.

## Checks after import

Open `dials_bridge_summary.json`. A good conversion has a non-zero `rows_written`, a sensible number of observations with partiality ≥ 0.90, an explicit sigma mode, and `sg_validation.median_abs_sg_inv_A` close to zero. The import refuses to write `INTEGRATE.HKL` when the 95th percentile of |Sg| exceeds 0.005 Å⁻¹, which indicates mismatched or unreliable geometry. This check is stricter than the 0.01 Å⁻¹ beam-support limit of the Bloch calculation. Observations flagged `dont_integrate`, overloaded, in a powder ring, with bad pixels, a bad shoebox, or failed background or summation are dropped. `--keep-dont-integrate` and `--keep-bad-flags` (`--dials-keep-dont-integrate` and `--dials-keep-bad-flags` with `grond auto`) keep them, for diagnostics only.

Fitted thickness depends on the observations it was fitted to. If XDS and DIALS use different cells, rotation axes, orientation models, filters or intensity completion, their thickness values can differ even when both models are reasonable; it is not a direct measurement of crystal thickness.
