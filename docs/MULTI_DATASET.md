# Several crystals

GROND treats each crystal as its own experiment until the last step. Thickness, frame offset, frame scales, mosaic spread, damage and uncertainties belong to one crystal. A weak crystal should not set the thickness or the error model of a good one, and a good crystal should not hide a weak crystal that adds reflections nobody else measured.

Put one integration per crystal in a preparation file and run `grond auto` on it:

```bash
grond prep \
  --structure model.res \
  --baseline-hkl model.hkl \
  --dataset crystal_a=/data/project/crystal_a/xds_noprf \
  --dataset crystal_b=/data/project/crystal_b/xds_noprf \
  --out prep_multi.json

grond auto \
  --prep prep_multi.json \
  --config grond.config.json \
  --out-dir grond_multi
```

For XDS, give the directory that contains `INTEGRATE.HKL` and `XPARM.XDS` or `GXPARM.XDS`. For DIALS, convert each crystal once with `grond import-dials` and give the converted directory ([DIALS_WORKFLOW.md](DIALS_WORKFLOW.md)). XDS and DIALS crystals can be mixed. Do not concatenate `INTEGRATE.HKL` files or merge the data before GROND: the Bloch calculation needs each crystal's signed observations and scan geometry.

## What is per crystal and what is shared

Per crystal:

- the indexing transform (all 24 proper signed-axis permutations that pass the unit-cell metric filter, ranked by matched Born predictions on working observations);
- the thickness distribution, mosaic spread, damage and frame-scale smoothness, chosen by cross-validation within the working observations;
- the frame scales and the error model, fitted on working observations.

Shared:

- the free set: one salted hash of the unique reflection, so a reflection is free in every crystal or in none. `decisions/cross_crystal_split_check.json` records the check, and the run stops before any refinement if a reflection would be free in one crystal and working in another. Give every crystal the same transform candidates (`--transform-candidates`) or verified transforms (`--hkl-transform ... --trust-hkl-transform`) if this happens;
- the hand: the held-out sign test pools the evidence of all crystals;
- the structural model.

A crystal whose cell matches the model under no transform is left out and recorded in `decisions/metric_excluded_datasets.json` (`--metric-incompatible-datasets exclude`, the default; `fail` stops the run instead).

## Dataset policy

`dataset_policy/structure_update/` and `dataset_policy/final/` give each crystal a role, computed from working observations only:

| Role | Effect |
|---|---|
| `primary` | full weight |
| `downweight` | stays in, with inflated uncertainties so it cannot dominate |
| `completeness_only` | left out of the model update; its reflections can still add coverage to the merged HKL |
| `exclude` | left out, and reported with the reason |

`--dataset-policy off` gives every crystal full weight.

## Each round

Each round recalculates every crystal from the current shared model with its own settings, corrects its working observations, and gives them the crystal's error model, σ'² = k²(σ² + (uP)²). The corrected observations of all crystals are put on one absolute scale (κ|F|² · Iobs/Icalc), so no inter-crystal scale is fitted; `combined_summary.json` reports an overlap estimate as `interdataset_scale_diagnostic`, which should be close to 1. Symmetry-equivalent observations are merged with inverse-variance weights (`--multi-endpoint-merge-key model_symmetry`; Friedel mates stay separate in acentric groups). Where merged observations disagree more than their uncertainties allow, the merged uncertainty is inflated (`chisq_overdispersion`). The merged HKL is not rescaled again.

SHELXL then refines one shared model against the merged HKL to convergence: `L.S. 10` per call, weights converged from `WGHT 0 0`, at most 20 calls, stopping when |max shift/esd| < 0.01. The new model goes back to every crystal. Rounds repeat until no non-hydrogen atom moves by more than 0.002 Å, at most five times. Next to the merged HKL, GROND also refines an unmerged HKL with one batch per crystal and refined `BASF` (`--unmerged-endpoint on`).

With the default `--free-set off`, the chosen model is refined once more against all reflections, working and free, into `final_all_data/`.

`--endpoint-min-d-spacing` applies an additional resolution cutoff to the merged HKL and the combined row statistics, for example to match a conventional refinement. Membership is recalculated from the model cell.

## Continuing from your own model

Completed runs write `grond_continue_checkpoint.json`. After editing the shared model, send it back with `grond refine` ([USAGE.md](USAGE.md)):

```bash
grond refine \
  --checkpoint grond_multi/grond_continue_checkpoint.json \
  --start-res edited_model.res \
  --out-dir refine_edited \
  --cycles 3
```

`grond refine` (also called `grond push-endpoint`) keeps each crystal's settings, dataset-policy weights and merge key from the checkpoint. Each of its `--cycles` (default 3) recalculates every crystal for the current model, merges the corrected observations and runs SHELXL, and it keeps the last cycle. The report is `MULTI_ENDPOINT_PUSH_REPORT.md`.

## Files to read first

```text
FINAL_SUMMARY.md
HELDOUT_VALIDATION.md
dataset_policy/final/REPORT.md
runs/combined/combined_summary.json
resolution_limits/combined/REPORT.md
endpoint_chooser/PUBLICATION_ENDPOINT_CHOOSER.md
```

In a multi-crystal run, the merged model of all accepted crystals is the result. A single crystal with a lower residual does not replace it.
