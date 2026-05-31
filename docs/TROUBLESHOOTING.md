# Troubleshooting

If `GROND doctor --probe-bloch` fails, the Python environment is not ready. Reinstall the package in a clean Python 3.11 or 3.12 environment and check that NumPy, SciPy, and Gemmi import correctly.

If the run stops during preflight, inspect `decisions/<dataset>/preflight.json`. Common causes are missing numeric reflection rows, too few positive intensities, non-positive sigmas, or geometry columns that are absent or badly tailed.

If endpoint statistics or maps are missing, check whether external crystallographic programs are on `PATH`. GROND can still run native row-level modeling without those tools, but merged endpoint statistics and maps require external programs.

If `GROND auto` stops immediately with a missing-tool message, run `GROND init-config --out grond.config.json`, fill in absolute paths for `shelxl` and `shelx2map`, and rerun with `--config grond.config.json`. `GROND doctor --config grond.config.json --probe-bloch` prints the resolved executable path, whether it came from config, environment, or `PATH`, and what aliases were checked. For a deliberate row-only run, add `--native-only`; for endpoint statistics without maps, add `--no-maps`; for incomplete diagnostics only, add `--allow-missing-tools`.

If DIALS import fails with an error about `dials` or `dxtbx`, pass the DIALS Python executable explicitly with `--dials-python /path/to/dials/python`. GROND itself can live in a separate Python 3.11/3.12 environment; the import bridge launches the DIALS Python only to read `.refl` and `.expt` files.

If a DIALS run imports but gives implausible row geometry, inspect `prepared/dials_<name>/dials_bridge_summary.json`. The `sg_validation.median_abs_sg_inv_A` value should be close to zero for predicted row positions. A large value usually means the wrong `.expt` was paired with the `.refl`, the scan-varying crystal model was not preserved, or the reflection table came from a different integration/regularization stage.

If DIALS gives reasonable R1 values but much worse weighted residuals than an XDS reduction of the same frames, inspect the `sigma_model` block in `dials_bridge_summary.json`. For electron-diffraction CBF data, DIALS may store a detector gain on the detector model while the summation variances are controlled by `integration.summation.detector_gain`. GROND's default DIALS sigma auto mode keeps raw DIALS sigmas unless the rows look grossly over-precise; `--dials-sigma-mode detector_gain` forces a `sqrt(gain)` multiplier for diagnostics, and `--dials-sigma-mode raw` reproduces the unmodified DIALS variances. If `dials_bridge_summary.json` shows the median `I/sigma` collapsing by roughly `sqrt(gain)`, the gain was probably applied twice and the run should be repeated with raw sigmas. Also check whether many rows were dropped by `dont_integrate` or bad-pixel flags, because failed background modelling can otherwise look like unrealistically precise weak data.

If `wR2all` looks much worse than `R1all`, first check the endpoint `iteration_history` in `summary.json` or the endpoint `.lst` files. Full auto should show SHELXL's recommended `WGHT` being promoted into the active input on later cycles. A first-cycle endpoint can look artificially poor if the starting model carried a stale high `WGHT`; the final reported endpoint should come from the last promoted-weight cycle.

If adding crystals makes the combined raw-row statistics worse, inspect `dataset_policy/final/REPORT.md`. GROND should keep the strongest crystals as `primary`, keep useful but noisier crystals as `downweight`, and move weak nonredundant crystals to `completeness_only` so they can help endpoint coverage without steering the shared raw-row regularization.

If a multi-crystal corrected-HKL endpoint is worse than the best single crystal or an XSCALE-style kinematic baseline, inspect `runs/combined/combined_summary.json` and confirm that `interdataset_scale.mode` is `overlap`. Large scale factors are a warning that the raw integrations were on very different arbitrary scales, but this is exactly why GROND now normalizes corrected reflections after the dynamical correction and before endpoint merging.

If a full run is slow, use `--fast` first, then inspect `logs/commands.jsonl` to see which stage dominates. The final high-basis prediction is usually the expensive step.

If a branch gives a suspiciously good strong-reflection R factor but poor all-row statistics, inspect the forensic reports before adding more beams. Weak-row, resolution, background, scale, and geometry trends often explain the gap better than a larger blind grid.
