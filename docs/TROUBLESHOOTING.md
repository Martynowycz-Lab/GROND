# Troubleshooting

## Installation

If `grond doctor --workflow small-molecule --config grond.config.json --probe-bloch --strict` fails, the Python environment or SHELXL is not ready. Reinstall the package in a clean Python 3.11 to 3.14 environment, check that NumPy, SciPy and Gemmi import, and put the absolute SHELXL path in the config. Plain `grond doctor` checks only the core and only reports.

If the doctor output has no `grond` block, or `grond.install_status` is `stale_or_incomplete_install`, an old installation is in use even though the command runs. Remove the old environment and reinstall rather than installing over it.

If `grond auto` stops at once with a missing-tool message, run `grond init-config --out grond.config.json`, fill in the absolute path of `shelxl`, and rerun with `--config grond.config.json`. `--allow-missing-tools` is for incomplete diagnostic runs only. `shelx2map` is needed only for `--make-maps` or `grond finish --map-mode shelx2map`; the native maps of `grond finish` need no external program.

## Inputs

If the run stops during the input check, open `decisions/<dataset>/preflight.json`. Common causes are missing numeric rows, too few positive intensities, non-positive sigmas, or geometry columns that are missing or badly tailed.

If transform selection reports that every unit-cell metric candidate was rejected, compare the source and model cells printed in the message. GROND handles the usual XDS case in which `INTEGRATE.HKL` declares P1 while `CORRECT` and the model use a permuted, symmetry-constrained setting. That fallback still rejects a transform whose reciprocal metric differs by more than 5% or whose axis lengths differ by more than 3%. A failure beyond those limits usually means the wrong model and data pair, a non-isomorphous crystal, missing XDS header information, or indexing that should be repaired rather than bypassed with `--trust-hkl-transform`.

If `grond auto` stops with an SFAC message, run `grond sfac-check --structure model.res`. Element-only lines such as `SFAC C H N O` rely on SHELXL's built-in X-ray table and are not a valid electron model. Reopen the structure in the electron-scattering Olex2/SHELXL setup that produced it and save a RES with numeric electron SFAC rows (`SFAC C 0.1361 0.3731 ...`). Every element has its own coefficients; do not copy one element's row to another. The check also warns when the `CELL` wavelength is a placeholder or an X-ray value; GROND then writes 0.025079 Å into its `.ins` files.

GROND writes corrected reflections as `HKLF 4`. A model with `HKLF 3` can be used as the start model. `ACTA` is removed from the files GROND writes. Models with `HKLF 5`, `TWIN`, `BASF`, `SWAT` or a SQUEEZE/solvent mask are not supported and are rejected before the run starts.

## DIALS

If the import fails with an error about `dials` or `dxtbx`, pass the DIALS Python with `--dials-python /path/to/dials.python`. GROND can live in its own environment; the import runs the DIALS Python only to read the `.refl` and `.expt` files.

If the import reports more than one experiment, run `dials.split_experiments combined.expt combined.refl`, import each `split_N.refl`/`split_N.expt` pair into its own directory, and list those directories separately in `grond prep`.

If the import stops at the geometry check, open `dials_bridge_summary.json` in the output directory. GROND refuses to write `INTEGRATE.HKL` when the 95th percentile of |Sg| exceeds 0.005 Å⁻¹. This usually means the `.refl` and `.expt` do not belong together, the scan-varying crystal model was not kept, or the geometry is not supported. Fix the input rather than widening any search.

If DIALS gives reasonable R1 but much worse weighted residuals than XDS on the same frames, look at the `sigma_model` block of `dials_bridge_summary.json`. Put the detector gain on the detector model at import (`dials.import ... panel.gain=<gain>`), not in `integration.summation.detector_gain`. The default `--dials-sigma-mode raw` keeps the DIALS variance; `detector_gain` and `auto` are diagnostics. A median I/σ that drops by about √gain means the gain was applied twice. Also check how many rows were dropped for `dont_integrate` or bad-pixel flags.

## Refinement

If wR2 is much worse than R1, open the error-model record of the endpoint (`dataset_sigma` and `error_model_passes` in `combined_summary.json` for several crystals, the `post_bloch_sigma` block of the endpoint `manifest.json` for one) and `shelxl.wght_convergence`, which lists the applied and suggested WGHT of every SHELXL call. The default error model σ'² = k²(σ² + (uP)²) is fitted per crystal on working observations; if too few observations are usable it stops with a message that names the alternative (`--dataset-sigma-policy uniform_z_rms` or `none`). With the single-factor `uniform_z_rms` policy (also used by `grond manual` for one crystal), the status `ok_clipped` means the factor reached its bound (0.25 to 4); treat the weighted statistics with care and check the observation residuals, the transform, the resolution limit and the maps.

If SHELXL reports `NON POSITIVE DEFINITE` atoms, GROND repeats the same call from its unchanged input with `ISOR 0.01 0.02` on the NPD atoms, and with `ISOR 0.005 0.01` on an atom that stays NPD (`--shelxl-npd-recovery isor`, the default). The restraints stay in the model and are reported as warnings, since they can also hide a real error such as the wrong hand. Every attempt is kept under `shelxl_attempt_iter_*`. If the recovery fails, the round keeps the previous clean model. The fields to read are `npd_count`, `npd_atoms`, `npd_recovery`, `split_count` and `safe_for_feedback`.

If a round stops with `invalid_structure_model`, missing statistics or an empty `.res`, the previous SHELXL call did not write a usable model. GROND requires a parseable electron-SFAC model with atoms, complete R1/wR2/N statistics and no NPD atoms, and records `endpoint_res_health`. Read that round's `.lst`, `.res` and `.ins` files to see why SHELXL failed.

If adding crystals makes the statistics worse, read `dataset_policy/final/REPORT.md`. Strong crystals should be `primary`, useful but noisier ones `downweight`, and weak crystals that add unique reflections `completeness_only`. All crystals are on one absolute scale; `interdataset_scale_diagnostic` in `runs/combined/combined_summary.json` should be close to 1. A value far from 1 points to a crystal whose correction or scale disagrees with the others.

If a strong-reflection R factor looks very good but the all-observation statistics are poor, read the forensic reports in `forensics/<dataset>/` before adding beams. Weak-observation, resolution, background, scale and geometry trends usually explain the gap.

## Speed

If a run is slow, try `--preset fast` first, then read `logs/commands.jsonl` to see which stage takes the time. The final high-basis prediction is usually the most expensive step.
