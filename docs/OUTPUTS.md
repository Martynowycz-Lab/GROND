# Outputs

This page describes the files of a `grond auto` run directory. The files to open first are listed in [USAGE.md](USAGE.md).

## Summary files

`TABLE1_FINAL_STATISTICS.md` is the first statistics file to open after `grond auto` or `grond finish`; `TABLE1_FINAL_STATISTICS.json` is its machine-readable companion. It has three panels that must not be mixed:

- **A. SHELXL statistics**: `R1gt`, `R1all`, `wR2all`, GooF, reflection, parameter and restraint counts, `WGHT`, EXTI and the files of the conventional and the GROND models. A `held-out orbits` row next to a model gives SHELXL's statistics of that model on the free reflections ([below](#shelxl-endpoint-on-held-out-orbits---endpoint-free-validation-on)). With `--free-set off` the final all-data refinement is the last row.
- **B. Observation statistics** of the matched Born and Bloch predictions, unmerged (`Robs`, `Rall`, `wRobs`, `wRall`) and after merging symmetry equivalents (`MRobs`, `MRall`, `MwRobs`, `MwRall`).
- **C. Held-out validation**: R factors on the free set with their leak flags ([below](#held-out-validation-and-run-contracts)); in the JSON, `heldout_validation`.

Cells print `n/a` when a calculation was not run.

`JANA_ROW_STATISTICS.md`, `.json` and `.tex` hold the pooled observation statistics, calculated by GROND with the JANA definitions of Klar et al.: observed means `Iobs > 3σ(Iobs)`, `Rall` and `wRall` use all observations, the instability factor is `u = 0.01`, and observations with `Iobs < 0.01σ(Iobs)` use `Iobs = 0` and `σ(√Iobs) = 5√σ(Iobs)`. σ is the uncertainty of the integration, not a fitted one. The `MR` values merge symmetry equivalents with propagated uncertainties.

`FINAL_SUMMARY.md` is short: the hand and the evidence for it, the indexing transform, the chosen model, the held-out validation and the files to open next. With the default `--report-layout heldout-first`, held-out validation and hand evidence come first, and the Klar–Palatinus ZKP values are labelled as diagnostics. The hand is decided by the held-out sign test ([USAGE.md](USAGE.md)). Displayed ZKP values are oriented so that positive favours the selected coordinates; the raw signed statistic is in the decision files. `conditional_inversion_preference` means a proper transform was used but the sign convention of the acquisition and import has not been calibrated externally; `parity_unresolved` means no named configuration can be claimed.

`QUICK_STATS.md` is the longer audit behind it: transform and determinant, initial and final hand, the chosen thickness, mosaic spread, damage and scale settings, exact file paths, stage timings, and the requested-resolution tables. It also carries the held-out validation block.

At the end the terminal prints `GROND finished successfully`, the held-out R factors, the hand, the observation and SHELXL statistics, and the files to open first. The stage count shown during the run is an early estimate. `--json` prints a compact machine-readable payload instead.

`run_config.json` records the options, preset, worker count and dependency checks. `run_prep.json` records the structure and reflection files used. `logs/commands.jsonl` records every command with its timing and return code ([FILE_FORMATS.md](FILE_FORMATS.md)).

XDS and DIALS integrations of the same frames can differ in filtering, partiality, scan-varying geometry, uncertainties and the set of observations. GROND may then find the same transform and hand and similar R factors while thickness, frame offset and frame-scale settings differ. Read those values as settings fitted to one integration.

## Held-out validation and run contracts

`HELDOUT_VALIDATION.md` and `HELDOUT_VALIDATION.json` (schema `grond.heldout_validation.v1`) are written at the run root; `grond finish` writes them again. The free set is about 10% of the unique reflections, chosen by a salted hash (`--free-fraction 0.10`, `--split-salt`), the same in every crystal. The predictions are the run's own. Frame scales and the error model are fitted on working observations, and free reflections enter no SHELXL refinement up to the reported model; with the default `--free-set off` that model is then refined once more with them ([below](#final-all-data-refinement---free-set-off)), and the held-out numbers describe the model before that step. Reflections listed in `--exclude-orbits-json` are dropped first.

R is amplitude R1 on √max(I, 0) in percent: `R_all` on every observation and `R_obs` on `Io > 3σ`. The 95% intervals come from bootstraps over unique reflections (Friedel mates merged): 2000 resamples, stratified in 5 resolution shells, fixed seed. For each crystal and pooled over crystals the report gives the held-out values, the same model on working observations, and the gap (held-out minus working, in percentage points). The JSON also holds the free-set record per crystal, the excluded-reflection list and the observations it removed, whether the last SHELXL round and the final observation table used the same frame-scale settings (`scale_contract_consistency`), the hand decisions, the SHELXL round trajectory, a parameter ledger and the wall time per phase.

The parameter ledger counts physics parameters separately from frame-scale parameters; P-spline scales are reported as effective degrees of freedom (with `n_basis` and `lambda`), never added to integer counts. If the report cannot be built it records `status: failed` with the error, and the other reports are still written.

Each crystal and the pooled result carry leak flags. `yes` means the held-out value may be optimistic by an unknown amount; the evidence text names the record behind each flag.

| Flag | `yes` means | Default |
|---|---|---|
| frame-scale fit | a final table's frame scales were fitted with free observations | no |
| nuisance selection | a nuisance choice read free-observation statistics | no |
| dataset policy | crystal roles or weights used free-observation statistics | no |
| SHELXL HKL | free reflections were in an HKL that refined the model | no |
| round selection | the kept SHELXL round was chosen on free-observation R_obs (`--shelxl-feedback-keep best-free`) | no (`last`) |
| hand decision | a hand decision read free observations | yes when the hand was tested |
| start model (report only) | the start model was refined on the free reflections | no when `controls/leak_free_start` ended `ok` |

The hand flag is expected: the held-out sign test uses the free reflections as evidence, so free-set R after the decision is conditional on that one binary choice.

`decisions/hand_decision_heldout_early.json` (with the default `--hand-decision-stage early`) and `decisions/hand_decision_heldout_final.json` record each hand check: `evidence`, `decision_authority`, `rule`, the `pooled` and `per_crystal` evidence, the matched Born check (`born_control`), `supports_switch`, `rule_met`, `reasons`, `errors`, the hand before and after, `switched`, the all-observation ZKP (`legacy_all_row_zkp`) and, in the final file, `early_final_agreement`. `status` is `complete` or `error`; a failed check still writes the file and never switches the hand. A one-crystal run with `--no-hand-control` or a centrosymmetric model writes neither file.

Signs are relative to the hand that entered the check. `orbit_sign_z > 0` favours the inverse. `delta_r_obs.delta` and `delta_r_all.delta` are R(inverse) − R(selected) in percentage points, so a negative value favours the inverse. The hand switches when `orbit_sign_z >= --hand-flip-orbit-z` (3.0) and `delta_r_obs.ci95[1] < 0`. The final check applies the same rule to the refined model. The per-crystal `subsets` split the sign test into out-of-fold working reflections and free reflections. `legacy_all_row_zkp.z_positive_favors_selected` has the opposite orientation.

`decisions/auto_physics_contract.json` (schema `grond.auto-physics-contract.v1`) is written after the leak-free start, before the first Bloch stage. For each crystal it records whether the reflection file carries DIALS frame ranges, the requested and resolved `sg_window_mode`, `sg_window` and `sg_window_max`, the final and screen profile points with their source, `max_support_sg`, and the damage conversion. Run-wide it records the mosaic-spread and damage grids, `physics_gate_selection` and `thickness_gate_split`. Crystals from different integration programs can resolve to different rocking windows and point counts.

`decisions/cross_crystal_split_check.json` (schema `grond.cross_crystal_split_check.v1`) is written by multi-crystal runs after the transforms are chosen and before any hand, thickness or structure stage. No unique reflection may be free in one crystal and working in another, and no excluded reflection may be used. With `status: consistent` it records per-crystal counts. With `status: conflict` it lists the conflicting reflections and the run stops ([MULTI_DATASET.md](MULTI_DATASET.md)).

`controls/leak_free_start/` holds the conventional refinement of the supplied model against the baseline HKL with the free and excluded reflections removed (`--leak-free-start on`, the default; it needs `--baseline-hkl`):

- `start.hkl`: the baseline HKL without the free and excluded reflections;
- the SHELXL inputs and `start_callNN.res` of each call;
- `start_final.res`, the start model of every later stage;
- `summary.json`: input files, free-set settings, reflections kept and removed, shift/esd and NPD atoms of each call, `res_sha256` and `status`.

The refinement is conventional and does not depend on the hand. It keeps the supplied instructions, EXTI and restraints, promotes SHELXL's suggested weights, and runs up to 15 calls of `L.S. 10` until |max shift/esd| < 0.01. NPD atoms get the same ISOR recovery as the SHELXL rounds ([TROUBLESHOOTING.md](TROUBLESHOOTING.md)), recorded under `npd_recovery`. A status of `not_converged`, `shelxl_failed` or `npd` continues from the supplied model with a warning, and the held-out report then flags the start model. SHELXL output goes to `logs/leak_free_start_shelxl.log`. `grond_summary.json` holds the same record as `leak_free_start`, or `disabled` or `unavailable` (no baseline HKL) when the stage did not run. A start model refined on every reflection has already fitted the free reflections, which would make every held-out number optimistic; `--leak-free-start off` starts from the supplied model anyway.

### SHELXL endpoint on held-out orbits (`--endpoint-free-validation on`)

The held-out statistics above use the Bloch predictions. This block scores the SHELXL models themselves: each final SHELXL model is run through SHELXL with `L.S. 0` (no parameter moves; final WGHT, BASF and EXTI) on its own working HKL plus the free reflections, flagged with SHELXL's free-R flag (`-1`, `L.S. 0 -1`). One run gives the working statistics exactly as the refinement reported them and SHELXL's `R1(Free)` at the model's own scale. It is reporting only: nothing is chosen on these numbers.

- **GROND endpoints** (merged and unmerged, multi-crystal): the free observations are corrected exactly like the working ones, with the same final-model correction, the same exclusion rule, the crystal's error model fitted on working observations, the same weights, merge and HKL factor. Nothing is fitted to them.
- **Conventional** (`controls/leak_free_start`): the free reflections of the baseline HKL are scored with `start_final.res` and its final WGHT. This is the conventional R_free on the same reflections.

R1gt (Fo > 4σ(Fo)), R1all and N are SHELXL's printed values. SHELXL prints no free wR2 or GooF, so GROND calculates them from the `.fcf` of the same run (`LIST 4 1000`) with the final weights; the free GooF divides by N, since no parameter was fitted to these reflections. Each run checks that the `L.S. 0` working statistics reproduce the model's own within one unit of the last printed digit, and that no refined parameter moved beyond SHELXL's storage precision (one unit of the last written digit for parameters tied by a special position; riding hydrogens are reported separately). Any failure gives a status other than `ok`.

| File | Contents |
|---|---|
| `runs/combined/combined_endpoint_lam_1p0/free_validation/free.hkl`, `validation.{hkl,ins,res,lst,fcf}` | the corrected merged free reflections and the scoring run |
| `runs/combined/combined_unmerged_endpoint_lam_1p0/free_validation/free_unmerged.hkl`, `validation_unmerged.*` | the same for the unmerged HKL |
| `combined_summary.json` → `combined_endpoint.free_validation`, `combined_unmerged_endpoint.free_validation` | status, free and working statistics, the gap, the reproduction checks, the free observations per crystal and their RMS(z) |
| `controls/leak_free_start/free_validation/` and `summary.json` → `free_validation` | the conventional block |
| `HELDOUT_VALIDATION.md` (section `SHELXL endpoint on held-out orbits`), `HELDOUT_VALIDATION.json` → `shelxl_endpoint_free` | one row per model: N, R1gt, R1all, wR2, GooF on free and working reflections, and the gap |

`--endpoint-free-validation off` skips the block. One-crystal runs record only the conventional row during the run; `grond validate-endpoint` scores the GROND model afterwards.

`grond validate-endpoint --run-dir RUN --out-dir NEW [--endpoint published|final]` calculates the block for a finished run without changing it (`NEW` must be outside `RUN` and new or empty; the command checks that no file in `RUN` changed). With `--endpoint published` (default) it scores the model the run reported: it rebuilds the working HKL from the recorded commands, requires it to equal the published HKL byte for byte, and scores the published `.res` at `L.S. 0` with the weights its last refinement applied. With `--endpoint final` it reruns the final merge of a multi-crystal run with free validation switched on and requires the same HKL, model and statistics. In both modes it rebuilds the leak-free start's free set, requires it to reproduce `start.hkl` byte for byte, and scores `start_final.res`. `NEW/ENDPOINT_FREE_VALIDATION.md` and `.json` hold the results and the reproduction checks. The exit status is 1 when anything is not reproduced or not `ok`; the error names the cause (for example `published_endpoint_changed` or `work_hkl_not_reproduced`).

### SHELXL rounds

The rounds are recorded in `runs/shared_shelxl_feedback/multi_shelxl_feedback_loop_summary.json` (several crystals) or in `shelxl_feedback_loop_summary.json` under `runs/<dataset>/adaptive_shelxl_feedback/` (one crystal). `trajectory` has one entry per round with:

- free and working `R_obs`/`R_all` (percent) of the model entering the round;
- SHELXL R1gt, R1all, wR2 and GooF, the largest and mean shift/esd, the number of calls and whether they converged;
- the largest non-hydrogen coordinate shift from the entering model (Å);
- NPD count, ISOR atoms, observations excluded for an extreme correction ratio, and whether DAMP is present;
- the thickness re-selected for the new model (`thickness`);
- `adopted_for_feedback`.

`stop_reason` is `converged` (shift below `--shelxl-feedback-xyz-tol`, 0.002 Å) or `cap` (`--shelxl-feedback-cycles`, 5). Other statuses are stops such as `stopped_adp_pathology`, `stopped_unsafe_endpoint` or `failed_cycle`. `kept_cycle`, `keep_policy` and `kept_reason` name the model handed on (`--shelxl-feedback-keep`, default `last`; `best-free` keeps the round with the lowest free R_obs). Each round fixes the correction for the entering model, so its fixed point is not the full dynamical least-squares solution, and held-out R need not fall every round. Before trusting the kept model, read free R_obs across rounds, the coordinate shifts and any NPD or ISOR entries.

### Continuing a run

`grond refine` (also `grond push-endpoint`) writes `push_endpoint_continuation_contract.json` (one crystal) or `continuation_contract` in `push_endpoint_summary.json` (several crystals); with `--dry-run`, in `push_endpoint_plan.json`. It lists the free-set, frame-scale and SHELXL settings replayed from the run being continued (`split_scheme`, `split_salt`, `exclude_orbits_json`, `free_fraction`, `scale_model`, `scale_spline_*`, `endpoint_include_free_rows`, `endpoint_q_policy`, `rescale`, `shelxl_npd_recovery`, `shelxl_convergence_shift_esd`, `endpoint_wght_mode`, `keep_damp`) with the source of each value:

- `source_summary`: taken from the run being continued;
- `override`: given explicitly on the command line;
- `historical_default_not_recorded`: the run did not record the setting, so a built-in default is used and a warning lists it.

The current `grond auto` defaults are never substituted, so a run continues under its own settings. A one-crystal continuation cannot change the free-set or scale settings. If crystals disagree on a setting, the continuation is refused unless it is given explicitly. `--cycles`, `--shelxl-iterations`, `--endpoint-ls-cycles`, `--shelxl-feedback-xyz-tol` and `--shelxl-feedback-keep` have their own defaults (3, 1, 1, 0 and `last`) and are not replayed.

### Final all-data refinement (`--free-set off`)

With `--free-set off` (the default) the result is `final_all_data/all_data.res` and `all_data.hkl`: the reported model refined once more, with the run's SHELXL settings for that model, against its working reflections plus the free reflections corrected the same way. `final_all_data/all_data_refinement.json` records the source model, the working and free reflection counts, the SHELXL record and the statistics. Table 1 adds it as the last row of Panel A, and the terminal and `QUICK_STATS.md` name it as the final model. The held-out statistics describe the model before this step. The free reflections are available for the merged model of a multi-crystal run and are rebuilt for the corrected HKL of a one-crystal run; any other reported model is marked `not_available` and stays the result. `--free-set on` skips the step and keeps the free reflections out of the final model.

## Other decision records

| File | Written by | Contents |
|---|---|---|
| `decisions/<dataset>/nested_nuisance.json` | nested selection (default) | the request, the cross-validated R_all of every candidate, the choice, grid-edge flags, the best candidate per family; free observations are not used (`free_rows_used: false`) |
| `runs/<dataset>/*_nested_nuisance/`, `runs/<dataset>/*_nested_selected/` | nested selection | the anchor observations and spectra, and the replay of the choice |
| `thickness_update_<label>/` under the SHELXL round directory | thickness update each round | the re-selected thickness and smoothness for each new model |
| `decisions/cell_esd.json` | cell e.s.d. policy | the ZERR e.s.d.s, their source and the symmetry constraints |
| `runs/combined/combined_unmerged_endpoint_lam_1p0/multi_dyn_unmerged.{hkl,ins,res,lst,fcf}` | unmerged HKL (several crystals) | HKLF 4 with one batch per crystal, `MERG 0`, refined `BASF`; `combined_summary.json` → `combined_unmerged_endpoint` |
| `combined_summary.json` → `combined_endpoint.dataset_sigma`, `error_model_passes` | error model | k, u, K and RMS(z) before and after, per-bin tables for working and free observations, each refit with its SHELXL statistics, and the weight of each crystal |
| `shelxl.wght_convergence` in every endpoint summary | weight convergence | every call's applied and suggested WGHT, shift/esd and GooF |
| `<out-dir>/check_shelx.json` | `grond check-shelx` | Olex2/SHELXLE readiness checks of one SHELXL result |
| `decisions/metric_excluded_datasets.json` | metric filter (several crystals) | crystals left out because no transform passed the unit-cell metric filter |
| `decisions/split_summary.json`, `decisions/heldout_stage_<label>.json` | validation console | the free set and the held-out numbers printed after each stage |
| `split_record.json` | `grond splits make` | the free, working and excluded reflections for a new run; `grond splits inspect` checks a record or a run directory |
| `final_polish_diffbloch/`, `grond_summary.json` → `final_polish` | `--final-polish diffbloch` | the polished model, its SHELXL files and the record |
| `PETS_STATIC_AUTO_CONTRACT.json` | `grond auto` on a PETS2 import | the settings changed for PETS data ([INTEROPERABILITY.md](INTEROPERABILITY.md)) |

## Resume, continuation and maps

`grond_progress.json` stores the completed stages. Rerun the same command with the same `--out-dir` to continue after an interruption. A stage is reused only when its command and its recorded input and output fingerprints match; changed source, inputs or options rerun the affected stages.

`grond_continue_checkpoint.json` is written by completed runs. It is the starting point for `grond refine` with your own model ([USAGE.md](USAGE.md)). For one crystal it points to the chosen observation table and settings. For several crystals it records each crystal's final settings, dataset-policy weights and the merge key. It is not crash recovery; interrupted runs use `grond_progress.json`.

`grond finish --run-dir RUN` does not refine. It reloads `grond_summary.json` and `run_prep.json`, writes `FINAL_SUMMARY.md` and `QUICK_STATS.md` again, writes the native dynamical maps in `native_dynamic_maps/` and records everything in `finish_manifest.json`. `grond finish` and `grond publish` print the terminal summary; `--json` prints the compact JSON payload instead.

`native_dynamic_maps/` contains `native_dynamical_coefficients.csv`, `native_dynamical_coefficients.hkl`, `grond_native_dynamical.cif`, `grond_native_dynamic_fo.map`, `grond_native_dynamic_2fofc.map` and `grond_native_dynamic_fofc.map`. The observed amplitude is not read from a SHELXL `.fcf`. It is calculated from the chosen observation table as `|F_obs| = |F_model| √(I_obs / I_dyn_scaled)`, merged by signed index and given the model phase. For several crystals, `finish` uses the observation tables of the merged model and records each one. Fo − Fc uses `[|F_obs| − |F_model|] exp(iφ_model)` and 2Fo − Fc uses `[2|F_obs| − |F_model|] exp(iφ_model)`. These are model-phased Fourier maps with amplitudes inferred through the dynamical intensity ratio, not a measured exit wave or a likelihood-gradient map; judge them with model and omit checks.

`grond finish --run-dir RUN --map-mode shelx2map` (or `--make-maps` in `grond auto`) runs `shelx2map` on the `.fcf` of the corrected HKL. `grond_corrected_fo.map`, `grond_corrected_2fofc.map` and `grond_corrected_fofc.map` use `shelx2map -w 1`; `grond_corrected_fofc_unweighted_diagnostic.map` uses `-w 0`. The `.fcf` is not changed.

### Map conventions and model-box maps

Every unit-cell map GROND writes is a full-cell P1 CCP4 map (start 0, size = sampling), so no viewer has to choose a space-group setting. The native maps are the direct synthesis ρ(x) = (1/V) Σ C(h) exp(−2πi h·x) of the symmetry-expanded coefficients, without F(000) (mean zero) and without ML or σA weighting, on a grid no coarser than `--native-map-spacing` (0.25 Å) and with at least 2 h_max + 3 points on every axis. `shelx2map` maps (rate 1.5, about d_min/3) are expanded with the symmetry operators in the file and rewritten as full-cell P1. `shelx2map` drops reflections below the `.fcf` header `_reflns_d_resolution_high`, so a hand-written `.fcf` needs that value just below its smallest d. The native maps use every eligible observation, free ones included. Maps are not used for any held-out number.

A molecule that crosses a cell edge appears in pieces in a unit-cell map. Beside every unit-cell map, `grond finish`/`publish`, the manual `finish` stage and `--make-maps` also write `<map>_model_box.map`: the same map cut to a box around the whole model plus a margin (`--model-box-margin`, 3 Å; `--no-model-box-maps` turns it off). The box is Gemmi's `calculate_fractional_box(margin)` of the model, rounded outward to grid points. Its values come from the periodic full-cell map, so they equal the unit-cell map point for point. The box file is P1 with the full-cell sampling and a non-zero start index, which PyMOL, ChimeraX and Coot place in the crystal frame. `model_box_reference.pdb` and `.res` hold the model the box is cut around, the run's reported model; bonded fragments split by the stored coordinates are first made whole with lattice translations. `grond never-h` writes its boxes to `maps/model_box/` with `maps/MODEL_BOX_MANIFEST.json`. `finish_manifest.json` records the box, grid, source-map SHA-256, placement and the smallest atom-to-face distance under `maps.model_box`.

### Figure maps (`grond figure-maps`)

`grond figure-maps --run-dir RUN --out-dir NEW [--reference-model FULL_H.res] [--margin-A 3]` puts every map of a finished run in one frame for figures. RUN is only read; NEW must be new or empty and outside RUN, and the manifest records that no file in RUN changed (`run_dir_unchanged`). RUN is a `grond auto` directory or a `grond never-h` directory. The command gathers:

- the native Fo, 2Fo − Fc and Fo − Fc maps;
- the `shelx2map` maps of the corrected HKL (`-w 1`) and the raw Fo − Fc (`-w 0`);
- conventional `shelx2map` maps of the same kinds, when the run has a conventional `.res`/`.fcf` (or `--conventional-res`/`--conventional-fcf`);
- for a never-h directory, every map in `maps/MAP_MANIFEST.json`.

Maps that `finish` has not written are calculated in `NEW/source_maps/`.

Each map's model is then placed on the reference model: `--reference-model`, or by default the run's own model (for a never-h directory, the full-H reference it recorded). The search covers the space-group operations, lattice translations and the allowed origin shifts of the setting, and minimises the RMSD of the common non-hydrogen atoms, matched by label and element or by nearest neighbour. The cell and space group must match the reference. A placement above `--rmsd-max-A` (0.5 Å) is refused. In a non-centrosymmetric group a placement that needs an inversion is a change of hand: it is reported (`best_hand_change_not_applied`, `hand_change_indicated`) and never applied. An improper operation of the group itself (a glide, for example) is used only with `--allow-improper-space-group-operations`. When either fits better than every allowed placement, the placement is refused, because a nearly centrosymmetric molecule could otherwise be overlaid on its enantiomer. An origin shift moves the map with its model (written as `<map>_placed_unit_cell.map`). Every map is then boxed around the reference placement, so maps of different models overlay directly. When a placement is refused, that model's maps are boxed around the model itself (`box_reference: own_model_placement_refused`) and the status is `ok_with_refused_placements`. The command exits 0 for `ok` and `ok_with_refused_placements`.

`NEW` contains:

- `model_box/<map>_model_box.map`, in the map's own units;
- `model_box/<map>_model_box_sigma.map`, holding (value − unit-cell mean) / unit-cell RMS, so a contour of 1.0 is one unit-cell σ in any viewer (switch off per-file normalisation, for example PyMOL `set normalize_ccp4_maps, off`);
- `models/model_box_reference.pdb|res` and `models/<model>_placed.pdb|res`: each model moved by its operation, for figures, not for refinement;
- `FIGURE_MAPS_MANIFEST.json`: the operation, the RMSD and per-atom distances, the box and grid, and each source map's SHA-256;
- `README.md`: a table of maps, weighting, box grid and unit-cell RMS.

## Observation tables and SHELXL files

`runs/<dataset>/.../native_dyn_calc.csv` is the observation table: intensity, uncertainty, dynamical and Born predictions, scale, resolution factor, the `split` (`work` or `free`) and the settings. P-spline runs add `scale_angle_deg`, the scan angle of the frame-scale model. `summary.json` files hold machine-readable results and `REPORT.md` files the same for reading.

Each SHELXL result records `iteration_history`: per call, `R1gt`, `R1all`, `wR2all`, GooF, the applied and suggested `WGHT`, and the mean and largest shift/esd of the last cycle. The record also states `convergence_shift_esd`, `converged`, the NPD recovery, `keep_damp` and the DAMP/ISOR state of the written model. With the default `--endpoint-wght-mode converge`, each SHELXL step starts at `WGHT 0 0` and applies SHELXL's suggested weights on every later call, until the applied and suggested weights agree (|Δa| < 0.002, |Δb| < 0.02) and |max shift/esd| < 0.01, at most `--wght-converge-max-calls` (20) calls. Judge a result from its last call and the `.lst` named in `QUICK_STATS.md`.

The uncertainties written for SHELXL come from the per-crystal error model σ'² = k²(σ² + (uP)²), P = (max(Iobs, 0) + 2K Fc²)/3, fitted on working observations (`--dataset-sigma-policy error_model`): u makes the residuals flat across intensity bins and k makes their RMS 1. The merged HKL of several crystals is not rescaled again. R1gt and R1all depend least on the weights; read wR2all and GooF together with the error model.

The endpoint manifest also records `mod_p95`: for each observation, the multiplicative correction actually applied, divided by its geometric median, expressed as a symmetric fold change max(c/c₀, c₀/c); `mod_p95` is its 95th percentile. A large or missing value is a warning, not a failure.

A corrected-HKL directory holds `dyn_nuisance_corrected.hkl`, the unmerged companion HKL and the `.ins/.res/.lst/.fcf` of the kept SHELXL call. Every safe call is also kept as `shelxl_safe_iter_NN.{ins,res,lst,fcf}`. When a later call is rejected (NPD after the recovery, a failed or invalid result), the last safe call's files are put back under the endpoint's name and the rejected call's are kept as `shelxl_rejected_iter_NN.*`. By default the corrected HKL:

- is on the absolute scale κ|F|² · Iobs/Icalc (κ = 10, times `hkl_safe_factor`); the SHELXL overall scale is reset to match at each step;
- leaves out the free reflections;
- drops observations whose correction ratio lies outside 0.05–20 times the crystal's median, counted in `n_excluded_q_rows`.

The endpoint `.res` files are tidied for reopening: the `L.S.` count of the step (1 for per-crystal stage results, `--shelxl-feedback-endpoint-ls-cycles` (10) for the SHELXL rounds and merged results); no DAMP unless `--shelxl-carry-damp`; `HklSrc` pointing to the corrected HKL; the active `WGHT` before `END`; no `ACTA`; `HKLF 4`; no SHELXL suggestions after `END`. EXTI stays off.

`hkl_safe_factor` is one common factor on intensity and sigma, used only to fit the `F8.2` fields; it keeps every I/σ. `hkl_precision` reports positive written sigmas and the rounding bound relative to sigma. Invalid written values block the model from being reported; large rounding is a warning.

## Other directories

- `forensics/<dataset>/`: residual, pre-merge, damage and observation-correction audits. Observation polish is not applied under the default SHELXL settings and records `not_applicable_handoff_contract`.
- `scale_polish/<dataset>/`: only with `--scale-model per-frame`. Under the default `--scale-model pspline`, every stage fits smooth P-spline frame scales on working observations (32 basis functions, λ chosen per crystal by inner cross-validation over 0.3–100), and `--scale-polish`, `--multi-scale-contract-selection` and `--endpoint-calibration` resolve to off.
- `endpoint_calibration/`: not written under the defaults; the run records `disabled` with reason `scale_model_pspline`. `grond calibrate-endpoint` runs the replay directly for per-frame scales.
- `dataset_policy/structure_update/` and `dataset_policy/final/` (several crystals): the role of each crystal and its `REPORT.md` ([MULTI_DATASET.md](MULTI_DATASET.md)). `weighted_runs/<dataset>/native_dyn_calc.csv` holds the weighted tables. `combined_summary.json` records the merge key, merge-sigma policy, error model, weights and `interdataset_scale` (mode `none` on the default absolute scale, with the overlap estimate as `interdataset_scale_diagnostic`).
- `resolution_limits/<dataset>/` (and `resolution_limits/combined/` for several crystals): `resolution_shells.csv`, `summary.json` and `REPORT.md` with signal, split-half stability, Born and Bloch `Rall`/`wRall`/`MRall`/`MwRall` per shell, and working/free stability where there are enough free observations.
- `endpoint_chooser/`: `publication_endpoint_choice.json`, `PUBLICATION_ENDPOINT_CHOOSER.md` and `publication_endpoint_chooser.tex` name the reported model, the best single-crystal model and the best rejected model, with the reason for each rejection. A reported model must keep EXTI off, have finite statistics and valid HKL values. In a multi-crystal run a single crystal with a lower residual does not replace the merged model. Large modulation, a GooF far from 1 or a corrected R1all worse than the conventional refinement are warnings.
- `tables/`: `publication_table.md`, `publication_table.tex` and, with a LaTeX engine, `publication_table.pdf`, with the observation statistics by scale and resolution setting. These are observation statistics, not SHELXL statistics, and the two must not be substituted for each other.

[FILE_FORMATS.md](FILE_FORMATS.md) shows small examples of the input and output files.
