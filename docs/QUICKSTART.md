# Quickstart

Install GROND first using [INSTALL.md](INSTALL.md). For XDS data, prepare a no-profile-fitting integration directory as described in [XDS_WORKFLOW.md](XDS_WORKFLOW.md). The short version is that the GROND input directory should contain `INTEGRATE.HKL` from an XDS run with:

```text
PROFILE_FITTING= FALSE
MINPK= 90.0
```

Keep that no-profile-fitting integration separate from the conventional profile-fitted structure-solution path. The starting `model.res` and `model.hkl` can come from the best conventional profile-fitted XDS/XSCALE/XDSCONV/SHELXT/SHELXL workflow. A CIF may be used by `GROND run` when it embeds the complete SHELX instruction block. GROND's Bloch-wave calculations should then point at the separate summation `INTEGRATE.HKL` directory, because profile-fitted rows already contain profile and partiality assumptions that usually make rowwise dynamical correction worse even when their conventional statistics look better.

For DIALS data, keep `integrated.refl` and `integrated.expt` from `dials.integrate`. If the detector gain is known, put it on the detector model during import with `panel.gain=<gain>` so the summation variances are correct upstream. GROND imports the summation rows directly, preserves the signed `miller_index`, uses the DIALS scan-varying orientation matrices for excitation-error calculations, records the `bbox` z-range, and drops severe integration-quality flags. The default bridge keeps the DIALS sigma values unchanged. Empirical equivalent-row and detector-gain sigma modes are available only when you request them as diagnostics.

Run the automatic path on one dataset:

```bash
GROND auto \
  --config /data/my_dataset/grond.config.json \
  --xds-dir /data/my_dataset/xds \
  --structure /data/my_dataset/model.res \
  --baseline-hkl /data/my_dataset/model.hkl \
  --out-dir grond_out
```

The equivalent DIALS command is:

```bash
GROND auto \
  --config /data/my_dataset/grond.config.json \
  --dials-refl /data/my_dataset/dials/integrated.refl \
  --dials-expt /data/my_dataset/dials/integrated.expt \
  --dials-python /path/to/dials/python \
  --structure /data/my_dataset/model.res \
  --baseline-hkl /data/my_dataset/model.hkl \
  --out-dir grond_dials_out
```

For other integration programs, first write the portable row-contract files and
turn them into a GROND compatibility directory:

```bash
GROND row-validate \
  --rows /data/my_dataset/rows.csv \
  --metadata /data/my_dataset/rows_metadata.json \
  --write-xds-dir /data/my_dataset/grond_bridge

GROND auto \
  --xds-dir /data/my_dataset/grond_bridge \
  --structure /data/my_dataset/model.res \
  --baseline-hkl /data/my_dataset/model.hkl \
  --out-dir grond_portable_out
```

See [PORTABLE_INPUTS_AND_ENDPOINTS.md](PORTABLE_INPUTS_AND_ENDPOINTS.md) before
writing a wrapper. The signed indices and orientation matrix have to be in the
same solved-model basis, otherwise GROND can compute a very precise answer for
the wrong geometry.

Create the config with `GROND init-config --out grond.config.json`, then fill in any external executables that are not already on `PATH`. For the default full small-molecule endpoint, the only required external executable is `shelxl`. Native dynamical maps from `GROND finish` use the installed Python/Gemmi stack; `shelx2map` is optional for quick corrected-HKL endpoint maps. The command can omit `--config` when SHELXL is already discoverable, or when `GROND_CONFIG` points to the file.

Use `GROND endpoints --config grond.config.json` to see the available model-update endpoints. SHELXL is the implemented production endpoint for the normal small-molecule feedback loop. `custom` is a command-template adapter for a lab-specific one-cycle refiner. `bloch_only` is always available and means GROND will simulate Bloch intensities, optimize nuisance terms, write diagnostics, and report hand/residual behavior without updating the model through an external refiner. Macromolecular tests should use the explicit MTZ layer described in [MACROMOLECULES.md](MACROMOLECULES.md): assign unique-HKL FreeR flags, export corrected `IMEAN/SIGIMEAN`, then run one configured `protein-ls1` endpoint macrocycle.

This is the recommended production command. It starts with cheap checks and only spends time on expensive Bloch calculations after the data pass those checks.

When a baseline HKL is available, GROND uses it to reduce transform ambiguity before running Bloch scouts. Any remaining transform candidates are tested cheaply first, then the best candidates are rerun with a stronger basis and quadrature.

The automatic path checks hand, thickness, simple geometry terms, raw-row resolution behavior, absorption-like terms, jitter, sigma behavior, detector-position scales, full-correction endpoints, maps, forensic reports, and adaptive scale compression. The final polish keeps Q=20. It chooses N from the observed rowwise support scan as `max_observed_basis + 10`, with a default cap of 500. The support top-up is at least as deep as the selected data resolution.

For multi-dataset endpoints, GROND applies one overlap-derived post-dynamical scale per crystal before merging. This removes arbitrary raw integration scale differences from the final combined HKL. The default endpoint-B grid is `0.0`. Nonzero endpoint-B scans are only used when you request them with `--endpoint-b-grid`; GROND treats them as diagnostic because they can lower endpoint R factors while making ADPs and maps worse.

On a shared server, set `--workers` to the number of cores you want GROND to consume. Transform scouts use that budget as an outer queue by default, so a multi-dataset job can test remaining transform candidates across crystals concurrently. Leave `--transform-scout-workers-per-job 1` unless you deliberately reduce `--transform-scout-jobs`; otherwise each scout subprocess would compete with the others for the same cores.

Run a fast triage pass:

```bash
GROND auto \
  --xds-dir /data/my_dataset/xds \
  --structure /data/my_dataset/model.res \
  --baseline-hkl /data/my_dataset/model.hkl \
  --out-dir grond_fast \
  --fast
```

Fast mode intentionally skips the expensive final polish decisions. Use it to catch wrong paths, wrong transforms, bad integrations, or implausible thickness/hand quickly; use the plain `GROND auto` command for the final answer.

After a completed run, open the root-level `TABLE1_FINAL_STATISTICS.md` first for statistics. Its SHELXL endpoint panel reports conventional and promoted corrected-HKL `R1gt`, `R1all`, `wR2all`, GooF, and associated counts. Its separate JANA/PETS-style row panel reports matched Born and Bloch dynamical `Robs`, `Rall`, `wRall`, `MRobs`, `MRall`, and `MwRall`, with optional observed-subset weighted factors. `MR` values are formed after model-symmetry merging; they are not endpoint SHELXL values. When a model/statistic combination was not calculated, Table 1 says `n/a` instead of substituting a different residual family.

Use `JANA_ROW_STATISTICS.md` or `.json` for the canonical pooled row table and `JANA_ROW_STATISTICS_PER_DATASET.*` for individual-crystal values. `FINAL_SUMMARY.md` names the final hand, paired t-style row contrast with `n` and `df=n-1`, selected files, and promotion decision. The paired contrast is diagnostic support, not a calibrated normal Z score or confidence. `QUICK_STATS.md` keeps the fuller audit trail; any internal native-objective residuals there are optimizer diagnostics rather than publication factors. To regenerate these files later without rerunning the physics stages, run:

```bash
GROND finish --run-dir grond_out
```

This writes `native_dynamic_maps/` with native row-level dynamical Fo, 2Fo-Fc, and Fo-Fc CCP4 maps, a coefficient table, and `grond_native_dynamical.cif`. For multi-dataset runs, the manifest and CIF list every selected combined-endpoint input row table used for the native map coefficients. The alias `GROND publish --run-dir grond_out` does the same packaging. If you only want a quick corrected-HKL endpoint-map check from SHELXL's FCF output, use `GROND finish --run-dir grond_out --map-mode shelx2map`.

Run the middle-speed automatic profile:

```bash
GROND auto \
  --xds-dir /data/my_dataset/xds \
  --structure /data/my_dataset/model.res \
  --baseline-hkl /data/my_dataset/model.hkl \
  --out-dir grond_full_fast \
  --full-fast
```

`--full-fast` is different from `--fast`. It still runs the automatic transform, hand, thickness, nuisance, lightweight structure-update, scale-compression, forensic, and endpoint-reporting steps. It pins final Bloch work to N=24/Q=10 and uses one endpoint polish pass.

Before a real run, check the model file:

```bash
GROND sfac-check --structure /data/my_dataset/model.res
```

GROND needs explicit electron scattering coefficient rows in SHELX models. A line such as `SFAC C H N O` is not enough, because SHELXL then uses its built-in table while the Bloch calculation has no explicit electron coefficients to read. The checker also reports whether the `CELL` wavelength looks like a placeholder. Endpoint `.ins` files are normalized to the GROND electron default `0.025079 A` when the source model says `CELL 1 ...` or another clearly non-electron wavelength.

Obtain the numeric SFAC block from the electron-diffraction refinement that
created the model. Olex2/SHELXL electron workflows can write explicit rows into
the saved RES; retain those rows when moving the model. A valid row is numeric,
for example `SFAC C 0.1361 0.3731 ...`, rather than an element list. Coefficients
are element-specific: do not invent them or copy the carbon row to another
element. If the only file available contains `SFAC C H N O`, reopen it in the
electron-scattering refinement used for the solution and save/export a RES with
its numeric electron SFAC records before running GROND.

The lightweight structure-update budget is chosen from the number of non-hydrogen atoms in the model. Use `--full-fast-structure-rows` and `--full-fast-structure-max-nfev` only when you need to override that schedule. Structure blocks use the default data-RMS plateau gate, so a stale xyz or ADP loop should stop instead of spending days on calls that no longer improve the row objective.

For a cheaper stress test, add `--full-fast-n-basis 10 --full-fast-n-points 5`. Treat that result as a scouting endpoint, not the preferred final answer.

Prepare a reusable multi-dataset file:

```bash
GROND prep \
  --structure /data/project/model.res \
  --baseline-hkl /data/project/model.hkl \
  --dataset crystal_a=/data/project/crystal_a/xds \
  --dataset crystal_b=/data/project/crystal_b/xds \
  --out prep.json

GROND auto --prep prep.json --out-dir grond_multi
```

For multi-dataset runs, `GROND auto` writes `dataset_policy/structure_update/` and `dataset_policy/final/`. These reports explain which crystals were kept at full weight, downweighted by sigma inflation, used only for endpoint completeness, or excluded. Leave `--dataset-policy auto` on unless you are deliberately reproducing an older all-crystals-equal run; use `--dataset-policy off` for that comparison.

Multi-dataset runs keep nuisance models per dataset and refine one shared structure model. Each dataset gets its own thickness, frame offset, Sg jitter, absorption decision, scale model, and sigma calibration before corrected rows are merged. The merged endpoint then fits one overlap scale per dataset after the Bloch/Born correction and inflates merged sigmas when datasets disagree more than their formal errors predict. The production shared-model update recalculates the accepted datasets from one model, merges one corrected endpoint HKL, runs one shared SHELXL LS step, and repeats. The slower native finite-difference updater is available with `--structure-update native_gradient` for a matched diagnostic. See [MULTI_DATASET.md](MULTI_DATASET.md) for the full contract.

If a run is interrupted, rerun the identical command. GROND records completed stages in `grond_progress.json` and keeps per-stage commands and timings in `logs/commands.jsonl`.

A completed run prints `GROND finished successfully` followed by the final hand call, JANA/PETS-style Born and dynamical row factors, SHELXL publication endpoint factors, total completed stage time, and the main files to open. The step counter during the run is an estimate; optional gates can make the final completed count smaller than the early denominator. For scripting, add `--json` to print the compact machine-readable payload instead of the human final panel.

## Hydrogen-omit controls

Do not use the built-in strip-H branch as the primary hydrogen-discovery
experiment. `--h-omit on` is a single-dataset retrospective sensitivity
control: it removes H from a final model that has already contained those H
atoms. Multi-dataset use fails explicitly, and the historical `only`/`both`
spellings are deprecated aliases for `on`.

For prospective evidence, solve and refine a separate conventional model that
has never contained H atoms, then run ordinary GROND from that no-H RES and the
same row data. Preserve the zero-H starting model, the complete GROND run,
native dynamical Fo-Fc map, corrected-endpoint Fo-Fc map, conventional no-H
map, and an independently aligned full-H coordinate reference used only after
the maps exist to measure peak distances.

Build the conventional comparator from the same prospective no-H RES/HKL in a
separate directory, keep H absent, run the declared conventional refinement
(for example ten SHELXL cycles), and generate its ordinary Fo-Fc map. Do not
align the reference by maximizing H peaks. If the no-H and full-H solutions do
not already share an origin and proper HKL setting, derive the rigid/origin
alignment from the common non-H atoms and record it before sampling any H site.

Quantify all maps with the public audit command. It verifies that the
prospective model contains zero H atoms, checks map/reference cell
compatibility, preserves hashes, and reports every local positive peak within
0.40 A of the independent H coordinates:

```bash
GROND audit-hydrogen-omit \
  --reference-res /data/project/independent_full_h.res \
  --never-h-res /data/project/never_h_grond/final_no_h.res \
  --map native=/data/project/never_h_grond/native_dynamic_fofc.map \
  --map grond_endpoint=/data/project/never_h_grond/endpoint_fofc.map \
  --map conventional=/data/project/conventional_never_h/fofc.map \
  --radius-A 0.40 \
  --out-dir /data/project/hydrogen_omit_audit
```

The output contains `hydrogen_omit_peaks.csv`, `summary.json`, `REPORT.md`, and
`input_contract.json`. Before reading a map, the command requires compatible
cells and numeric symmetry operations and checks the common labeled non-H atoms
at their minimum-image positions (default RMS at most 0.35 A and no atom beyond
0.75 A). This fails closed on an equivalent but origin-shifted reference rather
than silently erasing the H peaks. Any required alignment must be derived from
the non-H atoms without consulting the omit maps.
