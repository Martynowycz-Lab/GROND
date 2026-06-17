# Quickstart

Install GROND first using [INSTALL.md](INSTALL.md). For XDS data, prepare a no-profile-fitting integration directory as described in [XDS_WORKFLOW.md](XDS_WORKFLOW.md). The short version is that the GROND input directory should contain `INTEGRATE.HKL` from an XDS run with:

```text
PROFILE_FITTING= FALSE
MINPK= 90.0
```

Keep that no-profile-fitting integration separate from the conventional profile-fitted structure-solution path. The starting `model.res`, `model.cif`, and `model.hkl` can come from the best conventional profile-fitted XDS/XSCALE/XDSCONV/SHELXT/SHELXL workflow. GROND's Bloch-wave calculations should then point at the separate summation `INTEGRATE.HKL` directory, because profile-fitted rows already contain profile/partiality assumptions that usually make rowwise dynamical correction worse even when their conventional data statistics look better.

For DIALS data, keep `integrated.refl` and `integrated.expt` from `dials.integrate`. If the detector gain is known, put it on the detector model during import with `panel.gain=<gain>` so the summation variances are correct upstream. GROND will import the summation rows directly, preserve the signed `miller_index`, use the DIALS scan-varying orientation matrices for excitation-error calculations, record the DIALS `bbox` z-range for diagnostics, drop severe DIALS integration-quality flags, and record the empirical sigma calibration or fallback detector-gain decision. If the gain was already supplied to DIALS, the default bridge policy should still estimate whether the reported sigmas match equivalent-row scatter rather than applying the gain a second time. The starting model and optional baseline HKL can still come from any conventional solution route.

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

Create the config with `GROND init-config --out grond.config.json`, then fill in any external executables that are not already on `PATH`. The important ones for the default full endpoint are `shelxl` for corrected-HKL polishing/statistics and `shelx2map` for maps. The command can omit `--config` when the needed programs are already discoverable, or when `GROND_CONFIG` points to the file.

Use `GROND endpoints --config grond.config.json` to see the available model-update endpoints. SHELXL is the implemented production endpoint. `custom` is a command-template adapter for a lab-specific one-cycle refiner. `bloch_only` is always available and means GROND will simulate Bloch intensities, optimize nuisance terms, write diagnostics, and report polarity/residual behavior without updating the model through an external refiner.

This is the recommended production command. It starts with cheap checks and only spends time on expensive Bloch calculations after the data pass those checks.

When a baseline HKL is available, GROND uses it to reduce transform ambiguity before running Bloch scouts. Any remaining transform candidates are tested cheaply first, then the best candidates are rerun with a stronger basis and quadrature.

The automatic path checks polarity, thickness, simple geometry terms, raw-row resolution behavior, absorption-like terms, jitter, sigma behavior, detector-position scales, full-correction endpoints, maps, forensic reports, and adaptive scale compression. The final polish keeps Q=20. It chooses N from the observed rowwise support scan as `max_observed_basis + 10`, with a default cap of 500. The support top-up is at least as deep as the selected data resolution.

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

Fast mode intentionally skips the expensive final polish decisions. Use it to catch wrong paths, wrong transforms, bad integrations, or implausible thickness/polarity quickly; use the plain `GROND auto` command for the final answer.

Run the middle-speed automatic profile:

```bash
GROND auto \
  --xds-dir /data/my_dataset/xds \
  --structure /data/my_dataset/model.res \
  --baseline-hkl /data/my_dataset/model.hkl \
  --out-dir grond_full_fast \
  --full-fast
```

`--full-fast` is different from `--fast`. It still runs the automatic transform, polarity, thickness, nuisance, lightweight structure-update, scale-compression, forensic, and endpoint-reporting steps. It pins final Bloch work to N=24/Q=10 and uses one endpoint polish pass.

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

If a run is interrupted, rerun the identical command. GROND records completed stages in `grond_progress.json` and keeps per-stage commands and timings in `logs/commands.jsonl`.
