# Quickstart

Install GROND first using [INSTALL.md](INSTALL.md). For XDS data, prepare a no-profile-fitting integration directory as described in [XDS_WORKFLOW.md](XDS_WORKFLOW.md). The short version is that the GROND input directory should contain `INTEGRATE.HKL` from an XDS run with:

```text
PROFILE_FITTING= FALSE
MINPK= 90.0
```

Keep that no-profile-fitting integration separate from the conventional profile-fitted structure-solution path. The starting `model.res`, `model.cif`, and `model.hkl` can come from the best conventional profile-fitted XDS/XSCALE/XDSCONV/SHELXT/SHELXL workflow. GROND's Bloch-wave calculations should then point at the separate summation `INTEGRATE.HKL` directory, because profile-fitted rows already contain profile/partiality assumptions that usually make rowwise dynamical correction worse even when their conventional data statistics look better.

For DIALS data, keep `integrated.refl` and `integrated.expt` from `dials.integrate`. If the detector gain is known, put it on the detector model during import with `panel.gain=<gain>` so the summation variances are correct upstream. GROND will import the summation rows directly, preserve the signed `miller_index`, use the DIALS scan-varying orientation matrices for excitation-error calculations, record the DIALS `bbox` z-range for diagnostics, drop severe DIALS integration-quality flags, and record whether any additional detector-gain sigma scaling was still needed. If the gain was already supplied to DIALS, the default bridge policy should keep the raw DIALS sigmas rather than applying the gain a second time. The starting model and optional baseline HKL can still come from any conventional solution route.

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

Create the config with `GROND init-config --out grond.config.json`, then fill in any external executables that are not already on `PATH`. The important ones for the default full endpoint are `shelxl` for corrected-HKL polishing/statistics and `shelx2map` for maps. The command can omit `--config` when those programs are already discoverable, or when `GROND_CONFIG` points to the file.

This is the recommended production command. It automatically uses the solved/XSCALE/SHELX baseline HKL to prefilter indexing transforms when that metadata is available, then tests any surviving common transforms with a cheap pass followed by a better N/Q rerank of the top candidates. It also checks the alternate polarity when appropriate, thickness and geometry terms, raw-row resolution envelopes, absorption-like terms, jitter, sigma/weak-row behavior, detector-position scale fields, adaptive final basis size, no-EXTI endpoint lambda candidates, active SHELXL `WGHT` promotion, maps, forensic reports, adaptive frame-scale compression, a publication endpoint chooser, and publication tables. The final polish keeps Q=20 and chooses N from the observed rowwise support scan as `max_observed_basis + 10`, limited to 500 by default, with support top-up at least as deep as the selected data resolution. Multi-dataset endpoints get one overlap-derived post-dynamical scale per crystal before merging, so arbitrary raw integration scales do not survive into the final combined HKL. The default endpoint-B grid is `0.0`; nonzero endpoint-B scans are available only when explicitly requested with `--endpoint-b-grid`, and GROND treats them as diagnostic rather than final because they can improve endpoint R factors while making ADPs and maps worse.

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

`--full-fast` is different from `--fast`. It still runs the automatic transform, polarity, thickness, nuisance-feature, lightweight structure-update, scale-compression, forensic, and endpoint-reporting steps, but it pins final Bloch work to N=24/Q=10 and uses a single endpoint polish pass. The lightweight structure-update budget is chosen automatically from the number of non-hydrogen atoms in the model; explicit `--full-fast-structure-rows` and `--full-fast-structure-max-nfev` values override that schedule. Structure blocks also use the default data-RMS plateau gate, so a run should stop a stale xyz or ADP loop instead of spending days on calls that no longer improve the row objective. For an even cheaper stress test, add `--full-fast-n-basis 10 --full-fast-n-points 5`, then treat the result as a scouting endpoint rather than the preferred default.

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
