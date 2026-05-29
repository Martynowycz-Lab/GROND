# XDS Workflow For GROND

GROND is designed to start from integrated observations and a solved structure. In practice, the cleanest workflow is to keep the conventional structure-solution path separate from the GROND no-profile-fitting integration path. The starting `RES` or `CIF` can be solved from any reasonable intensities, including the best profile-fitted XDS/XSCALE/XDSCONV/SHELXT/SHELXL path. The Bloch-wave row calculations should use summation intensities from a separate no-profile-fitting XDS pass.

## 1. Solve The Structure Conventionally

Use your ordinary small-molecule electron-diffraction path to obtain a chemically sensible starting model. A typical XDS/SHELX route is:

```text
XDS with PROFILE_FITTING= TRUE or the local default
XSCALE
XDSCONV to SHELX HKLF4
SHELXT
SHELXL initial regularization
```

The output from this path should include a starting `model.res` or `model.cif`. It is fine, and usually preferred, if this model was solved against profile-fitted intensities with the best conventional XDS statistics. If you have a solved-model HKL file from XDSCONV/SHELX, keep it even when it came from that profile-fitted path. GROND uses that file as `--baseline-hkl` to compare signed HKLs and intensity overlap before it spends time on expensive Bloch transform scouts.

## 2. Make A Separate GROND Integration Directory

Copy the XDS processing directory to a new directory, for example:

```bash
cp -R xds xds_noprf
cd xds_noprf
```

Edit `XDS.INP` in the copied directory. Reuse the updated geometry and indexing from the successful conventional processing, but turn off profile fitting for the rows GROND will model. The usual GROND input integration is:

```text
PROFILE_FITTING= FALSE
MINPK= 90.0
```

Then rerun XDS in that copied directory:

```bash
xds
```

or, on a machine configured for parallel XDS:

```bash
xds_par
```

If the copied directory already contains the accepted geometry and indexing from the profile-fitted solution run, it is usually enough to rerun the integration/correction part of XDS after changing the profile-fitting flags. When in doubt, rerunning the full copied XDS directory is slower but safer than mixing files from two different reductions.

The important output is `INTEGRATE.HKL`, but GROND also benefits from nearby XDS metadata such as `XDS.INP`, `XPARM.XDS`, `GXPARM.XDS`, `INTEGRATE.LP`, `CORRECT.LP`, and `XSCALE.LP` when available. Keep the directory intact rather than copying `INTEGRATE.HKL` alone.

This no-profile-fitting pass may look slightly worse by ordinary integration statistics, especially at high resolution, because profile fitting is designed to improve conventional intensity estimates. That is expected. GROND needs intensities that are closer to raw summation counts for each observation row, because the Bloch calculation is trying to predict the rocking/partiality behavior itself. A profile-fitted `INTEGRATE.HKL` will often run, but it can double-count or distort partiality assumptions and usually gives poorer dynamical endpoint behavior.

## 3. Check The Inputs

At minimum, you should know these paths:

```text
/path/to/xds_noprf/INTEGRATE.HKL
/path/to/solved/model.res
/path/to/solved/model.hkl
```

The `model.hkl` path is optional but strongly recommended. Without it, GROND can still run transform scouts, but it may spend more time testing possible HKL alignments.

Run:

```bash
GROND doctor --config grond.config.json --probe-bloch
```

before launching a long regularization. This confirms Python dependencies and reports whether SHELXL and `shelx2map` are visible for endpoint statistics and maps.

## 4. Run One Dataset

```bash
GROND auto \
  --config grond.config.json \
  --xds-dir /path/to/xds_noprf \
  --structure /path/to/solved/model.res \
  --baseline-hkl /path/to/solved/model.hkl \
  --out-dir /path/to/grond_out
```

Use `--fast` first if you are checking a new dataset, a new detector setup, or a questionable integration:

```bash
GROND auto \
  --config grond.config.json \
  --xds-dir /path/to/xds_noprf \
  --structure /path/to/solved/model.res \
  --baseline-hkl /path/to/solved/model.hkl \
  --out-dir /path/to/grond_fast \
  --fast
```

## 5. Run Multiple Datasets

For several crystals of the same structure, create a prep file:

```bash
GROND prep \
  --structure /path/to/solved/model.res \
  --baseline-hkl /path/to/solved/model.hkl \
  --dataset crystal_001=/data/project/001/xds_noprf \
  --dataset crystal_002=/data/project/002/xds_noprf \
  --dataset crystal_003=/data/project/003/xds_noprf \
  --out prep.json
```

Then run:

```bash
GROND auto --config grond.config.json --prep prep.json --out-dir grond_multi
```

GROND writes per-dataset decisions, dataset inclusion/downweighting reports, combined endpoint reports, maps when possible, and publication-style tables.

## Common Input Problems

If GROND stops during preflight, inspect `decisions/<dataset>/preflight.json`. Common causes are missing numeric reflection rows, absent sigmas, too few positive observations, badly tailed geometry columns, wrong input directory, or accidentally using a profile-fitted integration when the intended GROND input was the no-profile-fitting directory. The solved structure and `--baseline-hkl` may come from the profile-fitted path; the GROND `--xds-dir` should point to the no-profile-fitting summation path.

If transform scouting is slow, provide `--baseline-hkl`. GROND can often use the solved-model HKL to reduce the candidate transform list before running Bloch calculations.

If endpoint statistics or maps are missing, run `GROND doctor --config grond.config.json --probe-bloch`. Full corrected-HKL endpoint polishing requires SHELXL, and map generation requires `shelx2map`.
