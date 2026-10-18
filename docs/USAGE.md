# Using GROND

There are three ways to run GROND:

| Command | Use it when | What you control |
|---|---|---|
| `grond auto` | you want a result; this is the normal route | nothing: every decision is made and recorded |
| `grond manual <stage>` | you want to inspect or change one stage | each stage: inspect it, change a setting, rerun, continue |
| SHELXL or olex2.refine, then `grond refine` | you want to refine the model yourself | every coordinate update; GROND recalculates the corrections for your model |

`grond`, `GROND` and `python -m grond` are the same program.

## Inputs

You need:

- a SHELX model (`.res` or `.ins`) with numeric electron scattering factors on its SFAC lines; check it with `grond sfac-check --structure model.res`;
- the merged HKLF 4 file that model was refined against;
- the unmerged observations of each crystal:
  - **XDS**: a directory with `INTEGRATE.HKL` from a summation run (`PROFILE_FITTING=FALSE`) and `XPARM.XDS` or `GXPARM.XDS`;
  - **DIALS**: `integrated.refl` and `integrated.expt`.

Put the path to SHELXL in a configuration file (`grond init-config --out grond.config.json`) or pass `--shelxl`.

For one crystal, `grond auto` reads an XDS directory or DIALS files directly. For several crystals, or for `grond manual`, convert each DIALS crystal once and list all crystals in a preparation file:

```bash
grond import-dials --refl c1/integrated.refl --expt c1/integrated.expt \
    --dials-python /path/to/dials.python --out-dir bridges/c1
grond prep --structure model.res --baseline-hkl model.hkl \
    --dataset c1=bridges/c1,dmin=0.80 --dataset c2=/data/c2/xds,dmin=0.80 --out prep.json
```

A converted DIALS directory is used exactly like an XDS directory, and XDS and DIALS crystals can be mixed. `dmin` is 0.70 Å when it is left out. See [XDS_WORKFLOW.md](XDS_WORKFLOW.md), [DIALS_WORKFLOW.md](DIALS_WORKFLOW.md) and [MULTI_DATASET.md](MULTI_DATASET.md).

## Automatic: `grond auto`

```bash
grond auto --prep prep.json --config grond.config.json --workers 40 --out-dir run
grond auto --prep prep.json --config grond.config.json --workers 6 --preset fast --out-dir run_fast
grond auto --xds-dir xds/ --structure model.res --baseline-hkl model.hkl --config grond.config.json --out-dir run
grond auto --dials-refl integrated.refl --dials-expt integrated.expt --dials-python /path/to/dials.python \
    --structure model.res --baseline-hkl model.hkl --config grond.config.json --out-dir run
grond finish --run-dir run
```

The first line is the standard run. `--preset fast` runs every stage, including the held-out validation, at lower fidelity; use it to check a new data set or setup. The third and fourth lines are the one-crystal forms for XDS and DIALS. `grond finish` writes the maps and the final CIF. Add `--dry-run` to any `auto` command to see the plan without calculating.

**What `grond auto` does.**

1. Checks the inputs and the model, and matches each crystal's indexing to the model.
2. Holds out a free set of about 10% of the unique reflections, the same in every crystal, and refines the supplied model kinematically without them (the leak-free start).
3. For a non-centrosymmetric model, compares the model with its inverse on held-out reflections, before any dynamical refinement. For each unique reflection it asks which hand fits better, and switches to the inverse only when the sign test gives z ≥ 3 in favour of the inverse and the paired 95% interval of R_obs(inverse) − R_obs(model) lies entirely below zero. Both hands come from the same calculation. The result states which of the two coordinate sets the data prefer; naming an absolute configuration also needs the sense of the rotation axis and of the processing chain.
4. For each crystal, chooses the thickness distribution (gamma or wedge), mosaic spread, radiation damage and frame-scale smoothness by cross-validation within the working observations.
5. Writes corrected intensities, refines them with SHELXL to convergence, recalculates the corrections for the new model, and repeats until no non-hydrogen atom moves by more than 0.002 Å (at most five rounds).
6. Reports R factors on the free set, with bootstrap confidence intervals, and confirms the hand on the final model.
7. With the default `--free-set off`, refines the chosen model once more against all reflections, working and free, and writes that as the result. With `--free-set on`, the free reflections stay out of the final model as well.

If a run stops, repeat the same command with the same `--out-dir`; completed stages are skipped.

## Results

At the end the terminal prints the held-out R factors, the hand decision and a list of files to open first:

| File in the run directory | What it is |
|---|---|
| `TABLE1_FINAL_STATISTICS.md` (and `.json`) | SHELXL statistics of the conventional and GROND models, R factors of the observations, held-out validation |
| `FINAL_SUMMARY.md`, `QUICK_STATS.md` | hand, indexing, chosen model, one-screen statistics |
| `HELDOUT_VALIDATION.md` | held-out R factors with intervals, and where free observations could have entered |
| the model and corrected reflection file named in `FINAL_SUMMARY.md` | the refined model and its HKLF 4 file |
| `final_all_data/` | the final all-data refinement (with `--free-set off`) |
| `native_dynamic_maps/` | after `grond finish`: Fo, 2Fo − Fc and Fo − Fc maps and a CIF |
| `grond_continue_checkpoint.json` | the saved settings, for `grond refine` |

[OUTPUTS.md](OUTPUTS.md) describes every file.

## Stage by stage: `grond manual`

`grond manual` runs the calculation one stage at a time. Each stage reads `manual_workflow.json`, writes one numbered directory with a `handoff.json`, and prints the next command. `grond manual status --workflow $W` shows what has run.

```bash
grond manual setup --out-dir man --structure model.res --baseline-hkl model.hkl \
    --dataset c1=bridges/c1,dmin=0.80 --dataset c2=/data/c2/xds,dmin=0.80 \
    --config grond.config.json --shelxl /path/to/shelxl --workers 6
W=man/manual_workflow.json
grond manual scout     --workflow $W
grond manual thickness --workflow $W
grond manual hand      --workflow $W                    # writes the evidence and stops
grond manual hand      --workflow $W --choice inverse   # or --choice input, or --skip
grond manual feedback  --workflow $W --cycles 5
grond manual scales    --workflow $W
grond manual finish    --workflow $W
```

| Stage | Open this | It holds |
|---|---|---|
| `00_setup` | `prep.json`, `fresh_conventional_baseline/` | the crystals, the model check, the conventional refinement |
| `01_scout` | `scouted_prep.json` | the indexing of each crystal |
| `02_thickness` | `REPORT.md`, `thickness_scout_table.csv` | thickness and frame offset per crystal, with every trial |
| `03_hand` | `combined_hand_evidence.json` | the evidence for each hand and the recommendation |
| `04_feedback` | `MULTI_ENDPOINT_PUSH_REPORT.md` | the corrected files and the SHELXL model of each cycle |
| `05_scales` | `<crystal>/REPORT.md` | a frame-scale audit (not applied) |
| `06_finish` | `MANUAL_FINAL_SUMMARY.md`, `native_dynamic_maps/` | the final statistics and maps |

To change a setting, rerun that stage with the new option and run forward again; the later stages are marked as out of date. A rerun stage rewrites its own directory, so copy it first if you want to compare. A new model or new data needs a new `setup`.

`grond manual` is simpler than `grond auto`. It has no leak-free start and no cross-validated choice of mosaic spread and damage, its hand stage recommends from the all-observation Klar–Palatinus statistic (|z| ≥ 3.5) rather than the held-out test, and it makes one SHELXL least-squares cycle per round. Use it to look inside the calculation and `grond auto` for the reported result. [MANUAL_WORKFLOW.md](MANUAL_WORKFLOW.md) lists every option.

## Your own refinement: `grond refine`

Start from the checkpoint of an `auto` run, `run/grond_continue_checkpoint.json`, and give each step a new directory.

1. Refine the model in SHELXL or olex2.refine against GROND's corrected HKLF 4 file, as you would any HKLF 4 data. The files contain the numeric electron SFAC lines, `WGHT` and `L.S.`; nothing else is needed.
2. Send the refined model back:

```bash
grond refine --checkpoint run/grond_continue_checkpoint.json --start-res step1/model.res \
    --out-dir recycle1 --cycles 1 --workers 6 --config grond.config.json
```

GROND recalculates every crystal for your model with the saved thickness, mosaic spread, damage, beams, indexing, hand and free set. It refits only the frame scales, the error model and the weights, as the automatic loop does, and writes a new corrected reflection file. Refine that file and repeat. Recycling the automatic run's own last model reproduces its final observations exactly.

`--dataset-thickness-nm c1=NM` and `--dataset-n-basis c1=N` change a saved value on purpose; `crystal_parameters.json` then records the override. Add `--dry-run` to see the plan first.

## Where things are

| Path | Written by |
|---|---|
| `run/` | `grond auto`; `grond finish` adds `native_dynamic_maps/` |
| `man/00_setup` … `man/06_finish` | `grond manual` |
| `recycleN/` | `grond refine` (use a new `--out-dir` each time) |
| `logs/commands.jsonl`, `grond_progress.json` | every command and stage that ran, for resuming |

## Two-axis mosaic cone

For a two-axis Gaussian cone, use `--mosaicity-mode gaussian_cone`,
`--profile-coordinate direct_angle` and
`--profile-normalization target_sg_equivalent_v1`. Set
`--mosaicity-support-sigma-deg` to cover the largest tested cone width. AUTO
uses nested selection so that the zero-width and finite-width candidates share
one declared beam-support envelope. Fixed native calculations and `grond refine`
also retain this convention in their saved configurations and checkpoints.

This opt-in convention uses the exact angular endpoints of the historical
excitation-error window and preserves its intensity normalization. At zero cone
width, the direct-angle and target-Sg calculations agree as quadrature converges.
If the interval exceeds `--profile-angle-max-half-width-deg`, the calculation
refuses to truncate it. Inspect the recorded interval and acquisition geometry
before increasing that bound. Beam support and quadrature still require
convergence checks at the fitted thickness distribution.

The convention preserves the existing reduced-observation model; it does not
reconstruct detector foreground masks or remove assumptions in DIALS partiality
completion. Legacy calculations remain unchanged unless this option is selected.
PETS-owned integration wedges and the experimental `native_structure_update`
path do not support this convention and reject it explicitly.
