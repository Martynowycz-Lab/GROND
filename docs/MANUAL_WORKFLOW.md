# Stage by stage: `grond manual`

`grond manual` runs a simplified calculation as short, resumable stages. Setup writes one `manual_workflow.json`; every later stage reads it, writes its own `handoff.json`, updates the workflow state and prints the next command. No paths, transforms or thicknesses need to be copied by hand.

Use it to look inside the calculation and to try one decision at a time. It is not a stage-for-stage copy of `grond auto`, and its results are not the reported result. Compared with `grond auto` it:

- has no leak-free start and no cross-validated choice of thickness distribution, mosaic spread and damage;
- recommends the hand from the all-observation Klar–Palatinus statistic before refinement, where `grond auto` decides it from held-out reflections ([USAGE.md](USAGE.md));
- makes one SHELXL least-squares cycle per round, where `grond auto` refines SHELXL to convergence every round;
- has no dataset policy, resolution-limit gate or final all-data refinement.

The stages are:

```text
setup -> scout -> thickness -> hand -> feedback -> scales -> finish
```

`grond manual status --workflow RUN/manual_workflow.json` shows what has run, the timings, the chosen model, the final summary and the next command. Add `--json` for the machine-readable form. Every stage has its own help page, for example `grond manual thickness --help`.

## Setup

For two XDS directories or converted DIALS directories:

```bash
grond manual setup \
  --out-dir manual_run \
  --structure model.res \
  --baseline-hkl model.hkl \
  --dataset crystal_1=/path/to/crystal_1,dmin=0.70 \
  --dataset crystal_2=/path/to/crystal_2,dmin=0.70 \
  --config /absolute/path/grond.config.json \
  --shelxl /absolute/path/shelxl \
  --workers 10
```

Setup checks the model's SFAC lines and stores the Python, worker count, config and SHELXL path, so the printed commands of later stages use the same settings. When a baseline HKL and an explicit `--shelxl` path are given, setup also runs a fresh conventional SHELXL refinement in `00_setup/fresh_conventional_baseline/` for the final comparison. For DIALS, convert each experiment first with `grond import-dials` and give the converted directories.

## Indexing transform

```bash
grond manual scout --workflow manual_run/manual_workflow.json
```

This ranks the proper (determinant +1) transforms for every crystal without using the hand or known chemistry. `--skip` keeps the transforms given at setup and records that choice. Skip mode still rejects improper transforms, runs the input check, and keeps the requested partiality and geometry limits.

## Thickness

```bash
grond manual thickness --workflow manual_run/manual_workflow.json
```

The default uses all accepted observations, frame scales without detector terms, no absorption or Sg jitter, and a thickness with a uniform gradient (wedge). It searches the thickness and gradient grids at N=24 beams, then confirms at N=48 within 30 nm of the first choice. `--thickness-grid-nm`, `--gradient-grid`, `--n-basis-grid` and `--n-points` change the search; a single `--thickness-grid-nm` value fixes the thickness at every basis size. `--expanded` adds sensitivity branches.

The thickness cannot be skipped silently, because the feedback stage needs fixed settings. To reuse existing summaries:

```bash
grond manual thickness --workflow manual_run/manual_workflow.json --skip \
  --summary crystal_1=/path/to/summary.json \
  --summary crystal_2=/path/to/summary.json
```

Each reused summary must match its crystal's current HKL, transform, resolution cutoff, partiality floor and geometry percentile. When the summary records the SHA-256 of its source HKL, that must match too, so an HKL edited at the same path fails before feedback. A summary without a recorded hash is accepted with a warning. Use a new setup when the observations change.

## Hand

```bash
grond manual hand --workflow manual_run/manual_workflow.json
```

This compares the thickness-stage model and its inverse with the negative-thickness identity, writes per-crystal evidence and `combined_hand_evidence.json`, and stops. Known chemistry is never used. The recommendation follows the combined all-observation Klar–Palatinus adjusted Z with a threshold of 3.5 (`--min-adjusted-z-all`); held-out Z and the number of held-out observations are shown and give warnings, but do not veto it. Weak evidence gives an ambiguous report. Record your choice:

```bash
grond manual hand --workflow manual_run/manual_workflow.json --choice input
grond manual hand --workflow manual_run/manual_workflow.json --choice inverse
grond manual hand --workflow manual_run/manual_workflow.json --skip      # centrosymmetric, or no hand test
```

The inverse choice is accepted only when every crystal produced the same inverted model.

`grond auto` decides the hand differently: right after the screen, from held-out reflections, switching only when the orbit sign test favours the inverse with z ≥ 3 and the 95% interval of ΔR_obs lies below zero; the all-observation statistic is only reported.

## Feedback

```bash
grond manual feedback \
  --workflow manual_run/manual_workflow.json \
  --cycles 5 \
  --workers 10 \
  --shelxl /path/to/shelxl
```

Each round recalculates every crystal with its fixed thickness and settings, merges the corrected observations of all crystals, and makes one shared SHELXL least-squares cycle. Transforms and thickness are not refitted. Every round is kept. The earliest round without NPD atoms that is within the observation-residual tolerance of the best is chosen, and its observations are recalculated without another SHELXL step so that maps and statistics describe the same model. A round with NPD atoms is never chosen. `grond auto` instead refines SHELXL to convergence every round, for up to five rounds, and stops when no non-hydrogen atom moves by more than 0.002 Å.

For a one-crystal run, setup may omit `--baseline-hkl`; a baseline stored in a reused thickness summary is then not used for the one-crystal comparison. `--n-basis N` fixes the number of beams (0, the default, keeps the adaptive choice). `--n-points` defaults to 20.

## Scales

```bash
grond manual scales --workflow manual_run/manual_workflow.json
```

This audits compact frame and detector scale models on each crystal's final observations and reports their parameter counts. Nothing is applied to the model. `--skip` skips the audit. `grond auto` fits P-spline frame scales; compact per-frame scale models are used only with `--scale-model per-frame`.

## Finish

```bash
grond manual finish --workflow manual_run/manual_workflow.json
```

This writes `06_finish/MANUAL_FINAL_SUMMARY.md` with a JSON companion, the chosen RES/HKL/LST paths, per-crystal statistics, the native Fo, 2Fo−Fc and Fo−Fc maps, a CIF and a final `handoff.json`. Use `--map-mode off` or `--skip` to leave out the maps. The summary is copied to `RUN/MANUAL_TOP_LEVEL_SUMMARY.md`. It shows the start and chosen SHELXL statistics, the observation statistics per crystal and combined, the hand evidence, the thickness and transform of each crystal, and the chosen round.

## Directories and rerunning

Each stage writes one numbered directory:

```text
00_setup/  01_scout/  02_thickness/  03_hand/  04_feedback/  05_scales/  06_finish/
```

Every directory contains `handoff.json`. The full output of each child calculation is in a `manual_*.log` file; the terminal shows milestones, warnings, the result and the next command. Add `--verbose-child-output` to see everything. Repeating an unchanged stage reuses its completed calculations.

Rerunning a stage marks the later stages as out of date (`invalidated_downstream_stages`); their files stay on disk. The rerun stage rewrites its own directory, so copy it first if you want to compare. A new model or new observations need a new setup.

To refine a manual result yourself and send it back ([USAGE.md](USAGE.md)), rerun the feedback stage's own `grond refine` command (`command.feedback.command_text` in `04_feedback/handoff.json`) with the new `--start-res` and a new `--out-dir`. `04_feedback/input_checkpoint.json` alone does not carry the feedback options (`--n-basis`, `--n-points`, the sigma policy).

## Lower-level commands

`grond scout`, `grond thickness`, `grond hand-check`, `grond refine`, `grond scale-polish`, `grond calibrate-endpoint` and `grond finish` are the commands the manual stages call. They remain available for unusual cases.
