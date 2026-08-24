# Simple staged workflow

`grond manual` exposes the ordinary workflow as short, resumable stages. Setup
writes one `manual_workflow.json`; every later command reads that file, writes
its own `handoff.json`, updates the workflow state, and prints the next command.
No paths, transforms, thicknesses, or nuisance grids need to be copied by hand.

The normal sequence is:

```text
setup -> scout -> thickness -> hand -> feedback -> scales -> finish
```

`grond manual status --workflow RUN/manual_workflow.json` reports what ran,
elapsed times, selected model, the final summary, and the next command. Add
`--json` for the machine-readable form.

Every stage has its own short help page. For example:

```bash
grond manual thickness --help
grond manual feedback --help
```

The thickness help shows only thickness controls. The feedback help also says
explicitly which work is per crystal and which coordinate update is shared.

## Setup

For two existing XDS or GROND bridge directories:

```bash
grond manual setup \
  --out-dir manual_run \
  --structure model.res \
  --baseline-hkl model.hkl \
  --dataset crystal_1=/path/to/crystal_1,dmin=0.70 \
  --dataset crystal_2=/path/to/crystal_2,dmin=0.70
```

The command prints the only state path needed later:

```text
manual_run/manual_workflow.json
```

For DIALS, first make one bridge per experiment with `grond import-dials`, then
use those bridge directories as the dataset paths.

## Scout proper transforms

```bash
grond manual scout --workflow manual_run/manual_workflow.json
```

This runs chemistry-blind, determinant-positive transform scouting independently
for every dataset. To assert that transforms already recorded during setup are
correct, use `--skip`; the handoff records that user assertion.

## Select effective thickness

```bash
grond manual thickness --workflow manual_run/manual_workflow.json
```

The default is the compact primary experiment: all accepted rows, detector-free
frame scales, no absorption or Sg jitter, N=24 then N=48 confirmation, and a
broad thickness/gradient grid. `--expanded` also runs the sensitivity branches.

Thickness cannot be silently skipped because feedback needs a pinned nuisance
contract. An expert can reuse existing audited summaries explicitly:

```bash
grond manual thickness --workflow manual_run/manual_workflow.json --skip \
  --summary crystal_1=/path/to/summary.json \
  --summary crystal_2=/path/to/summary.json
```

## Check hand and pause for a choice

```bash
grond manual hand --workflow manual_run/manual_workflow.json
```

This evaluates every crystal with the exact elastic negative-thickness identity,
writes per-dataset diagnostics plus `combined_hand_evidence.json`, and pauses.
Known chemistry is never used. Record the deliberate coordinate choice with one
short command:

```bash
grond manual hand --workflow manual_run/manual_workflow.json --choice input
```

or:

```bash
grond manual hand --workflow manual_run/manual_workflow.json --choice inverse
```

For a centrosymmetric structure or an explicitly omitted hand experiment:

```bash
grond manual hand --workflow manual_run/manual_workflow.json --skip
```

The inverse choice is accepted only when every dataset handoff materialized the
same hash-identical inverted coordinate model.

## Shared Bloch/LS1 feedback

```bash
grond manual feedback \
  --workflow manual_run/manual_workflow.json \
  --cycles 5 \
  --workers 10 \
  --shelxl /path/to/shelxl
```

The controller creates a hash-bound multi-dataset checkpoint from the selected
thickness summaries and hand. During each macrocycle it recomputes Bloch/Born
rows and pinned nuisance scales separately for each crystal. Those jobs do not
run SHELXL and do not create separate coordinate models. GROND then determines
the cross-dataset overlap scales, merges model-symmetry equivalents with
inverse-variance weights, and performs exactly one shared SHELXL LS1 update.
It does not rescout transforms or refit thickness. Every cycle is preserved.
The earliest clean no-NPD cycle within the row-residual tolerance is selected,
and its rows are recomputed without another coordinate step so maps and row
statistics use the same model.

For a one-crystal sensitivity or leave-one-out workflow, setup may omit
`--baseline-hkl`. The controller then prevents a larger-cohort baseline stored
inside a reused thickness summary from entering the one-crystal endpoint. If a
genuine conventional HKL for that crystal is available, supply it at setup and
it overrides the summary provenance. Final reports print starting conventional
statistics only when that HKL is hash-identical to the HKL beside the starting
model LST.

Use `--n-basis N` only for a deliberate fixed-basis experiment. Zero retains
adaptive production basis selection. `--n-points` defaults to 20.

## Audit scale compression

```bash
grond manual scales --workflow manual_run/manual_workflow.json
```

This audits each frozen final row table independently and records frame and
detector parameter counts. Compact scales are not silently applied to the
shared endpoint. Use `--skip` to keep the full pinned scale grouping without
running the audit.

## Write maps and the final handoff

```bash
grond manual finish --workflow manual_run/manual_workflow.json
```

This writes `06_finish/MANUAL_FINAL_SUMMARY.md`, a machine-readable JSON summary,
the selected RES/HKL/LST paths, per-dataset row summaries, the native dynamical
Fo, 2Fo-Fc, and Fo-Fc maps, a dynamical CIF, and a final `handoff.json`. Use
`--map-mode off` or `--skip` when maps are not wanted.

The same report is copied to `RUN/MANUAL_TOP_LEVEL_SUMMARY.md` so it is the
first result visible in the workflow root. It includes starting-versus-selected
SHELXL endpoint statistics, combined and per-crystal JANA/PETS-style row
statistics, selected-coordinate hand evidence, thickness/gradient and
transform per crystal, the feedback-cycle decision, and separate applied frame
and detector parameter counts. The scale section says explicitly whether a
compact model was merely audited or actually applied.

## Status and resuming

At any point:

```bash
grond manual status --workflow manual_run/manual_workflow.json
```

Each stage has a fixed numbered directory:

```text
00_setup/
01_scout/
02_thickness/
03_hand/
04_feedback/
05_scales/
06_finish/
```

Every directory contains `handoff.json`. Child GROND calculations retain their
normal command and progress ledgers; the manual controller keeps their complete
terminal output in a named `manual_*.log` and prints only milestones, warnings,
the stage result, and the next command. Add `--verbose-child-output` to a
computational stage when the complete child stream is wanted interactively.
Repeating an unchanged stage resumes its fingerprint-matching calculations.

Going backward is safe. Rerunning a stage removes later stages from the active
workflow state and lists them under `invalidated_downstream_stages`; their files
remain on disk for audit. Run forward again to regenerate them. A changed input
model or observation set requires a new setup workflow rather than editing an
old handoff in place.

## Advanced low-level interface

The original expert commands remain available for unusual experiments:
`grond scout`, `grond thickness`, `grond hand-check`, `grond refine`,
`grond scale-polish`, `grond calibrate-endpoint`, and `grond finish`. They are
the implementation boundaries used by the manual controller, not the expected
day-to-day user interface.
