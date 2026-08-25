# Simple staged workflow

`grond manual` exposes a simplified expert workflow as short, resumable stages. Setup
writes one `manual_workflow.json`; every later command reads that file, writes
its own `handoff.json`, updates the workflow state, and prints the next command.
No paths, transforms, thicknesses, or nuisance grids need to be copied by hand.

This interface is for inspecting and handing off individual decisions. It is
not a stage-for-stage reimplementation of `grond auto`: it does not run every
feature gate, dataset policy, resolution/endpoint calibration, final
post-feedback shared-hand selection, or compact-scale promotion. Use `auto` for
the canonical production endpoint and use the manual path for controlled expert
experiments.

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
  --dataset crystal_2=/path/to/crystal_2,dmin=0.70 \
  --config /absolute/path/grond.config.json \
  --shelxl /absolute/path/shelxl \
  --workers 10
```

The command prints the only state path needed later. It also stores the Python,
worker, config, and SHELXL defaults, so subsequent printed commands retain the
same runtime on that machine. When a baseline HKL and explicit SHELXL path are
available, setup writes a fresh one-cycle conventional control for the final
comparison.

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
correct, use `--skip`; the handoff records that user assertion. Skip mode still
rejects improper transforms, runs input preflight, and preserves the requested
MINPK/partiality and geometry support rather than silently relaxing it.

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

Every reused summary is verified against the named source-HKL hash, proper
transform, resolution cutoff, PEAK/partiality floor, and geometry percentile.
Swapping two crystals' summaries fails before feedback.

## Check the pre-feedback hand and pause for a choice

```bash
grond manual hand --workflow manual_run/manual_workflow.json
```

This evaluates the thickness-stage model with the exact elastic
negative-thickness identity, writes per-dataset diagnostics plus
`combined_hand_evidence.json`, and pauses. Known chemistry is never used. The
automatic recommendation requires sign-consistent all/free residual evidence,
an adjusted hand-Z threshold, and enough free rows; weak evidence defaults to an
ambiguous report rather than an automatic inversion. Record the deliberate
coordinate choice with one short command:

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

This manual hand check occurs before feedback. It is not `auto`'s authoritative
final post-feedback shared-hand selector. If the final hand is scientifically
important, use the `auto` decision or run a separate final `grond hand-check`
against the selected final row summaries.

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
statistics use the same model. A best-native cycle containing an NPD atom is
diagnostic only and is never used as the manual handoff fallback.

For a one-crystal sensitivity or leave-one-out workflow, setup may omit
`--baseline-hkl`. The controller then prevents a larger-cohort baseline stored
inside a reused thickness summary from entering the one-crystal endpoint. If a
genuine conventional HKL for that crystal is available, supply it at setup and
it overrides the summary provenance. With an explicit configured SHELXL path,
setup runs and hashes a fresh one-cycle conventional control; final reporting
uses that control instead of relying on an unrelated sibling LST.

Use `--n-basis N` only for a deliberate fixed-basis experiment. Zero retains
adaptive production basis selection. `--n-points` defaults to 20.

## Audit scale compression

```bash
grond manual scales --workflow manual_run/manual_workflow.json
```

This audits each frozen final row table independently and records frame and
detector parameter counts. Compact scales are not applied to the shared
endpoint by this simplified workflow. Use `--skip` to keep the full pinned scale
grouping without running the audit. Run production `auto` when the smallest
validated compact scale contract is required.

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
