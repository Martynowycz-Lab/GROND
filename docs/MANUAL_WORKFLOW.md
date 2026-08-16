# Expert staged workflow

`grond auto` is a policy runner, not the only interface. The same calculation
can be run as explicit, inspectable stages. Each expert stage writes a JSON
handoff containing its immutable input paths, hashes, selected contract, and a
ready-to-copy command for the next stage. An expert can stop, edit the next
command, or replace a decision without rerunning completed physics.

The recommended order is `prep -> scout -> thickness -> hand-check -> refine ->
finish`. Transform selection, coordinate hand, thickness, nuisance terms,
model refinement, scale compression, and reporting remain separate decisions.

## Prepare the observations

For XDS, point each dataset at the directory containing the GROND row bridge or
the audited `INTEGRATE.HKL` products. For DIALS, first make one bridge per
experiment with `grond import-dials`; the DIALS workflow documents the
summation and partiality choices. Then bind the structure, conventional HKL,
and datasets once:

```bash
grond prep \
  --structure model.res \
  --baseline-hkl model.hkl \
  --dataset crystal_01=crystal_01_bridge \
  --dataset crystal_02=crystal_02_bridge \
  --out 00_prep.json
```

`00_prep.json` is the input to every later stage. Do not change the RES, HKL,
or bridge in place after this point; make a new prep file when the inputs
change.

## Fix the indexing transform

Run only the chemistry-blind proper-transform scout:

```bash
grond scout --prep 00_prep.json --out-dir 01_scout --workers 16
```

The output is `01_scout/scouted_prep.json`. The detailed candidate evidence is
under `01_scout/decisions/`, and `01_scout/scout_handoff.json` records the exact
next commands. Known absolute configuration is not used to choose an operator.
If the RES and baseline HKL are known to remain in the exact post-cosym DIALS
basis, the expert may instead freeze the transform already recorded in the prep
file:

```bash
grond scout --prep 00_prep.json --out-dir 01_scout \
  --trust-hkl-transform
```

That is an explicit user assertion. It is not inferred merely because DIALS
`cosym` was run.

## Inspect the fixed model and thickness

A cheap frozen-model Bloch/Born calculation can be run independently:

```bash
grond run --prep 01_scout/scouted_prep.json \
  --out-dir 02_fixed_model --mode bloch \
  --no-auto-decide --no-hand-control --no-maps --native-only
```

The row table and its full configuration are
`02_fixed_model/runs/DATASET/screen/native_dyn_calc.csv` and `summary.json`.
This stage does not move atoms or ADPs.

Thickness sensitivity is exposed directly under either command spelling:

```bash
grond thickness --prep 01_scout/scouted_prep.json \
  --out-dir 02_thickness --workers 16
```

`thickness` is an alias for `thickness-scout`. Inspect `REPORT.md`,
`thickness_scout_table.csv`, and `q_field_stability.csv`; then use
`thickness_handoff.json` to find the selected per-dataset `summary.json` and
the next hand command. GROND thickness is an effective row-contract thickness,
not an independent microscope measurement unless externally calibrated.

## Check absolute structure without recomputing the eigensystem

For an acentric structure, run the matched, fixed-nuisance hand check from the
accepted native summary:

```bash
grond hand-check \
  --source-summary 02_thickness/runs/DATASET/SELECTED/summary.json \
  --out-dir 03_hand --workers 16
```

The default `--hand-check-method negative-thickness` uses the elastic identity

\[
I_{\mathrm{inverse}}(t)=I_{\mathrm{input}}(-t).
\]

For an elastic Hermitian Bloch matrix, coordinate inversion gives
`H_inverse = H*`; with a real direct-beam incident vector, the two propagation
signs share one eigendecomposition. GROND therefore computes the supplied and
inverse coordinate hands together at the source N/Q, on the same signed rows,
with the same proper transform, split, thickness distribution, and
nonabsorptive nuisance parameters. It independently profiles the declared
scale model for each hand. `hand_handoff.json` records both row tables, the
materialized inverse-coordinate RES, the diagnostic, hashes, and commands for
refining either coordinate hand.

The shortcut is exact only without absorption. The default
`--hand-check-absorption elastic` zeros absorptive terms in both branches and
records the source absorption contract. GROND refuses negative thickness with
`--hand-check-absorption preserve`. To retain absorptive terms, request two
ordinary coordinate-model calculations explicitly:

```bash
grond hand-check \
  --source-summary 02_thickness/runs/DATASET/SELECTED/summary.json \
  --out-dir 03_hand_explicit \
  --hand-check-method explicit --hand-check-absorption preserve
```

The fast hand check is evidence at a fixed model and nuisance state. It does
not replace a fully refined inverse-hand control when that stronger experiment
is required, and it does not use a known chemical configuration to select the
answer.

## Refine with repeated Bloch/LS1 updates

`refine` is an expert-facing alias for `push-endpoint`. It starts from an
accepted native summary and a chosen coordinate model, pins that row/nuisance
contract, performs the requested number of updated-q Bloch/SHELXL macrocycles,
and preserves every cycle:

```bash
grond refine \
  --source-summary 02_thickness/runs/DATASET/SELECTED/summary.json \
  --start-res model.res \
  --out-dir 04_refine \
  --cycles 5 --n-basis 108 --n-points 20 \
  --shelxl /path/to/shelxl --workers 16
```

One macrocycle means: calculate a fresh Bloch row model from the current RES,
form the corrected HKL under the fixed scale contract, perform the declared
SHELXL update, and return to Bloch physics. The production default keeps one
internal LS/CGLS cycle per frozen correction. `--cycles 5` is five fresh Bloch
updates, not five conventional cycles against one stale correction. A branch
that develops NPD atoms or fails its endpoint gates is retained as a diagnostic
rather than silently promoted.

For a full inverse-hand refinement, use the `inverse_model` path and
`refine_inverse_hand` command written by `03_hand/hand_handoff.json`. Both hands
must retain the same proper HKL transform and row contract.

## Scale and report

The final native row table may be audited for a smaller scale topology without
rerunning Bloch propagation:

```bash
grond scale-polish \
  --row-table 04_refine/PATH/native_dyn_calc.csv \
  --summary 04_refine/PATH/summary.json \
  --out-dir 05_scale_polish
```

Use `grond scale-replay` or `grond calibrate-endpoint` only for scale contracts
that pass the frozen-row gates. Finish the accepted run with:

```bash
grond finish --run-dir 04_refine
```

The endpoint SHELXL statistics, unmerged and merged JANA/PETS-style row
statistics, maps, paths, and provenance remain separate in the final report.
Do not compare a row `wRall` directly with an endpoint `wR2` or use a lower
corrected-HKL R factor as the sole reason to choose a physics branch.

## Resuming and auditing

Every stage has its own output directory. Repeating an unchanged command there
resumes fingerprint-matching work. The command ledger is `logs/commands.jsonl`;
the progress ledger is `grond_progress.json`; stage-specific `*_handoff.json`
files are the supported boundaries between expert decisions. If an input file
changes, start a new stage directory rather than reusing a handoff whose hashes
no longer match.
