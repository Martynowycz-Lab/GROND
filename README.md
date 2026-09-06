# GROND

**Global Regularization Of Non-kinematical Diffraction**

GROND applies row-level Bloch-wave calculations to rotation electron-diffraction data. It starts from an XDS or DIALS integration and a solved structure, keeps each signed observation tied to its scan geometry, and writes an auditable regularization and refinement bundle.

The package is alpha research software. It is useful in the small-molecule and peptide-sized workflows for which it has been tested, but it is not a replacement for careful integration, chemical judgment, or established crystallographic refinement programs. Read [LIMITATIONS.md](LIMITATIONS.md) before treating a result as final.

## Why row-level data matter

A merged reflection table hides the geometry in which each observation was measured. That is acceptable for many kinematic workflows, but multiple scattering depends on orientation, thickness, coupled beams, and the exact signed reflection. GROND therefore works from summation-style integration rows for as long as possible.

For each row, GROND calculates a Bloch intensity and a matched Born intensity from the same model and rocking-profile support. Their ratio,

```text
q = Idyn / Iborn
```

is used to regularize the observed intensity. The calculation preserves the measured intensity and uncertainty as observations; the physical model and every accepted nuisance term are recorded separately.

## Supported inputs

The established XDS route uses an `INTEGRATE.HKL` together with `XPARM.XDS` or `GXPARM.XDS`. We recommend a summation integration produced with `PROFILE_FITTING=FALSE`; the starting structure may still come from the best conventional profile-fitted solution and refinement path.

The DIALS route uses matching `integrated.refl` and `integrated.expt` files. GROND imports summation intensities, preserves the signed `miller_index`, carries the scan-varying orientation model into the excitation-error calculation, and records its row filtering. The detector-model variance is kept unchanged by default; empirical and gain-based sigma modes are explicit diagnostics.

Other integration programs can target the documented `grond-row-v1` CSV, TSV, or JSON Lines contract plus its geometry sidecar. See [Portable inputs and endpoints](docs/PORTABLE_INPUTS_AND_ENDPOINTS.md).

GROND can also export either integration to the shared PETS `cif_pets`
observation-and-geometry format used by PETS/Jana2020/Dyngo and diffBloch. See
[PETS and diffBloch interoperability](docs/INTEROPERABILITY.md).

Native PETS2 input is available through `GROND import-pets`. It preserves the
recorded PETS zone geometry for native row-space Born/Bloch, thickness, and hand
calculations. When an ordinary `cif_pets`, its companion 13-column `.hkl`, and
PETS' own static SHELX HKL agree row-for-row, the importer records that proof
and `grond auto` can run the normal q-correction and frozen-weight LS1 workflow.
PETS `*_dyn.cif_pets` virtual-frame occurrences remain native row-space data and
are never silently presented as a conventional SHELXL endpoint.

A solved SHELX `RES` or `INS` model with explicit electron scattering coefficients is the normal starting point. `GROND run` also accepts a CIF when it embeds the complete SHELX instruction block needed for model updates. The lower-level Bloch parser can read a coefficient-based CIF only when the file explicitly identifies those coefficients as electron factors. A solved-model HKL file is strongly recommended because it lets GROND settle indexing alignment before spending time on Bloch transform scouts.

## What a normal run does

`GROND auto` validates the inputs, resolves indexing ambiguity, searches thickness and selected nuisance terms, updates the structure, and calculates matched Bloch and Born predictions. Indexing and hand are deliberately separated. Automatic transform scouting considers proper transforms only and ranks them with a hand-invariant matched-Born work-row target. Hand is deferred until the final coordinates and nuisance contracts are fixed. GROND then disables absorption only for this Hermitian control and uses the exact negative-thickness identity to evaluate the inverse coordinates while sharing each eigendecomposition. Rows, N/Q, thickness, jitter, scales, and the proper transform remain fixed. If the all-row Klar--Palatinus adjusted Z favors the inverse by at least 3.5, GROND inverts the model once and continues the pinned final refinement. Held-out hand evidence is reported and warns when weak or conflicting, but does not veto a decisive all-row result. Final displayed ZKP values are reoriented so positive favors the handed model actually carried forward; the raw signed comparison remains in the decision JSON. Known chemistry is never used to choose the operator or hand.

Compact scale topology is selected separately from the SHELXL endpoint. Frame scales are fitted on work rows with the JANA-style row weight, then the smallest detector-free or independently validated detector model that preserves held-out and work `Rall` is promoted. Held-out `wRall` is a secondary tie-break and warning because it is more sensitive to the imported sigma model. SHELXL `R1`, `wR2`, and GooF are downstream endpoint reports and pathology checks; they do not make a more flexible scale field win. The adjusted reflection-count hand statistic uses immutable experimental sigmas in the target intensity units. Method version 4 assigns half a win to exact and numerical residual ties; the historical strict-count statistic remains available separately.

The final report names both the all-row preferred hand and the exported coordinate representation. A weak inverse preference remains visible even when it does not cross the automatic switch threshold; identical evidence reports no preference. Free-set weakness is a diagnostic warning. Invalid physical rows, geometry or a failed calculation produce explicit errors with the available metrics and logs, rather than a fabricated hand answer. Nominal Z values are conditional count contrasts, not calibrated sigma confidence under correlated observations. See [Hand results and calculation errors](docs/HAND_RESULT_POLICY.md).

A proper transform makes the coordinate-hand comparison identifiable, but it does not by itself calibrate the sign of an acquisition and import chain. GROND therefore reports a conditional inversion preference by default. A named absolute configuration requires a separate, hash-bound sign calibration from a reserved known crystal processed through the same acquisition/import setup; that calibrant must not also be counted as validation. An explicitly requested mixed-parity transform search is diagnostic only and cannot establish a named configuration.

After these decisions, GROND writes row-level residuals, corrected reflection files, endpoint diagnostics, maps when the required external tools are available, and a complete command and decision ledger.

The default fast structure-update route is an alternating Bloch/LS1 calculation. GROND freezes the current row-wise correction, performs one conventional least-squares model update, and recalculates the Bloch rows from the updated model. This is a practical surrogate, not exact native dynamical least squares. `GROND validate-ls1` supplies fixed-q, q=1, shuffled-q, alternate-hand, and optional native finite-difference controls for judging that approximation.

The complete `auto` stage order, the switches that disable optional stages, and the corresponding direct commands are documented in [docs/AUTO_WORKFLOW.md](docs/AUTO_WORKFLOW.md). The staged interface in [docs/MANUAL_WORKFLOW.md](docs/MANUAL_WORKFLOW.md) is a simplified expert decomposition for inspecting transforms, thickness, a pre-feedback hand control, feedback, and scale sensitivity. It is not a manual reimplementation of every `auto` policy gate; use `auto` for the canonical production result.

For several crystals, GROND keeps thickness, geometry, scale, and uncertainty decisions attached to each dataset. Accepted corrected rows are placed on a common post-dynamical scale and merged only at the shared endpoint. The production shared-model loop recalculates every dataset from one model, writes one combined corrected HKL, performs one shared SHELXL update, and returns that model to every dataset. The slower native finite-difference updater remains available as a diagnostic.

## Install

GROND supports Python 3.11, 3.12, and 3.13. Python 3.12 is the simplest default for a new environment.

```bash
git clone https://github.com/Martynowycz-Lab/GROND.git
cd GROND
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
GROND init-config --out grond.config.json
# Fill in the absolute SHELXL path when it is not already on PATH.
GROND doctor --workflow small-molecule --config grond.config.json --probe-bloch --strict
```

Native Bloch calculations use Python, NumPy, SciPy, and Gemmi. The default small-molecule LS1 endpoint needs a separately installed SHELXL executable. Native dynamical maps are written by `GROND finish` with no external map program; `shelx2map` is optional and is used only for quick corrected-HKL endpoint maps. XDS, XDSCONV, SHELXT, DIALS, and Olex2 are not general GROND runtime requirements. A DIALS Python is needed only to import `.refl`/`.expt` observations or to write the optional PETS/Jana/diffBloch interoperability format; the XDS export uses DIALS solely for the PETS coordinate-frame writer.

The complete installation and update instructions are in [docs/INSTALL.md](docs/INSTALL.md).

## Run one dataset

For XDS data:

```bash
GROND auto \
  --config /path/to/grond.config.json \
  --xds-dir /path/to/xds_directory \
  --structure /path/to/model.res \
  --baseline-hkl /path/to/model.hkl \
  --out-dir grond_out
```

For DIALS data:

```bash
GROND auto \
  --config /path/to/grond.config.json \
  --dials-refl /path/to/integrated.refl \
  --dials-expt /path/to/integrated.expt \
  --dials-python /path/to/dials/python \
  --structure /path/to/model.res \
  --baseline-hkl /path/to/model.hkl \
  --out-dir grond_dials_out
```

Use `--fast` to check a new integration, indexing choice, or environment before starting the full calculation. Use plain `GROND auto` for the final high-support run. The middle-cost `--full-fast` profile keeps the automatic decision path but deliberately uses a smaller final calculation; treat it as an intermediate result.

## Run the staged expert workflow

The manual interface uses one state file and one short command per stage:

```bash
GROND manual setup --out-dir manual_run --structure model.res --baseline-hkl model.hkl \
  --dataset crystal_1=/path/to/crystal_1 --dataset crystal_2=/path/to/crystal_2 \
  --config /absolute/path/grond.config.json --shelxl /absolute/path/shelxl --workers 10
GROND manual scout --workflow manual_run/manual_workflow.json
GROND manual thickness --workflow manual_run/manual_workflow.json
GROND manual hand --workflow manual_run/manual_workflow.json
GROND manual hand --workflow manual_run/manual_workflow.json --choice input
GROND manual feedback --workflow manual_run/manual_workflow.json --cycles 5 --shelxl /path/to/shelxl
GROND manual scales --workflow manual_run/manual_workflow.json
GROND manual finish --workflow manual_run/manual_workflow.json
```

Every stage writes a numbered directory and `handoff.json`, updates `manual_workflow.json`, records commands and elapsed time, and prints the next command. Setup persists the Python, worker, tool-config, and SHELXL defaults so a printed command can be copied on the same machine. `GROND manual STAGE --help` shows only the controls relevant to that stage. `GROND manual status --workflow manual_run/manual_workflow.json` shows the current decision, timings, selected model, final summary, and next command; add `--json` for machine-readable output. A skipped scout still checks proper transform parity and preflight. Reused thickness summaries are hash-bound to the named HKL and row-selection contract. Manual feedback refuses an NPD diagnostic cycle. Compact scales are audited but not applied automatically.

Complete child output is saved in stage-local `manual_*.log` files while the terminal shows milestones and the stage result; `--verbose-child-output` restores the full interactive stream. `manual finish` writes `MANUAL_TOP_LEVEL_SUMMARY.md` in the run root with the conventional-versus-GROND endpoint, combined and per-crystal JANA/PETS-style rows, hand evidence, thickness, transforms, feedback cycles, and separate applied frame/detector scale counts. The manual hand stage is evaluated before feedback and is intentionally conservative; it is not the final shared-hand selector used by `auto`. Rerunning an earlier stage preserves old artifacts but automatically invalidates every later stage in the active workflow state.

## Read the result

Open the root-level `TABLE1_FINAL_STATISTICS.md` first for the publication-facing numbers. Its two panels deliberately keep SHELXL endpoint statistics (`R1gt`, `R1all`, `wR2all`, and GooF) separate from GROND-computed JANA/PETS-style row statistics (`Robs`, `Rall`, `wRall`, `MRobs`, `MRall`, and `MwRall`). The table compares the conventional kinematical, matched Born, and Bloch dynamical models wherever those quantities were actually calculated; an unavailable cross-family comparison is printed as `n/a`, never inferred from an unrelated residual.

`JANA_ROW_STATISTICS.md` and `.json` retain the canonical unmerged and post-symmetry-merged row factors, and `JANA_ROW_STATISTICS_PER_DATASET.*` gives the same factors per crystal when dataset labels are available. Born and Bloch use one common finite observation support; the reports expose input, matched, excluded, and merge-excluded counts and record the actual JANA/PETS instability factor `u`. `FINAL_SUMMARY.md` then names the selected model, reflection file, hand decision, indexing transform, and promotion checks. `QUICK_STATS.md` remains the compact audit report. Internal native-objective residuals may appear there for optimization diagnostics, but they are not publication SHELXL or JANA/PETS factors.

If an older diagnostic endpoint used an improper transform, do not relabel that RES in place. `GROND canonicalize-hand` materializes the exact gauge-equivalent representation by inverting the model coordinates and explicit symmetry translations, changing the transform from `T` to the proper sign-paired operator `-T`, and negating every endpoint HKLF4 index while preserving its intensity, sigma, batch, and trailing fields byte-for-byte. Its manifest records input/output hashes and claim status. Replay the canonical RES/HKL pair through the configured endpoint program to regenerate publication RES/CIF/LST/FCF artifacts. Canonicalization is calibration-free and therefore remains a conditional representation:

```bash
GROND canonicalize-hand \
  --input-res legacy.res \
  --input-hkl legacy.hkl \
  --hkl-transform=-h,k,l \
  --output-res canonical.res \
  --output-hkl canonical.hkl
```

Sign calibration is separate from the conditional final hand comparison. Record setup, physical-crystal, and acquisition identities before scouting when a calibration study is planned, and reserve a known crystal that is not counted as validation. The current final negative-thickness `auto` artifact does not yet feed `create-parity-calibration`; that command remains available only for the older hash-bound branch-symmetric/common-sigma decision format. Do not describe a current `auto` hand as a named absolute configuration solely from residual sign. Until the final-elastic calibration adapter is implemented and validated, report the conditional inversion preference together with the external known-chemistry assignment.

An interrupted run can be resumed by repeating the same command with the same output directory. GROND records completed stages and their commands in `grond_progress.json` and `logs/commands.jsonl`. To rebuild final summaries and maps from a completed calculation without rerunning the physics, use:

```bash
GROND finish --run-dir grond_out
```

## Documentation

Start with the [quickstart](docs/QUICKSTART.md), then use the guide that matches the data:

- [XDS workflow](docs/XDS_WORKFLOW.md)
- [DIALS workflow](docs/DIALS_WORKFLOW.md)
- [PETS, Jana2020, Dyngo, and diffBloch interoperability](docs/INTEROPERABILITY.md)
- [Experimental diffBloch endpoint polish](docs/DIFFBLOCH_ENDPOINT.md)
- [Multi-dataset workflow](docs/MULTI_DATASET.md)
- [Simple staged workflow](docs/MANUAL_WORKFLOW.md)
- [Outputs](docs/OUTPUTS.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Performance and resource use](docs/PERFORMANCE.md)
- [Source-code map](docs/CODE_MAP.md)
- [Scientific and software references](docs/REFERENCES.md)

The public repository is limited to the installable source under `src/grond`, its tests, documentation, synthetic examples, schemas, and release tooling. The name `src/grond/legacy/engine` is historical; it currently contains much of the mature numerical engine used by the CLI. Standalone experiment workspaces, dataset-derived benchmark outputs, article-drafting material, and private data stay outside this repository.

## Development

Install the development dependencies and run the public checks before committing:

```bash
python -m pip install -e ".[dev]"
python -m ruff check src/grond tests scripts --no-cache
python -m pytest
python -m grond validate
python -m grond doctor --probe-bloch --strict
```

Release checks, contribution rules, security reporting, and the repository data policy are documented in [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md), [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [DATA_POLICY.md](DATA_POLICY.md).

## License and citation

GROND is released under the [MIT License](LICENSE). The clean-source boundary is recorded in [PROVENANCE.md](PROVENANCE.md). Cite the software using [CITATION.cff](CITATION.cff) and cite the scientific methods relevant to the calculation from [docs/REFERENCES.md](docs/REFERENCES.md).

Project acknowledgments are in [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).
