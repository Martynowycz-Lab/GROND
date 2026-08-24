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

A solved SHELX `RES` or `INS` model with explicit electron scattering coefficients is the normal starting point. `GROND run` also accepts a CIF when it embeds the complete SHELX instruction block needed for model updates. The lower-level Bloch parser can read a coefficient-based CIF only when the file explicitly identifies those coefficients as electron factors. A solved-model HKL file is strongly recommended because it lets GROND settle indexing alignment before spending time on Bloch transform scouts.

## What a normal run does

`GROND auto` validates the inputs, resolves indexing ambiguity, runs the meaningful hand control, searches thickness and selected nuisance terms, and calculates matched Bloch and Born predictions. Indexing and hand are deliberately separated. Automatic transform scouting considers proper transforms only and ranks them with a hand-invariant matched-Born work-row target. The chosen transform is then fixed while the two coordinate hands are compared. The hand control first evaluates the same predeclared nuisance grid for both labels, selects one common state by the mean work-row dynamical wR2 with a deterministic nuisance-hash tie-break, and replays both hands at that singleton state. This requires four bounded hand-scout evaluations instead of two, but makes the nuisance choice invariant when an input model is deliberately inverted and the branch labels exchange. The known chemistry or expected configuration is never an operator- or nuisance-selection input.

A proper transform makes the coordinate-hand comparison identifiable, but it does not by itself calibrate the sign of an acquisition and import chain. GROND therefore reports a conditional inversion preference by default. A named absolute configuration requires a separate, hash-bound sign calibration from a reserved known crystal processed through the same acquisition/import setup; that calibrant must not also be counted as validation. An explicitly requested mixed-parity transform search is diagnostic only and cannot establish a named configuration.

After these decisions, GROND writes row-level residuals, corrected reflection files, endpoint diagnostics, maps when the required external tools are available, and a complete command and decision ledger.

The default fast structure-update route is an alternating Bloch/LS1 calculation. GROND freezes the current row-wise correction, performs one conventional least-squares model update, and recalculates the Bloch rows from the updated model. This is a practical surrogate, not exact native dynamical least squares. `GROND validate-ls1` supplies fixed-q, q=1, shuffled-q, alternate-hand, and optional native finite-difference controls for judging that approximation.

The complete `auto` stage order, the switches that disable optional stages, and the corresponding direct commands are documented in [docs/AUTO_WORKFLOW.md](docs/AUTO_WORKFLOW.md). Experts who want to inspect and hand off every decision can instead follow the explicit [`prep -> scout -> thickness -> hand-check -> refine -> finish` workflow](docs/MANUAL_WORKFLOW.md).

For several crystals, GROND keeps thickness, geometry, scale, and uncertainty decisions attached to each dataset. Accepted corrected rows are placed on a common post-dynamical scale and merged only at the shared endpoint. The production shared-model loop recalculates every dataset from one model, writes one combined corrected HKL, performs one shared SHELXL update, and returns that model to every dataset. The slower native finite-difference updater remains available as a diagnostic.

## Install

GROND supports Python 3.11, 3.12, and 3.13. Python 3.12 is the simplest default for a new environment.

```bash
git clone git@github.com:Martynowycz-Lab/GROND.git
cd GROND
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
GROND doctor --probe-bloch --strict
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

## Run the same workflow manually

The manual interface uses one state file and one short command per stage:

```bash
GROND manual setup --out-dir manual_run --structure model.res --baseline-hkl model.hkl \
  --dataset crystal_1=/path/to/crystal_1 --dataset crystal_2=/path/to/crystal_2
GROND manual scout --workflow manual_run/manual_workflow.json
GROND manual thickness --workflow manual_run/manual_workflow.json
GROND manual hand --workflow manual_run/manual_workflow.json
GROND manual hand --workflow manual_run/manual_workflow.json --choice input
GROND manual feedback --workflow manual_run/manual_workflow.json --cycles 5 --shelxl /path/to/shelxl
GROND manual scales --workflow manual_run/manual_workflow.json
GROND manual finish --workflow manual_run/manual_workflow.json
```

Every stage writes a numbered directory and `handoff.json`, updates `manual_workflow.json`, records commands and elapsed time, and prints the next command. `GROND manual status --workflow manual_run/manual_workflow.json` shows the current decision and all output paths. Optional scout, hand, scale, and map stages have explicit skip controls; thickness can be skipped only by binding one existing audited summary per dataset.
Verbose child output is saved in stage-local `manual_*.log` files. Rerunning an earlier stage preserves old artifacts but automatically invalidates every later stage in the active workflow state.

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

Sign calibration is deliberately separate. Before running either scout, record `--parity-setup-id` and `--parity-setup-manifest-sha256`; for a direct single-dataset run also record `--physical-crystal-id` and `--acquisition-fingerprint`, or put `physical_crystal_id=...` and `acquisition_fingerprint=...` in each `GROND prep --dataset` specification. GROND freezes those values into both decisions. It enumerates all 24 proper signed-axis permutations, filters them only by unit-cell metric compatibility, uses one candidate-invariant held-out split, and replays the ranking from bound row artifacts. Calibration replay additionally requires every transform candidate to share the exact N/Q, row budget, selected/work/free support, nuisance, kernel, and backend contract. The matched hand call is likewise replayed with one source-sigma vector and must have a sign-consistent Klar--Palatinus adjusted Z above `--hand-min-adjusted-z` before it can train a setup calibration. `GROND create-parity-calibration` then learns either setup sign from one reserved known crystal; `GROND apply-parity-calibration` accepts only a different frozen physical crystal and acquisition from the same setup. Those setup and crystal labels and their supplied manifest/fingerprint hashes remain caller-attested provenance. GROND detects stale artifacts and exact declared reuse, but it cannot cryptographically prove that a user did not assign false identifiers to the same raw acquisition.

An interrupted run can be resumed by repeating the same command with the same output directory. GROND records completed stages and their commands in `grond_progress.json` and `logs/commands.jsonl`. To rebuild final summaries and maps from a completed calculation without rerunning the physics, use:

```bash
GROND finish --run-dir grond_out
```

## Documentation

Start with the [quickstart](docs/QUICKSTART.md), then use the guide that matches the data:

- [XDS workflow](docs/XDS_WORKFLOW.md)
- [DIALS workflow](docs/DIALS_WORKFLOW.md)
- [PETS, Jana2020, Dyngo, and diffBloch interoperability](docs/INTEROPERABILITY.md)
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
