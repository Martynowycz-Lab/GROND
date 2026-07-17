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

A solved SHELX `RES` or `INS` model with explicit electron scattering coefficients is the normal starting point. `GROND run` also accepts a CIF when it embeds the complete SHELX instruction block needed for model updates. The lower-level Bloch parser can read a coefficient-based CIF only when the file explicitly identifies those coefficients as electron factors. A solved-model HKL file is strongly recommended because it lets GROND settle indexing alignment before spending time on Bloch transform scouts.

## What a normal run does

`GROND auto` validates the inputs, resolves indexing ambiguity, runs the meaningful hand control, searches thickness and selected nuisance terms, and calculates matched Bloch and Born predictions. It then writes row-level residuals, corrected reflection files, endpoint diagnostics, maps when the required external tools are available, and a complete command and decision ledger.

The default fast structure-update route is an alternating Bloch/LS1 calculation. GROND freezes the current row-wise correction, performs one conventional least-squares model update, and recalculates the Bloch rows from the updated model. This is a practical surrogate, not exact native dynamical least squares. `GROND validate-ls1` supplies fixed-q, q=1, shuffled-q, alternate-hand, and optional native finite-difference controls for judging that approximation.

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

Native Bloch calculations use Python, NumPy, SciPy, Pandas, and Gemmi. The full small-molecule endpoint needs a separately installed SHELXL executable; map generation needs `shelx2map`. GROND does not bundle either program. Configure external tools with `GROND init-config` and confirm the resolved paths with `GROND doctor`.

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

## Read the result

Open the root-level `TABLE1_FINAL_STATISTICS.md` first for the publication-facing numbers. Its two panels deliberately keep SHELXL endpoint statistics (`R1gt`, `R1all`, `wR2all`, and GooF) separate from GROND-computed JANA/PETS-style row statistics (`Robs`, `Rall`, `wRall`, `MRobs`, `MRall`, and `MwRall`). The table compares the conventional kinematical, matched Born, and Bloch dynamical models wherever those quantities were actually calculated; an unavailable cross-family comparison is printed as `n/a`, never inferred from an unrelated residual.

`JANA_ROW_STATISTICS.md` and `.json` retain the canonical unmerged and post-symmetry-merged row factors, and `JANA_ROW_STATISTICS_PER_DATASET.*` gives the same factors per crystal when dataset labels are available. Born and Bloch use one common finite observation support; the reports expose input, matched, excluded, and merge-excluded counts and record the actual JANA/PETS instability factor `u`. `FINAL_SUMMARY.md` then names the selected model, reflection file, hand decision, indexing transform, and promotion checks. `QUICK_STATS.md` remains the compact audit report. Internal native-objective residuals may appear there for optimization diagnostics, but they are not publication SHELXL or JANA/PETS factors.

An interrupted run can be resumed by repeating the same command with the same output directory. GROND records completed stages and their commands in `grond_progress.json` and `logs/commands.jsonl`. To rebuild final summaries and maps from a completed calculation without rerunning the physics, use:

```bash
GROND finish --run-dir grond_out
```

## Documentation

Start with the [quickstart](docs/QUICKSTART.md), then use the guide that matches the data:

- [XDS workflow](docs/XDS_WORKFLOW.md)
- [DIALS workflow](docs/DIALS_WORKFLOW.md)
- [Multi-dataset workflow](docs/MULTI_DATASET.md)
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
