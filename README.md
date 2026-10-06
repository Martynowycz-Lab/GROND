# GROND

**Global Regularization Of Non-kinematical Diffraction**

GROND adds dynamical (Bloch-wave) corrections to electron-diffraction data integrated with **XDS or DIALS**, and hands the corrected intensities to **SHELXL or olex2.refine** as an ordinary HKLF 4 file. You keep your integration program and your refinement program; GROND sits between them.

GROND is research software. Read [LIMITATIONS.md](LIMITATIONS.md) before relying on a result.

## How it works

For every unmerged observation, GROND calculates two intensities from the current model at the orientations where the observation was recorded:

- the dynamical intensity $I_\mathrm{dyn}$, from a Bloch-wave calculation;
- the kinematic intensity $I_\mathrm{kin}$, from the same beams, rocking range and thickness.

It fits the few experimental settings that this calculation needs, on working observations only:

- smooth frame scales for each crystal;
- a thickness distribution (gamma or wedge);
- a mosaic spread;
- a radiation-damage term.

It then writes corrected intensities

```text
I_corr = κ |Fc|² · I_obs / (s · I_dyn)
```

which take the calculated dynamical effect out of each measurement and keep its measured deviation from the model. SHELXL refines the corrected data like X-ray data, GROND recalculates the correction for the new model, and the two alternate until no non-hydrogen atom moves by more than 0.002 Å (at most five rounds). Because the corrected amplitudes are the usual dynamical map coefficients, SHELXL's own Fo − Fc maps are dynamically corrected difference maps.

Validation is built in. GROND holds out a free set of about 10% of the unique reflections, starts from a model refined without them, chooses every setting by cross-validation on the working observations, and reports R factors on the free set with bootstrap confidence intervals. For non-centrosymmetric structures it compares the model with its inverse on held-out reflections.

## Install

GROND needs Python 3.11 or later and a SHELXL executable. Importing DIALS data also needs a Python that can import `dials` (for example the one shipped with DIALS).

```bash
git clone https://github.com/Martynowycz-Lab/GROND.git
cd GROND
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
grond init-config --out grond.config.json     # then set the path to SHELXL in this file
grond doctor --workflow small-molecule --config grond.config.json   # checks the installation and SHELXL
```

More detail is in [docs/INSTALL.md](docs/INSTALL.md).

## Inputs

- A SHELX `.res` model with numeric electron scattering factors on its SFAC lines (`grond sfac-check` checks this), and the merged HKLF 4 file it was refined against.
- The unmerged observations of each crystal:
  - **XDS**: a directory with `INTEGRATE.HKL` from a summation run (`PROFILE_FITTING=FALSE`) and `XPARM.XDS` or `GXPARM.XDS`;
  - **DIALS**: the `integrated.refl` and `integrated.expt` files, converted once per crystal with `grond import-dials`.

XDS and DIALS crystals can be mixed in one run.

## Run

```bash
grond import-dials --refl c1/integrated.refl --expt c1/integrated.expt \
    --dials-python /path/to/dials.python --out-dir bridges/c1
grond prep --structure model.res --baseline-hkl model.hkl \
    --dataset c1=bridges/c1,dmin=0.80 --dataset c2=/data/c2/xds,dmin=0.80 --out prep.json
grond auto --prep prep.json --config grond.config.json --workers 40 --out-dir run
grond finish --run-dir run
```

- `grond auto --preset fast` gives a quick answer with every check kept; use it to try a new data set.
- Repeating a stopped command with the same output directory resumes it.
- `grond manual` runs the same calculation one stage at a time, and `grond refine` recalculates the corrections for a model you refined yourself in SHELXL or Olex2. See [docs/USAGE.md](docs/USAGE.md).

**Free set.** By default (`--free-set off`) GROND holds out the free set while it chooses every setting and reports the held-out validation, then refines the chosen model once more against all reflections and writes that as the result. With `--free-set on` the free reflections stay out of the final model as well.

## Results

Open these first in the output directory:

- `TABLE1_FINAL_STATISTICS.md`: SHELXL statistics of the conventional and GROND models, R factors of the observations, and the held-out validation;
- `FINAL_SUMMARY.md` and `HELDOUT_VALIDATION.md`;
- the refined model and the corrected reflection file;
- the maps written by `grond finish`.

[docs/OUTPUTS.md](docs/OUTPUTS.md) describes every file.

## Documentation

- [Installation](docs/INSTALL.md)
- [Usage](docs/USAGE.md) and the [stage-by-stage workflow](docs/MANUAL_WORKFLOW.md)
- [XDS data](docs/XDS_WORKFLOW.md), [DIALS data](docs/DIALS_WORKFLOW.md) and [several crystals](docs/MULTI_DATASET.md)
- [Outputs](docs/OUTPUTS.md) and [file formats](docs/FILE_FORMATS.md)
- [Other integration programs](docs/PORTABLE_INPUTS_AND_ENDPOINTS.md) and [export to PETS, Jana2020 and diffBloch](docs/INTEROPERABILITY.md)
- [Macromolecules (experimental)](docs/MACROMOLECULES.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Source-code map](docs/CODE_MAP.md) and [development](docs/DEVELOPMENT.md)
- [References](docs/REFERENCES.md)

## Development

```bash
python -m pip install -e ".[dev]"
python -m ruff check src/grond tests scripts
python -m pytest
```

## License and citation

GROND is released under the [MIT License](LICENSE). Please cite the software with [CITATION.cff](CITATION.cff) and the methods listed in [docs/REFERENCES.md](docs/REFERENCES.md).

The name comes from Tolkien: "Grond, the Hammer of the Underworld" (*The Silmarillion*).

## Acknowledgements

See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md). The code and documentation were developed with the help of the AI assistants Gemini (Google), ChatGPT (OpenAI) and Claude (Anthropic).
