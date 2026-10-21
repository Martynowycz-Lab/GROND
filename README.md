# GROND

GROND (Global Regularization Of Non-kinematical Diffraction) corrects electron-diffraction intensities for multiple
scattering. It reads unmerged observations integrated by DIALS or XDS and a solved SHELX model, calculates every
observation with Bloch waves, and writes an HKLF 4 file of corrected intensities that SHELXL or Olex2 refine as
usual. GROND recalculates the correction for each new model until the model stops moving.

The method is described in the paper (preprint link to follow). This repository holds the code used for it.

## Install

Python 3.11 or later on Linux or macOS, and SHELXL. NumPy, SciPy and gemmi are installed with the package.

```bash
git clone https://github.com/Martynowycz-Lab/GROND.git
cd GROND
python3 -m venv .venv && source .venv/bin/activate
python -m pip install .
grond init-config --out grond.config.json      # put the SHELXL path in this file
grond doctor --workflow small-molecule --config grond.config.json
```

Details are in [docs/INSTALL.md](docs/INSTALL.md).

## Inputs

- A SHELX `.res` model with numeric electron scattering factors on its SFAC lines (`grond sfac-check` checks this),
  and the HKLF 4 file it was refined against.
- For each crystal, the unmerged summation intensities and the scan geometry:
  - DIALS: `integrated.refl` and `integrated.expt`, converted once with `grond import-dials`
    ([docs/DIALS.md](docs/DIALS.md));
  - XDS: a directory with `INTEGRATE.HKL` from a run with `PROFILE_FITTING= FALSE` ([docs/XDS.md](docs/XDS.md)).

## Run

```bash
grond import-dials --refl c1/integrated.refl --expt c1/integrated.expt \
    --dials-python /path/to/dials.python --out-dir bridges/c1
grond prep --structure model.res --baseline-hkl model.hkl \
    --dataset c1=bridges/c1,dmin=0.80 --dataset c2=/data/c2/xds,dmin=0.80 --out prep.json
grond auto --prep prep.json --config grond.config.json --workers 40 --out-dir run
grond finish --run-dir run
```

`grond auto --preset fast` gives a quick answer with every check kept. Rerunning a stopped command with the same
`--out-dir` resumes it. `grond <command> -h` lists the common options; `-h -v` and `-h -vv` list the rest.

| Command | What it does |
|---|---|
| `auto` | Full run: indexing check, held-out set, settings by cross-validation, hand test, SHELXL rounds. |
| `manual` | The same calculation one stage at a time, for inspection. |
| `refine` | Runs the calculation again on a model you refined yourself, with the run's data, indexing, hand and held-out set. |
| `finish` | Writes the maps and summary tables of a finished run. |
| `prep` | Writes the input file for one model and one or more crystals. |
| `import-dials` | Converts a DIALS `integrated.refl`/`.expt` pair. |
| `sfac-check`, `init-config`, `doctor` | Check the model, write the tool config, check the installation. |

## What `auto` does

1. Holds out 10% of the unique reflections (a salted hash of each reflection under the Laue class) and refines the
   supplied model again without them.
2. For each observation, selects the beams with |S_g| ≤ 0.01 Å⁻¹ and diagonalizes the Bloch-wave matrix with SciPy
   (`eigh`, single precision) at each rocking point and mosaic tilt. One diagonalization gives every thickness and
   both hands.
3. Chooses the thickness distribution (gamma or wedge), mosaic spread, radiation damage and the smoothness of a
   B-spline frame scale for each crystal by threefold cross-validation on the working reflections.
4. Writes I_corr = κ |F_c|² I_obs / (s I_dyn), fits a two-parameter error model and runs SHELXL. It recalculates
   I_dyn from the refined model, for up to five rounds or until no non-hydrogen atom moves by more than 0.002 Å.
5. Tests the hand of non-centrosymmetric models with a sign test over unique reflections, and reports R factors on
   the held-out reflections with bootstrap confidence intervals.

The run ends by listing the files to open first: `TABLE1_FINAL_STATISTICS.md`, `FINAL_SUMMARY.md`,
`HELDOUT_VALIDATION.md`, the refined model with its corrected HKL file, and `grond_continue_checkpoint.json` for
`grond refine`.

`grond refine --checkpoint run/grond_continue_checkpoint.json --start-res model.res --out-dir run2` takes a model you
refined in SHELXL or Olex2 from GROND's HKL file and runs `grond auto` on it with the run's data, options, indexing,
hand and held-out set. Beams, rocking points, thickness, mosaic spread and damage are chosen again from the new model,
for any number of crystals. Options of `grond auto` given to `refine` override the run's. `--reuse-settings` keeps the
run's settings instead and refits only the frame scales, error model and weights.

By default (`--free-set off`) the chosen model is refined once more against all reflections at the end. With
`--free-set on` the held-out reflections stay out of the final model. All runs in the paper used `--free-set on`.

Large cells (SU-102 has about 800 beams per matrix) need `--shared-grid auto`, which diagonalizes each matrix once
per scan orientation and interpolates onto the rocking points of every reflection. The SU-102 runs of the paper used

```bash
GROND_SHARED_GRID_TABLE_PER_DEGREE=50 GROND_SHARED_GRID_TABLE_PRECISION=single \
grond auto --prep su102.json --out-dir run --workers 40 --free-set on --nested-response-tables \
    --shared-grid auto --basis-centring-coset on --n-points 240 --nested-n-points 60 --chunksize 1 \
    --max-support-sg 0.01 --final-basis-scan-limit 4000 --final-basis-max 4000 --nested-n-basis 1000 \
    --row-plan-cache off
```

With matrices of 2,000 beams or more (for example SU-102 at |S_g| ≤ 0.04 Å⁻¹), `--shared-grid-gpus 0,1,2` solves
them on NVIDIA GPUs with JAX (`python -m pip install ".[gpu]"`); smaller matrices stay on the CPU cores.

Read [LIMITATIONS.md](LIMITATIONS.md) before relying on a result.

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## License and citation

MIT ([LICENSE](LICENSE)). Please cite the paper and the software ([CITATION.cff](CITATION.cff)). AI coding assistants
were used to write and test this code; the authors checked the algorithms and the numbers.
