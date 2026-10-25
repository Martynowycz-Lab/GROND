# GROND

GROND (Global Regularization Of Non-kinematical Diffraction) corrects electron-diffraction intensities for multiple
scattering. It calculates each observation with Bloch waves and writes corrected intensities. SHELXL or Olex2 then
refines the structure as for X-ray data. GROND repeats the correction until the model stops moving.

GROND does not integrate images or solve structures. It starts from DIALS or XDS integration and a solved SHELX model.

The paper describes the method (preprint link to follow).

## Install

You need:

- Linux or macOS with Python 3.11 to 3.14.
- SHELXL on your `PATH`.
- Optional: `shelx2map` (CCP4) for SHELXL Fo−Fc maps.
- Optional: DIALS, only to convert DIALS files.

Do these steps:

```bash
git clone https://github.com/Martynowycz-Lab/GROND.git
cd GROND
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
grond doctor
```

`grond doctor` must show `Ready.` and the path of `shelxl`. You do not set environment variables. GROND sets its own
thread limits. The PyPI package `grond` is a different program. Install from this repository.

## Run the example

The example has two L-tyrosine crystals (DIALS, 300 kV) and their kinematic model. Download it from Zenodo (DOI to
follow) and extract it. Then do these steps:

```bash
cd grond_example_tyrosine
grond prep --structure model.res --baseline-hkl model.hkl \
    --dataset noef_4_2=noef_4_2,dmin=0.80 --dataset noef_6_2=noef_6_2,dmin=0.80 --out prep.json
grond auto --prep prep.json --out-dir run --preset fast
```

The fast run takes approximately 15 minutes on 10 cores. The held-out R_obs is approximately 9.7%. Without
`--preset fast`, the run gives the paper value, 9.3%.

## Use your data

1. Integrate with DIALS or XDS.
   - DIALS: convert each crystal one time. GROND finds DIALS from the `dials.*` commands on your `PATH`.

     ```bash
     grond import-dials --refl integrated.refl --expt integrated.expt --out-dir c1
     ```

   - XDS: set `PROFILE_FITTING= FALSE` in `XDS.INP` and integrate again. Use the XDS directory.
2. Prepare the model. The `.res` file must have numeric electron scattering factors on its SFAC lines. Check it:

   ```bash
   grond sfac-check --structure model.res
   ```

3. Write the prep file. Give each crystal as `NAME=DIRECTORY,dmin=RESOLUTION`. Also give the HKLF 4 file that the
   model was refined against.

   ```bash
   grond prep --structure model.res --baseline-hkl model.hkl \
       --dataset c1=c1,dmin=0.80 --dataset c2=c2,dmin=0.80 --out prep.json
   ```

4. Run GROND:

   ```bash
   grond auto --prep prep.json --out-dir run
   ```

   The defaults are the settings of the paper. GROND keeps 10% of the unique reflections out of all fits and out of
   the final model, and reports R factors on them. To refine the final model against all reflections, add
   `--free-set off`. The cross-validation still runs, but the held-out R factors are then biased. To continue a stopped
   run, give the same command again.

5. For a large cell (for example, a metal–organic framework), add `--shared-grid auto`. GROND then calculates each
   orientation one time for all reflections. The shared grid runs on the CPU. A GPU is optional: for matrices of
   2,000 beams or more, install `python -m pip install ".[gpu]"` and add `--shared-grid-gpus 0`.

## Results

`grond auto` writes these files in the output directory. Open them first:

- `TABLE1_FINAL_STATISTICS.md`: SHELXL R factors and the held-out R factors.
- `FINAL_SUMMARY.md`: the selected settings, the hand and the final model.
- `HELDOUT_VALIDATION.md`: R factors on the held-out reflections.
- The refined model and its corrected HKL file.

`grond finish --run-dir run` writes Fo−Fc maps. `grond <command> -h` shows the options. `-h -v` and `-h -vv` show more.

## Limits

- GROND keeps the correction fixed during each SHELXL refinement. SHELXL standard uncertainties do not include the
  uncertainty of the correction.
- The thickness distribution, mosaic spread and damage term are parameters of the calculation, not measurements.
- GROND does not model absorption, twinning or solvent masks. It refuses HKLF 5, TWIN, BASF and SWAT.
- The hand test compares the model with its inverse. Calibrate the rotation axis and detector on a crystal of known
  hand.
- Do not accept a lower R factor if the maps, displacement parameters or held-out statistics get worse.

## License and citation

MIT ([LICENSE](LICENSE)). Cite the paper and the software ([CITATION.cff](CITATION.cff)). AI coding assistants helped
to write and test this code. The authors checked the algorithms and the numbers.
