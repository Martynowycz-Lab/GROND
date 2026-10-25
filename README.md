"Then Morgoth hurled aloft Grond, the Hammer of the Underworld, and swung it down like a bolt of thunder." <br>
_The Silmarillion, Chapter 18, "Of the Ruin of Beleriand and the Fall of Fingolfin," by J.R.R. Tolkien_ 



"Grond! Grond! Grond!" <br>
_The Lord of the Rings: The Return of the King, Book V, Chapter 4, "The Siege of Gondor," by J.R.R. Tolkien_ 

# GROND

GROND stands for the **G**lobal **R**egularization **O**f **N**on-kinematical **D**iffraction. This is an open source python package that corrects summation integrated electron-diffraction intensities for multiple scattering so that they ca be passed to popular refinement programs without special treatment otherwise. It operates as a layer between integration and refinement, after you have solved the structure and refined kinematically, same as other dynamical refinement. It calculates Bloch wave math for every reflection and writes corrected intensities. You can then use SHELXL or Olex2 or whatever reads hkl files to refines the structure without changing anything else. GROND repeats this correction after each time it is refined in SHELXL/OLEX2 until the model stops moving, or after 5 rounds. 

GROND does not integrate data, solve structures, or refine structures itself. You start with your data integrated from DIALS or XDS and a solved model.

Our paper describes how this actually works and its limitations, the SI is fairly well laid out (preprint link).

## Installation

Requirements:

- Linux or macOS with Python 3.11 to 3.14
- Windows with WSL and the same
- SHELXL on your `PATH`
- Optional: `shelx2map` (CCP4) for SHELXL Fo−Fc maps

Do this in your terminal:

```bash
git clone https://github.com/Martynowycz-Lab/GROND.git
cd GROND
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
grond doctor
```

`grond doctor` should show `Ready.` and the path of your `shelxl` binary.GROND sets its own thread limits at 80% of available, but this can be overridden. Hilariously, the PyPI package `grond` is a different program (a geology program?), so install from this repository. We are constantly fixing stuff, so update often. 

## Run an example

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
   - DIALS: convert each crystal one time. GROND reads the DIALS files itself. You do not need DIALS for this.

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

- SHELXL standard uncertainties do not include the uncertainty of this correction. This matters when the initial model is far from being converged. 
- Our thickness distribution, mosaic spread, and damage terms are  parameters not measurements.
- GROND does not currently model absorption, twinning or solvent masks. It refuses if your file has HKLF 5, TWIN, BASF and SWAT.
- Our hand test compares the model with its mirror. Do not trust this unless your rotation axis and detector orientation are **KNOWN**
- Do not accept a lower R factor if the maps, displacement parameters or held-out statistics get worse.

## License and citation

MIT ([LICENSE](LICENSE)). Cite the paper and the software ([CITATION.cff](CITATION.cff)). AI coding assistants helped
to write and test this code. The authors checked the algorithms and the numbers.
