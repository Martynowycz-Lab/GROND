"Then Morgoth hurled aloft Grond, the Hammer of the Underworld, and swung it down like a bolt of thunder." <br>
_The Silmarillion, Chapter 18, "Of the Ruin of Beleriand and the Fall of Fingolfin," by J.R.R. Tolkien_ 



"Grond! Grond! Grond!" <br>
_The Lord of the Rings: The Return of the King, Book V, Chapter 4, "The Siege of Gondor," by J.R.R. Tolkien_ 

# GROND

GROND is the **G**lobal **R**egularization **O**f **N**on-kinematical **D**iffraction. 

This is an open source python package that corrects summation integrated electron-diffraction intensities for multiple scattering so that they ca be passed to popular refinement programs without special treatment otherwise. It operates as a layer between integration and refinement, after you have solved the structure and refined kinematically, same as other dynamical refinement. It calculates Bloch wave math for every reflection and writes corrected intensities. You can then use SHELXL or Olex2 or whatever reads hkl files to refines the structure without changing anything else. GROND repeats this correction after each time it is refined in SHELXL/OLEX2 until the model stops moving, or after 5 rounds. 

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

## Examples

Zenodo deposits coming soon. 

## Use your own data

1. Integrate your data with DIALS or XDS.
   - DIALS: convert each crystal one time. GROND reads the DIALS files itself. You do not need DIALS for this any more. 

     ```bash
     grond import-dials --refl integrated.refl --expt integrated.expt --out-dir c1
     ```

   - XDS: set `PROFILE_FITTING= FALSE` in `XDS.INP` and integrate again. Use the XDS directory. We also use MINPK=90 in XDS.INP. 
2. The `.res` file must have electron scattering factors on the SFAC lines. Check it:

   ```bash
   grond sfac-check --structure model.res
   ```

3. Write a prep file. For each crystal say `NAME=DIRECTORY,dmin=RESOLUTION`. Give the HKLF 4 file that the
   model was refined against.

   ```bash
   grond prep --structure model.res --baseline-hkl model.hkl \
       --dataset c1=c1,dmin=0.80 --dataset c2=c2,dmin=0.80 --out prep.json
   ```

4. Run GROND on your data:

   ```bash
   grond auto --prep prep.json --out-dir run
   ```

   The defaults are the settings in the paper. GROND holds back 10% of the unique reflections from all fits and
   the final model. It reports R factors on these. If you want to refine the final model against all reflections, add
   `--free-set off`. Cross-validation still runs, but the held-out R factors are biased. If you want to continue a stopped
   run, give the same command again.

5. For large unit cells (say, a metal–organic framework), use `--shared-grid auto`. GROND then calculates a Bloch-wave matrix for each angle through the dataset at a sampling of 100 calculations per physical degree. The shared grid runs on the CPU. A GPU is optional. For matrices that have 2,000 beams or more, install `python -m pip install ".[gpu]"` and add `--shared-grid-gpus 0`. CPUs are typically fine. 

## Results

`grond auto` writes the following files in the output directory. Read them.

- `TABLE1_FINAL_STATISTICS.md`: SHELXL R factors and held-out R factors
- `FINAL_SUMMARY.md`: selected settings, the hand and the final model
- `HELDOUT_VALIDATION.md`: R factors on held-out reflections
- The refined model and its corrected HKL file

`grond finish --run-dir run` writes Fo−Fc maps. `grond <command> -h` shows options `-h -v` and `-h -vv` show more

## Limits

- SHELXL standard uncertainties do not include the uncertainty of this correction. This matters when the initial model is far from being converged. 
- The thickness distribution, mosaic spread, and damage terms are  parameters not measurements.
- GROND does not currently model absorption, twinning or solvent masks. It refuses if your file has HKLF 5, TWIN, BASF and SWAT.
- Our hand test compares the model with its mirror. Do not trust this unless your rotation axis and detector orientation are **KNOWN**
- Do not believe a lower R factor if the maps, ADPs or held-out statistics get worse. Skepticism is important in crystallography. 

## License and citation

MIT ([LICENSE](LICENSE)). Cite the paper and the software ([CITATION.cff](CITATION.cff)). AI coding assistants helped
to write and test this code. The authors checked the algorithms and the numbers.
