# PETS, Jana2020, Dyngo and diffBloch

`grond export-pets` writes an unmerged DIALS or XDS integration as a `cif_pets` file, the observation-and-geometry format read by PETS, Jana2020 and Dyngo and used by diffBloch as its experimental data. The export does not refine anything, does not merge Friedel mates and does not choose a hand.

`grond import-pets` goes the other way. It reads a PETS2 integration and keeps the PETS observations, uncertainties, zone geometry, UB matrix, rocking width and wavelength as they are. It does not invent detector coordinates that PETS did not write.

## Importing PETS2 data

An ordinary `sample.cif_pets` does not record the frame of each observation, so also give its 13-column `sample.hkl`. GROND checks every index, intensity, uncertainty and frame against it before writing anything.

The cell must have positive lengths and a valid angle/volume combination, and `UB.T @ UB` must agree with the declared reciprocal metric. The allowance is a half-last-digit uncertainty of 5e-6 Å⁻¹ per UB entry (five-decimal PETS output) plus 1e-4 of the reciprocal metric scale. A unit or cell mismatch stops the import.

```bash
grond import-pets \
  --cif-pets sample.cif_pets \
  --pets-hkl sample.hkl \
  --space-group-number 19 \
  --structure model.res \
  --baseline-hkl model.hkl \
  --dataset-name crystal_01 \
  --out-dir crystal_01_pets_bridge
```

PETS also writes `*_dyn.cif_pets`, in which one observation can appear in several virtual frames. These occurrences are the right comparison target for Jana/Dyngo and diffBloch, but they are correlated, not independent measurements, so GROND refuses them unless you say so explicitly. A Klar–Palatinus selection can be applied at import:

```bash
grond import-pets \
  --cif-pets sample_dyn.cif_pets \
  --allow-virtual-frames \
  --klar-rsg 0.66 \
  --klar-dsg 0.0 \
  --space-group-number 19 \
  --structure model.res \
  --baseline-hkl model.hkl \
  --dataset-name crystal_01_dyn \
  --out-dir crystal_01_dyn_bridge
```

## `grond auto` with PETS2 data

For an ordinary (not virtual-frame) import, `--baseline-hkl` may name PETS's own SHELX HKL export. GROND requires its indices, intensities and sigmas to match the `cif_pets` and `.hkl` exactly, in the same order. When that check passes, `grond auto` runs on the import:

```bash
grond auto \
  --prep crystal_01_pets_bridge/grond_prep.json \
  --config grond.config.json \
  --out-dir crystal_01_grond_auto
```

Only one crystal is supported. GROND skips the transform scout, because the import has already verified the setting and symmetry. PETS observations have no continuous scan angle and no detector coordinates, so GROND changes some settings and records the changes in `PETS_STATIC_AUTO_CONTRACT.json`: per-frame scales on angular groups instead of P-spline scales, `--nuisance-selection sequential`, `--mosaicity-mode none`, no detector scales and no frame-offset search. It chooses the beams, quadrature, thickness, frame scales and hand itself and runs the default SHELXL rounds. Non-positive observations are kept when they were in the import.

A `*_dyn.cif_pets` import cannot go through `grond auto`: dividing correlated virtual-frame occurrences by a Bloch/Born ratio does not give a valid SHELXL data set. Use `grond pets-evaluate` for a fixed-model calculation on it. Beams, quadrature, reciprocal support and thickness grid must be given explicitly, because the defaults for XDS and DIALS data are too shallow for virtual-frame rocking curves:

```bash
grond pets-evaluate \
  --prep crystal_01_dyn_bridge/grond_prep.json \
  --out-dir crystal_01_pets_native \
  --n-basis 400 \
  --n-points 41 \
  --topup-d-min 0.6 \
  --max-support-sg 0.01 \
  --thickness-grid 800,1000,1200,1400,1600,1800,2000 \
  --workers 16
```

The thickness grid is in Å. The command keeps non-positive observations, fits one scale per PETS zone on working observations, chooses the thickness by held-out wR2, compares the inverse hand by negative-thickness propagation, and writes the observation statistics. It never runs SHELXL. Increase N, Q and the reciprocal support until the result no longer changes before calling it final. For the conventional comparison, refine the PETS data separately in SHELXL.

## Export from DIALS

The DIALS route calls the PETS writer of the selected DIALS installation and exports the signed `miller_index`, intensity, uncertainty, wavelength, UB matrix and virtual-frame geometry. Summation intensities are the default. With a `scaled.refl`, `--row-selection scaled` keeps only the observations accepted by `dials.scale` while still exporting the summation intensities.

```bash
grond export-pets \
  --dials-refl scaled.refl \
  --dials-expt scaled.expt \
  --dials-python /path/to/dials.python \
  --row-selection scaled \
  --intensity sum \
  --min-partiality 0.90 \
  --dataset-name crystal_01 \
  --out-dir crystal_01_pets
```

PETS stores one unit cell. When a DIALS experiment has a scan-varying cell as well as a scan-varying orientation, the default (`--scan-varying-cell project`) keeps `U(scan)` and replaces `B(scan)` by the single declared cell, and records the range of the cell in the manifest. It refuses a length change above 3% or an angle change above 1° (`--max-cell-length-deviation`, `--max-cell-angle-deviation`). `--scan-varying-cell error` refuses any scan-varying cell.

## Export from XDS

XDS defines

```text
IOBS = Irecorded * RLP / (PEAK / 100)
```

so GROND reconstructs the recorded intensity and its uncertainty as

```text
Irecorded = IOBS * (PEAK / 100) / RLP
```

For an unambiguous summation export, `XDS.INP` or `INTEGRATE.HKL` must declare `PROFILE_FITTING=FALSE`; profile-fitted input is refused unless `--allow-profile-fitted-xds` is given. The default `--xds-row-selection correct` keeps only the `INTEGRATE.HKL` observations accepted into the unmerged `XDS_ASCII.HKL`, matched by position and signed index after the XDS-to-export basis transform; `integrate` keeps every valid observation. The transform, determinant, rounding error, source hashes, variance handling and row counts are written to `pets_export_manifest.json`.

The default `--xds-sigma-mode reported` keeps the XDS uncertainty and applies the same PEAK/RLP factor as to the intensity. `--xds-sigma-mode raw` instead inverts the declared XDS `VARIANCE_MODEL` (`v0 = v_reported/a - b Irecorded^2`); observations where that inversion is not positive keep the reported variance. Weighted residuals from the two modes are not comparable.

```bash
grond export-pets \
  --xds-dir /path/to/xds_noprf \
  --dials-python /path/to/dials.python \
  --xds-row-selection correct \
  --min-partiality 0.90 \
  --dataset-name crystal_01 \
  --out-dir crystal_01_pets
```

The XDS route uses DIALS to express the XDS geometry in the PETS frame. Keep the manifest with the file.

The reconstructed recorded intensity is the target for an absolute rocking-intensity program such as diffBloch. `grond auto` works on `IOBS` and writes its corrected HKL files in `IOBS` units, so they can be compared with the conventional refinement.

## Virtual frames

By default PETS writes an observation into every virtual frame within the excitation-error cutoff (`--frame-ownership overlap`). This is what PETS, Jana2020 and Dyngo expect, but one observation can then appear many times, and a downstream program must not treat the copies as independent.

For diffBloch refinement, `--frame-ownership nearest-excitation` assigns each observation to exactly one virtual frame: the one with the smallest absolute excitation error, the lower frame on a tie. The manifest records the observation count, the candidate occurrences, the kept occurrences and the removed overlaps. Only frame ownership changes; indices, intensities, uncertainties, partiality filtering and geometry do not. Keep the default overlap for Jana unless its import calls for unique ownership.

## Output

The output directory contains one `*.cif_pets` file per experiment, `pets_export_manifest.json` and a short `README.md`. GROND reads every cell, wavelength, UB matrix, zone axis, reflection, intensity and uncertainty back after writing. It also replaces the zero `dstarmax` written by the DIALS PETS writer with the largest reciprocal spacing in the file, which diffBloch's input checks need.

For a multi-experiment DIALS pair, omit `--experiment-id` to write every experiment, or repeat `--experiment-id N` to choose some. A diffBloch calculation also needs a structure CIF and its own YAML settings; the `cif_pets` file is its `exp_data` input. Give Jana2020/Dyngo the `cif_pets` file with the manifest and the same indexed structure setting.
