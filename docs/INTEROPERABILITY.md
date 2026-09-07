# PETS, Jana2020, Dyngo, and diffBloch interoperability

`GROND export-pets` turns a row-preserving DIALS or XDS integration into the
`cif_pets` observation-and-geometry contract. PETS/Jana2020/Dyngo use this
format, and diffBloch reads the same file as its experimental-data input. The
export does not perform a GROND refinement, merge Friedel mates, or choose an
absolute hand.

`GROND import-pets` goes in the other direction. It reads a native PETS2
integration and preserves PETS ownership of the signed observation, uncertainty,
zone geometry, UB matrix, rocking width, and wavelength. It does not reconstruct
detector coordinates that PETS did not write.

## Importing native PETS2 data

An ordinary `sample.cif_pets` does not record the acquisition frame for each
row. Supply its companion 13-column `sample.hkl`; GROND checks every HKL,
intensity, uncertainty, and frame before writing the bridge.

The cell must have finite positive lengths and a physically valid angle/volume
combination. GROND checks that `UB.T @ UB` agrees with the declared reciprocal
cell metric; a normalized zone-axis direction alone cannot establish its scale.
The check propagates a half-last-digit allowance of `5e-6` inverse Angstroms
per UB entry, appropriate to five-decimal PETS output, and adds `1e-4` of the
reciprocal metric scale for cell/model serialization. A legitimate orientation
rotation preserves this metric. A unit or cell mismatch raises an explicit error.
The general-cell reciprocal-basis sign was corrected during the numerical audit;
regenerate earlier PETS bridges for nonorthogonal cells before comparing results.

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

PETS also writes `*_dyn.cif_pets`, in which one source reflection can appear in
several PETS-owned virtual frames. These occurrences are the correct direct
comparison target for Jana/Dyngo and diffBloch, but they are correlated rather
than new independent measurements. GROND refuses them unless this is
acknowledged explicitly. A Klar--Palatinus support can be selected at import:

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

The bridge supports native Born/Bloch row statistics, thickness scans, elastic
negative-thickness hand checks, and experimental native structure updates. Keep
detector scales disabled because PETS does not provide detector coordinates.
For an ordinary import, `--baseline-hkl` may name PETS' own SHELX export. GROND
requires its ordered HKL, intensity, and sigma columns to match the ordinary
`cif_pets` and companion `.hkl` exactly. A successful audit is stored in the
bridge and permits the normal automatic correction-and-endpoint workflow:

```bash
grond auto \
  --prep crystal_01_pets_bridge/grond_prep.json \
  --out-dir crystal_01_grond_auto
```

AUTO skips transform scouting only because the importer has already verified
the proper identity setting and model symmetry. It selects the screen and final
beam bases, quadrature, thickness distribution, compact angular frame scales,
hand, and q/LS1 feedback itself. Detector fields remain disabled because PETS
does not export detector coordinates. Every feedback cycle preserves signed
nonpositive observations when they were present in the imported integration.

PETS virtual-frame rocking intensities are not a conventional static HKLF4
integration. A `*_dyn.cif_pets` calculation therefore does not acquire a valid
SHELXL endpoint merely by dividing those correlated occurrences by a
Bloch/Born ratio. Report the separate conventional SHELXL refinement as the
endpoint comparator for that native-row route.

Use `grond pets-evaluate` for the safe fixed-model path. Beam count, angular
quadrature, reciprocal support, and thickness grid are deliberately explicit;
the generic XDS/DIALS scout defaults are too shallow for PETS virtual-frame
rocking calculations and are refused. For the tyrosine example matched to its
Jana settings, the command is:

```bash
grond pets-evaluate \
  --prep crystal_01_dyn_bridge/grond_prep.json \
  --out-dir crystal_01_pets_native \
  --n-basis 420 \
  --n-points 41 \
  --topup-d-min 0.6134969 \
  --max-support-sg 0.01 \
  --thickness-grid 800,1000,1200,1400,1500,1600,1800,2000 \
  --workers 48
```

The command always includes non-positive observations retained by PETS, fits
one scale per populated PETS zone on work rows, keeps detector scales at zero,
selects thickness by held-out row `wR2`, evaluates the inverse hand by elastic
negative-thickness propagation, and writes the JANA/PETS-style row table. It
always skips SHELXL feedback and endpoint generation. Repeat at larger N/Q and
wider reciprocal support until the result is numerically stable before calling
the calculation final.

## DIALS

The DIALS route calls the PETS writer shipped with the selected DIALS
installation. It exports the requested signed `miller_index`, intensity,
uncertainty, wavelength, UB matrix, and virtual-frame geometry. Summation is
the default. A `scaled.refl` can be used with `--row-selection scaled` to keep
only observations accepted by `dials.scale` while still exporting the original
summation column.

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

PETS stores one unit cell. Some refined DIALS experiments carry a
scan-varying cell as well as a scan-varying orientation. The default bounded
projection retains `U(scan)` and replaces `B(scan)` by the one declared static
cell, recording the source range and maximum cell variation in the manifest.
It refuses a length change above 3% or an angle change above 1 degree. Use
`--scan-varying-cell error` to require a strictly representable experiment, or
change a limit explicitly when the approximation has been independently
justified.

## XDS

XDS does not need to be rerun with a hidden “Lorentz off” switch. The documented
`INTEGRATE.HKL` relationship is

```text
IOBS = Irecorded * RLP / (PEAK / 100)
```

so GROND reconstructs the recorded observation and its linearly transformed
uncertainty using

```text
Irecorded = IOBS * (PEAK / 100) / RLP.
```

For an unambiguous summation export, `XDS.INP` or `INTEGRATE.HKL` must declare
`PROFILE_FITTING=FALSE`. The default `--xds-row-selection correct` then binds
the raw INTEGRATE observations to the rows accepted into unmerged
`XDS_ASCII.HKL`. Position and signed HKL must both match exactly after the
derived XDS-to-export basis transform. The transform, determinant, rounding
error, source hashes, variance handling, and row counts are written to
`pets_export_manifest.json`.

By default, `--xds-sigma-mode reported` retains XDS's calibrated reported
uncertainty and applies the same PEAK/RLP linear transformation as the
intensity. The labeled `--xds-sigma-mode raw` sensitivity option instead
inverts a declared XDS `VARIANCE_MODEL`, using
`v0 = v_reported/a - b Irecorded^2`; rows for which that inversion is not
positive retain the reported recorded-scale variance. The mode, coefficients,
fallback count, and exact policy are recorded. Do not compare weighted
residuals across the two sigma contracts as if they used the same likelihood.

```bash
grond export-pets \
  --xds-dir /path/to/xds \
  --dials-python /path/to/dials.python \
  --xds-row-selection correct \
  --min-partiality 0.90 \
  --dataset-name crystal_01 \
  --out-dir crystal_01_pets
```

The XDS route uses DIALS to express XDS geometry in the PETS coordinate frame.
That conversion is an interoperability bridge, not evidence that the two
integration programs produce interchangeable measurements. The manifest must
travel with the file. Profile-fitted XDS input is refused unless the diagnostic
`--allow-profile-fitted-xds` override is given.

The recorded-intensity reconstruction is the appropriate direct target for an
absolute rocking-intensity calculation such as diffBloch. GROND's production
operation is different: it fits a matched Bloch/Born ratio on the declared row
contract and writes a conventional corrected HKLF4 endpoint. When an internal
XDS target-transform diagnostic fits the recorded-intensity reconstruction,
GROND therefore removes that diagnostic target factor again before every
SHELXL feedback step. The fit can change thickness and relative nuisance
scales, but the resulting endpoint remains in ordinary XDS IOBS units and can
be compared with the conventional endpoint. A recorded-intensity HKLF4 file
that has not been restored in this way is not a valid endpoint comparison.

## Virtual-frame ownership

The stock PETS convention writes an observation into every eligible virtual
frame within the excitation-error cutoff. This overlap is retained by default
for PETS/Jana2020/Dyngo compatibility. It can repeat one integrated observation
many times in the materialized `cif_pets`, so it must not be treated as
independent repeated evidence by a downstream optimizer.

For direct diffBloch refinement, add `--frame-ownership nearest-excitation`.
GROND then assigns each selected source observation to exactly one eligible
virtual frame: the frame with minimum absolute excitation error, with a
deterministic lower-frame tie break. The source-row count, candidate occurrence
count, unique owned count, and removed overlap count are recorded in the
manifest. This changes only virtual-frame ownership; it does not alter HKL,
intensity, uncertainty, partiality filtering, or geometry. Overlapping occurrences still have their recorded zone geometry and induce
correlated evidence. Compare predictions only after matching both the source
observation and its retained zone geometry; removing overlap alone does not
establish equivalence of downstream objectives. Use the default overlap representation for Jana
unless that program's import contract explicitly calls for unique ownership.

## Outputs and validation

The output directory contains one `*.cif_pets` file per selected experiment,
`pets_export_manifest.json`, and a short `README.md`. GROND parses every cell,
wavelength, UB matrix, zone-axis identifier, signed reflection, intensity, and
positive uncertainty after export. It also replaces the zero `dstarmax` emitted
by the DIALS PETS writer with the actual maximum reciprocal spacing in the
materialized rows. This makes the file usable by diffBloch's input and
convergence checks without changing any observation.

For a multi-experiment DIALS pair, omit `--experiment-id` to write all
experiments separately, or repeat `--experiment-id N` to select a subset. A
diffBloch experiment still needs a structure CIF and its own YAML run settings;
the exported `cif_pets` file is the `exp_data` input. Jana2020/Dyngo should be
given the `cif_pets` file together with the manifest and the same indexed
structure setting.
