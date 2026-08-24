# PETS, Jana2020, Dyngo, and diffBloch interoperability

`GROND export-pets` turns a row-preserving DIALS or XDS integration into the
`cif_pets` observation-and-geometry contract. PETS/Jana2020/Dyngo use this
format, and diffBloch reads the same file as its experimental-data input. The
export does not perform a GROND refinement, merge Friedel mates, or choose an
absolute hand.

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
intensity, uncertainty, partiality filtering, or geometry. A fixed-geometry
calculation should therefore give the same scored observations and predictions
under either representation. Use the default overlap representation for Jana
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
