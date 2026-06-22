# Portable Inputs and Refinement Endpoints

GROND has two native input readers today: XDS `INTEGRATE.HKL` plus `XPARM.XDS`/`GXPARM.XDS`, and DIALS `integrated.refl` plus `integrated.expt`. Those are the supported paths for normal `GROND auto` runs. Other integration programs can still feed GROND if they can write the same information into the portable `grond-row-v1` row contract. The point of this contract is not to make every external file format part of GROND. The point is to give external wrappers, custom scripts, and future readers one small target format that says exactly what GROND needs.

The JSON metadata schema is in `schemas/grond-row-v1.schema.json` in the source tree. Wheel installs also include the same file under `grond/schemas/grond-row-v1.schema.json`.

## What A GROND Row Means

A row is one signed observation of one reflection in one exposure or rotation-integration support window. It is not a merged unique reflection. GROND needs rows because the Bloch correction depends on the signed index, the crystal orientation at the time that spot was integrated, the excitation error near the rocking condition, and row-level quality information. Once all observations of the same `hkl` are merged, much of that geometry has been thrown away.

The minimal row table can be CSV, TSV, or JSON Lines. A CSV example looks like this:

```csv
row_id,dataset_id,h,k,l,Iobs,sigma_Iobs,frame,x_px,y_px,partiality,lp,profile_correlation,bbox_z0,bbox_z1
0,crystal_a,1,0,2,1284.5,42.1,18.0,1912.4,2105.7,1.00,1.0,91.4,17.5,18.5
1,crystal_a,0,1,3,422.8,27.6,18.0,2063.9,1988.2,0.95,1.0,88.7,17.4,18.6
```

The required row fields are `h`, `k`, `l`, `Iobs`, and `sigma_Iobs`. The Miller indices must be signed integers and must be in the same basis as the solved structure after any known reindexing transform has been applied. `Iobs` should be the background-subtracted summation-style integrated intensity on a consistent scale. `sigma_Iobs` is the standard uncertainty of that intensity in the same scale; GROND does not silently flip negative sigma values to positive values. The row must also provide either `frame` or `alpha_deg`. `frame` is the calculated or centroid frame coordinate in the scan convention given by the metadata sidecar. `alpha_deg` can be used instead when the integration program naturally reports the goniometer angle directly.

The recommended optional fields are `x_px`, `y_px`, `frame_obs`, `partiality`, `lp`, `profile_correlation`, `rlp`, `peak`, `corr`, `bbox_z0`, `bbox_z1`, and `panel`. GROND can run without most of them, but they make geometry audits, partiality gates, detector-scale diagnostics, and rocking-width estimates much better. If `partiality` is present, GROND writes `100 * partiality` into the compatibility `PEAK` field. If `bbox_z0` and `bbox_z1` are present, GROND can use their width as an auditable rocking-window estimate.

## Geometry Sidecar

The row table is paired with a JSON metadata file. This file tells GROND how to convert a signed row into a reciprocal vector and excitation error:

```json
{
  "schema_version": "grond-row-v1",
  "dataset_id": "crystal_a",
  "geometry": {
    "wavelength_A": 0.02508,
    "incident_beam_direction": [0.0, 0.0, 1.0],
    "rotation_axis": [1.0, 0.0, 0.0],
    "starting_frame": 1.0,
    "starting_angle_deg": 0.0,
    "oscillation_range_deg": 0.5,
    "unit_cell": [7.0, 8.0, 9.0, 90.0, 90.0, 90.0],
    "reciprocal_A_matrix": [
      [0.1428571429, 0.0, 0.0],
      [0.0, 0.125, 0.0],
      [0.0, 0.0, 0.1111111111]
    ]
  }
}
```

`reciprocal_A_matrix` is the most important field. It is the 3 by 3 matrix that maps a signed column vector `(h,k,l)` into a reciprocal-space vector in the laboratory coordinate system before the goniometer rotation for that frame is applied. If your program instead gives real-space unit-cell axes in the same laboratory basis, write them as `real_space_axes_A` with the real axes as columns; GROND will invert them. For scan-varying orientation models, add `A_scan_points` as a list of 3 by 3 reciprocal matrices. GROND linearly interpolates those matrices by frame before applying the goniometer rotation.

With this sidecar, GROND computes

```text
alpha(frame) = starting_angle_deg + (frame - starting_frame) * oscillation_range_deg
g_h(alpha)  = R(rotation_axis, alpha) A(frame) h
Sg(alpha)   = (k^2 - |s0 + g_h(alpha)|^2) / (2k)
```

where `s0` is the incident wave vector and `k = 1 / wavelength_A`. This is why a row-only merged HKL file is insufficient for Bloch regularization: it does not define `alpha`, `A(frame)`, or the signed row geometry.

## Commands For Wrapper Authors

Generate a template pair:

```bash
GROND row-template \
  --rows-out my_rows.csv \
  --metadata-out my_rows_metadata.json
```

Validate a table:

```bash
GROND row-validate \
  --rows my_rows.csv \
  --metadata my_rows_metadata.json
```

Write the current engine compatibility directory:

```bash
GROND row-validate \
  --rows my_rows.csv \
  --metadata my_rows_metadata.json \
  --write-xds-dir grond_portable_bridge
```

The compatibility directory contains an `INTEGRATE.HKL` whose header declares `FORMAT=GROND_PORTABLE_XDS_COMPAT`, plus `GROND_ROW_CONTRACT_MANIFEST.json`. This file is XDS-like only as an internal compatibility layer. It is not claiming that the data were processed by XDS. GROND will not write this bridge unless all rows have finite intensities and positive sigmas. Once the bridge exists, it can be used as the row-preserving dataset path in the same way as an XDS directory:

```bash
GROND auto \
  --xds-dir grond_portable_bridge \
  --structure solved_model.res \
  --baseline-hkl solved_model.hkl \
  --out-dir grond_out
```

## Endpoint Contract

GROND separates Bloch regularization from the model-update program. The default implemented endpoint is SHELXL because it accepts the corrected HKLF-style reflection files GROND already writes, can perform a single least-squares cycle cleanly, and returns an updated `RES` model that GROND can simulate again. If no external refiner is available, `bloch_only` remains useful: GROND still runs the Bloch simulation, nuisance and thickness decisions, row/MR residuals, corrected-reflection diagnostics, and polarity checks from the supplied model.

The general endpoint contract is deliberately simple. A refiner receives the current model, a corrected reflection target, an output directory, and a small cycle count. It must write a refined model that GROND can read again as `RES`, `CIF`, `PDB`, or mmCIF. For a one-cycle feedback loop, the endpoint should update coordinates and displacement parameters against the current corrected target without adding an empirical extinction correction unless the user explicitly requested a diagnostic comparison.

GROND exposes the endpoint registry:

```bash
GROND endpoints --config grond.config.json
GROND endpoints --config grond.config.json --json
```

The registry distinguishes production support, no-refiner fallback support, diagnostic native update support, and the generic adapter:

```text
shelxl          implemented_default
bloch_only      implemented_no_refiner
native_gradient implemented_diagnostic
phenix_refine   macromolecular_adapter
refmac5         macromolecular_adapter
servalcat       macromolecular_adapter
custom          adapter_template
```

`shelxl` is wired into the full automatic feedback loop. `bloch_only` is the no-refiner fallback. `native_gradient` is the slow row-space diagnostic. `phenix_refine`, `refmac5`, and `servalcat` are macromolecular command-template adapters for corrected MTZ files with fixed FreeR flags. `custom` is the escape hatch for anything else: if a lab can turn GROND's corrected reflections and model into a one-cycle external update and back into a readable model, the command template can describe it without changing GROND's input readers.

For a custom endpoint, add a block to `grond.config.json`:

```json
{
  "tools": {
    "custom_refiner": "/path/to/my_refiner"
  },
  "endpoints": {
    "custom": {
      "command": "/path/to/my_refiner --model {model} --hkl {reflections} --cycles {cycles} --out {out_dir}",
      "output_model": "{out_dir}/refined_model.cif"
    }
  }
}
```

The placeholders are `{model}`, `{reflections}`, `{out_dir}`, and `{cycles}`. GROND can probe and report this adapter now. Fully automatic replacement of the SHELXL feedback loop requires a tested adapter for the chosen file format and will be enabled endpoint by endpoint rather than assumed globally. Programs that prefer MTZ, PDB/mmCIF, or their own restraint dictionaries can still be used through `custom` if the wrapper script performs the conversion and writes a model file GROND can parse again.

For macromolecular work, use the MTZ layer instead of forcing a protein through HKLF4. `GROND make-free-flags` assigns FreeR labels at unique-HKL level, `GROND export-mtz` writes corrected `IMEAN` and `SIGIMEAN` values with `FreeR_flag`, and `GROND protein-ls1` runs one configured endpoint macrocycle. The row-level work/free split remains a GROND row-contract validation split; it is not a substitute for macromolecular Rfree. The full macromolecular wrapper contract is documented in [docs/MACROMOLECULES.md](MACROMOLECULES.md).
