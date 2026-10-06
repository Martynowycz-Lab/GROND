# Portable inputs and refinement endpoints

GROND reads XDS (`INTEGRATE.HKL`) and DIALS (`integrated.refl` with `integrated.expt`) directly, and PETS2 data through `grond import-pets` ([INTEROPERABILITY.md](INTEROPERABILITY.md)). Other integration programs can feed GROND by writing the portable `grond-row-v1` format: a table of observations and a JSON file with the geometry. It states exactly what GROND needs, so a wrapper script can produce it.

The JSON schema is `schemas/grond-row-v1.schema.json` in the source tree; the wheel installs it as `grond/schemas/grond-row-v1.schema.json`.

## What a row is

A row is one signed observation of one reflection in one exposure or integration window, not a merged reflection. The Bloch correction depends on the signed index, the crystal orientation when the spot was recorded and its excitation error, and merging throws that away.

The table can be CSV, TSV or JSON Lines:

```csv
row_id,dataset_id,h,k,l,Iobs,sigma_Iobs,frame,x_px,y_px,partiality,lp,profile_correlation,bbox_z0,bbox_z1
0,crystal_a,1,0,2,1284.5,42.1,18.0,1912.4,2105.7,1.00,1.0,91.4,17.5,18.5
1,crystal_a,0,1,3,422.8,27.6,18.0,2063.9,1988.2,0.95,1.0,88.7,17.4,18.6
```

Required fields are `h`, `k`, `l`, `Iobs` and `sigma_Iobs`, and each row needs either `frame` or `alpha_deg`.

- `h`, `k`, `l` are signed integers in the setting of the solved model, after any known reindexing.
- `Iobs` is the background-subtracted summation intensity on one consistent scale, and `sigma_Iobs` its standard uncertainty on the same scale. GROND does not turn negative sigmas positive; the converted file requires positive sigmas on every row.
- `frame` is the calculated or centroid frame coordinate in the scan convention of the geometry file; `alpha_deg` is the goniometer angle, for programs that report it directly.

Optional fields are `row_id`, `dataset_id`, `x_px`, `y_px`, `frame_obs`, `partiality`, `lp`, `profile_correlation`, `rlp`, `peak`, `corr`, `bbox_z0`, `bbox_z1`, `panel` and `source`. They help the geometry checks and the partiality limit. `partiality` is written as `100 * partiality` into the `PEAK` column, and `bbox_z0`/`bbox_z1` are kept as the observation's frame range.

## Geometry file

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

`reciprocal_A_matrix` maps a signed column vector `(h,k,l)` to a reciprocal-space vector in the laboratory frame before the goniometer rotation of that frame. If your program gives real-space axes in the same frame, write them as `real_space_axes_A` (axes as columns) and GROND inverts them. For a scan-varying orientation, add `A_scan_points`, a list of 3×3 reciprocal matrices; GROND interpolates them linearly by frame.

GROND then computes

```text
alpha(frame) = starting_angle_deg + (frame - starting_frame) * oscillation_range_deg
g_h(alpha)  = R(rotation_axis, alpha) A(frame) h
Sg(alpha)   = (k^2 - |s0 + g_h(alpha)|^2) / (2k)
```

where `s0` is the incident wave vector and `k = 1 / wavelength_A`. A merged HKL file cannot give `alpha`, `A(frame)` or the signed observations, which is why it is not enough for the Bloch calculation.

## Commands

```bash
grond row-template --rows-out my_rows.csv --metadata-out my_rows_metadata.json
grond row-validate --rows my_rows.csv --metadata my_rows_metadata.json
grond row-validate --rows my_rows.csv --metadata my_rows_metadata.json --write-xds-dir grond_portable_bridge
```

The first writes a template pair, the second checks a table, and the third also writes a directory GROND can read. It contains an `INTEGRATE.HKL` with the header `FORMAT=GROND_PORTABLE_XDS_COMPAT` and `GROND_ROW_CONTRACT_MANIFEST.json`. The file is in GROND's own format; it does not claim the data came from XDS. GROND writes it only when every row has a finite intensity and a positive sigma. Use the directory like an XDS directory:

```bash
grond auto \
  --xds-dir grond_portable_bridge \
  --structure solved_model.res \
  --baseline-hkl solved_model.hkl \
  --config grond.config.json \
  --out-dir grond_out
```

## Refinement endpoints

GROND keeps the Bloch calculation separate from the program that refines the model. `grond auto` uses SHELXL: it reads the corrected `HKLF 4` files GROND writes and returns a `.res` that GROND can calculate again. Each round, SHELXL is run to convergence (`L.S. 10` per call, converged weights, at most 20 calls); see [USAGE.md](USAGE.md).

For SHELXL, GROND writes corrected `HKLF 4` intensities. A model refined against `HKLF 3` can be the start model, but GROND's files are always `HKLF 4`. `ACTA` is removed from the files GROND writes. Models with `HKLF 5`, `TWIN`, `BASF`, `SWAT` or a SQUEEZE/solvent mask are not supported and are rejected before the run starts.

`grond endpoints` lists the registry:

```bash
grond endpoints --config grond.config.json
grond endpoints --config grond.config.json --json
```

```text
shelxl          implemented_default
bloch_only      implemented_no_refiner
native_gradient implemented_diagnostic
phenix_refine   macromolecular_adapter
refmac5         macromolecular_adapter
servalcat       macromolecular_adapter
custom          adapter_template
```

`shelxl` is the refinement program of `grond auto`. `bloch_only` describes a calculation without model refinement. `native_gradient` is a slow finite-difference update in observation space, for diagnostics. `phenix_refine`, `refmac5` and `servalcat` are command templates for corrected MTZ files with fixed FreeR flags, run by `grond protein-ls1` ([MACROMOLECULES.md](MACROMOLECULES.md)). `custom` is a command template for any other program.

A custom endpoint receives the current model, a corrected reflection file, an output directory and a cycle count, and must write a refined model that GROND can read (`RES`, `CIF`, `PDB` or mmCIF). Add it to `grond.config.json`:

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

The placeholders are `{model}`, `{reflections}`, `{out_dir}` and `{cycles}`. `grond endpoints` probes and reports the adapter. `grond auto` does not use custom endpoints; it always refines with SHELXL. A wrapper script can convert to MTZ, PDB/mmCIF or its own restraint files as long as it writes a model GROND can read.

For macromolecules, use the MTZ commands instead of HKLF 4: `grond make-free-flags` assigns FreeR flags to unique reflections, `grond export-mtz` writes corrected `IMEAN`/`SIGIMEAN` with `FreeR_flag`, and `grond protein-ls1` runs one configured refinement macrocycle ([MACROMOLECULES.md](MACROMOLECULES.md)).
