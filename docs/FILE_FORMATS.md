# File formats

Small examples of the files GROND reads and writes, to recognise their shape before opening a real run. The numbers are made up.

## Inputs

```text
my_xds_directory/
  INTEGRATE.HKL          # summation integration (PROFILE_FITTING=FALSE)
  XPARM.XDS              # or GXPARM.XDS
  CORRECT.LP             # read for cell e.s.d.s when present
  XDS.INP

my_structure_directory/
  model.res              # or a CIF with embedded SHELX instructions
  model.hkl              # the HKL the model was refined against (--baseline-hkl)

my_dials_directory/
  integrated.refl        # signed observations with intensity.sum.value
  integrated.expt        # beam, scan, detector, crystal, scan-varying A
```

Each numeric line of `INTEGRATE.HKL` is one observation: signed indices, intensity, uncertainty and geometry columns. Its header carries the detector, beam, rotation axis and cell that GROND uses to place every observation in one coordinate system.

```text
!FORMAT=XDS_ASCII
!NX=4148 NY=4362 QX=0.0750 QY=0.0750
!ORGX=2074.2 ORGY=2181.6 DETECTOR_DISTANCE=720.0
!H K L     IOBS      SIGMA   XCAL    YCAL    ZCAL    RLP    PEAK  CORR
   1  0  2   1284.50    42.10  1912.4  2105.7   18.0  0.021   142  91.4
   0  1  3    422.80    27.60  2063.9  1988.2   18.0  0.034    61  88.7
  -2  1  1     96.30    19.20  2251.5  2320.4   19.0  0.058    18  74.9
```

`grond import-dials` (or `grond auto --dials-refl ... --dials-expt ...`) converts a DIALS pair into a directory GROND reads like an XDS directory:

```text
dials_grond_bridge/
  INTEGRATE.HKL                  # GROND's own format: signed DIALS observations
  DIALS_A_AT_SCAN_POINTS.txt     # one 3x3 A matrix per DIALS scan point
  dials_bridge_summary.json      # row counts, flags, sigma mode, Sg check
  DIALS_BRIDGE_README.md
```

The intensity mapping and sigma modes are described in [DIALS_WORKFLOW.md](DIALS_WORKFLOW.md). `grond import-pets` writes a similar directory for PETS2 data, and `grond export-pets` writes `cif_pets` files ([INTEROPERABILITY.md](INTEROPERABILITY.md)). Integrations from other programs use the portable `grond-row-v1` table and metadata file ([PORTABLE_INPUTS_AND_ENDPOINTS.md](PORTABLE_INPUTS_AND_ENDPOINTS.md)).

## Preparation file

For one crystal the command line can point at the files directly. For several crystals, `grond prep` writes a preparation file. Each crystal has its own directory and settings; the model and baseline HKL are shared.

```json
{
  "structure": "/data/project/model.res",
  "baseline_hkl": "/data/project/model.hkl",
  "default_hkl_transform": "identity",
  "geometry_percentile": 99.0,
  "datasets": [
    {"name": "crystal_a", "hkl": "/data/project/crystal_a/INTEGRATE.HKL", "xds_dir": "/data/project/crystal_a",
     "hkl_transform": "identity", "min_d_spacing": 0.7, "min_peak": 90.0},
    {"name": "crystal_b", "hkl": "/data/project/crystal_b/INTEGRATE.HKL", "xds_dir": "/data/project/crystal_b",
     "hkl_transform": "identity", "min_d_spacing": 0.7, "min_peak": 90.0}
  ]
}
```

`examples/multi_dataset_prep.json` is a complete example.

## Observation table

The main numerical output of each stage is `native_dyn_calc.csv`, one line per observation. It holds the observation, its uncertainty, the dynamical and Born predictions, the correction and the `split` (`work` or `free`). P-spline runs add `scale_angle_deg`, the scan angle of the frame-scale model. Selected columns:

```csv
row_index,h,k,l,d_A,target_I,sigma_eff,row_resolution_factor,dyn_raw_unmodulated,dyn_raw,born_raw,dyn_scaled,born_scaled,split
0,1,0,2,0.842,1284.50,42.10,1.118,1091.10,1219.80,1038.40,1257.62,1070.58,work
1,0,1,3,0.914,422.80,27.60,1.061,415.83,441.20,389.50,454.22,401.25,work
2,-2,1,1,1.284,96.30,19.20,0.912,130.15,118.70,101.60,122.29,104.65,free
```

Use this table to look at all-observation R factors, weak-observation bias, and trends with resolution, detector position or frame.

## Corrected HKL

The corrected reflections for SHELXL are `HKLF 4` files: `3I4,2F8.2`, ending with a `0 0 0` line. The unmerged multi-crystal file adds a batch number per crystal.

```text
   1   0   2 1245.30   43.20
   0   1   3  430.10   28.00
  -2   1   1  110.60   20.10
   0   0   0    0.00    0.00
```

Intensities are on an absolute scale, κ|F|² · Iobs/Icalc. When the values would not fit the `F8.2` fields, GROND multiplies intensity and sigma by one common factor and records it as `hkl_safe_factor`; this keeps every I/σ. The uncertainties are those of the per-crystal error model, σ'² = k²(σ² + (uP)²), fitted on working observations, and the endpoint records the fit (`dataset_sigma`, `error_model_passes`, or the `post_bloch_sigma` block of a single-crystal endpoint's `manifest.json`). Each endpoint also records the transform, hand, thickness, beams and settings that produced it.

## Summaries and reports

Each stage writes a `summary.json` with its machine-readable results and, for most stages, a `REPORT.md` with the same information for reading. Keys differ between stages; scripts should read the summary of the stage they need.

## Command log and resume file

`logs/commands.jsonl` lists every command GROND launched, one JSON object per line: a start record and an end record.

```jsonl
{"label": "crystal_a_preflight", "cwd": "/path/to/engine", "start_time": "2025-05-18T14:03:02+0000", "command": ["python", "...", "--out-dir", "..."]}
{"label": "crystal_a_preflight", "end_time": "2025-05-18T14:03:04+0000", "elapsed_seconds": 2.41, "returncode": 0}
```

`grond_progress.json` is the resume file. `completed` holds one record per finished stage with `returncode`, `elapsed_seconds`, `completed_at`, `command`, `log`, `fingerprint`, `output_paths` and `output_fingerprints`; `events`, `current`, `invocation` and `interrupted_stage` record the run's progress. A rerun reuses a stage only when its command, the GROND source and its input and output fingerprints still match.

```json
{
  "completed": {
    "crystal_a_preflight": {"returncode": 0, "elapsed_seconds": 2.41, "completed_at": "2025-05-18T14:03:04+0000",
                            "command": ["..."], "log": "logs/crystal_a_preflight.log", "fingerprint": {}, "output_paths": [],
                            "output_fingerprints": []}
  },
  "events": [{"event": "completed", "label": "crystal_a_preflight", "returncode": 0, "elapsed_seconds": 2.41,
              "time": "2025-05-18T14:03:04+0000"}]
}
```

## Maps

Native dynamical maps are calculated with the installed Python and Gemmi. Maps from the corrected HKL need `shelx2map`. See [OUTPUTS.md](OUTPUTS.md).
