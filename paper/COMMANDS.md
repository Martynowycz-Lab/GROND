# Paper Command Ledger

Commands in this file should be copied from completed runs or filled in before rerunning. Do not use this as a wish list: every command should eventually point to a run folder and a manifest row.

## 370 Current Anchor: LS1 Validation N108/Q20

This command is reconstructed from the existing `ls1_validation_summary.json`. Recheck flags against the current CLI before rerunning in the clean paper folder.

```bash
grond validate-ls1 \
  --source-summary /Users/Mike/Desktop/MartynowyczLab/data_processing/2026.04.10/370/cbf_v7/dials/xds_bloch/outputs/20260610_366370_multi_ls1_feedback/long_free_h_stability_50cycle/370_single/cycle_02/370_native/summary.json \
  --start-res /Users/Mike/Desktop/MartynowyczLab/data_processing/2026.04.10/370/cbf_v7/dials/xds_bloch/outputs/20260610_366370_multi_ls1_feedback/long_free_h_stability_50cycle/370_single/cycle_02/370_native/endpoint_lam_1p0_B_0p0_noexti/dyn_nuisance_corrected.res \
  --out-dir /Users/Mike/Desktop/MartynowyczLab/data_processing/2026.04.10/370/cbf_v7/dials/xds_bloch/outputs/20260611_validate_ls1_370_N108Q20 \
  --cycles 5 \
  --n-basis 108 \
  --n-points 20 \
  --compute-backend scipy_bethe \
  --workers 10 \
  --chunksize 8 \
  --native-polish \
  --native-polish-rows 200 \
  --native-polish-blocks adp_ueq,xyz \
  --native-polish-max-nfev 12
```

Current anchor result: selected cycle 1; native row wR2all 10.76%; q=1/Born endpoint wR2all 21.33%; alternate polarity native wR2all 16.59%; shuffled-q native wR2all 39.93%, 37.43%, and 26.57% for global, shell, and shell-plus-scan shuffles. Native polish was accepted but did not become the main endpoint claim.

## 370 Current Anchor: Endpoint Calibration

The existing endpoint calibration report is:

```text
/Users/Mike/Desktop/MartynowyczLab/data_processing/2026.04.10/370/cbf_v7/dials/xds_bloch/outputs/20260611_endpoint_calibration_370_calibrate_endpoint_verify/REPORT.md
```

Current anchor result: selected publication endpoint `adaptive_120_det_8`, `WGHT 0 0` branch, 184 scale parameters, total row-model parameters 194, R1gt 5.81%, R1all 10.97%, wR2all 10.07%, GooF 0.906, held-out row wR2all 15.08%. The low-R diagnostic branch used 708 scale parameters and was rejected for publication promotion because GooF and scale complexity failed the declared endpoint logic.

Paste the exact rerun command here after the clean paper-folder replay:

```bash
# TODO: grond calibrate-endpoint ...
```

## Clean 370 Paper Rerun

Use this section for the clean flagship dossier. The run should create a single folder with auto output, validation output, endpoint calibration output, maps, and command logs.

```bash
# TODO: grond auto ...
```

```bash
# TODO: grond validate-ls1 ...
```

```bash
# TODO: grond calibrate-endpoint ...
```

## Tyrosine Replication

Run the same command path without hand tuning.

```bash
# TODO: tyrosine 366 XDS command
```

```bash
# TODO: tyrosine 366+370 XDS command
```

## DIALS Route

DIALS should be shown as supported and auditable. It does not need to beat XDS in Paper 1.

```bash
# TODO: DIALS bridge or auto command for 370
```

## Histidine

Run counting-mode and linear-mode datasets through the same frozen path.

```bash
# TODO: histidine counting-mode command
```

```bash
# TODO: histidine linear-mode command
```

## External Survey Package

The external survey should use one command and one result schema. It should not become a tuning exercise.

```bash
# TODO: grond auto-full --validate-ls1 ...
```
