# Report Index

This index points to current known reports. Entries marked `current anchor` should be replaced or confirmed by a clean paper rerun before manuscript submission.

## Tyrosine 370

| Status | Report | Path | Key current result |
|---|---|---|---|
| current anchor | Endpoint calibration | `/Users/Mike/Desktop/MartynowyczLab/data_processing/2026.04.10/370/cbf_v7/dials/xds_bloch/outputs/20260611_endpoint_calibration_370_calibrate_endpoint_verify/REPORT.md` | Promoted `adaptive_120_det_8`, 184 scales, R1all 10.97%, wR2all 10.07%, GooF 0.906. |
| current anchor | LS1 validation N108/Q20 | `/Users/Mike/Desktop/MartynowyczLab/data_processing/2026.04.10/370/cbf_v7/dials/xds_bloch/outputs/20260611_validate_ls1_370_N108Q20/LS1_VALIDATION_REPORT.md` | Selected cycle 1; native row wR2all 10.76%; shuffled-q controls fail. |
| current anchor | LS1 validation N108/Q20 JSON | `/Users/Mike/Desktop/MartynowyczLab/data_processing/2026.04.10/370/cbf_v7/dials/xds_bloch/outputs/20260611_validate_ls1_370_N108Q20/ls1_validation_summary.json` | Machine-readable validation metrics and config. |
| current anchor | LS1 validation N30/Q20 smoke | `/Users/Mike/Desktop/MartynowyczLab/data_processing/2026.04.10/370/cbf_v7/dials/xds_bloch/outputs/20260611_validate_ls1_370_N30Q20/LS1_VALIDATION_REPORT.md` | Software check before final-support validation. |

## Needed Reports

| Dataset or workflow | Required report |
|---|---|
| Clean tyrosine 370 XDS rerun | Full dossier with auto, validation, endpoint calibration, maps, and manifest entry. |
| Tyrosine 366 XDS | Same frozen path as 370. |
| Tyrosine 366+370 XDS | Multi-dataset endpoint and dataset-policy report. |
| Tyrosine 370 DIALS | DIALS bridge report, sigma policy, validation, and endpoint calibration. |
| Histidine counting mode | Full endpoint and maps. |
| Histidine linear mode | Full endpoint and maps. |
| External survey | Standard result schema and failure categories. |

Use `REPORT_TEMPLATE.md` for each report and `DOSSIER_CHECKLIST.md` to decide whether a folder is paper-ready.
