# GROND Paper Workspace

This folder is the working control board for the first GROND paper. It is intentionally practical: each claim should trace to a command, a run folder, a table row, a figure panel, and a reproducible artifact.

The frozen scientific claim for Paper 1 is:

**GROND is a fast, open, row-level Bloch-to-Born regularization loop that brings dynamical scattering information into standard refinement endpoints, with explicit null controls and endpoint calibration.**

The paper should not claim that the default GROND loop is exact native dynamical least squares. It should not depend on internal tutorial parity data, private comparison datasets, or unsupported input formats. The core story is our data, our open workflow, XDS and DIALS row contracts, endpoint calibration, row-wise dynamical null controls, and optional matched native finite-difference polish.

## File Map

| File | Purpose |
|---|---|
| `MAIN_TEXT.md` | Manuscript narrative scaffold. |
| `PAPER_TODO.md` | Ordered paper work plan with stop conditions. |
| `ARTIFACT_REQUIREMENTS.md` | Exact artifact checklist for each workflow and figure. |
| `COMMANDS.md` | Reproducible command ledger and command templates. |
| `DATASET_MANIFEST.csv` | Dataset-level provenance table. Every result should point here. |
| `TABLE1_MASTER.csv` | Master endpoint-statistics table for manuscript and SI. |
| `FIGURE_PLAN.md` | Figure-by-figure panel plan and required source artifacts. |
| `METHODS_MATH.md` | Methods and SI math outline for the regularization loop. |
| `REVIEWER_OBJECTIONS.md` | Pre-written answers to predictable concerns. |
| `REPORT_INDEX.md` | Index of current known run reports and their status. |
| `DOSSIER_CHECKLIST.md` | Required contents of a paper-ready dataset dossier. |
| `REPORT_TEMPLATE.md` | Standard template for writing each dataset or method-test report from artifacts. |

## Current Anchor Artifacts

These are current working anchors, not a substitute for the clean paper rerun. They seed the manifest and figure plan.

| Artifact | Current path |
|---|---|
| 370 endpoint calibration report | `/Users/Mike/Desktop/MartynowyczLab/data_processing/2026.04.10/370/cbf_v7/dials/xds_bloch/outputs/20260611_endpoint_calibration_370_calibrate_endpoint_verify/REPORT.md` |
| 370 LS1 validation, N108/Q20 | `/Users/Mike/Desktop/MartynowyczLab/data_processing/2026.04.10/370/cbf_v7/dials/xds_bloch/outputs/20260611_validate_ls1_370_N108Q20/LS1_VALIDATION_REPORT.md` |
| 370 LS1 validation summary, N108/Q20 | `/Users/Mike/Desktop/MartynowyczLab/data_processing/2026.04.10/370/cbf_v7/dials/xds_bloch/outputs/20260611_validate_ls1_370_N108Q20/ls1_validation_summary.json` |
| 370 LS1 validation, N30/Q20 smoke | `/Users/Mike/Desktop/MartynowyczLab/data_processing/2026.04.10/370/cbf_v7/dials/xds_bloch/outputs/20260611_validate_ls1_370_N30Q20/LS1_VALIDATION_REPORT.md` |

## Paper Boundary

Use "electron diffraction", "electron crystallography", and "continuous-rotation electron diffraction" for public text. Keep the method framed as row-level Bloch regularization with standard endpoint refinement, not as an unqualified native dynamical refiner. Comparisons to existing native dynamical workflows belong in the introduction and discussion as context, not as the central proof.
