# Paper To-Do List

This list is ordered to stop the project from sprawling. Each task has an output and a stop condition. If a task cannot meet its stop condition, the failure becomes a documented result instead of a new open-ended optimization branch.

## Priority Tracks

| Priority | Task | Required output | Stop condition |
|---:|---|---|---|
| 1 | Freeze the exact paper command path. | `COMMANDS.md` contains clean commands for `grond auto`, `grond validate-ls1`, and `grond calibrate-endpoint` on 370. | One clean 370 rerun reproduces the promoted endpoint and LS1 validation within declared tolerance. |
| 2 | Build the 370 flagship dossier. | One run folder containing final endpoint, Table 1 row, row metrics, null controls, validation report, endpoint calibration report, maps, command log, and manifest entry. | This folder can be handed to a reader as the canonical example. |
| 3 | Fill the benchmark manifest. | `DATASET_MANIFEST.csv` has one row per dataset or dataset combination, with paths to all source artifacts. | Every figure panel and table row traces to one manifest row. |
| 4 | Finish the 370 proof figure. | Figure draft with conventional comparator, GROND endpoint, endpoint calibration, q controls, and native polish. | The figure explains why the correction is row-specific and why the endpoint is not a low-GooF trick. |
| 5 | Run the minimal tyrosine replication set. | 370, 366, and 366+370 through the same frozen path. | The method either transfers without tuning or the failure mode is documented. |
| 6 | Make the DIALS route auditable. | At least one complete DIALS route with row-contract report, sigma policy, endpoint report, and controls. | The paper can claim DIALS support as explicit and inspectable, even if XDS gives better numbers for a dataset. |
| 7 | Add histidine. | Counting-mode and linear-mode histidine runs with the same endpoint and validation reports. | The method is no longer a tyrosine-only story. |
| 8 | Generate final figures. | Five figure drafts with source paths and panel provenance. | A reader can understand the method before reading Methods. |
| 9 | Write the SI math and audit section. | `METHODS_MATH.md` becomes a manuscript-ready derivation plus limitations text. | The model-conditioned observation objection is answered before it is raised. |
| 10 | Prepare the external survey package. | One frozen command recipe and result-schema CSV for collaborators. | External users can run GROND without manual tuning or side-channel instructions. |
| 11 | Freeze release artifacts. | GitHub tag, archive, example data or reduced example, command log, and environment notes. | The preprint can point to a reproducible software and data snapshot. |

## Required Paper Claims

| Claim | Evidence required | Artifact source |
|---|---|---|
| Row-wise q carries real dynamical information. | Updated-q beats q=1, alternate polarity, and shuffled-q controls in native row metrics. | `grond validate-ls1` report and summary JSON. |
| The LS1 loop is a useful surrogate, not exact native refinement. | Fixed-q derivation, missing-gradient diagnostic, and native finite-difference polish. | `METHODS_MATH.md` plus validation report. |
| The publication endpoint is calibrated and conservative. | Compact scale selection, fixed `WGHT 0 0`, GooF gate, held-out row metrics, and low-R diagnostic rejection. | `grond calibrate-endpoint` report. |
| The workflow is reproducible. | Commands, input paths, output paths, environment, and run metadata. | `COMMANDS.md`, `DATASET_MANIFEST.csv`, run folders. |
| The method is not only a single favorable run. | 370, 366, 366+370, histidine, and DIALS route results. | Manifest plus Table 1 master. |

## Immediate Next Actions

1. Make a clean 370 paper rerun folder from the current package command path.
2. Write the 370 flagship report using `REPORT_TEMPLATE.md`, with current anchors clearly marked until the clean rerun exists.
3. Copy the promoted endpoint and validation artifacts into a 370 flagship dossier or link them in the manifest.
4. Fill `TABLE1_MASTER.csv` from the clean rerun rather than hand-transcribing from older exploratory folders.
5. Build Figure 2 first. If Figure 2 is not convincing, fix the evidence chain before adding more datasets.
6. Run 366 and 366+370 with no tuning changes.
