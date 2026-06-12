# Figure Plan

Figures should be built from manifest rows and source reports. No panel should be manually assembled from memory.

## Figure 1: Workflow And Row Contract

Purpose: explain what GROND does before any statistics are shown.

Required panels:

| Panel | Content | Required source |
|---|---|---|
| 1A | Supported inputs: XDS and DIALS row data plus starting structure. | Input-row contract from `ARTIFACT_REQUIREMENTS.md`. |
| 1B | Row object: signed hkl, intensity, sigma, geometry, excitation error, split label. | Example row table. |
| 1C | Bloch/Born calculation: Hamiltonian, support beams, quadrature, q. | `METHODS_MATH.md` plus row prediction table. |
| 1D | Endpoint loop: corrected HKL, one least-squares step, recomputed q, endpoint calibration. | `COMMANDS.md` and clean 370 run. |
| 1E | Audit outputs: null controls, native polish, maps, manifest, Table 1. | `validate-ls1` and `calibrate-endpoint` reports. |

Stop condition: a non-expert crystallographer can follow the data path from integration rows to final endpoint.

## Figure 2: 370 Flagship Proof

Purpose: prove the correction is row-specific and the endpoint is calibrated.

Required panels:

| Panel | Content | Current anchor |
|---|---|---|
| 2A | Conventional/Born comparator versus GROND row metrics. | N108/Q20 validation report. |
| 2B | Updated-q cycles and selected best cycle. | N108/Q20 validation report. |
| 2C | Controls: q=1, alternate polarity, shuffled global, shuffled shell, shuffled shell-plus-scan. | N108/Q20 validation report. |
| 2D | Endpoint calibration curve: scale count versus R1all, wR2all, GooF, held-out row wR2. | Endpoint calibration report. |
| 2E | Promoted endpoint versus low-R diagnostic endpoint. | Endpoint calibration report. |

Current anchor values: selected cycle 1, native row wR2all 10.76%, q=1 endpoint wR2all 21.33%, alternate-polarity native wR2all 16.59%, shuffled-q native wR2all 39.93%, 37.43%, and 26.57%, promoted endpoint R1all 10.97%, wR2all 10.07%, GooF 0.906.

Stop condition: the figure makes clear that row identity matters and that the promoted endpoint is not the lowest-R diagnostic branch.

## Figure 3: Maps And Chemistry

Purpose: show what changes structurally, not only statistically.

Required panels:

| Panel | Content | Required source |
|---|---|---|
| 3A | Conventional comparator map. | Conventional endpoint and map script. |
| 3B | GROND promoted endpoint map. | Promoted endpoint map script. |
| 3C | H/no-H or omit behavior. | Omit branch report. |
| 3D | Difference-map noise and peak/hole summary. | Map statistics table. |
| 3E | Coordinate and ADP stability versus comparator. | Model comparison script. |

Stop condition: map claims are modest and tied to exact model-conditioned outputs.

## Figure 4: Input Route And Dataset Scaling

Purpose: show that the workflow is not single-route or single-crystal only.

Required panels:

| Panel | Content | Required source |
|---|---|---|
| 4A | 370 XDS versus 370 DIALS route. | Manifest rows and endpoint reports. |
| 4B | Tyrosine 370 versus 366 versus 366+370. | Multi-dataset reports. |
| 4C | Per-dataset inclusion/downweight decisions. | Dataset policy reports. |
| 4D | Completeness and resolution behavior. | Merge reports. |
| 4E | Runtime per dataset and combined run. | Timing logs. |

Stop condition: the reader can see where multiple datasets help and where they do not.

## Figure 5: Robustness, Speed, And External Survey

Purpose: put the method in practical context.

Required panels:

| Panel | Content | Required source |
|---|---|---|
| 5A | Runtime versus rows, N, Q, backend, and cache status. | Timing benchmark logs. |
| 5B | Endpoint chooser pass/fail categories across datasets. | Manifest and Table 1 master. |
| 5C | Histidine counting versus linear mode. | Histidine reports. |
| 5D | External survey success/failure distribution. | Collaborator result schema. |
| 5E | Failure modes and no-call categories. | Reviewer-objection and survey notes. |

Stop condition: the figure supports practical adoption without overclaiming universal success.
