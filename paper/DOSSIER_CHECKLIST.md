# Paper Dossier Checklist

This checklist defines what a run folder must contain before it can support a manuscript result. The goal is to make each dataset boring to audit: a reader should be able to start from the command, open the manifest row, follow the row table and validation reports, and recover every number in the paper without asking which exploratory branch was used.

## Dossier Folder Layout

Each paper-ready dataset or dataset combination should have one dossier folder. The folder may contain copies of small files and links or text paths to large run products.

| Required item | Purpose | Paper-ready condition |
|---|---|---|
| `COMMANDS.md` | Exact commands used for the dataset. | Commands can be pasted into the same environment and produce the named output folders. |
| `MANIFEST.json` or manifest row | Dataset identity, input paths, package commit, runtime environment, and status. | Every table and figure panel points back to this record. |
| `INPUTS.md` | Integration route, starting model, row contract, and preprocessing choices. | Summation/profile policy, gain or offset notes, row count, dmin, and model source are explicit. |
| `AUTO_SUMMARY/` | Main GROND automatic or row-regularization output. | Contains `summary.json`, row prediction table, endpoint exports, and timing. |
| `VALIDATE_LS1/` | Null-control and native-polish audit. | Updated-q, frozen-q, q=1, polarity, shuffled-q controls, and native-polish status are reported. |
| `ENDPOINT_CALIBRATION/` | Publication endpoint chooser. | Selected endpoint, low-R diagnostic endpoint, scale count, GooF, held-out row behavior, and rejection reasons are reported. |
| `MAPS/` | Map-producing inputs and outputs. | Conventional comparator and GROND endpoint maps are labeled and reproducible. |
| `TABLES/` | Table-ready CSV/JSON summaries. | Table 1, validation table, endpoint-calibration table, and map statistics are populated from scripts or reports. |
| `REPORT.md` | Human-readable interpretation. | The report states what worked, what failed, what was promoted, and why. |

## Flagship 370 Dossier

The 370 dossier is the canonical example. It should be built first and should not depend on memory of exploratory runs. The clean folder should contain the promoted public endpoint, the low-R diagnostic endpoint, the N108/Q20 validation report, the endpoint-calibration report, maps, timing, and the command log.

Stop when the clean dossier reproduces the current anchor logic: the row-wise updated-q branch beats q=1, alternate polarity, and shuffled-q controls; native polish does not find a large downhill move; and the promoted endpoint is the calibrated fixed-weight branch rather than the lowest-R scale-heavy branch.

## Replication Dossiers

The next dossiers should be produced without hand tuning. If a dataset fails, the failure is still a result and should be written up as a no-call or limitation. The minimum replication set is tyrosine 366, tyrosine 366+370, tyrosine 370 through DIALS, histidine counting mode, and histidine linear mode.

Each replication dossier should answer the same questions as the 370 dossier: did the row-wise correction beat nulls, did endpoint calibration choose a conventional-looking public endpoint, did maps remain stable, and did the frozen command path work without per-dataset intervention?

## Multi-Dataset Dossier

A multi-dataset dossier must keep each crystal visible. The combined result should include per-dataset thickness, scale contract, sigma behavior, row count, row metrics, contribution to completeness, and include/downweight/exclude decisions. The combined endpoint should not hide a weak dataset inside a merged HKL without explaining what it contributed and what it cost.

The report should explicitly separate per-dataset row calibration from the second-stage merge calibration. This is the easiest way to prevent a marginal crystal from dragging the endpoint while still allowing it to improve completeness.

## DIALS Dossier

The DIALS dossier is an auditability claim, not a promise that DIALS beats XDS for every dataset. It must include the DIALS version, `integrated.refl`, `integrated.expt`, import gain, summation/profile behavior, flags dropped or retained, scan-varying orientation handling, sigma policy, and the GROND row-contract report.

The report should say plainly whether the DIALS endpoint is promoted, diagnostic, or no-call. If XDS is better for a dataset, that belongs in the result rather than being hidden.

## Map And Chemistry Dossier

Any map or H/omit claim needs its own small report. The report should state which model was omitted, whether q was recomputed from the omitted model, which endpoint was used, and whether the evidence is visible in row-level metrics, maps, or both. GROND-corrected maps should be labeled as model-conditioned endpoint maps, not raw experimental maps.

## External Survey Dossier

The external survey should be one command plus one result schema. It should not become a collaborator-specific tuning exercise. The returned table should include dataset id, row count, supported input route, success/failure status, promoted endpoint metrics, validation-control status, runtime, and no-call reason if any.
