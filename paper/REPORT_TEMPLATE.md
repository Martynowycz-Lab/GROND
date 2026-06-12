# Experiment Report Template

Copy this template for each dataset, dataset combination, or focused method test. Keep the prose factual. The report should be written from artifacts, not from memory.

## One-Sentence Verdict

State the outcome in one sentence. Use one of these words somewhere in the sentence: promoted, diagnostic, no-call, failed, or needs rerun.

## Dataset And Inputs

Name the sample, dataset id, crystal count, integration route, starting model, space group, resolution used by GROND, and the exact input paths. State whether the rows came from XDS, DIALS, or the portable GROND row contract. State whether the intensities were summation-style rows and whether the starting structure came from a separate conventional solution route.

## Command And Environment

Paste the exact command or point to the command ledger entry. Include the GROND git commit, Python environment, external endpoint programs, backend, worker count, N/Q, and wall time.

## Row Contract And Preflight

Report row count, unique count, dmin, intensity and sigma behavior, rejected-row counts, transform or polarity decision, and any warning from the importer. If DIALS was used, include the sigma policy and scan-varying geometry handling. If XDS was used, include the summation/profile policy and the geometry files used.

## Bloch Regularization Result

Report the selected thickness, thickness model, jitter or absorption choices, lambda policy, scale contract, scale-parameter count, row-level R1all, row-level wR2all, row-level CC, Born comparator metrics, shell behavior, and work/free behavior. State whether the result is physically interpretable or only diagnostic.

## LS1 Validation

Summarize updated-q, frozen-q, q=1, alternate-polarity, shuffled global q, shuffled shell q, shuffled shell-plus-scan q, and native-polish results. The central question is whether row identity matters. If shuffled q is competitive, the branch is not ready for a main-text claim.

## Endpoint Calibration

Report the promoted public endpoint and the lowest-R diagnostic endpoint separately. Include R1gt, R1all, wR2all, GooF, scale count, total parameter count, sigma policy, held-out row wR2all, and rejection reasons for non-promoted branches. State whether the endpoint passes the publication gate.

## Maps And Chemistry

Report map paths, map statistics, peak/hole values, H/no-H or omit behavior, coordinate stability, ADP stability, NPD warnings, and restraint policy. If the maps are not strong enough for a main-text claim, say that.

## Runtime

Report wall time by stage when available: import/preflight, transform or polarity scout, thickness/nuisance scan, Bloch prediction, LS1 cycles, validation controls, endpoint calibration, and map generation. Note backend and cache status.

## Files To Cite

List exact paths to the source reports and machine-readable files. At minimum include `summary.json`, row prediction table, `LS1_VALIDATION_REPORT.md`, `ls1_validation_summary.json`, endpoint calibration `REPORT.md`, endpoint calibration `summary.json`, final `RES/CIF`, final `HKL`, map files, and the command log.

## Paper Use

State where this result belongs: Figure 2, Figure 3, Figure 4, Figure 5, Table 1, SI only, or no-call appendix. State what number or claim it supports.

## Remaining Action

State the next action. If no action is needed, write "freeze this result."
