# What `GROND auto` does

`GROND auto` is a resumable stage runner. It does not hide a second refinement program. Each stage is an ordinary GROND calculation whose command, timing, log, inputs, and output fingerprints are recorded in the run directory. `grond_progress.json` is the stage ledger and `logs/commands.jsonl` is the command ledger. Repeating the same command with the same output directory resumes only fingerprint-matching completed stages.

The normal sequence is input validation and a fresh conventional control; indexing-transform selection; a matched original/inverse-hand diagnostic when the space group is acentric; coarse row-level Bloch/Born screening; continuous thickness and thickness-gradient polishing; gated tests of row-resolution, absorption, rocking-width jitter, sigma, and dose terms; a final beam-support calculation; Bloch/LS1 model feedback; compact scale selection; resolution and endpoint calibration; and final SHELXL-style, JANA/PETS-style, map, hand, and provenance reports. Multi-crystal jobs do those physics checks separately for each crystal before making one shared corrected-HKL endpoint.

Transform and hand are separate decisions. Transform scouting uses proper operators and no known chemistry. Once a proper transform is fixed, the two coordinate hands are compared on the same rows and nuisance contract. If the structure and baseline HKL were exported from the exact same post-cosym DIALS experiment and no later program changed the setting or hand, the scout can be bypassed explicitly with:

```bash
grond auto ... --hkl-transform identity --trust-hkl-transform
```

That option is an assertion by the user, not an automatic inference from `dials.cosym`. If the RES or baseline HKL passed through another program, keep the scout or supply the verified transform explicitly.

## Disabling optional stages

The important switches are listed below. A disabled stage remains visible in the command and final configuration; GROND does not silently substitute a different calculation.

| stage | control |
|---|---|
| indexing scout | `--hkl-transform OP --trust-hkl-transform` |
| alternate-hand diagnostic | `--no-hand-control` |
| structure update | `--structure-update off` |
| continuous thickness polish | `--continuous-thickness-polish off` |
| late rocking-width scout | `--sg-window-scout off` |
| endpoint feedback | `--endpoint-feedback off` |
| endpoint calibration | `--endpoint-calibration off` |
| scale compression | `--scale-polish off` |
| multi-crystal compact replay | `--multi-scale-contract-selection off` |
| resolution-limit gate | `--no-resolution-limit-gate` |
| dataset weighting/exclusion policy | `--dataset-policy off` |
| forensic gate | `--no-forensic-gate` |
| legacy quick endpoint maps | `--no-maps` |
| hydrogen-omit experiment | `--h-omit off` |

Disabling a control can change what may be claimed. For example, `--no-hand-control` is appropriate for a centrosymmetric structure or a deliberate speed test, but it removes hand evidence. `--trust-hkl-transform` is appropriate only when the external indexing basis is genuinely known. `--structure-update off` produces a frozen-model Bloch result, not a refined GROND endpoint.

## Running parts directly

The complete expert sequence, handoff artifacts, and the elastic inverse-hand identity are documented in [MANUAL_WORKFLOW.md](MANUAL_WORKFLOW.md). Use `grond import-dials` to make and inspect a DIALS row bridge, and `grond prep` to bind a structure, baseline HKL, and one or more row directories. `grond scout` performs only proper-transform selection. `grond hand-check` performs a matched fixed-nuisance hand comparison. `grond run --mode bloch` is the low-level frozen-model calculation. `grond run --mode regularize` performs the row correction and endpoint calculation without asking the full automatic policy to choose every stage. The more focused commands are intended for completed or partially completed runs:

```bash
grond scout ...
grond thickness ...
grond hand-check ...
grond scale-polish --row-table native_dyn_calc.csv --summary summary.json --out-dir scale_polish
grond scale-replay ...
grond calibrate-endpoint ...
grond validate-ls1 --source-summary summary.json --start-res model.res --out-dir ls1_check
grond refine --checkpoint grond_continue_checkpoint.json --start-res model.res --out-dir continuation
grond finish --run-dir completed_run
```

`grond validate-ls1` is the specific check for the lagged fixed-point approximation. It compares updated-q macrocycles with fixed-q, q=1/Born, shuffled-q, alternate-hand, and optional native-gradient controls. `--endpoint-ls-cycles N` changes the number of internal SHELXL LS/CGLS cycles performed while one Bloch correction is frozen. `--shelxl-iterations N` repeats external one-cycle SHELXL calls against that same corrected HKL. Neither option changes the number of Bloch recalculations; the updated-q macrocycle count is controlled by `--cycles` in `validate-ls1` or `--shelxl-feedback-cycles` in `auto`.

Continuation replays normally inherit the checkpoint's native row weights. For an older checkpoint whose feedback history was written before native weights were pinned, `push-endpoint --native-wght-a A --native-wght-b B` declares one replacement contract for every dataset in the shared replay. This is a recovery/diagnostic option; it does not tune the weights per cycle.

The production defaults deliberately use one internal LS cycle per Bloch feedback step. More fixed-q cycles are a supported diagnostic, not forbidden. They should be promoted only when the next Bloch recomputation improves the native row objective and the model remains free of NPD atoms and split warnings. A lower corrected-HKL endpoint residual by itself is not enough, because several LS cycles can optimize the frozen surrogate after it has drifted away from the current dynamical model.
