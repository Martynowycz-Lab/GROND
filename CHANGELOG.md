# Changelog

This file records user-visible changes to the installable package. Version `0.2.2` has not yet been tagged as a GitHub release.

## 0.2.2 (unreleased)

The DIALS bridge now preserves cropped-scan frame origins by exporting DIALS' zero-based array-range start as `STARTING_FRAME` and applying the same origin when validating omega and scan-point A matrices. Imports whose predicted rows remain grossly off the Ewald sphere fail before writing a new reflection bridge, with an auditable diagnostic summary instead of a misleading `status: ok`.

Repeated Bloch/LS1 feedback cycles now share their content-addressed row-prediction cache, so a final recomputation of an unchanged model reuses the exact preceding prediction instead of repeating the matrix propagation. Prediction-cache metadata now compares NaN sentinels canonically and fingerprints the prediction-engine source plus numerical runtime, preventing both false misses and unsafe reuse after a code or library change.

`GROND doctor --python PATH` now tests the requested interpreter instead of silently substituting another Python. The new `--strict` mode returns a failing exit status when required packages, installed GROND features, or requested SciPy Bloch operations are not ready, and reports machine-readable `ready` and `failures` fields. Failed workflow stages write a concise recovery artifact and log tail without masking the original subprocess error, while malformed or structurally invalid resume records are warned about and quarantined before a fresh record is started.

`grond auto` and `grond finish` now compare a corrected publication endpoint with the fresh conventional baseline rerun from the same final native stage. A corrected endpoint with worse `R1all` is retained as a diagnostic but is no longer promoted, and the reports point directly to the conventional RES/HKL/LST that should be kept. Thickness reporting now states both the mean in nanometres and the support implied by the selected thickness-distribution model.

Automatic XDS runs now measure the scan-datum difference between the global header orientation and row-level `ZCAL` positions before testing rocking-profile offsets. This prevents an unvalidated zero-frame assumption from forcing thickness to compensate for a profile-center error while keeping nuisance selection on work rows. Sparse, inconsistent, or extreme geometry estimates fall back without recentering, and the higher-quality thickness polish retains the original geometry-derived probes. Fixed-nuisance row runs also correctly translate the public `sigma_a`, `sigma_b_corr`, and `sigma_c_peak` names into the frozen Bloch configuration instead of failing while writing their summary. Calibrated scale-contract endpoints now enter the final publication chooser with their actual HKL safety evidence, including combined multi-dataset replays.

GROND now supports the combined multi-dataset Bloch/LS1 feedback path, model-symmetry endpoint merging, resumable endpoint continuation, native final maps, and explicit endpoint-selection reports. Publication endpoints reject non-positive-definite model updates and use the declared post-Bloch uncertainty model with fixed `WGHT 0 0`; inherited or promoted SHELXL weights remain diagnostic choices. Automatic DAMP retries were added experimentally and then removed after they proved unsuitable as a silent repair policy.

The command-line workflow now exposes diagnostic SHELXL cycle control, clearer stale-install checks, portable XDS and DIALS input preparation, expanded row and endpoint reports, and guarded experimental acceleration backends. The public package continues to use dense SciPy as its numerical reference.

The package audit for this release corrected wavelength-dependent relativistic scaling, nonorthogonal anisotropic displacement factors, symmetry-mate tensors, alternate-hand symmetry translations, special-position expansion, local excitation-error root finding, and DIALS `S R F A` goniometer geometry. The DIALS bridge now applies LP and quantum-efficiency corrections together and keeps raw detector-model variances by default. SHELX parsing resolves FVAR parameters, excludes Q peaks and instruction records, and rejects models without explicit electron scattering coefficients.

Validation scales now fit work reflections by default, while symmetry and Friedel mates remain on the same side of the held-out split. Sparse or free-only scale groups borrow a finite work-derived scale instead of producing a zero denominator. Multi-dataset overlap scales require overlap between distinct datasets and use leave-self-out equations. Invalid sigma rows no longer acquire artificial weight.

Dense Fourier-potential allocation is checked against a declared memory ceiling before NumPy allocates it. Fourier-potential, row-plan, and prediction caches now validate their schema, input fingerprints, shapes, and finite values before reuse, and cache writes are atomic. Resume records fingerprint the GROND source and declared inputs. Failed external refinements cannot reuse stale output files.

The numerical engine is included in Ruff checks and every shipped legacy module is tested as an ordinary package import and as a direct script where supported. The README and user guides were rewritten as concise package documentation, and release metadata, citation data, policy files, and artifact checks now agree on the package boundary.

## 0.2.1

The environment doctor gained explicit stale-install and required-feature diagnostics so an older app bundle or partial site-packages tree can be distinguished from the current checkout.

## 0.2.0

This was the first GitHub-facing package layout. It organized the Python code into input/output, Bloch, refinement, map, forensic, workflow, and historical engine namespaces; removed dataset-specific helper scripts from the installed package; added package standards files; and introduced documentation for a general crystallography lab audience.

At this stage the endpoint path promoted SHELXL's recommended `WGHT` between endpoint-polish cycles. That behavior is retained here as release history; version 0.2.2 changed the production endpoint to fixed `WGHT 0 0` after GROND's endpoint sigma calibration.
