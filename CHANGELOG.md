# Changelog

This file records user-visible changes to the installable package. Version `0.2.2` has not yet been tagged as a GitHub release.

## 0.2.2 (unreleased)

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
