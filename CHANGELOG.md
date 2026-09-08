# Changelog

This file records user-visible changes to the installable package. Version `0.2.2` has not yet been tagged as a GitHub release. The unreleased entry describes the final behavior; superseded development policies remain in Git history.

## 0.2.2 (unreleased)

### Runtime recovery

A global fixed-width HKL rescale no longer rejects an otherwise usable endpoint. Changing the original intensity units can change this factor while producing exactly the same HKL bytes. Small factors are reported; publication selection checks the actual written fields for finite intensities and positive sigmas and records their rounding bound in sigma units. Missing or invalid scale factors still fail. Compact-contract and balanced-endpoint decisions use the same unit-invariant rule.

Missing XDS centroids now report limited diagnostic coverage without claiming the configured geometry filter was disabled. Observations remain eligible under the same row-support rules. Optional centroid-derived profile widths reject unavailable centroids with a specific row/HKL error rather than producing a NaN width or silently treating missing position as zero residual. This input error is never retried under another multiprocessing context; the default fixed profile is unchanged.

Rejected non-passive absorption candidates now propagate their physical error immediately in both cached and uncached prediction paths. Retrying the same invalid candidate under another multiprocessing context caused stalled worker startup on a production run. Accepted scattering calculations are unchanged.

Endpoint map export handles the legacy shelx2map reader's short header buffer by shortening an overlong descriptive title in a separate, hash-recorded FCF copy. Cell, symmetry, reflection coefficients and the original file remain unchanged. Other incompatible headers report their exact cause. Converter calls have a 120-second limit, terminate stalled processes, discard partial maps and retain a warning alongside the usable structural result.

### Source organization and public package

The command line, physics, regularization, endpoint, input and reporting implementations now have explicit domain owners. `grond.cli` and historical engine entry points retain compatibility reexports and necessary dispatch boundaries. Argument registration is split by command family; the public parser, defaults and help remain unchanged. Scientific cache fingerprints include the new implementation files, so moving a numerical helper does not make later edits invisible to an old cache.

Tests are grouped by domain, with CLI behavior separated into focused files and explicit shared synthetic helpers. Repository paths no longer depend on each test's folder depth, and tests no longer make bare engine imports work by modifying `sys.path`. The public tree contains package source, portable regression tests, documentation, synthetic examples, schemas and release tools. Real-data workspaces, exploratory experiments, drafts and caches remain separate. README and code-map documentation explain the ordinary XDS/DIALS integration route and the distinction between production behavior, interfaces and experimental approaches.

### Physics and numerical contracts

The numerical audit repairs wavelength-dependent relativistic scaling, nonorthogonal anisotropic displacement factors, symmetry-mate tensors, alternate-hand symmetry translations and SHELX special-position occupancies. SHELX FVAR values are resolved, Q peaks and instruction records are excluded from atoms, and explicit electron scattering coefficients are required. Coefficient-based coreCIF input must identify electron factors; isotropic and anisotropic B values are converted to U. Malformed supplied displacement parameters fail explicitly. Native coordinate updates freeze special-position coordinates until symmetry-constrained derivatives are available.

PETS import corrects the reciprocal-basis off-diagonal sign for general unit cells. It rejects nonphysical cells and UB matrices inconsistent with the declared reciprocal metric while allowing documented serialization precision. Existing PETS bridges with nonorthogonal cells should be regenerated before comparison; XDS and DIALS geometry do not use this conversion.

Fixed-XDS excitation crossings and angular derivatives use exact sinusoidal geometry with conservative numerical fallbacks. Scan-varying and fitted geometry retain the numerical route. Base-profile continuation is confined to a certified monotonic Ewald branch; the Bragg-assignment and active-jitter gates remain strict. Infeasible jitter candidates retain their configuration and error while feasible candidates continue. Entirely invalid grids fail explicitly. Multiprocessing retries cannot retain partial rows from a failed context.

General absorptive propagation requires a passive loss operator. Exactly uniform scalar imaginary absorption factors into Hermitian propagation and analytic damping. The latter shortcut does not apply to beam-dependent or off-diagonal loss. Experimental JAX, real-embedded exponential and Bethe/Schur backends remain explicit alternatives; dense SciPy is the reference. The common-basis incoherent mosaic cone and external geometry-path controls remain default-off experiments.

Fourier-potential allocation has a memory guard. Potential, row-plan and prediction caches validate schema, model and geometry fingerprints, shapes and finite values, and write atomically. Exact fixed-model row-plan reuse is enabled automatically only when expected reuse justifies construction; cross-model support freezing is explicit. Sparse-active potential storage and batched/packed thickness response preserve the underlying calculation. Packed propagation has a memory-bounded scalar fallback. Thickness grids include their declared bounds, and local searches remain inside the selected basin with gradient in `[0,1]`. Reports describe broad or weakly identified optima as effective thickness/dephasing, not calibrated specimen measurements.

### Workflow and structural endpoints

Automatic preflight preserves declared resolution, row, partiality and geometry support. It can report alternatives but does not silently lower a user's acceptance floor. Every accepted final structural model receives a matching fixed-nuisance Bloch row recomputation before row reports and native maps. Feedback preserves profile, mosaicity, scale, sigma, source-row and split contracts. Diagnostic fit-target transformations are restored to ordinary endpoint intensity units, with source sigmas in the same units as the target.

The fast structural path alternates a frozen Bloch/Born correction with one conventional least-squares update. The inner LS1 step remains a dynamical surrogate; it does not include the structural derivative of q. `validate-ls1` exposes updated-q, fixed-q, q=1, permuted-q and optional native-gradient controls. Internal SHELXL cycle count, repeated external SHELXL calls and Bloch macrocycles are separate controls. Native finite differences use explicit absolute steps and preserve their physical bounds.

Shared refinement recomputes each accepted crystal from one model, applies per-dataset nuisance and dataset-policy weights, scales corrected observations through overlaps, merges model-symmetry equivalents with the declared uncertainty rule, and performs one shared structural update. Completeness-only crystals may add endpoint coverage without entering shared raw-row model statistics. Continuation checkpoints preserve this contract; after a manually changed model, compact scales require reselection rather than inheriting their earlier promotion automatically.

Corrected small-molecule endpoints use HKLF 4, no EXTI, calibrated endpoint uncertainties and fixed `WGHT 0 0` by default. Empirical weights, endpoint-B and observation-polish corrections remain explicit diagnostics with their own acceptance conditions. Model/endpoint selection verifies source hashes, current metrics and artifact health. An NPD step is retried from its unchanged input with the documented bounded DAMP ladder; the first complete zero-NPD step is retained, every attempt is archived, and failed later steps cannot reuse stale output. Damping does not add restraints or relax the NPD requirement.

General model handoff, compact-scale promotion and scientific evidence are separate decisions. Finite, provenance-valid, otherwise usable endpoints remain available when their GooF, modulation or R1all comparison is unfavorable; those values are reported as warnings. Missing required baseline evidence, nonfinite metrics, invalid support/provenance, disallowed NPD pathology or a failed specialized contract can still block the relevant path. Compact scale and detector promotion additionally require prospective row preservation, coverage and zero split/NPD pathology. Frame and detector parameter counts remain separate, and a more flexible topology cannot win solely through a lower SHELXL endpoint R factor. Detector-free models are preferred unless a field earns its gates; 8x8 requires an explicit independent calibration.

### Hand, maps and reports

Proper indexing-transform selection is separated from coordinate inversion and uses a hand-invariant matched-Born work-row target. The final elastic negative-thickness hand calculation shares rows, basis, geometry, thickness and nuisance settings. Absorption is disabled only for this Hermitian comparison. Method version 4 assigns half credit to exact/numerical residual ties and retains the historical strict-count statistic separately. The all-row adjusted count contrast controls the automatic inversion threshold; weak, missing or conflicting free evidence warns without vetoing it. Invalid thresholds or physical row/geometry mismatches produce explicit errors. A failed hand calculation retains the preceding usable structural endpoint; centrosymmetric not-applicable results are handled explicitly.

The manual final report now uses the same tie-neutral effective count as the v4 hand decision, with strict counts retained only as a fallback for older diagnostics. Reports distinguish preferred hand, conditional support and the exported coordinate representation. Nominal count Z values are not calibrated sigma confidence under correlated observations. Named configuration still requires acquisition/import sign calibration; the legacy branch-symmetric calibration replay does not yet accept the current final-elastic AUTO artifact.

Table 1 separates SHELXL endpoint factors from GROND-computed JANA/PETS-style Born/Bloch row factors. Row reporting uses immutable source sigmas in the target intensity units, and unavailable quantities remain unavailable. Endpoint and row files are bound to their selected model. Native maps use ratio-derived amplitudes and model phases; optional `shelx2map` endpoint maps explicitly request weighting and retain an unweighted diagnostic. Finish regenerates reports/maps without structural refinement and exposes partial/missing map families with their precise cause.

The prospective hydrogen-omit audit checks a zero-H model, compatible cells/symmetry, alignment from common non-H atoms, input hashes and actual map peaks near independently supplied H coordinates. The built-in strip-H branch remains a retrospective single-dataset sensitivity control and is rejected for multiple datasets.

### Input interfaces and operation

DIALS import preserves signed summation rows, raw detector-model variances, scan-varying crystal geometry and row-quality decisions; empirical/gain sigma changes are explicit diagnostics. Cropped scans retain their array-frame origin. One bridge represents one experiment. The helper runs in a configured DIALS Python, validates launchers and inputs, and reports concise errors. Portable row input retains explicit units and geometry rather than reconstructing them from a merged HKL.

Optional PETS export records source hashes, signed-row mapping, intensity/variance transformations and virtual-frame ownership. Native PETS2 import preserves PETS-owned observation and zone geometry. An ordinary import can enter normal AUTO only after exact ordered agreement with its companion PETS HKL and static SHELX export. Correlated virtual-frame rows remain a native row-space target; they are not presented as a conventional static endpoint. Unsupported settings, precession and incompatible contracts fail explicitly. The work-only diffBloch adapter preserves original model instructions and requires a matched GROND replay before any returned structural candidate can be promoted. Macromolecular FreeR/MTZ and endpoint templates remain explicit experimental interfaces.

The staged manual interface records setup, scout, thickness, pre-feedback hand, shared feedback, scale audit and finish handoffs. It is an expert decomposition, not every AUTO policy in manual form. Resuming requires matching fingerprints; going backward invalidates downstream state while retaining artifacts. Doctor tests the requested interpreter and reports readiness for the chosen workflow. Explicit executable paths are frozen and forwarded, subprocess failures retain recovery information, and dry runs produce planning summaries without requiring unmaterialized outputs.

Reused manual thickness summaries reject a recorded source-HKL hash mismatch, including edits at the same path. Older summaries without a saved source hash remain explicit legacy reuse and report historical identity as unverified.

## 0.2.1

The environment doctor gained explicit stale-install and required-feature diagnostics so an older app bundle or partial site-packages tree can be distinguished from the current checkout.

## 0.2.0

This was the first GitHub-facing package layout. It organized the Python code into input/output, Bloch, refinement, map, forensic, workflow, and historical engine namespaces; removed dataset-specific helper scripts from the installed package; added package standards files; and introduced documentation for a general crystallography lab audience.

At this stage the endpoint path promoted SHELXL's recommended `WGHT` between endpoint-polish cycles. That behavior is retained here as release history; version 0.2.2 changed the production endpoint to fixed `WGHT 0 0` after GROND's endpoint sigma calibration.
