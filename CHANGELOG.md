# Changelog

This file records user-visible changes to the installable package. Version `0.2.2` has not yet been tagged as a GitHub release.

## 0.2.2 (unreleased)

Detector-position scaling is no longer fixed unconditionally at 8x8. Physical
nuisance selection is detector-free by default; the subsequent frozen-row
scale audit evaluates detector-free, 2x2, 4x4, and diagnostic 8x8 fields. Added fields must improve
multiple held-out splits, preserve the Born-to-Bloch advantage, have adequate
rows in every tile, reproduce across data halves, agree when inferred from
Born and Bloch predictions, and survive the fixed-weight endpoint GooF,
NPD, and split gates. An 8x8 field cannot be promoted without explicitly
declared independent calibration. Detector fits now include dimensionless
shrinkage toward unity and nearest-neighbor smoothness, and reports separate
frame-scale and detector-scale parameter counts. DIALS imports also preserve a
present inverse-scale-factor column as a fixed control sidecar rather than
silently substituting a GROND detector field.

`grond auto` can now use a model-keyed exact row-plan cache inside sufficiently
large fixed-model nuisance grids. Repeated thickness and absorption candidates
reuse detector geometry, the signed direct/target beam basis, rocking-profile
points, and difference-HKL lookup only while the structural-model hash is
unchanged. Because a cold cache is built serially, automatic mode enables it
only when at least 16 predictions share one geometry key; smaller searches stay
uncached. Moving-model coordinate/ADP updates remain uncached by the automatic
policy. The older cross-model frozen-basis behavior is still available only
through an explicit `--row-plan-cache readwrite`. The requested mode, resolved
mode, reuse estimate, threshold, and cache key remain recorded in the row
summary.

Automatic thickness promotion now compares the complete nuisance contract
correctly after mosaicity and geometry-path provenance were added. String-valued
fields no longer make an unchanged starting trial fail to match itself and abort
an otherwise healthy non-fast run. XDS all-zero `XOBS/YOBS/ZOBS` sentinels are
also treated as missing centroids rather than detector measurements at the
origin. Such rows remain available through predicted geometry, are excluded
from observed-geometry tail estimates, and are counted explicitly in preflight
and row-selection diagnostics.

Experts can now run the principal automatic decisions as explicit handoff
stages. `grond scout` performs only proper-transform preflight and scouting,
`grond thickness` aliases the existing thickness audit, `grond hand-check`
replays a completed native row contract, and `grond refine` aliases pinned
endpoint continuation. Scout, thickness, and hand stages write hash-bound JSON
handoffs containing the accepted inputs and ready-to-copy commands for the next
stage. The new expert workflow guide documents the complete `prep -> scout ->
thickness -> hand-check -> refine -> finish` sequence.

Elastic hand checking now implements the negative-thickness identity. One
Hermitian eigendecomposition produces both the supplied-coordinate and
coordinate-inverted intensities on the same signed rows, proper transform,
split, thickness distribution, and nonabsorptive nuisance contract. Absorption
is disabled explicitly in both branches and recorded; GROND refuses to apply
the shortcut when absorptive terms are requested. A separate explicit
coordinate-inversion mode remains available when absorption must be preserved
or when a full two-model control is required.

The native Bloch engine now has an opt-in direct-angle rocking-profile
coordinate. It projects the existing finite profile width into scan angle,
applies an explicit angular half-width bound, and can convolve that profile
with one bounded Gaussian projected-mosaic width per dataset. Unlike the older
target-Sg jitter experiment, this mode never asks for extra offset Ewald roots.
The default target-Sg production path is unchanged. XDS reflecting range and
beam divergence, DIALS profile-model parameters, and the presence of DIALS
frame-sliced shoeboxes are now retained as provenance; a DIALS GaussianRS
`sigma_m` or XDS reflecting-range width is exposed only as an optional prior
centre and is never activated automatically.

An experimental geometry-only scan-path fitter can derive a bounded,
zero-median along-scan slew and two transverse Chebyshev rotations from XDS
centroids. The global frame datum remains a separate nuisance. The fitter uses
no intensities, structure, hand label, or GROND residuals, records contiguous-
frame validation and source hashes, and rejects transverse corrections that do
not improve held-out detector centroids or hit a coefficient bound. Geometry
contracts are opt-in and fail closed on source-HKL or bound mismatches.

The new `GROND export-pets` command writes audited `cif_pets` files from DIALS
or summation-only XDS integrations for PETS/Jana2020/Dyngo and diffBloch. DIALS
exports use its native PETS geometry writer, with an explicit bounded projection
when a scan-varying cell cannot be represented by the one-cell PETS contract.
For XDS, GROND inverts the documented `RLP/PEAK` correction, verifies the raw-to-
CORRECT signed-row mapping and basis transform, and refuses unlabeled or
profile-fitted input by default. Every output is reparsed, receives an actual
nonzero `dstarmax`, and is accompanied by source hashes and a complete transform,
filter, variance, and cell-projection manifest.

PETS exports now have an explicit virtual-frame ownership contract. The default
`overlap` mode preserves the stock PETS/Jana candidate-frame representation.
The opt-in `--frame-ownership nearest-excitation` mode assigns every selected
source observation to exactly one minimum-absolute-excitation-error frame for
diffBloch refinement and records the complete ownership accounting in the
manifest. This prevents overlapping virtual frames from being mistaken for
independent repeat observations without changing the underlying measurement.
XDS PETS exports now also declare their uncertainty contract. The default
retains the calibrated XDS-reported sigma after the same PEAK/RLP transform as
the intensity; `--xds-sigma-mode raw` is an explicit sensitivity mode that
inverts the declared XDS variance model. The raw-mode nonpositive fallback now
retains variance on the correct recorded-intensity scale.

`GROND validate-ls1` now exposes and correctly forwards `--endpoint-ls-cycles` and `--endpoint-wght-mode` to every updated-q and fixed-q endpoint calculation. This repairs a diagnostic-only wrapper failure that prevented a controlled comparison of one versus several conventional LS/CGLS cycles under one frozen Bloch correction; production automatic feedback remains at one internal cycle by default. The new auto-workflow guide lists the automatic stage order, optional-stage switches, direct subcommands, and the separate meanings of inner LS cycles, repeated SHELXL calls, and updated-q macrocycles.

Corrected-HKL endpoint maps now pass their weighting mode to `shelx2map` explicitly. The primary Fo, Fo-Fc, and 2Fo-Fc maps use `-w 1`; a raw `-w 0` Fo-Fc map is retained under an unambiguous diagnostic filename. This avoids silently receiving unweighted maps from `shelx2map` builds whose executable default does not match the documented default, while preserving the untouched LIST 6 FCF and the unweighted control for audit.

The diagnostic native-gradient structure updater now uses explicit absolute finite-difference steps in each parameter's physical unit instead of passing one nominal step to SciPy as a relative multiplier. Coordinates and Ueq therefore receive stable, separately configurable perturbations even when a coordinate is zero or an ADP is small; the former `--diff-step` spelling remains as a deprecated absolute shorthand, and every block records the applied scheme and steps. This change is confined to the optional native finite-difference updater and does not alter the production Bloch/LS1 feedback path. The opt-in row-plan cache is also now wired through both single- and multi-dataset native updates. Its validator accepts NaN sentinels only for audit quantities that are explicitly inapplicable under the fixed-window/no-jitter contract, so valid caches now produce real geometry-cache hits while stale, incompatible, or geometry-nonfinite entries still rebuild.

The install and external-tool surface now matches the workflows GROND actually executes. Pandas was removed because no shipped runtime imports it. XDS, XDSCONV, and SHELXT were removed from the executable registry because GROND reads their existing products but never launches those programs; Olex2 likewise remains an optional user interface rather than a runtime dependency. The default config and doctor report now show only the small-molecule tools, while optional macromolecular/custom endpoint adapters are available through explicit expanded views. Doctor readiness can be scoped to the portable core, the SHELXL small-molecule endpoint, native maps, or the optional `shelx2map` endpoint-map path. Automatic runs no longer request legacy `shelx2map` maps by default. The production small-molecule runtime therefore needs only NumPy, SciPy, Gemmi, and SHELXL; native maps from `GROND finish` require no external map executable. DIALS Python remains necessary only when converting `.refl`/`.expt` input.

The DIALS bridge can now select rows accepted by `dials.scale` independently of
the intensity source. `--row-selection scaled` therefore preserves raw
summation intensities and variances from a `scaled.refl` table while respecting
documented scaling exclusions; requesting it without scaled rows fails closed.
New DIALS bridges also record the experiment UUID, actual space-group number,
and source Miller-index basis. A DIALS cell-metric failure now names the
offending dataset, prints both cells, and explains when a post-cosym identity
transform may safely be trusted. The bounded distorted-P1 fallback remains
specific to native XDS input and is no longer suggested for a DIALS bridge.

HKL-transform candidates now use one internally consistent cell metric. GROND preserves the input `UNIT_CELL_CONSTANTS` as raw provenance, derives the working identity and transformed model-basis cells from the same oriented `UNIT_CELL_[ABC]-AXIS` vectors used by the Ewald geometry, and reports both sources with an explicit basis label. This removes a rounding-dependent identity/nonidentity asymmetry from resolution filtering, Fourier-potential normalization, support construction, and transform scoring. Fourier-potential and row-plan cache schemas were advanced so results created under the former mixed-cell convention cannot be reused.

Automatic indexing alignment and absolute-structure testing now have an explicit parity contract. Calibration-grade scouting enumerates the complete 24 determinant-positive signed-axis permutations, applies only a recorded source/model unit-cell metric filter, and sends every retained operator through one matched-Born work-row nuisance trial. One candidate-union symmetry/Friedel split is rebuilt deterministically from the bound HKL, model, retained operators, and seed, so operator changes cannot move observations between work and free. Rankings are replayed from hash-bound row artifacts with a fixed candidate priority; every candidate must reproduce one exact N/Q, row-budget, selected/work/free support, nuisance, kernel, and backend contract. Expected chemistry and metadata overlap cannot select, eliminate, or break ties. A mixed-parity search remains available only as an explicit diagnostic mode and cannot authorize a named absolute configuration.

The new `GROND canonicalize-hand` command repairs a legacy improper-transform endpoint without changing its physical content. It materializes the exact inverted-coordinate and inverted-symmetry RES, changes `T` to the proper `-T` representative, negates every model-basis HKLF4 index, preserves the observation and trailing fields byte-for-byte, and writes a hash-bound gauge-equivalence manifest. This operation is calibration-free and remains conditional. Hand control now evaluates the same predeclared nuisance grid once for each coordinate hand, chooses one common nuisance by the label-symmetric mean work-row dynamical wR2 with a nuisance-hash tie-break, and performs two singleton evidence replays at that state. This doubles the initial hand-scout evaluations but removes the previous dependence on a nuisance selected from the as-supplied hand. Matched decisions are then replayed from both hash-bound row artifacts with one source-sigma vector and one scale-design rule; the calibration gate requires a sign-consistent Klar--Palatinus adjusted Z above its frozen threshold. The separate `GROND create-parity-calibration` and `GROND apply-parity-calibration` commands require the setup ID, setup-manifest hash, physical-crystal ID, and raw-acquisition fingerprint to have been frozen before either decision. Exact declared reuse of the calibrant reflection content or decision artifacts fails closed. Setup, physical-crystal, and acquisition identifiers are nevertheless caller-attested provenance, not cryptographic proof of raw-data identity; a deliberately false relabeling remains outside the package trust boundary.

The Klar--Palatinus adjusted reflection-count hand diagnostic now treats exact equal-residual rows as neutral half wins. This versioned finite-precision extension restores exact sign antisymmetry when the selected and other hand are swapped. Diagnostic JSON retains the former strict-lower-residual statistic and probabilities as explicitly labeled legacy fields. Ordinary diagnostics retain the paired-row contrast, while calibration-grade decisions require the independently replayed common-source-sigma adjusted-Z evidence described above.

Multi-dataset `push-endpoint` can now take one immutable policy-selected `native_dyn_calc.csv` contract per dataset. The continuation verifies every recorded row index, transformed HKL, target intensity, raw XDS sigma, and d spacing against the live XDS input, then restores the table's declared dataset role plus raw-row and endpoint weights rather than falling back to an older checkpoint policy. It records content hashes in each cycle and refuses partial mappings, stale input, duplicate rows, inconsistent policy annotations, or a `max-rows` truncation. This preserves a promoted multi-dataset support exactly through later Bloch/LS1 controls instead of silently recreating membership from raw eligibility filters; compact-scale selection remains a separate explicit replay after the selected safe model.

Every successful multi-dataset Bloch/LS1 continuation now finishes with one fixed-nuisance, no-extra-LS1 Bloch row recomputation from the accepted shared endpoint model. Final row metrics, maps, and compact-scale replays therefore no longer risk pairing a cycle's input-model predictions with the coordinate/ADP-updated SHELXL output model.
`push-endpoint --cycles 0 --recompute-final-rows` exposes that same exact replay for an already accepted multi-dataset endpoint without performing another SHELXL step.
Multi-dataset continuations can now bind one compact scale topology per dataset with `--dataset-scale-spec`. GROND materializes those refitted scale rows after every fresh Bloch calculation and before the shared corrected-HKL/SHELXL step, so an endpoint control cannot quietly broaden a previously promoted compact scale contract.

Native dynamical map export now uses the correct crystallographic Fourier sign, so off-origin map peaks appear at the model rather than its inverted hand. Map coefficients merge model-symmetry equivalents with propagated amplitude uncertainties, retain independent Friedel observations through dynamical amplitude inference, and then make the necessary uncertainty-weighted Hermitian projection only while synthesizing a real electrostatic-potential map. The resulting manifest records the symmetry expansion, Friedel disagreement, and projection rather than silently discarding a branch. Hydrogen-omit peak auditing now samples Gemmi/CCP4 grids at their actual `ijk/grid_shape` positions and uses the nearest sample at a site. The alternate-hand row diagnostic additionally reports the Klar--Pálatiňus adjusted reflection-count statistic for valid, matched single-dataset rows, while failing closed on mismatched observations or undeclared multi-dataset pooling; the legacy paired residual diagnostic remains separately labeled.

Final statistics now have an explicit reporting contract. Every completed report bundle writes root-level `TABLE1_FINAL_STATISTICS.md` and `.json` with separate SHELXL endpoint and GROND-computed JANA/PETS-style row panels. Canonical `JANA_ROW_STATISTICS.*` and per-dataset `JANA_ROW_STATISTICS_PER_DATASET.*` artifacts distinguish unmerged `Robs/Rall/wRall` from post-symmetry-merged `MRobs/MRall/MwRall`; optional observed-subset weighted factors remain labeled separately. Born and Bloch factors are evaluated on one enforced finite row support, the actual instability factor `u` is recorded, and invalid HKL indices are surfaced as merge exclusions instead of disappearing from `MR` counts. Conventional kinematical, matched Born, and Bloch dynamical entries are shown only where computed, with explicit `n/a` elsewhere. Legacy native-objective residuals remain available for optimization audit but are no longer presented as publication endpoint or row factors.

The DIALS bridge now preserves cropped-scan frame origins by exporting DIALS' zero-based array-range start as `STARTING_FRAME` and applying the same origin when validating omega and scan-point A matrices. Imports whose predicted rows remain grossly off the Ewald sphere fail before writing a new reflection bridge, with an auditable diagnostic summary instead of a misleading `status: ok`.

Multi-dataset automatic runs now evaluate alternate-hand evidence for every crystal but make one conservative decision for the shared structure. Each decision compares matched original- and alternate-hand scouts on the same selected, work, and positive held-out row counts, with work-only scale fitting even when production scales use all rows. The alternate hand is selected only when every dataset passes the all-row and raw held-out quality gates; weak, failed, missing-free, or mixed evidence keeps the input hand for all downstream thickness, nuisance, LS1, and endpoint stages. Materialized SHELX alternate models now invert both atomic coordinates and explicit `SYMM`/`EQIV` translations, validate the written model against the scouted geometry, and use source/output hashes so a changed same-name input cannot reuse an old variant. Models with signed `CHIV` restraints retain the input hand because automatic CHIV-target inversion is not yet supported.

Multi-dataset automatic runs now materialize and test full, BIC-selected compact, and minimum-strict scale contracts before writing the canonical combined endpoint. Compact scales reuse the exact frozen Bloch predictions and may be promoted only when their row identity and physics contract match the full control, their aggregate fitted-variance BIC improves, all/work/free and per-dataset row costs remain bounded, pooled Bloch predictions still beat the matched Born control, merged coverage is unchanged, and the SHELXL endpoint remains healthy, publication-like, and free of new ADP pathology. If more than one compact contract passes, GROND chooses the smallest scale count; every incomplete or failed gate keeps the full-scale result. Reports, publication tables, run configuration, and continuation checkpoints record the selected materialized contract rather than mixing compact metadata with full-scale rows.

Combined corrected-HKL endpoints now accept a separate paper-facing resolution cutoff. Row membership is recalculated from signed HKL and the shared structure cell, and the same retained rows drive per-dataset, joint, held-out, merge, and SHELXL statistics. Reports preserve the unfiltered source metrics separately, record requested and discrete achieved resolution plus retained/dropped counts, and keep full-scale controls distinct from the selected compact publication rows. Multi-dataset endpoint continuation also states its actual contract: `push-endpoint` refits the pinned full grouping, while compact reselection is a separate post-push publication step.

Repeated endpoint refinement now snapshots each matching NPD-free SHELXL RES/LST/FCF set. If a later least-squares pass fails, omits final statistics, writes a structurally incomplete model, or develops a non-positive-definite ADP, GROND restores the last safe endpoint for refinement and maps while retaining the rejected pass as an explicit diagnostic artifact. Feedback endpoints must contain a parseable explicit electron-SFAC model with atoms, complete R-factor statistics, and a path/hash record that still matches the summary-selected lambda. Single- and multi-dataset feedback therefore cannot pair current metrics with a stale endpoint file left by a resumed run.

When the first corrected-HKL SHELXL step is otherwise complete but develops NPD atoms, single-dataset feedback, shared multi-dataset feedback, and scale-contract replay now retry that same step from the unchanged input model with the bounded `DAMP 100`, `300`, `1000`, `3000`, and `10000` ladder. GROND accepts the first complete zero-NPD result and archives the undamped and every damped RES/LST/FCF attempt. This is step control only: it does not add restraints, enable XNPD, alter the corrected reflection data, or relax the zero-NPD gate. If no safe step exists, the endpoint remains rejected.

Discrete thickness polishing continues to rank candidate grids using work rows and work-fitted scales alone, but its single winner must now pass a post-selection held-out no-harm gate against the exact starting nuisance already evaluated in the same grid. A negligible training gain, missing or changed row counts, changed valid-row identity, nonfinite validation statistics, failure to improve the full-row composite score, excessive free-wR2 loss, or excessive all-row R1 loss restores the starting trial and its already-computed Bloch arrays with no extra propagation. Continuous thickness polishing now enforces the Bloch engine's physical `0 <= thickness_grad_frac <= 1` domain in its CLI bounds, every objective evaluation, quadrature construction, and final promotion gate. Its search interval is the local fractional basin around the coarse-grid winner, including for the final Powell step, so it cannot jump to a separated dynamical-thickness alias. The former numerical 400--2600 A limits are no longer treated as physical bounds; any finite positive coarse winner is accepted, while explicit hard bounds may only clamp its local basin. A bounded optimizer that returns a nonfinite or slightly out-of-bounds point is recorded as rejected instead of passing an invalid nuisance value into later physics gates or shared LS1 cycles.

Optional Sg-jitter searches now reject a nuisance candidate when one of its convolved quadrature targets has no crossing inside the strict local rotation bracket. Feasible no-jitter and jitter candidates continue normally, the rejected configuration and row are recorded, and GROND does not widen the geometry bound or silently truncate the requested orientation distribution.

Required zero-jitter base-profile targets that lie beyond the short Bragg-assignment bracket can now be reached by exact continuation through a bounded sequence of strict local solves on one interior-certified monotonic Ewald branch. The Sg=0 Bragg gate and active-jitter gate remain unchanged and fail closed. Row-plan and prediction caches, native row tables, and summaries record whether continuation was used and how many local segments were required. Multiprocessing prediction retries are also transactional, so a failed context cannot leak partial rows into a fallback, duplicate progress, or produce a mixed-backend prediction.

Repeated Bloch/LS1 feedback cycles now share their content-addressed row-prediction cache, so a final recomputation of an unchanged model reuses the exact preceding prediction instead of repeating the matrix propagation. Prediction-cache metadata now compares NaN sentinels canonically and fingerprints the prediction-engine source plus numerical runtime, preventing both false misses and unsafe reuse after a code or library change.

`GROND doctor --python PATH` now tests the requested interpreter instead of silently substituting another Python. The new `--strict` mode returns a failing exit status when required packages, installed GROND features, or requested SciPy Bloch operations are not ready, and reports machine-readable `ready` and `failures` fields. An explicit `--shelxl /path/to/shelxl` now satisfies automatic-run tool preflight, is resolved once to an absolute executable, and is forwarded through every normal native, feedback, combination, observation-polish, baseline, and endpoint-calibration stage. Endpoint-enabled workers fail closed if that executable is missing or not launchable, row-only scouts skip redundant SHELXL baseline work, and resume fingerprints include the executable contents. Failed workflow stages write a concise recovery artifact and log tail without masking the original subprocess error, while malformed or structurally invalid resume records are warned about and quarantined before a fresh record is started. Automatic dry runs now finish with a planning-only summary instead of trying to read row tables that intentionally were not materialized; real runs retain the strict missing-row refusal.

`grond auto` and `grond finish` now compare a corrected publication endpoint with the fresh conventional baseline rerun from the same final native stage. A corrected endpoint with worse `R1all` is retained as a diagnostic but is no longer promoted, and the reports point directly to the conventional RES/HKL/LST that should be kept. Single-dataset thickness reporting states both the mean in nanometres and the support implied by the selected distribution. Multi-dataset reports instead list each crystal's independent center, gradient, support, scale field, and nuisance contract; they no longer assign a fabricated support interval to an arithmetic aggregate.

Publication modulation gates now measure the correction that was actually applied instead of accepting a placeholder. The recorded `mod_p95` is the 95th percentile of the symmetric fold change after removing the arbitrary overall intensity scale in log space. Native, compact-scale replay, observation-polish, and combined endpoints all write this audit; combined values include the fitted inter-dataset overlap scales. Missing or excessive modulation fails publication selection, and a shared multi-dataset LS1 loop keeps the preceding clean model rather than feeding an unmeasured or over-modulated endpoint into another Bloch cycle.

Automatic XDS runs now measure the scan-datum difference between the global header orientation and row-level `ZCAL` positions before testing rocking-profile offsets. This prevents an unvalidated zero-frame assumption from forcing thickness to compensate for a profile-center error while keeping nuisance selection on work rows. Sparse, inconsistent, or extreme geometry estimates fall back without recentering, and the higher-quality thickness polish retains the original geometry-derived probes. Fixed-nuisance row runs also correctly translate the public `sigma_a`, `sigma_b_corr`, and `sigma_c_peak` names into the frozen Bloch configuration instead of failing while writing their summary. A selected continuous-thickness nuisance is now materialized on the full native row contract before feature comparison or multi-dataset policy, preventing a summary-only optimizer directory from silently disabling structure-update weighting. Calibrated single-dataset scale-contract endpoints enter the final publication chooser with their actual HKL safety evidence. Multi-dataset automatic calibration is explicitly marked not applicable because a one-table replay would incorrectly share scale fields across crystals; the validated combined endpoint remains authoritative.

GROND now supports the combined multi-dataset Bloch/LS1 feedback path, model-symmetry endpoint merging, resumable endpoint continuation, native final maps, and explicit endpoint-selection reports. Publication endpoints reject non-positive-definite model updates and use the declared post-Bloch uncertainty model with fixed `WGHT 0 0`; inherited or promoted SHELXL weights remain diagnostic choices. Earlier small-value automatic DAMP retries were removed after they proved unsuitable as a silent repair policy; the current bounded first-safe-step recovery is explicit in the endpoint provenance and never accepts an NPD model.

The command-line workflow now exposes diagnostic SHELXL cycle control, clearer stale-install checks, portable XDS and DIALS input preparation, expanded row and endpoint reports, and guarded experimental acceleration backends. The public package continues to use dense SciPy as its numerical reference.

Multi-dataset feedback and its final row recomputation now retain the pinned native row-weight contract instead of re-reading the preceding SHELXL endpoint's `WGHT` line. This prevents a `WGHT 0 0` endpoint from making the next cycle's weighted row statistic look artificially different while the unweighted row fit is unchanged or worse. `push-endpoint --native-wght-a/--native-wght-b` supplies an explicit global recovery contract for checkpoints written before this fix.

The package audit for this release corrected wavelength-dependent relativistic scaling, nonorthogonal anisotropic displacement factors, symmetry-mate tensors, alternate-hand symmetry translations, special-position expansion, local excitation-error root finding, and DIALS `S R F A` goniometer geometry. The DIALS bridge now applies LP and quantum-efficiency corrections together and keeps raw detector-model variances by default. SHELX parsing resolves FVAR parameters, excludes Q peaks and instruction records, and rejects models without explicit electron scattering coefficients.

Validation scales now fit work reflections by default, while symmetry and Friedel mates remain on the same side of the held-out split. Sparse or free-only scale groups borrow a finite work-derived scale instead of producing a zero denominator. Multi-dataset overlap scales require overlap between distinct datasets and use leave-self-out equations. Invalid sigma rows no longer acquire artificial weight.

Dense Fourier-potential allocation is checked against a declared memory ceiling before NumPy allocates it. Fourier-potential, row-plan, and prediction caches now validate their schema, input fingerprints, shapes, and finite values before reuse, and cache writes are atomic. Resume records fingerprint the GROND source and declared inputs. Failed external refinements cannot reuse stale output files.

The numerical engine is included in Ruff checks and every shipped legacy module is tested as an ordinary package import and as a direct script where supported. The README and user guides were rewritten as concise package documentation, and release metadata, citation data, policy files, and artifact checks now agree on the package boundary.

## 0.2.1

The environment doctor gained explicit stale-install and required-feature diagnostics so an older app bundle or partial site-packages tree can be distinguished from the current checkout.

## 0.2.0

This was the first GitHub-facing package layout. It organized the Python code into input/output, Bloch, refinement, map, forensic, workflow, and historical engine namespaces; removed dataset-specific helper scripts from the installed package; added package standards files; and introduced documentation for a general crystallography lab audience.

At this stage the endpoint path promoted SHELXL's recommended `WGHT` between endpoint-polish cycles. That behavior is retained here as release history; version 0.2.2 changed the production endpoint to fixed `WGHT 0 0` after GROND's endpoint sigma calibration.
