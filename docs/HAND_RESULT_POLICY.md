# Hand results and calculation errors

GROND uses the full matched observation table for its primary hand comparison.
The supplied coordinates and their inverse retain the same signed rows,
geometry, beam basis, thickness, profile, scale design and source-sigma
convention. The final elastic control disables absorption to use the exact
negative-thickness identity. Its residual summaries include both coordinate
representations and the matched Born control.

The primary adjusted reflection-count statistic assigns half a win to an
exact or numerical residual tie. Swapping the hands changes only its sign;
identical predictions give zero and are reported as no preference. Numerical
ties use relative tolerance `1e-12` and absolute tolerance
`1e-12 * source_sigma`. Method version 4 records those tolerances and retains
the earlier strict-count value in `z_strict_lower_residual_klar_2023`.

All-row evidence determines the reported preference. Work/free values remain
diagnostics, and missing, small, weak or conflicting free subsets give
specific warnings without vetoing the primary result. The default 3.5
threshold controls whether AUTO changes the coordinate representation. Thus a
weak inverse preference can be reported alongside retained input coordinates.
`FINAL_SUMMARY.md` and console output distinguish preferred hand, conditional
support and exported representation.

The adjusted count statistic is conditional on the input model, geometry,
scaling and acquisition/import sign. Shared nuisance parameters, repeated
observations and model selection can correlate rows. Its nominal normal-tail
transform is not a calibrated probability or a calibrated number of sigma for
absolute configuration. A separately reserved sign calibrant is still needed
for a named absolute-configuration claim.

Statistical uncertainty does not make a completed physical calculation
disappear. The calculated values and warnings remain available. In contrast,
mismatched observations, source sigmas, physical geometry or provenance cannot
establish a hand preference. Those cases report `calculation_errors` with
specific causes. A failed single-dataset final-hand stage writes
`adaptive_hand_calculation_error.json`, retains the preceding usable
structural endpoint and reports hand preference as unavailable. An entirely
invalid absorption/geometry candidate grid reports the rejected configurations
and their errors; feasible candidates can continue when only part of the grid
is invalid.

A centrosymmetric structure has no applicable absolute-hand comparison. Its
explicit not-applicable result is accepted by endpoint delivery. This does not
override model-health, geometry or artifact-provenance checks.
