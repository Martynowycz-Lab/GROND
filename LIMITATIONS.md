# Limitations

GROND writes its decisions, tables and logs so that each result can be checked. An automatic choice can still be
wrong for a given data set. Inspect the output, especially for weak, incomplete, badly integrated, strongly damaged
or very thick crystals.

- GROND starts from integrated summation intensities and a solved model. It does not integrate images, solve
  structures or replace SHELXL.
- The correction is held fixed during each SHELXL refinement and recalculated afterwards. The result is close to a
  full dynamical refinement but not identical to one, and the standard uncertainties from SHELXL do not include the
  uncertainty of the correction.
- The fitted thickness distribution, mosaic spread and damage term are effective parameters of the calculation. They
  are not measurements of the crystal.
- Absorption, twinning, disorder models and solvent masks are not modelled. HKLF 5, TWIN, BASF and SWAT in the input
  model are refused.
- An observation uses at most 500 beams by default. Large cells need a convergence check and higher limits
  (`--final-basis-max`, `--nested-n-basis`) and, for speed, `--shared-grid auto`.
- The hand test says whether the data prefer the supplied coordinates or their inverse. An absolute configuration also
  depends on the sense of the rotation axis and of the processing chain, which GROND cannot check. A crystal of known
  configuration processed the same way fixes the sign.
- `grond manual` is a simpler version of the calculation for inspection. It has no leak-free start, no
  cross-validated choice of mosaic spread and damage, and one SHELXL call per round. Use `grond auto` for reported
  results.
- R_complete is not implemented. Holding out reflections (`--free-set`) is the validation GROND provides.

Validate GROND on known structures before using it on new chemistry, microscopes, detectors or integration
strategies. Do not accept a result that lowers an R factor while the maps, displacement parameters or held-out
statistics get worse.
