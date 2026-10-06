# Limitations

GROND is research software. It writes its decisions, tables and logs so that every result can be checked, but an automatic choice is not guaranteed to be right for every data set. Inspect the output, especially when the data are weak, incomplete, poorly integrated, strongly damaged or very thick.

## Scope

GROND is written for small-molecule and peptide electron-diffraction data that start from XDS or DIALS integrated intensities and a solved structure. It does not integrate images and does not replace XDS, DIALS, SHELXL or Olex2. The macromolecular interface (`grond export-mtz`, `grond protein-ls1`) is experimental.

For DIALS data GROND reads the summation intensities and keeps the DIALS variances (`--sigma-mode raw`); a detector-gain model is available with `--sigma-mode detector_gain`. Check `dials_bridge_summary.json` for row counts, partiality and dropped flags.

## The model is an approximation

- The correction freezes the dynamical-to-kinematic ratio during each SHELXL refinement and recalculates it afterwards. The refined model is therefore close to, but not the same as, the result of a full dynamical refinement, and the standard uncertainties reported by SHELXL do not include the uncertainty of the correction.
- The fitted thickness distribution, mosaic spread and damage term are effective parameters of the calculation, not measurements of the crystal.
- Absorption is off by default. Twinning, disorder and solvent masks are not modelled; HKLF 5, TWIN, BASF and SWAT are not supported.
- By default an observation uses at most 500 beams. Large or strongly coupled cells need a convergence check and, if necessary, a higher limit (`--final-basis-max`, `--nested-n-basis`).

Residuals can come from integration, background, detector effects, geometry, partiality, scaling, indexing or the structure itself. GROND can point to some of these, but more beams do not repair bad observations.

## Hand

The held-out hand test states whether the data prefer the supplied coordinates or their inverse. It does not name an absolute configuration: that also requires the sense of the rotation axis and of the whole processing chain, which GROND cannot check. A crystal of known configuration processed the same way is needed to fix the sign.

## `grond manual`

`grond manual` is a simpler, stage-by-stage version of the calculation for inspection. It has no leak-free start and no cross-validated choice of mosaic spread and damage, its hand stage uses the all-observation statistic, and it makes one SHELXL cycle per round. Use `grond auto` for reported results.

## Validation

Validate GROND on known structures before trusting it on new chemistry, microscopes, detectors or integration strategies. A result that lowers one R factor while degrading the maps, displacement parameters or the held-out statistics should not be accepted.

GROND is provided without warranty under the MIT License. External programs such as SHELXL are not bundled and have their own licenses.
