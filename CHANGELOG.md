# Changelog

## 0.2.2 (unreleased)

Validation is now part of every run:

- A free set of about 10% of the unique reflections, the same in every crystal, is held out from every fit and choice. The supplied model is first refined without it (leak-free start).
- Thickness, mosaic spread, damage and frame-scale smoothness are chosen by cross-validation within the working observations.
- Every run reports R factors on the free set with bootstrap confidence intervals.
- `--free-set off` (default) refines the chosen model once more against all reflections at the end; `--free-set on` keeps the free set out of the final model.

Physics and numerics:

- Thickness distributions are gamma or wedge on 10 nm–1 µm. The gamma average is evaluated in closed form.
- Mosaic spread is a three-point tilt with re-centred observations. Radiation damage is a displacement parameter that grows with frame number.
- Absorption is off by default.
- Frame scales are smooth penalised splines fitted on working observations, replacing per-frame scales.
- DIALS observations use the integration box as their rocking range.
- Centrosymmetric models use the real symmetric eigensolver.
- Identical Bloch-wave matrices are solved once and reused.

Refinement:

- Corrected intensities are written on an absolute scale.
- SHELXL refines to convergence in every round, with weights iterated and no damping. Up to five rounds run, until non-hydrogen atoms move by less than 0.002 Å.
- Uncertainties follow an error model σ′² = k²(σ² + (uP)²) fitted per crystal.

Hand: decided early by a sign test on held-out reflections and confirmed on the final model.

Interface:

- `grond auto --preset fast` gives a quick result with every check kept.
- `grond manual` runs the calculation stage by stage.
- `grond refine` recalculates the corrections for a model refined outside GROND.

## 0.2.1

`grond doctor` reports stale installations and missing features.

## 0.2.0

First public package layout.
