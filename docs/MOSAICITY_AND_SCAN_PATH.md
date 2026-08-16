# Projected mosaicity and scan-path experiments

These controls are opt-in native-row experiments. They do not change the
default `grond auto` production contract.

`grond import-dials` records the DIALS profile model and its serialized
parameters when available. For a GaussianRS model, `sigma_m` is retained in
degrees as an optional projected-mosaic prior centre. The bridge also records
whether David Waterman's experimental `frame_sliced_shoebox` column is present.
It does not yet turn those frame slices into an observed rocking curve. Native
XDS input similarly records `REFLECTING_RANGE_E.S.D.`,
`BEAM_DIVERGENCE_E.S.D.`, and `PROFILE_FITTING`. XDS reflecting range is not a
pure mosaicity measurement, so neither importer activates a width silently.

The direct-angle control is selected with:

```text
--profile-coordinate direct_angle \
--profile-angle-max-half-width-deg 2.0 \
--mosaicity-mode gaussian_omega \
--mosaicity-sigma-deg 0.02 \
--mosaicity-max-sigma-deg 0.25 \
--mosaicity-points 3
```

The ordinary finite profile width is projected from inverse Angstrom into
degrees using the local absolute `dSg/domega`. Integration is then performed at
explicit angles around the recorded row centre. A projected Gaussian mosaic
distribution is added by normalized Gauss-Hermite convolution. No offset-Sg
root is requested, and unusually shallow crossings are clipped only at the
declared angular half-width. The mosaic width counts as one dataset nuisance
parameter when nonzero. It should be chosen with a predeclared bounded grid and
held-out rows, not by the final merged endpoint.

For XDS centroids, a small geometry-only path can be generated independently of
the structure and intensities:

```text
python -m grond.legacy.engine.fit_geometry_path \
  --hkl INTEGRATE.HKL \
  --out geometry_path.json \
  --degree 3 \
  --min-peak 90
```

The fitted along-scan term is `-(ZOBS-ZCAL-median(ZOBS-ZCAL))`. Its median is
fixed to zero so it cannot duplicate the global frame-offset nuisance. Two
transverse rotations are fitted from `XOBS-XCAL` and `YOBS-YCAL` against the
laboratory detector axes. The output uses contiguous-frame validation, bounded
Chebyshev coefficients, and a source-HKL hash. A transverse path is written as
zero unless it improves held-out detector-centroid RMS by at least one percent
without touching a bound. Pass an accepted contract with
`--geometry-path geometry_path.json`.

DIALS scan-varying A matrices already enter GROND directly through the bridge;
they remain preferable to reconstructing a second orientation spline. The
next integration-aware step is to consume frame-sliced shoebox summation and
variance arrays as an observed rocking profile. That requires the unmerged
per-frame values and validity mask, and is deliberately not inferred from a
single integrated intensity.
