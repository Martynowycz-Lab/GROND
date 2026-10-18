# DIALS data

GROND reads `integrated.refl` and `integrated.expt` from `dials.integrate`. It uses the summation intensity
(`intensity.sum.value` and its variance), so profile fitting can stay on. The `.expt` file supplies the beam,
detector, goniometer, scan and the scan-varying crystal model. A merged MTZ or HKL file does not carry the geometry of
each observation and cannot be used.

## Convert each crystal once

```bash
grond import-dials --refl integrated.refl --expt integrated.expt \
    --dials-python /path/to/dials.python --dataset-name c1 --out-dir bridges/c1
```

`--dials-python` is a Python that can import DIALS (for example the `dials.python` of a DIALS installation). It is
needed only for this step. The output directory holds `INTEGRATE.HKL` in GROND's own format, the scan-point
orientation matrices and `dials_bridge_summary.json`. Use the directory in `grond prep`:

```bash
grond prep --structure model.res --baseline-hkl model.hkl \
    --dataset c1=bridges/c1,dmin=0.80 --dataset c2=bridges/c2,dmin=0.80 --out prep.json
grond auto --prep prep.json --config grond.config.json --out-dir run
```

One directory holds one DIALS experiment. Split multi-experiment files first with `dials.split_experiments`.

## What the import writes

With the default `--intensity-mode sum_complete`:

```text
correction = lp / qe
I          = intensity.sum.value * correction / partiality
sigma      = sqrt(intensity.sum.variance) * correction / partiality
PEAK       = 100 * partiality
```

The default `--min-peak 90` of `grond auto` therefore keeps observations with partiality ≥ 0.90. The DIALS variance
is kept as it is (`--sigma-mode raw`). Observations flagged `dont_integrate`, overloaded, in a powder ring, with bad
pixels or with failed background or summation are dropped.

`--row-selection scaled` with a `scaled.refl` keeps only the observations that `dials.scale` accepted, still with
summation intensities. `--intensity-mode scale` uses the scaled intensities instead; it is for comparison only.

The shoebox of each observation gives its rocking range. GROND samples it at 60 points with the width kept between
0.003 and 0.03 Å⁻¹ in excitation error.

## Checks

GROND does not use the DIALS `zeta` column. It recalculates the excitation error of every observation from the
geometry, interpolating the scan-varying A matrix between scan points. The import stops when the 95th percentile of
|S_g| at the predicted centres exceeds 0.005 Å⁻¹, which means the `.refl` and `.expt` files do not match. Open
`dials_bridge_summary.json` to see the counts of written and dropped observations and the |S_g| statistics.

DIALS does not need to integrate in the model's setting. GROND keeps the signed Miller indices and finds the proper
(determinant +1) transform to the model itself, before it compares hands.
