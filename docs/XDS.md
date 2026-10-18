# XDS data

GROND needs summation intensities. Solve and refine the structure the usual way, then integrate the same frames again
without profile fitting and give GROND that directory.

## Summation integration

Copy the XDS directory that holds the accepted geometry and edit `XDS.INP` in the copy:

```text
PROFILE_FITTING= FALSE
MINPK= 90.0
```

Rerun the integration and correction steps. GROND reads `INTEGRATE.HKL` (its header holds the detector, beam,
rotation axis and cell) and `CORRECT.LP` (cell e.s.d.s) from the directory, so keep the directory together. A
profile-fitted `INTEGRATE.HKL` runs with a warning, and the corrections are worse.

Each XDS observation gets a fixed rocking window of 0.010 Å⁻¹ in excitation error, sampled at 50 points.

## Run

```bash
grond sfac-check --structure model.res
grond prep --structure model.res --baseline-hkl model.hkl \
    --dataset c1=/data/c1/xds_noprf,dmin=0.80 --dataset c2=/data/c2/xds_noprf,dmin=0.80 --out prep.json
grond auto --prep prep.json --config grond.config.json --out-dir run
```

For one crystal, `grond auto --xds-dir /data/c1/xds_noprf --structure model.res --baseline-hkl model.hkl --out-dir run`
works without a prep file. XDS and DIALS crystals can be mixed in one prep file.

`model.res` must have numeric electron scattering factors on its SFAC lines; `SFAC C H N O` alone is refused. The
`--baseline-hkl` file is the HKLF 4 file the model was refined against. GROND uses it for the leak-free start (the
model refined again without the held-out reflections). Without it the leak-free start is skipped, and the held-out
R factors are optimistic.

## Indexing

`INTEGRATE.HKL` is often in the P1 integration setting while the model is in its space group. GROND tries the 24
proper (determinant +1) relabellings of the axes, keeps those whose reciprocal metric agrees with the model within
2%, and picks one with kinematic predictions that do not depend on the hand. The choice is written to
`decisions/<crystal>/transform_decision.json`. If the setting is known, give it in the prep entry
(`hkl_transform=identity`) and skip the search with `--trust-hkl-transform`.

An operator with determinant −1 inverts the structure. GROND never uses one to relabel the data; the hand is decided
by the hand test.

## Frame numbering

XDS puts image n between z = n − 1 and z = n. GROND reads the rotation angle of an observation as
`STARTING_ANGLE + (ZCAL − STARTING_FRAME + 1) × OSCILLATION_RANGE` for files written by XDS.
