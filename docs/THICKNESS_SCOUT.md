# Effective Thickness Scout

GROND reports an effective dynamical thickness for the signed-row contract it is given. That value is the thickness model that makes the Bloch/Born row prediction best match the reduced scalar observations after the declared scale family has been fitted. It should not be read as a direct microscope thickness unless it is independently calibrated.

This matters when comparing XDS and DIALS. The same frames can become different scalar row contracts after integration, partiality handling, sigma modeling, profile or summation choices, Lorentz handling, and scale conventions. Those differences can move fitted nuisance parameters such as thickness, detector scales, frame scales, Sg jitter, and absorption even when the final hand and row-level R factors agree.

The `thickness-scout` command makes that sensitivity explicit. It runs a small set of fixed-structure row fits over thickness and gradient grids, compares production detector scales against a no-detector-scale branch, includes a single final-feature diagnostic branch by default, and writes selected row metrics plus q-field stability diagnostics.

```bash
GROND thickness-scout \
  --xds-dir . \
  --structure temp_a.res \
  --baseline-hkl temp_a.hkl \
  --out-dir thickness_scout
```

For a narrower diagnostic, keep only the early bare contract:

```bash
GROND thickness-scout \
  --xds-dir . \
  --structure temp_a.res \
  --baseline-hkl temp_a.hkl \
  --out-dir thickness_scout_bare \
  --feature-modes bare
```

For a broader scan, edit the thickness and gradient grids:

```bash
GROND thickness-scout \
  --xds-dir . \
  --structure temp_a.res \
  --baseline-hkl temp_a.hkl \
  --out-dir thickness_scout_wide \
  --thickness-grid-nm 50,75,110,150,180,220,260,300,400,500 \
  --gradient-grid 0.0,0.4,0.8,1.0,1.2
```

The output directory contains `REPORT.md`, `thickness_scout_summary.json`, `thickness_scout_table.csv`, `thickness_scout_trials.csv`, and `q_field_stability.csv`. The selected-run table is the quick human view. The trials table is the full nuisance grid. The q-field table compares the selected row-wise `log(Idyn/Iborn)` fields against the promoted reference and reports the shared-row correlation and the 95th percentile absolute `delta log q`.

The production rule is conservative. Use the early bare production-scale thickness to guide `GROND auto`, pin it through the LS1 loop, and treat final-feature movement as an identifiability warning. If Sg jitter or absorption moves the apparent thickness by tens of nanometers while the row metrics remain similar, the thickness is absorbing row-contract or nuisance-model differences. In that case the stable object to trust is the q-field, row residuals, final hand, maps, and endpoint validation, not the absolute fitted thickness number.
