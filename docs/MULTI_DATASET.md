# Multi-Dataset Runs

GROND treats each crystal as its own row-level experiment until the last possible step. That is the safest default because thickness, frame offset, scan scale, detector-position scale, sigma behavior, row completeness, and integration quality are properties of an individual dataset. A weak crystal should not set the thickness or sigma scale for a good crystal, and a good crystal should not erase a weak crystal that contributes otherwise missing reflections.

The command-line entry point is still `GROND auto`. Put one row-preserving dataset per line in a prep file:

```bash
GROND prep \
  --structure model.res \
  --baseline-hkl model.hkl \
  --dataset crystal_370=/data/370/no_prf/INTEGRATE.HKL \
  --dataset crystal_366=/data/366/no_prf/INTEGRATE.HKL \
  --out prep_366_370.json

GROND auto \
  --prep prep_366_370.json \
  --out-dir grond_366_370
```

For each dataset, GROND runs the same row-contract checks and nuisance searches it uses for one crystal. Transform scouting, hand checks, thickness, thickness-gradient, frame offset, row-resolution terms, absorption-like terms, Sg jitter, detector-position scales, and sigma behavior are fitted independently. The shared structure model supplies the atomic scattering factors for every dataset, but the nuisance model remains dataset-specific.

After the per-dataset passes, `dataset_policy/structure_update/` decides which datasets are safe to use in the shared-model update. Primary datasets keep full row weight. Downweighted datasets remain in the row target, but their effective sigmas are inflated so they cannot dominate the model. Completeness-only datasets are excluded from row-level structure statistics and shared-model gradients, but they can still contribute endpoint reflections if they add real coverage. Excluded datasets are kept in the reports and left out of the combined endpoint.

The current shared-model update for multi-dataset jobs is the native finite-difference diagnostic path in `native_multi_structure_update.py`. The single-dataset `Bloch -> SHELXL LS1 -> Bloch` feedback loop is not yet used as the multi-dataset shared-model engine. This is deliberate. The multi-dataset path must keep per-crystal nuisance models, per-crystal sigma calibration, and one shared atomic model synchronized; until the combined LS1 loop is implemented and validated, the native-gradient route is the more auditable shared-model update.

The final combination happens only after each dataset has its own corrected row table. GROND first applies per-dataset endpoint sigma calibration when requested. It then estimates one overlap scale per dataset from common corrected reflections. This scale is applied after the Bloch/Born correction, so arbitrary raw integration scale differences are not mistaken for structure or thickness differences.

Corrected rows are then merged into a single endpoint HKL. The default merge policy is `chisq_overdispersion`: if several datasets contribute to the same merged reflection and they disagree more than their propagated sigmas predict, the merged sigma is inflated. This is the second sigma layer. The first layer makes each dataset internally self-consistent; the merge layer prevents a bad inter-dataset disagreement from entering SHELXL or another endpoint program with overconfident weights.

The files to read first are:

```text
dataset_policy/structure_update/POLICY.md
dataset_policy/final/POLICY.md
runs/<dataset>/final/summary.json
runs/combined/combined_summary.json
resolution_limits/combined/REPORT.md
endpoint_chooser/PUBLICATION_ENDPOINT_CHOOSER.md
```

The practical interpretation is simple. Per-dataset row metrics answer whether each crystal has a believable row-level dynamical correction. The combined endpoint answers whether the accepted datasets refine one shared model better than the individual-crystal alternatives. A lower residual from one single crystal should not automatically replace the combined endpoint; the combined endpoint is promoted when it passes the physical, sigma, and endpoint gates.
