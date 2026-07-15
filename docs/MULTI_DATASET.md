# Multi-Dataset Runs

GROND treats each crystal as its own row-level experiment until the last possible step. That is the safest default because thickness, frame offset, scan scale, detector-position scale, sigma behavior, row completeness, and integration quality are properties of an individual dataset. A weak crystal should not set the thickness or sigma scale for a good crystal, and a good crystal should not erase a weak crystal that contributes otherwise missing reflections.

The command-line entry point is still `GROND auto`. Put one row-preserving dataset per line in a prep file:

```bash
GROND prep \
  --structure model.res \
  --baseline-hkl model.hkl \
  --dataset crystal_a=/data/project/crystal_a/no_prf \
  --dataset crystal_b=/data/project/crystal_b/no_prf \
  --out prep_multi.json

GROND auto \
  --prep prep_multi.json \
  --out-dir grond_multi
```

That is the current gold-standard path. For XDS, give the directory that contains `INTEGRATE.HKL` plus `XPARM.XDS` or `GXPARM.XDS`; do not copy `INTEGRATE.HKL` away from its geometry sidecars. A direct `INTEGRATE.HKL` path is accepted only as shorthand when the XDS metadata files remain in the same parent directory. Do not concatenate `INTEGRATE.HKL` files, merge DIALS tables, or make one artificial HKL before GROND. The row-level Bloch prediction needs the signed row identity and the scan geometry from each dataset. The combined endpoint is made only after every accepted dataset has its own final `native_dyn_calc.csv`.

For each dataset, GROND runs the same row-contract checks and nuisance searches it uses for one crystal. Transform scouting, thickness, thickness-gradient, frame offset, row-resolution terms, absorption-like terms, Sg jitter, detector-position scales, and sigma behavior are fitted independently. Alternate-hand evidence is also measured per dataset, but hand is a property of the shared structure. Each hand comparison uses matched original and alternate scouts with the same selected/work/free row counts and work-only scale fitting; a switch requires a positive held-out set with finite raw free-row metrics. GROND changes the one shared hand only when every dataset independently passes those all-row and held-out gates. Weak, failed, missing-free, or mixed evidence keeps the input hand for every dataset. The selected shared structure, including its symmetry translations, supplies the atomic scattering factors for every downstream dataset, while the nuisance model remains dataset-specific.

After the per-dataset passes, `dataset_policy/structure_update/` decides which datasets are safe to use in the shared-model update. Primary datasets keep full row weight. Downweighted datasets remain in the row target, but their effective sigmas are inflated so they cannot dominate the model. Completeness-only datasets are excluded from row-level structure statistics and shared-model gradients, but they can still contribute endpoint reflections if they add real coverage. Excluded datasets are kept in the reports and left out of the combined endpoint.

The production shared-model update for multi-dataset jobs is the same lagged LS1 idea used for one dataset. GROND recomputes Bloch rows for every dataset from the current shared model, applies the dataset-policy row weights, merges the corrected rows into one endpoint HKL, runs one shared SHELXL least-squares update, then sends that updated model back through every dataset. The nuisance terms remain per dataset; only the atomic model is shared. The native finite-difference updater in `native_multi_structure_update.py` remains available with `--structure-update native_gradient`, but it is a slower diagnostic path rather than the default production route.

The final combination happens only after each dataset has its own corrected row table. GROND first applies per-dataset endpoint sigma calibration when requested. In the automatic multi-dataset path this defaults to `uniform_z_rms`, with a held-out floor so a dataset is not made overconfident just because its work rows fit well. It then estimates one overlap scale per dataset from common corrected reflections using model symmetry. Friedel opposites remain separate by default in acentric groups; the diagnostic `model_symmetry_friedel` mode pairs them. This scale is applied after the Bloch/Born correction, so arbitrary raw integration scale differences are not mistaken for structure or thickness differences.

The combined endpoint also audits how strongly those operations reshape individual intensities. It multiplies each row's Bloch, scale-field, and median-rescale correction by that dataset's fitted overlap scale, removes one shared global intensity gauge, and reports the 95th-percentile symmetric fold change as `combined_endpoint.mod_p95`. This makes relative scale disagreements visible without penalizing an arbitrary common scale. A missing value or one above `--publication-endpoint-max-modulation-p95` is diagnostic-only, and it cannot be fed into the next shared LS1/Bloch macrocycle.

Corrected rows are then merged into a single endpoint HKL before the shared LS1 model update. The default endpoint merge key is `model_symmetry`, which means symmetry-equivalent corrected observations are inverse-variance merged by GROND while Friedel opposites remain separate in acentric space groups. The diagnostic `model_symmetry_friedel` mode also merges Friedel mates, and `exact_signed_hkl` reproduces the older signed-row-only endpoint merge. The default merge-sigma policy is `chisq_overdispersion`: if several corrected endpoint rows contribute to the same written reflection and they disagree more than their propagated sigmas predict, the merged sigma is inflated. This is the second sigma layer. The first layer makes each dataset internally self-consistent; the merge layer prevents a bad inter-dataset disagreement from entering SHELXL or another endpoint program with overconfident weights. The combined endpoint report states both the overlap-scale key and the endpoint merge key so this step is auditable.

The shared corrected-HKL endpoint uses the same SHELXL contract as the single-dataset production endpoint. GROND writes a fresh HKLF4 file, strips EXTI from the generated endpoint input, forces one internal `L.S.` or `CGLS` cycle by default, and uses `WGHT 0 0` unless the user explicitly requests `--endpoint-wght-mode inherit` or `--endpoint-wght-mode promote` for a diagnostic run. This matters because the corrected HKL sigmas have already been transformed by GROND's per-dataset sigma calibration, merge overdispersion, and optional post-merge sigma calibration. Letting an old kinematic `WGHT` line or a SHELXL recommendation leak into the combined endpoint would mix two different uncertainty models.

The shared LS1 updater uses row-level weights from the per-dataset sigma model and the dataset-policy weight before the corrected rows are merged. This is the fast production route. If you request `--structure-update native_gradient`, the finite-difference shared-model updater uses the same per-dataset sigma and dataset-policy weights directly in the native row target. Use that mode when you specifically want a diagnostic check on whether the LS1 approximation is leaving a native row-level downhill direction.

Completed multi-dataset `auto` runs write `grond_continue_checkpoint.json`. Use it when you have manually edited the shared model, removed restraints, or want to push the same endpoint for more macrocycles without redoing transform scouting, hand checks, thickness search, or nuisance gates:

```bash
GROND push-endpoint \
  --checkpoint grond_multi/grond_continue_checkpoint.json \
  --start-res edited_joint_model.res \
  --out-dir grond_multi/push_edited_joint_model \
  --cycles 10
```

The continuation keeps the recorded per-dataset nuisance contracts and dataset-policy weights. Each push macrocycle recomputes Bloch rows for every dataset from the edited shared model, merges the corrected endpoint observations using the checkpoint merge key, runs one shared SHELXL LS step, and feeds that one model back to all datasets. The output report is `MULTI_ENDPOINT_PUSH_REPORT.md`; it lists the row-level dynamical and matched Born metrics, merged endpoint size, endpoint R values, GooF, and selected cycle for the push.

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
