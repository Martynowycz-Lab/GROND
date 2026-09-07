# Benchmarks and experimental approaches

GROND's reference is the complete row calculation with dense SciPy propagation. Timing an eigensolver alone is insufficient: geometry, beam selection, potential construction, profile integration, thickness averaging, scale fitting and endpoint work all contribute. The measurements below are development results on specified frozen contracts. They are not runtime promises or validation on new chemistry.

The source repository contains regression tests for portable behavior. Dataset-derived results and exploratory harnesses live in separate research archives under the [data policy](../DATA_POLICY.md). Those archives preserve input hashes, source commits, commands, numerical settings and negative results. Repeated proposals are indexed by the same experimental question rather than counted as independent confirmations. The current package tests do not rerun those historical data campaigns.

## Exact improvements retained

| Approach | Evidence | Scope |
|---|---|---|
| Analytic fixed-XDS excitation roots and angular derivatives | Complete 256-row N64/Q20 predictions were 1.91–1.94 times faster on two contracts; identical supports and maximum relative intensity difference below `9e-9`. | Fixed rotation geometry. Scan-varying and fitted geometry retain the numerical solver. |
| Sparse-active Fourier potential with fixed row plans | Complete moved-model evaluation improved 1.68–1.92 times with byte-identical predictions. | Avoids the dense potential cube when the exact active differences are known. |
| Batched thickness candidates | Propagation improved 3.68–3.72 times with differences below `9e-16`. | Reuses each eigensystem over all candidate thicknesses. |
| Packed scalar/Powell thickness response | All-eligible-row stages improved 1.61–1.93 times; 500-row stages improved 2.71–3.61 times. | Bounded memory, fixed Hamiltonian and retained scalar fallback. The corrected grid includes its declared boundaries. |
| Negative-thickness hand pairing | Agrees with explicit inversion to numerical precision. | Shared elastic Hermitian calculation; not a non-Hermitian absorption shortcut or external sign calibration. |
| Uniform scalar absorption factorization | A 512-row control changed from 16.0 to 3.24 seconds with approximately `1e-13` relative agreement. | Only `H = H_elastic + i mu I`; not beam-dependent or off-diagonal loss. |

Exact row-plan caching is retained with a reuse threshold. Small grids can be slower after serial cache construction, and moving coordinates or ADPs invalidate an exact model-bound plan. Explicit cross-model support freezing is a different, diagnostic contract.

One historical 3,786-row CPU experiment improved from 252.1 to 4.74 seconds between one and 96 workers with reused geometry, with identical predictions. Geometry rebuilding changed from 408.9 to 6.57 seconds. These were three-repeat measurements on an AMD EPYC 7742 with controlled BLAS threads. They establish scaling on that machine and contract, not universal linear scaling.

## Approximate and external approaches

| Approach | Decision and reason |
|---|---|
| Bethe/Schur strong–weak reduction | Keep opt-in for genuinely large bases. Forced 512-beam tests gave 10.6–12.4 times speedup at 0.30–0.40% intensity NRMSE; ordinary adaptive bases often already fit inside the strong block. Require dense sentinels. |
| Physical-optics propagation for a 5x5 mosaic cone | Promising external prototype. On 128 rows per contract, complete plan-plus-propagation gains were 1.90 and 2.09 times, with p95 absolute log-intensity errors `3.72e-5` and `3.04e-5`. Full-row width selection, hand and endpoint parity remain untested. |
| Physical-optics propagation for ordinary rows | Not promoted. The accurate five-thickness calculation improved only 1.26–1.38 times; the fast approximation missed the error gate. |
| Native analytic adjoint | Promising research direction. Bounded derivatives were 14.7–20.4 times faster than 24 central differences, with relative gradient errors `7e-6`–`3.5e-5`. This does not accelerate an external SHELXL LS1 call. |
| Exact single-atom potential update | About 21 times faster when one atom moves, but slower when all atoms move. Useful for a native derivative implementation, not ordinary multi-atom LS1 updates. |
| Detector fields | Fit and validate prospectively against a detector-free control. Fine grids can lower endpoint R while failing held-out support and physics-preservation checks. No detector-response claim follows from a stable fitted surface alone. |
| Common-basis incoherent mosaic cone | Retain as a default-off physical extension. Some held-out residuals improve, others do not, and the tested cost is about tenfold. A fitted width is not an independently measured mosaic distribution. |
| diffBloch endpoint adapter | Interoperability works; tested coordinate/ADP candidates were not promoted. Improvement in the external objective did not transfer robustly to the exact GROND rows and endpoints. See [the adapter contract](DIFFBLOCH_ENDPOINT.md). |
| JAX/CUDA/Metal | Experimental. An earlier three-GPU CUDA calculation agreed numerically but was slower than ten CPU workers. Later hardware failures prevented a separate fused-pipeline experiment. These are historical snapshots, not a current hardware diagnosis. |

## Approaches that did not earn a production role

Adjacent agreement is not a safe beam-count or quadrature stopping rule. In 128-row tests, low-support neighbors falsely accepted 17–22% of rows for beam count and 5–7% for quadrature at a `1e-3` log-q tolerance. Evaluating the entire ladder cost about 3.1–3.2 times one final calculation. A cheap scout can suggest a basin; it cannot certify its own final correction.

Partial eigenspectra, warm-start Arnoldi, generic Krylov action, FFT embeddings, H-matrices and tensor compression did not beat dense propagation with acceptable target intensities on ordinary supports. Entrywise pruning retained 98–99% matrix density at useful accuracy. Changing entries in a dense matrix does not remove the cubic eigensolve. RK4 remains a possible large-basis validation path, not a demonstrated small-molecule default. LAPACK driver and stacked-matrix gains were modest and platform-dependent.

Cheap 3x3 or cross-shaped mosaic rules did not reproduce the 5x5 reference. Strong-row-only thickness scouts selected different aliases from broad/all-row fits. Quasi-two-beam scouts, truncated dominant modes and scale-insensitive q fingerprints failed basin or q-field agreement. A coreset fitted to the complete objective curve compresses an already known result; it does not supply a prospective shortcut.

Broad thickness searches can improve residuals by dephasing coherent cross terms while leaving the mean poorly identified. Work/free aliases and changing q fields support reporting an effective thickness distribution, not a precise physical thickness. Exact phase recurrences and specialized batched scale fits may still save time, but their large microbenchmark gains do not yet cover the detector fields and priors used by every production route. See [thickness interpretation](THICKNESS_SCOUT.md).

A new propagation notation alone cannot repair a mismatched observation model. Raw-frame likelihoods, free specimen-shape fields, learned coherent modes, explicit inelastic channels and learned thickness models require calibrated observations or independent priors that ordinary integrated rows do not retain. Ionic or multipolar potential corrections remain interesting for targeted charge/protonation science; the tested held-out gain was too small to identify the general residual floor.

## What a new experiment must establish

Record the source commit, input hashes, immutable signed rows, model, geometry, N/Q and actual basis sizes, thickness/profile contract, scale and sigma families, backend, workers, cache state, hardware and complete wall time. An exact optimization must preserve the numerical observable. An approximation needs a declared error bound and full-reference checks across thickness, orientation and both hands. A physical extension must improve the intended row objective without purchasing the gain through unsupported flexibility.

Keep row residuals separate from SHELXL endpoint R factors. Recheck model health, corrected-HKL provenance, frame/detector parameter counts, weights and maps after any proposed structural improvement. Fixed-q, q=1 and permuted-q controls test whether the assigned correction matters. Synthetic LS1 tests support a useful surrogate but also show that direct dynamical optimization can remain better; shared-kernel synthetic recovery is not an independent physics implementation.

Small held-out subsets answer a narrower question than the complete fitted result. A conditional ten-percent hand experiment retained the full fitted predictions and reduced the nominal count contrast approximately by `sqrt(0.1)`, as expected; many subsets fell below the automatic inversion threshold. It was a power diagnostic, not refitting validation, and does not justify vetoing valid all-row hand evidence. See [hand-result policy](HAND_RESULT_POLICY.md).
