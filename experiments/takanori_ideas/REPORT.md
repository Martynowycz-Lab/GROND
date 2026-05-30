# Takanori Technical Notes And GROND Checks

This branch-local report answers the technical questions raised in the May 2026 email thread using the current GROND implementation and old local 370 logs. It is intentionally experimental and is not part of the clean public `main` framing.

**What GROND currently does**

GROND currently works from integrated observation rows, not raw per-frame pixel intensities. For XDS, the row contract is `INTEGRATE.HKL` or `XDS_ASCII.HKL` with signed `h,k,l`, `IOBS`, `SIGMA`, calculated detector coordinates, calculated frame coordinate, observed coordinates when available, and quality fields such as `RLP`, `PEAK`, and `CORR`. The XDS header plus `XPARM.XDS` or `GXPARM.XDS` provide the cell, beam, rotation axis, oscillation range, and orientation needed to replay the row geometry.

For DIALS, `src/grond/io/dials_bridge.py` reads `integrated.refl` and `integrated.expt`. It keeps the signed `miller_index`, uses summation intensity by default, completes it as `intensity.sum.value * lp / partiality` in `sum_complete` mode, propagates the variance through the same factor, writes DIALS partiality as the GROND `PEAK` proxy, drops severe DIALS bad-row flags by default, and writes `DIALS_A_AT_SCAN_POINTS.txt` so the Bloch engine can reconstruct scan-varying reciprocal orientation. The bridge recomputes `Sg` from geometry and writes an `sg_validation` block rather than treating DIALS `zeta` as excitation error.

For each selected row, `XDSBlochPort.row_prediction_plan()` computes the center angle from the row frame coordinate, solves for the nearby Bragg angle where the target reflection has `Sg = 0`, selects a coupled-beam basis at that Bragg angle, creates target excitation-error quadrature points, solves each target-Sg point back to an angle, recomputes all basis-beam excitation errors at that angle, builds the Hamiltonian, and predicts both Bloch and matched Born intensities. The row regularization is then based on the ratio `Idyn/Iborn`.

**Rotation angle versus excitation error**

GROND already does the efficient thing Takanori asked about: the rocking profile is sampled in target excitation error, not directly in rotation angle. `row_profile_sg_points()` makes uniform, trapezoid, or Gauss-Legendre points in target `Sg`; `solve_omega_for_sg()` maps each requested target `Sg` value back to the corresponding rotation angle; `sg_for_hkls()` then recomputes every coupled beam's excitation error at that solved angle. The default is uniform target-Sg quadrature with summed sub-exposure weights.

**Hamiltonian convention**

The production Hamiltonian uses the off-diagonal coupling `U(h_j - h_i) / (2k)`. The alternate `U(h_i - h_j) / (2k)` convention remains as a diagnostic flag. Old raw-CBF 370 smoke logs under `/Users/Mike/Desktop/Code/GROND/outputs/raw_polish_370_smoke50` show that `j_minus_i` was better than `i_minus_j` on the matched raw-forward-model diagnostic. On the calibrated XDS-target rows in that smoke test, `j_minus_i` gave native `R1_dyn = 24.61%`, `wR2_dyn = 21.37%`, and free `wR2_dyn = 27.71%`; `i_minus_j` gave `R1_dyn = 26.06%`, `wR2_dyn = 23.81%`, and free `wR2_dyn = 34.37%`. On raw-complete rows, `j_minus_i` was also better: `R1_dyn = 37.20%` versus `46.81%`, and `wR2_dyn = 69.83%` versus `78.51%`. That is the local evidence behind keeping `j_minus_i` as the production default.

**Matrix exponential test**

This experimental branch adds two optional SciPy propagation backends to test Takanori's direct-exponential suggestion: `--compute-backend scipy_expm` and `--compute-backend scipy_expm_multiply`. Both compute the same wave propagation as the eigensolver within numerical tolerance. The new unit test `test_expm_multiply_backend_matches_eigensolver_on_hermitian_matrix()` checks this on a Hermitian complex matrix with thickness quadrature.

The first 370 real-row benchmark used `N=40`, `Q=5`, 120 rows, `thickness=1200 A`, `uniform_gradient`, `thickness_grad_frac=0.4`, three thickness points, and the no-profile-fitting 370 XDS rows. The SciPy eigensolver predicted 120 rows in 0.3958 s, or 303 rows/s. `scipy_expm` predicted the same rows in 0.4315 s, or 278 rows/s. `scipy_expm_multiply` predicted the same rows in 3.5817 s, or 33.5 rows/s. The maximum rowwise `dyn_raw` difference from the eigensolver was about `1e-14` for both exponential backends.

The larger forced-support benchmark used `N=128`, `Q=5`, 30 rows, and `max_support_sg=0.05` so every row reached a 128 by 128 matrix. The SciPy eigensolver predicted 30 rows in 0.5833 s, or 51.4 rows/s. `scipy_expm` predicted them in 0.8124 s, or 36.9 rows/s. `scipy_expm_multiply` predicted them in 9.1649 s, or 3.27 rows/s. Again, the values matched the eigensolver but were not faster on local SciPy CPU.

The immediate conclusion is that `expm_multiply` is not a CPU improvement for the current dense small-molecule row matrices, probably because SciPy's dense Hermitian eigensolver is already very efficient and because our thickness quadrature reuses one eigensystem across several thickness points. Takanori's GPU point may still be correct: a batched GPU `expm` or `expm_multiply` implementation could beat batched GPU eigensolvers, but the SciPy CPU path is not the place where this pays off.

**Strong/weak beam splitting**

There is no production Bethe-potential strong/weak beam split in the current GROND package, and a code/log search did not find a maintained implementation of that approach. The current `select_basis()` scores candidate support beams as `abs(U_g)^2 / (abs(Sg) + 1e-5)` and then builds one full dense matrix containing the direct beam, target beam, and selected support beams. That is simple and robust, but it is still dense diagonalization or dense exponential propagation.

Takanori's proposed split is a real algorithmic opportunity, not a cosmetic change. A careful implementation would classify strongly excited beams and weak beams per row and angle, fold weak beams into a Bethe potential perturbation for the strong block, and then run small `(M+1)` calculations for weak-beam target intensities. The validation target should be rowwise `q = Idyn/Iborn`, not just speed: a useful implementation must match the current dense calculation over a held-out set of 370 rows across small, medium, and large support bases before it is allowed into automatic mode.

**Whole rotation-range integration**

GROND does not use overlapping virtual-frame observations in the usual program-specific sense. It starts from the actual integrated XDS or DIALS row and simulates a finite rocking profile around that row's Bragg condition. The current profile width is a fixed or metadata-derived target-Sg window. It is not a whole-dataset rotation-range integral for every reflection. A whole-range mode is scientifically plausible for a future DIALS/raw-forward path, especially if the upstream observation is a reflection total over a known scan interval, but it is not an immediate replacement for the current row contract unless GROND also knows the exact integration limits, partiality ownership, and row/spot assignment that produced the measured intensity.

**Practical answer to send back**

The factual short answer is that GROND already integrates in excitation-error space and already recomputes all basis-beam `Sg` values at each solved angle. It uses integrated XDS/DIALS rows rather than raw per-frame intensities for the production path, with raw-CBF forward modelling kept as a slower diagnostic that has not yet beaten the integrated-row path. The production off-diagonal convention is `U(h_j - h_i)/(2k)`, and old 370 logs support that choice over the alternate sign. Direct matrix exponentials are now implemented on this experimental branch and are numerically correct, but SciPy `expm_multiply` is slower than the current eigensolver on real 370 row matrices. The strong/weak Bethe split is the most promising speed idea from the email, but it needs a real implementation and parity audit rather than a quick flag.
