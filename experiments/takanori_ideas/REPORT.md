
**Matrix exponential test**

This experimental branch adds two optional SciPy propagation backends to test Takanori's direct-exponential suggestion: `--compute-backend scipy_expm` and `--compute-backend scipy_expm_multiply`. Both compute the same wave propagation as the eigensolver within numerical tolerance. The new unit test `test_expm_multiply_backend_matches_eigensolver_on_hermitian_matrix()` checks this on a Hermitian complex matrix with thickness quadrature.

The first 370 real-row benchmark used `N=40`, `Q=5`, 120 rows, `thickness=1200 A`, `uniform_gradient`, `thickness_grad_frac=0.4`, three thickness points, and the no-profile-fitting 370 XDS rows. The SciPy eigensolver predicted 120 rows in 0.3958 s, or 303 rows/s. `scipy_expm` predicted the same rows in 0.4315 s, or 278 rows/s. `scipy_expm_multiply` predicted the same rows in 3.5817 s, or 33.5 rows/s. The maximum rowwise `dyn_raw` difference from the eigensolver was about `1e-14` for both exponential backends.

The larger forced-support benchmark used `N=128`, `Q=5`, 30 rows, and `max_support_sg=0.05` so every row reached a 128 by 128 matrix. The SciPy eigensolver predicted 30 rows in 0.5833 s, or 51.4 rows/s. `scipy_expm` predicted them in 0.8124 s, or 36.9 rows/s. `scipy_expm_multiply` predicted them in 9.1649 s, or 3.27 rows/s. Again, the values matched the eigensolver but were not faster on local SciPy CPU.
