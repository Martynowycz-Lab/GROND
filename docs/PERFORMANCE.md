# Performance

GROND spends most of its time constructing and solving row-level Bloch matrices. The best speed improvements come from avoiding unnecessary candidate calculations, choosing an appropriate basis, and keeping multiprocessing from competing with threaded numerical libraries.

## Start With The Default Scheduler

`--workers 0` chooses a conservative fraction of the visible CPU cores. GROND uses process-level row parallelism and limits helper threads inside NumPy and SciPy to reduce oversubscription. Start with the automatic setting, then set `--workers` only after checking memory use, storage speed, and whether the machine is shared.

The Fourier-potential grid is dense. Before building or loading it, GROND estimates peak memory and refuses work above a 2 GiB default ceiling. The build estimate reserves 256 bytes per grid point, including the 16-byte complex result and the vectorized work arrays. If a large unit cell reaches this guard, increase `ug_d_min` first. Override `GROND_MAX_DENSE_UG_GIB` only after checking the available memory on the machine that will run the calculation.

Transform scouting has an outer job queue. By default, candidate transforms are distributed across the available worker budget and each scout subprocess receives one row worker. If the transform is already known from a well-recorded integration and solution path, an explicit `--hkl-transform` with `--trust-hkl-transform` skips that scout without disabling the other automatic checks.

For a shared server, keep the product of `--transform-scout-jobs` and `--transform-scout-workers-per-job` near the number of cores allocated to the job. Giving every concurrent scout the full core count makes the calculation slower and can disrupt other work on the node.

## Choose The Right Run Profile

`--fast` is a preflight and plausibility run. Use it to catch path, indexing, hand, integration, or gross thickness problems.

`--full-fast` keeps the automatic decision path and performs a lightweight structure update, but uses a deliberately smaller final Bloch calculation and a shorter endpoint path. It is useful for getting an intermediate model and a realistic runtime estimate. It is not the final high-support calculation.

Plain `GROND auto` runs the full endpoint path. Before the expensive final prediction, it narrows transform and hand candidates, scans the row support actually required by the selected data, and chooses a final basis with a safety margin. The support pool follows the selected data resolution so high-resolution observations are not evaluated against a shallower coupled-beam pool.

Structure-update blocks have a plateau stop. Leave it enabled for routine work; disabling it can spend a long time repeating coordinate or displacement-parameter probes after the data residual has stopped improving.

## Backends

Dense SciPy `eig` and `eigh` are the reference implementation and the default. They are the first choice for an unfamiliar dataset and the comparison standard for every accelerated backend.

`--compute-backend scipy_bethe` is the opt-in strong/weak-beam Schur path. The
signed target reflection is always retained in the strong block; weak beams
renormalize that block through a Schur/Bethe correction. It can save
substantial time when rows genuinely require large matrices, but gives little
benefit when adaptive support already keeps most matrices small. Agreement
depends on the strong-beam count and promotion thresholds, so new data should
be checked against dense SciPy before using it for a final result. This is the
GROND form of the strong-plus-target divide-and-conquer calculation; it does
not restore weak-to-weak multiple-scattering paths removed by the reduction.

On the 256-row 370 development contract at N=128/Q=20, the adaptive bases had
a median of 43 beams and a maximum of 74. A 48-beam Bethe block changed raw
intensities by at most 6.52e-5 relative and a 32-beam block by at most 2.16e-4,
but neither shortened the approximately 6.4 s whole calculation. Most rows
were already at or below the reduced matrix size, and process/setup overhead
dominated the few larger solves. This backend is therefore reserved for large-
matrix datasets until a full-reference sentinel demonstrates both agreement
and an end-to-end gain.

`scipy_expm_multiply` remains an agreement probe rather than a speed default.
On the 256-row 370 development contract described below it reproduced the
dense Hermitian result to numerical precision but took 24.90 s rather than
6.44 s with the warm reference eigensolver. Matrix exponential action may
still be useful for much larger sparse matrices, but GROND's present matrices
are dense and its thickness/profile quadrature reuses a complete Hermitian
eigensystem efficiently.

The JAX, CUDA, Apple-Metal, and real-embedded matrix-exponential backends are experimental. GROND raises an error when a requested backend cannot perform the required calculation; it does not silently replace that backend with SciPy. These paths should be judged by complete row intensities and endpoint products, not by an isolated matrix microbenchmark.

Development benchmarks and hardware-specific result files stay outside the public repository. The [JAX backend notes](EXPERIMENTAL_JAX_BACKEND.md) explain the retained numerical contracts and promotion gates without depending on a private dataset workspace. Any measured timing is evidence for a development decision, not a runtime promise for another system.

## Caches And Storage

Prediction caches belong to a run, not to the installed package. `GROND auto` writes them under the output directory unless `--prediction-cache-dir` is supplied. Standalone engine paths that need a reusable Fourier-potential cache use `GROND_CACHE_DIR` when it is set, then the operating-system user cache directory. On a server, point `GROND_CACHE_DIR` at fast scratch storage if the home filesystem is slow or quota-limited.

Installed package directories should remain read-only. Do not put run caches, model outputs, or benchmark arrays inside `src/grond`.

`grond auto` uses `--row-plan-cache auto`. The cache key includes the
structural-model content and RES hash, so a coordinate or ADP change selects a
different cache. Moving-model updates resolve `auto` to `off`. Fixed-model
stages resolve it to `exact` only when at least 16 thickness or absorption
predictions share one geometry key. Smaller searches stay uncached because the
cold row-plan builder is serial and can cost more than it saves. The decision,
reuse estimate, and threshold are recorded in the row summary.

On a 256-row 370 development contract at N=128/Q=20, a first exact-cache run
took 11.13 s and the repeated calculation took 5.98 s. The uncached reference,
cold exact-cache, and warm exact-cache row CSV files were byte-identical. A
six-candidate grid with only two predictions per geometry key was slower with
the cache (44.12 s rather than 35.14 s), which is why automatic mode uses the
conservative reuse threshold instead of enabling the cache for every grid.
These measurements demonstrate the decision boundary, not a universal runtime
claim.

The older cross-model behavior is retained as the explicit
`--row-plan-cache readwrite` mode for finite-difference development. It freezes
the support selected by the builder model while recomputing current structure
factors. Use it only when that approximation is intended and audited; `auto`
never selects it.

## Cheap Post-Processing

Scale compression, endpoint selection, and resolution-limit reporting operate on completed row tables and summaries. They do not repeat the Bloch calculation. Multi-dataset inclusion decisions and post-dynamical overlap scaling are similarly inexpensive compared with row prediction. If a completed run needs only refreshed reports or maps, use `GROND finish --run-dir ...` instead of rerunning `GROND auto`.

When comparing performance changes, record the GROND commit, input hashes, selected rows, N/Q contract, actual basis-size distribution, backend, worker count, cache state, CPU or accelerator model, and wall time. A faster run is promotable only when the row intensities, residuals, corrected reflections, and downstream maps still agree with the SciPy reference at the declared tolerance.
