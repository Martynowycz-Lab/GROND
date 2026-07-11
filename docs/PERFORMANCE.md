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

`--compute-backend scipy_bethe` is the opt-in strong/weak-beam Schur path. It can save substantial time when rows genuinely require large matrices, but gives little benefit when adaptive support already keeps most matrices small. Agreement depends on the strong-beam count and promotion thresholds, so new data should be checked against dense SciPy before using it for a final result.

The JAX, CUDA, Apple-Metal, and real-embedded matrix-exponential backends are experimental. GROND raises an error when a requested backend cannot perform the required calculation; it does not silently replace that backend with SciPy. These paths should be judged by complete row intensities and endpoint products, not by an isolated matrix microbenchmark.

Development benchmarks and hardware-specific result files stay outside the public repository. The [JAX backend notes](EXPERIMENTAL_JAX_BACKEND.md) explain the retained numerical contracts and promotion gates without depending on a private dataset workspace. Any measured timing is evidence for a development decision, not a runtime promise for another system.

## Caches And Storage

Prediction caches belong to a run, not to the installed package. `GROND auto` writes them under the output directory unless `--prediction-cache-dir` is supplied. Standalone engine paths that need a reusable Fourier-potential cache use `GROND_CACHE_DIR` when it is set, then the operating-system user cache directory. On a server, point `GROND_CACHE_DIR` at fast scratch storage if the home filesystem is slow or quota-limited.

Installed package directories should remain read-only. Do not put run caches, model outputs, or benchmark arrays inside `src/grond`.

The optional row-plan cache is enabled with `--row-plan-cache readwrite`. It freezes row identity, support, geometry, and quadrature while allowing structure factors to change between feedback cycles. Cache keys include the source rows and numerical contract, and incompatible entries are rebuilt. Current whole-run benchmarks have not shown a consistent enough improvement to make this the default, so use it as an explicit benchmark or LS1-development tool.

## Cheap Post-Processing

Scale compression, endpoint selection, and resolution-limit reporting operate on completed row tables and summaries. They do not repeat the Bloch calculation. Multi-dataset inclusion decisions and post-dynamical overlap scaling are similarly inexpensive compared with row prediction. If a completed run needs only refreshed reports or maps, use `GROND finish --run-dir ...` instead of rerunning `GROND auto`.

When comparing performance changes, record the GROND commit, input hashes, selected rows, N/Q contract, actual basis-size distribution, backend, worker count, cache state, CPU or accelerator model, and wall time. A faster run is promotable only when the row intensities, residuals, corrected reflections, and downstream maps still agree with the SciPy reference at the declared tolerance.
