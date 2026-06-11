# Experiments

This directory holds benchmark probes, negative controls, and prototype backends that informed GROND's production defaults but are not part of the installed package. The source distribution and wheel intentionally exclude `experiments/`; users get the stable CLI and engine code under `src/grond`.

Keep a prototype here when it is useful for reproducibility but not ready to be a default. Promote code into `src/grond` only after it matches the SciPy row-intensity reference, improves a full-dataset benchmark or a documented workflow, and has tests that protect the behavior.

Current subdirectories:

- `takanori_ideas/`: strong/weak Bethe-Schur probes and comparison scripts inspired by suggestions from Takanori Nakane.
- `mlx_backend/`: Apple MLX/Metal experiments. These remain diagnostic because they did not beat the production SciPy path on the tested 370 workload.
- `performance/`: local and server benchmark scripts/results used to decide whether SciPy, Bethe-Schur, JAX, or MLX should be the default.
