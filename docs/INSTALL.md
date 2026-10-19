# Install

## Requirements

- Linux or macOS, Python 3.11 to 3.14. NumPy, SciPy and gemmi come from wheels.
- SHELXL (SHELXL-2019/3 was used for the paper).
- Optional: `shelx2map` (CCP4) for the SHELXL Fo−Fc maps written by `grond finish`. The dynamical maps need no
  external program.
- Optional: JAX with CUDA (`python -m pip install ".[gpu]"`), only for `--shared-grid-gpus` on large cells.
- Optional: DIALS, only for `grond import-dials` and `grond export-pets`. GROND runs the DIALS Python in a
  subprocess (`--dials-python`), so DIALS can keep its own environment.

GROND does not integrate images or solve structures. It reads integrated observations and a solved model.

## Install

```bash
git clone https://github.com/Martynowycz-Lab/GROND.git
cd GROND
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The PyPI package called `grond` is unrelated; install from this repository. If the shell does not find `grond`, use
`python -m grond`.

## Tool configuration

```bash
grond init-config --out grond.config.json
```

```json
{
  "tools": {
    "shelxl": "/absolute/path/to/shelxl",
    "shelx2map": "/absolute/path/to/shelx2map"
  }
}
```

Pass the file with `--config`, or set `GROND_CONFIG`. Without either, GROND looks for `grond.config.json` in the
current directory and its parents, then `~/.config/grond/config.json`. Then check the installation:

```bash
grond doctor --workflow small-molecule --config grond.config.json --probe-bloch
```

## Long runs

Run long jobs inside `tmux` or a scheduler. If a run stops, repeat the same command with the same `--out-dir`; GROND
reuses the finished stages. Do not start two runs in one output directory.

- `--workers` sets the number of worker processes. The default uses 85% of the cores. Each worker uses one BLAS
  thread: GROND sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1 unless they are set.
- Prediction caches go to `OUT/cache` (`--prediction-cache-dir`). Other caches go to `~/.cache/grond`, or to
  `GROND_CACHE_DIR` when it is set.
- GROND refuses a dense U_g array larger than 2 GiB. `GROND_MAX_DENSE_UG_GIB` raises the limit.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests
```
