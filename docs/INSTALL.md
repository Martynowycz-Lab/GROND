# Install

GROND is a Python package. Install it in a clean Python 3.11, 3.12, 3.13 or 3.14 environment; NumPy, SciPy and Gemmi come from wheels. Python 3.12 is a good default. The test workflow runs all four versions.

## What you need

- Python 3.11 to 3.14, and Git to clone the repository.
- For each crystal, an XDS directory with a summation `INTEGRATE.HKL`, or a DIALS `integrated.refl`/`integrated.expt` pair.
- A solved SHELX model (`.res` or `.ins`) with numeric electron scattering factors on its SFAC lines. A CIF works when it embeds the SHELX instructions (`_shelx_res_file`).
- The HKL file the model was refined against (`--baseline-hkl`). GROND uses it to order the indexing candidates, and the leak-free start refines the model against it with the free reflections removed. Without it the leak-free start is skipped with a warning, and the free-set R factors are optimistic.
- SHELXL, which `grond auto` uses to refine the model.
- Optional: `shelx2map`, for maps from the corrected HKL. The native maps of `grond finish` do not need it.
- Optional: PyTorch (`python -m pip install torch`), only for `--final-polish diffbloch` (off by default).
- Optional: DIALS, only to convert `.refl`/`.expt` files (`grond import-dials`) or to export PETS files (`grond export-pets`). After conversion GROND does not need DIALS.
- Optional: a LaTeX engine, for PDF tables. Markdown and TeX are written without it.

GROND does not run XDS, XDSCONV, SHELXT, DIALS or Olex2 during a calculation. It reads integrated observations and a solved model.

## Install with venv

```bash
git clone https://github.com/Martynowycz-Lab/GROND.git
cd GROND
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
grond init-config --out grond.config.json
# Add the SHELXL path if shelxl is not on PATH.
grond doctor --workflow small-molecule --config grond.config.json --probe-bloch --strict
```

Use `python3.11`, `python3.13` or `python3.14` if 3.12 is not available. The package installs both `grond` and `GROND` commands. If the shell does not find them, use `python -m grond`.

Install from the cloned repository as shown. The name `grond` on PyPI belongs to an unrelated package.

To update, pull and reinstall:

```bash
cd /path/to/GROND
git pull --ff-only
source .venv/bin/activate
python -m pip install .
grond doctor --workflow small-molecule --config grond.config.json --probe-bloch --strict
```

To reproduce a calculation made with a given release, check out that release tag or commit before installing. Do not copy an environment between macOS and Linux; build it on each machine.

## Developer install

```bash
git clone https://github.com/Martynowycz-Lab/GROND.git
cd GROND
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m grond validate
python -m pytest
```

See [DEVELOPMENT.md](DEVELOPMENT.md).

## Conda or mamba

```bash
mamba env create -f environment.yml
mamba activate grond
grond init-config --out grond.config.json
grond doctor --workflow small-molecule --config grond.config.json --probe-bloch --strict
```

`environment.yml` installs Python 3.12 and GROND in editable mode with the development tools.

## Tool configuration

Write a config file once and fill in absolute paths for tools that are not on `PATH`:

```bash
grond init-config --out grond.config.json
```

```json
{
  "tools": {
    "shelxl": "/absolute/path/to/shelxl",
    "shelx2map": ""
  }
}
```

Pass it with `--config grond.config.json`, or set `GROND_CONFIG=/path/to/grond.config.json`. Without either, GROND looks for `grond.config.json` (or `.yaml`) in the current directory and its parents, then for `~/.config/grond/config.json` and `~/.grond.json`. YAML configs need PyYAML (`python -m pip install PyYAML`, or `python -m pip install ".[config]"` from the cloned repository).

Check what GROND finds:

```bash
grond doctor --config grond.config.json --probe-bloch
grond endpoints --config grond.config.json
```

`grond doctor` checks the core by default. `--workflow small-molecule` adds SHELXL, `--workflow native-maps` checks map regeneration (no external program), and `--workflow endpoint-maps` adds SHELXL and `shelx2map`. The template leaves out the macromolecular endpoint adapters; `grond init-config --with-endpoint-adapters` adds them ([MACROMOLECULES.md](MACROMOLECULES.md)).

`grond auto` stops at once if it needs SHELXL and cannot find it. It writes no maps by default; `grond finish --run-dir grond_out` writes the native maps afterwards, and `--make-maps` (auto) or `grond finish --map-mode shelx2map` adds the `shelx2map` maps.

## Long runs on a server

Build the environment and the config on the server itself, check it, and run inside `tmux`, a scheduler or another process supervisor:

```bash
cd /path/to/GROND
source .venv/bin/activate
grond doctor --workflow small-molecule --config /absolute/path/grond.config.json --probe-bloch --strict

tmux new -s grond
grond auto \
  --config /absolute/path/grond.config.json \
  --prep /absolute/path/prep.json \
  --out-dir /absolute/path/grond_out \
  --workers 48
```

Detach with `Ctrl-b d` and return with `tmux attach -t grond`. If the run is interrupted, repeat the same command with the same `--out-dir`; completed stages are reused. Do not start a second run in the same output directory while the first is running.

- `--workers 0` (the default) uses 85% of the detected cores (`--max-core-fraction`).
- Prediction caches go to `OUT/cache` (`--prediction-cache-dir` to change it). Other generated caches go to the user cache directory, or to `GROND_CACHE_DIR` when it is set.
- GROND refuses a dense potential array whose estimated peak memory exceeds 2 GiB; set `GROND_MAX_DENSE_UG_GIB` to raise the limit.

## Preparing the inputs

- XDS: solve the structure by your usual route, then integrate the same frames again with `PROFILE_FITTING= FALSE` and `MINPK= 90.0` in a copy of the XDS directory, and give that directory as `--xds-dir`. See [XDS_WORKFLOW.md](XDS_WORKFLOW.md).
- DIALS: keep the unmerged `integrated.refl` and `integrated.expt`, and pass the DIALS Python with `--dials-python` if DIALS has its own environment. See [DIALS_WORKFLOW.md](DIALS_WORKFLOW.md).
- Other integration programs: write the portable `grond-row-v1` row table and metadata file and convert them with `grond row-validate --write-xds-dir`. See [PORTABLE_INPUTS_AND_ENDPOINTS.md](PORTABLE_INPUTS_AND_ENDPOINTS.md).
