# Install

GROND is a normal Python package. The recommended installation is a clean Python 3.11 or 3.12 environment with NumPy, SciPy, Pandas, and Gemmi installed from wheels.

## Prerequisites

- Python 3.11 or 3.12.
- Git, if installing from GitHub.
- A working XDS processing directory, or a DIALS `integrated.refl`/`integrated.expt` pair, for each dataset you want to regularize.
- A solved starting structure, usually SHELX `model.res` or CIF.
- A baseline solved-model HKL file when available. This is strongly recommended because GROND uses it to reduce transform ambiguity before running Bloch scouts.
- Optional: SHELXL for corrected-HKL endpoint polishing/statistics.
- Optional: `shelx2map` for map generation.
- Optional: an Olex2 app installation or a configured custom endpoint if you are deliberately testing a non-default model-update adapter.
- Optional: a LaTeX engine if you want PDF tables; Markdown and TeX outputs are written without it.

Native row-level Bloch calculations are Python-first. The full default endpoint path needs SHELXL, and maps need `shelx2map`. Those programs are not bundled with GROND.

## Fresh Install With venv

```bash
git clone https://github.com/Martynowycz-Lab/GROND.git
cd GROND
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
GROND doctor --probe-bloch
```

If `python3.12` is not available but `python3.11` is, use `python3.11 -m venv .venv`.

## Developer Install

Use the editable install when modifying GROND:

```bash
git clone git@github.com:Martynowycz-Lab/GROND.git
cd GROND
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
GROND validate
pytest
```

## Conda Or Mamba

A conda or mamba environment is useful on shared servers:

```bash
mamba env create -f environment.yml
mamba activate grond
python -m pip install -e ".[dev]"
GROND doctor --probe-bloch
```

The package also works in a plain `venv`, which is usually simpler on a laptop.

## External Tool Config

For shared servers and teaching machines, prefer a small config file over relying on shell startup files:

```bash
GROND init-config --out grond.config.json
```

Fill in absolute paths for tools that are not already on `PATH`:

```json
{
  "tools": {
    "python": "/data/envs/grond/bin/python",
    "shelxl": "/programs/x86_64-linux/system/sbgrid_bin/shelxl",
    "shelx2map": "/programs/x86_64-linux/system/sbgrid_bin/shelx2map",
    "shelxt": "",
    "xds": "",
    "xdsconv": "",
    "olex2_app_dir": "",
    "olex2_console": "",
    "custom_refiner": ""
  },
  "endpoints": {
    "olex2_app": {
      "app_dir": "",
      "note": "Optional Olex2.app console directory, for example /Applications/olex2.app/Contents/MacOS."
    },
    "custom": {
      "command": "",
      "output_model": "",
      "note": "Optional command template using {model}, {reflections}, {out_dir}, and {cycles}."
    }
  }
}
```

Check the resolved environment:

```bash
GROND doctor --config grond.config.json --probe-bloch
GROND endpoints --config grond.config.json
```

Use the config explicitly:

```bash
GROND auto --config grond.config.json --xds-dir xds --structure model.res --baseline-hkl model.hkl --out-dir grond_out
```

or export it for a project:

```bash
export GROND_CONFIG=/path/to/grond.config.json
```

`GROND auto` stops early if a full endpoint workflow needs SHELXL or map generation needs `shelx2map` and neither is configured nor discoverable on `PATH`. Use `--native-only` for row-level Bloch statistics without corrected-HKL endpoint polishing, `--no-maps` for endpoint statistics without maps, or `--allow-missing-tools` only for an intentional incomplete diagnostic run.

YAML configs are accepted when `PyYAML` is installed with `python -m pip install "grond[config]"`; JSON works with no extra dependency.

The supported model-update endpoints today are SHELXL, an installed Olex2 app console, and a custom command template. SHELXL is the production default for full automatic feedback. Olex2 app support expects the app console directory, for example `/Applications/olex2.app/Contents/MacOS` on macOS; GROND will run the app console directly and set the app library path for that process. The custom endpoint is for labs that can provide a tested one-cycle command and a declared output model path. If none of those are available, use `--native-only`: GROND will still run the Bloch-wave simulation, thickness and nuisance decisions, residual diagnostics, corrected-HKL exports, and polarity checks from the supplied model, but it will not perform the external least-squares model-update loop.

## XDS Input Preparation

Most users should prepare two products from the same raw data:

1. A conventional solved structure path used to obtain `model.res` or `model.cif`, usually from XDS profile-fitted integration, XSCALE, XDSCONV, SHELXT, and initial SHELXL regularization.
2. A GROND input integration with no profile fitting, usually in a separate `noprf/` or `grond_input/` directory.

For the GROND input integration, edit `XDS.INP` in the copied processing directory:

```text
PROFILE_FITTING= FALSE
MINPK= 90.0
```

Then rerun XDS integration/correction in that copied directory. Keep the resulting `INTEGRATE.HKL` together with the XDS metadata files in the same folder. GROND should be pointed at that folder with `--xds-dir`; do not point it at a profile-fitted integration unless you are deliberately running a control.

The baseline solved-model HKL from the conventional path should be passed with `--baseline-hkl` when available. It helps GROND prefilter indexing transforms cheaply before spending time on Bloch-wave transform scouts.

More detail is in [XDS_WORKFLOW.md](XDS_WORKFLOW.md).

## DIALS Input Preparation

DIALS users should keep the row-preserving `integrated.refl` and matching `integrated.expt` from `dials.integrate`. GROND uses `intensity.sum.value` by default, so DIALS profile fitting does not need to be disabled, although `integration.profile.fitting=False` makes a cleaner summation-only table when that is what you want to test. Do not merge the data before GROND and do not rely on an exported HKL file when the `.refl` and `.expt` files are available. For electron-diffraction CBF data, check `dials_bridge_summary.json` after import: GROND records the empirical sigma calibration, how many DIALS quality-flagged rows were removed, and whether the scan-varying geometry reproduces near-zero predicted-row excitation errors.

If DIALS is installed in a separate environment, pass its Python executable with `--dials-python`:

```bash
GROND auto \
  --dials-refl /data/crystal/integrated.refl \
  --dials-expt /data/crystal/integrated.expt \
  --dials-python /path/to/dials/python \
  --structure /data/crystal/model.res \
  --baseline-hkl /data/crystal/model.hkl \
  --out-dir grond_dials_out
```

More detail is in [DIALS_WORKFLOW.md](DIALS_WORKFLOW.md).

## Portable Row Input

If your integration did not come from XDS or DIALS, write the portable `grond-row-v1` row table and metadata sidecar described in [PORTABLE_INPUTS_AND_ENDPOINTS.md](PORTABLE_INPUTS_AND_ENDPOINTS.md):

```bash
GROND row-template --rows-out rows.csv --metadata-out rows_metadata.json
GROND row-validate --rows rows.csv --metadata rows_metadata.json --write-xds-dir portable_bridge
```

The resulting `portable_bridge/INTEGRATE.HKL` is a GROND compatibility file, not an XDS claim. It carries the signed rows and geometry in the shape that the current engine consumes.
