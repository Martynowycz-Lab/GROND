# Macromolecular Endpoint Layer

GROND's row-level Bloch calculation still starts from signed observations. That part does not change for proteins, peptides, or larger assemblies. What changes is the endpoint object. Small-molecule GROND writes corrected HKLF-style files and normally updates the model through SHELXL. A macromolecular endpoint needs a corrected MTZ, a fixed unique-reflection FreeR set, bulk-solvent handling in the endpoint refiner, and a model file that can be read again after one controlled macrocycle.

This page describes the implemented wrapper layer. It is an interface layer, not a claim that protein GROND is already a solved default workflow.

## Two Validation Layers

GROND row-free and macromolecular Rfree are different validation objects. Row-free is a signed-row split used to test whether the row-level dynamical correction transfers to held-out observations. It is the right split for Born controls, frozen-q controls, shuffled-q controls, wrong-polarity controls, and row-level residual diagnostics.

Macromolecular Rfree is assigned at the unique-HKL level and carried into a merged MTZ. Every signed observation that maps to a free unique reflection must keep that free label through correction and merging. That is what conventional macromolecular refiners expect, and it must not be regenerated after GROND has corrected the observations.

## Create Fixed FreeR Flags

Use `make-free-flags` before any corrected-MTZ export. The command reduces signed rows to unique HKLs using the declared space group, optionally merges Friedel mates, balances flags approximately by resolution shell when d-spacings are available, and can propagate the flag back to every signed row.

```bash
GROND make-free-flags \
  --rows grond_rows.csv \
  --cell "50.1 61.2 72.3 90 90 90" \
  --space-group "P 21 21 21" \
  --fraction 0.05 \
  --min-free-count 500 \
  --out free_flags.csv \
  --write-rows grond_rows_with_free.csv
```

If the model file already contains cell and space-group metadata, use `--model model.pdb` instead of `--cell` and `--space-group`.

The default is `--merge-anomalous`, which assigns one FreeR label to Friedel mates. Use `--no-merge-anomalous` only when the endpoint refinement and validation plan intentionally keep anomalous mates separate.

## Export Corrected MTZ

`export-mtz` merges corrected GROND rows into unique reflections and writes `IMEAN`, `SIGIMEAN`, `FreeR_flag`, and `NOBS`. Negative corrected intensities are preserved. `F/SIGF` are not the primary output because the safest macromolecular handoff is still intensity-first.

```bash
GROND export-mtz \
  --row-table native_dyn_calc.csv \
  --free-flags free_flags.csv \
  --cell "50.1 61.2 72.3 90 90 90" \
  --space-group "P 21 21 21" \
  --out-mtz corrected_grond.mtz \
  --out-dir corrected_mtz_export
```

The exporter can apply a first-pass low-resolution q damping policy for protein tests:

```bash
GROND export-mtz \
  --row-table native_dyn_calc.csv \
  --free-flags free_flags.csv \
  --model model.pdb \
  --out-mtz corrected_grond_damped.mtz \
  --q-policy low_resolution_damped \
  --q-off-d-A 5.0 \
  --q-full-d-A 3.0
```

This does not put bulk solvent into the Bloch Hamiltonian. It simply prevents a strong ordered-model q correction from dominating the lowest-resolution region where bulk solvent, background, beam-stop effects, disorder, and anisotropic scale are entangled. The macromolecular endpoint refiner should still handle bulk solvent in its own target.

## Run One Protein Endpoint Macrocycle

`protein-ls1` is a command-template wrapper. It runs one external endpoint macrocycle against the corrected MTZ and records the exact command, stdout, stderr, timing, and declared output model. It does not guess Phenix, REFMAC, or Servalcat syntax because labs often need local restraint dictionaries, electron-scattering settings, solvent choices, and cluster paths.

```bash
GROND protein-ls1 \
  --model start_model.pdb \
  --mtz corrected_grond.mtz \
  --out-dir protein_cycle_001 \
  --command "phenix.refine {model} {mtz} main.number_of_macro_cycles={cycles} refinement.input.xray_data.free_r_flags.label={free_label} output.prefix={out_dir}/phenix_cycle" \
  --output-model "{out_dir}/phenix_cycle_001.pdb" \
  --cycles 1
```

For repeated use, put the command template in `grond.config.json`:

```json
{
  "tools": {
    "phenix_refine": "/programs/x86_64-linux/sbgrid/phenix/phenix.refine"
  },
  "endpoints": {
    "phenix_refine": {
      "command": "phenix.refine {model} {mtz} main.number_of_macro_cycles={cycles} refinement.input.xray_data.free_r_flags.label={free_label} output.prefix={out_dir}/phenix_cycle",
      "output_model": "{out_dir}/phenix_cycle_001.pdb"
    }
  }
}
```

Then run:

```bash
GROND protein-ls1 \
  --config grond.config.json \
  --endpoint phenix_refine \
  --model start_model.pdb \
  --mtz corrected_grond.mtz \
  --out-dir protein_cycle_001
```

Use `--dry-run` first. It writes the rendered command and checks template expansion without launching the external program.

## What Is Ready

The package now has the minimum wrapper layer needed for macromolecular experiments: deterministic unique-HKL FreeR flags, signed-row propagation of those flags, corrected-row MTZ export with negative intensities preserved, optional low-resolution q damping, endpoint registry entries for Phenix, REFMAC, and Servalcat command templates, and a dry-runnable one-macrocycle endpoint wrapper.

The production small-molecule `GROND auto` path is unchanged. Protein feedback loops are not yet promoted into `auto`; they should be run as explicit stages until benchmarked on real macromolecular data with Rfree, row-free controls, q-shuffle controls, solvent policy scans, and endpoint-map diagnostics.

## What Is Not Claimed

This wrapper does not implement bulk solvent inside the Bloch matrix. It does not turn row-free into Rfree. It does not choose water picking, TLS, NCS, occupancy refinement, or radiation-damage models. It does not guarantee that a scalar XDS or DIALS row table is sufficient for every protein dataset. Those choices belong in benchmarked endpoint templates and validation reports, not in silent defaults.
