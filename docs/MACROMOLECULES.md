# Macromolecular endpoints

The Bloch calculation on signed observations is the same for proteins, peptides and larger assemblies. What differs is the refinement step. For small molecules GROND writes corrected `HKLF 4` files and refines with SHELXL. A macromolecular refinement needs a corrected MTZ, a fixed FreeR set of unique reflections, bulk-solvent handling in the refinement program, and a model it can read back after one macrocycle.

This page describes the commands for that step. They are experimental and are not part of `grond auto`.

## Two free sets

The free set of `grond auto` is a salted hash of the unique reflection, the same in every crystal ([USAGE.md](USAGE.md)). It tests whether the dynamical correction transfers to reflections that were not fitted. With the default `--free-set off`, the final model is refined once more against all reflections.

A macromolecular Rfree set is assigned to unique reflections and carried into the merged MTZ. Every signed observation of a free reflection must keep that flag through correction and merging, and the flags must not be regenerated after correction. `grond make-free-flags` makes this set.

## Fixed FreeR flags

Run `make-free-flags` before any MTZ export. It reduces the signed observations to unique reflections in the declared space group, merges Friedel mates by default, spreads the free reflections over resolution shells when d-spacings are available, and can write the flag back to every observation.

```bash
grond make-free-flags \
  --rows grond_rows.csv \
  --cell "50.1 61.2 72.3 90 90 90" \
  --space-group "P 21 21 21" \
  --fraction 0.05 \
  --min-free-count 500 \
  --out free_flags.csv \
  --write-rows grond_rows_with_free.csv
```

If the model file has the cell and space group, use `--model model.pdb` instead of `--cell` and `--space-group`. `--merge-anomalous` (the default) gives Friedel mates one flag; use `--no-merge-anomalous` only when the refinement keeps them separate.

## Corrected MTZ

`export-mtz` merges corrected observations into unique reflections and writes `IMEAN`, `SIGIMEAN`, `FreeR_flag` and `NOBS`. Negative intensities are kept. Intensities are written rather than amplitudes.

```bash
grond export-mtz \
  --row-table native_dyn_calc.csv \
  --free-flags free_flags.csv \
  --cell "50.1 61.2 72.3 90 90 90" \
  --space-group "P 21 21 21" \
  --out-mtz corrected_grond.mtz \
  --out-dir corrected_mtz_export
```

The correction can be damped at low resolution:

```bash
grond export-mtz \
  --row-table native_dyn_calc.csv \
  --free-flags free_flags.csv \
  --model model.pdb \
  --out-mtz corrected_grond_damped.mtz \
  --q-policy low_resolution_damped \
  --q-off-d-A 5.0 \
  --q-full-d-A 3.0
```

This does not put bulk solvent into the Bloch calculation. It keeps the ordered-model correction from dominating the lowest-resolution range, where bulk solvent, background, beam stop, disorder and anisotropic scale are mixed. The refinement program still has to model bulk solvent.

## One refinement macrocycle

`protein-ls1` runs one external refinement macrocycle against the corrected MTZ from a command template and records the command, output, timing and the declared output model. It does not guess Phenix, REFMAC or Servalcat syntax, since local dictionaries, electron scattering settings, solvent choices and paths differ.

```bash
grond protein-ls1 \
  --model start_model.pdb \
  --mtz corrected_grond.mtz \
  --out-dir protein_cycle_001 \
  --command "phenix.refine {model} {mtz} main.number_of_macro_cycles={cycles} refinement.input.xray_data.free_r_flags.label={free_label} output.prefix={out_dir}/phenix_cycle" \
  --output-model "{out_dir}/phenix_cycle_001.pdb" \
  --cycles 1
```

For repeated use, put the template in `grond.config.json`:

```json
{
  "tools": {
    "phenix_refine": "/path/to/phenix.refine"
  },
  "endpoints": {
    "phenix_refine": {
      "command": "phenix.refine {model} {mtz} main.number_of_macro_cycles={cycles} refinement.input.xray_data.free_r_flags.label={free_label} output.prefix={out_dir}/phenix_cycle",
      "output_model": "{out_dir}/phenix_cycle_001.pdb"
    }
  }
}
```

Then:

```bash
grond protein-ls1 \
  --config grond.config.json \
  --endpoint phenix_refine \
  --model start_model.pdb \
  --mtz corrected_grond.mtz \
  --out-dir protein_cycle_001
```

Use `--dry-run` first: it writes the expanded command and checks the template without running the program.

## Limitations

- The feedback loop for macromolecules is not part of `grond auto`; run the stages explicitly.
- Bulk solvent is not part of the Bloch calculation.
- The free set of `grond auto` is not a macromolecular Rfree set.
- Water picking, TLS, NCS, occupancy refinement and radiation-damage models are left to the refinement program.
- A table of scalar XDS or DIALS observations may not be enough for every protein data set.
