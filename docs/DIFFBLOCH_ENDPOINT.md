# Experimental diffBloch endpoint polish

GROND can prepare and validate a diffBloch coordinate/ADP polish without making
diffBloch a required package dependency. This remains an experimental endpoint:
diffBloch and GROND use different rocking integrals, scattering factors, and
scale objectives, so acceptance is decided only after the candidate is replayed
through the unchanged GROND row contract.

## Validation contract

The external optimizer must never see GROND free rows. Export a PETS file from
the final native row table with the work split only:

```bash
grond export-pets \
  --xds-dir /path/to/xds \
  --dials-python /path/to/dials.python \
  --xds-row-selection correct \
  --xds-sigma-mode reported \
  --frame-ownership nearest-excitation \
  --min-partiality 0.90 \
  --virtual-frame-n-merged 10 \
  --virtual-frame-step 10 \
  --row-contract /path/to/final/native_dyn_calc.csv \
  --split-selection work \
  --dataset-name crystal_work \
  --out-dir pets_work
```

The manifest records the source row IDs, split counts, row-table hash, XDS
hashes, indexing transform, and the intersection that survived XDS acceptance
and partiality filtering. The current split-filtered exporter is intentionally
limited to XDS until DIALS reflection-table row identity receives the same
end-to-end test.

Run diffBloch with fixed orientation and a bounded fixed thickness, its
thickness neural network disabled, absorption disabled, and its own validation
split drawn only from the GROND work observations. Begin with one to four
coordinate-only Adam steps, `g_max=1.4 A^-1`, `rsg=0.99`, `dsg=0`, and 42
rocking samples. Treat ADP fitting as a separate sensitivity test rather than a
default. Materialize separate early-step candidates because an internal
diffBloch validation minimum does not guarantee transfer to GROND's different
row integral and pooled objective. More epochs are not evidence of convergence.

## Return to SHELX

Transfer the refined CIF into the exact source instruction model:

```bash
grond apply-diffbloch-model \
  --source-res final_grond.res \
  --refined-cif diffbloch/refined_structure.cif \
  --out-res diffbloch_candidate.res
```

Only matching atom coordinates and compatible displacement parameters are
transferred. The source CELL, SFAC, occupancies, FVAR, AFIX, WGHT, symmetry,
and all other instructions remain unchanged. Cell mismatch, missing heavy
atoms, element mismatch, duplicate labels, and non-positive ADPs are audited;
an NPD candidate returns a failing status by default. Hydrogen Uiso multipliers
and AFIX instructions remain source-owned.

## Acceptance

Recompute the candidate with `--row-contract` on the original GROND N/Q,
physics, exact row IDs, work/free split, and scale topology. Reapplying ordinary
geometry selection after coordinates move is not equivalent: borderline rows
can enter or leave the calculation. Scale values may be refitted on work rows
only; free rows must not choose coordinates, thickness, scale values, or the
saved epoch. Promote only a pathology-free candidate that improves the
predeclared work objective without a material free-row loss under a
detector-free or independently validated compact detector model. SHELXL R1gt,
R1all, wR2, and GooF are endpoint diagnostics, not substitutes for the held-out
row gate.

The current diffBloch objective profiles independent virtual-frame scales and
reports a mean of per-frame residuals. GROND therefore recomputes pooled row
statistics after handoff rather than comparing the two programs' headline wR
values directly. A proper PETS2-owned integration will make the physics
comparison cleaner but does not remove the need for this external validation.
