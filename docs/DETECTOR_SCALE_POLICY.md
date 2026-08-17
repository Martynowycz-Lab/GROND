# Detector-scale policy

Detector-position factors are optional nuisance parameters. They are not part
of the Bloch correction and they are not assumed to describe the detector just
because they lower an in-sample residual. GROND now treats a detector-free fit
as the reference and evaluates 2x2, 4x4, and 8x8 Cartesian fields on the same
frozen Born and Bloch predictions. Physical nuisance selection is detector-free
by default so that a flexible field cannot choose thickness, mosaic width, or
other physics. Once those predictions are frozen, scale polish evaluates no
field, 2x2, 4x4, and 8x8. The 8x8 field is diagnostic and cannot be promoted
without an explicitly declared independent calibration.

Each nontrivial field must earn promotion. The default gate requires an
absolute held-out wR2 improvement of at least 0.002 on at least two
deterministic splits, no material reduction of the held-out Born-to-Bloch
advantage, at least 20 fitted work rows in every tile, reproducibility between
independent data halves, and reasonable agreement between fields inferred from
the Born and Bloch predictions. The corrected-HKL endpoint must then pass the
fixed-weight GooF window with no NPD atoms or split warnings. Among fields that
pass, the smallest detector grid is used. A detector-free candidate remains
the default when no added field earns all gates.

The tile fit uses alternating weighted least squares with the declared frame
scales. Detector values are shrunk toward unity and nearest-neighbor jumps are
penalized. Penalty strengths are normalized by the median information in an
adequately populated tile, so the defaults are dimensionless. A tile below the
minimum row count receives no independent data term and is determined only by
the unity prior and neighboring populated tiles; such a field is still barred
from publication promotion because every fitted tile must have adequate
support. The hard factor range is 0.25 to 4.0.

The controls are exposed as `--detector-scale-ridge`,
`--detector-scale-smoothness`, `--detector-scale-min-rows`, and the associated
held-out and field-agreement thresholds shown by `grond auto --help` and
`grond scale-replay --help`. `--detector-scale-independent-calibration` is an
explicit assertion, not an automatic inference. Use it only when an external
flat-field or a genuinely independent detector-response experiment supports
the 8x8 surface.

Reports keep `frame_scale_count` and `detector_scale_count` separate. Their sum
is also retained for compatibility, but a single combined count should not be
used to describe model complexity in a paper. DIALS imports now preserve a
present `inverse_scale_factor` column and its variance in
`DIALS_INVERSE_SCALE_CONTROL.npz`. This is a fixed control or prior and is not
applied silently. Signed, unmerged XDS CORRECT factors should likewise be used
as a fixed comparison on matched rows rather than replaced by a flexible GROND
field. A merged or Friedel-merged XDS control cannot establish a signed-row
detector response.

For a stronger detector-response claim, learn one common 4x4 field from all but
one crystal and predict the held-out crystal while fitting only that crystal's
frame scales on its work rows. Repeat the leave-one-crystal-out test for the
whole acquisition cohort. A stable common surface that improves independent
crystals and preserves the Born-to-Bloch advantage is evidence for a physical
detector response. A crystal-specific field that only improves the same rows
used to estimate it is not.
