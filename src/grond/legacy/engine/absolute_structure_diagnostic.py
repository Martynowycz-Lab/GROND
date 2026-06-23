#!/usr/bin/env python3
"""Compare two native row tables with an alternate hand diagnostic.

This is intentionally row-level.  It does not pretend to be a SHELXL Flack
parameter or a merged absolute-structure statistic. It asks whether the
same signed observations have a lower scaled-residual sum for one native Bloch
prediction than for the alternate-hand check after each branch has had its
declared nuisance scales applied.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def block(label: str, frame: pd.DataFrame) -> dict:
    rb = np.asarray(frame["dyn_residual_over_sigma_best"], dtype=float)
    ro = np.asarray(frame["dyn_residual_over_sigma_other"], dtype=float)
    finite = np.isfinite(rb) & np.isfinite(ro)
    rb = rb[finite]
    ro = ro[finite]
    d = ro * ro - rb * rb
    n = int(d.size)
    if n == 0:
        return {"label": label, "n": 0}
    delta_chi2 = float(np.sum(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
    paired_t = float(np.mean(d) / (sd / math.sqrt(n))) if n > 1 and sd > 0 else float("nan")
    sqrt_positive_delta = float(math.sqrt(max(delta_chi2, 0.0)))
    return {
        "label": label,
        "n": n,
        "best_chi2": float(np.sum(rb * rb)),
        "other_chi2": float(np.sum(ro * ro)),
        "delta_chi2_other_minus_best": delta_chi2,
        "sqrt_positive_delta_chi2": sqrt_positive_delta,
        "paired_row_t_delta_chi2": paired_t,
        "mean_row_delta_chi2": float(np.mean(d)),
        "median_row_delta_chi2": float(np.median(d)),
        "best_lower_abs_residual_fraction": float(np.mean(np.abs(rb) < np.abs(ro))),
        "best_lower_squared_residual_fraction": float(np.mean(d > 0.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--best", required=True, help="Native row CSV for the active/best hand assignment.")
    ap.add_argument("--other", required=True, help="Native row CSV for the alternate-hand comparison.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    args = ap.parse_args()

    best = pd.read_csv(args.best)
    other = pd.read_csv(args.other)
    key = ["row_index", "h", "k", "l"]
    merged = best.merge(other, on=key, suffixes=("_best", "_other"), validate="one_to_one")
    out = {
        "best_csv": str(Path(args.best).resolve()),
        "other_csv": str(Path(args.other).resolve()),
        "row_join": {"key": key, "n_best": int(len(best)), "n_other": int(len(other)), "n_joined": int(len(merged))},
        "interpretation": {
            "diagnostic": "row-level alternate-hand check; not a Flack parameter and not a merged absolute-structure statistic",
            "delta_chi2": "sum((other residual/sigma)^2 - (best residual/sigma)^2); positive favors best",
            "paired_row_t": "paired t-style contrast of per-row delta_chi2 values",
            "sqrt_positive_delta_chi2": "square root of positive delta_chi2; a scale indicator, not a probability or hand-assignment score",
        },
        "all": block("all", merged),
        "work": block("work", merged[merged["split_best"] == "work"]),
        "free": block("free", merged[merged["split_best"] == "free"]),
    }
    Path(args.out).write_text(json.dumps(out, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
