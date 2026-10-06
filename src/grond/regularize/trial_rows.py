"""Serialize every evaluated nuisance trial's raw row predictions for workflow-side held-out scoring.

The native engine keeps only the selected trial's row table. Held-out physics gates must compare
every candidate on identical rows with scales refitted on training orbits only, so this writes the
unscaled dynamical prediction of each trial next to the fixed row identity (split, HKL, orbit key,
target, raw sigma, frame group, scan angle).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grond.validation.heldout import orbit_keys


def write_trial_rows_npz(path: Path, data, target, base_sig, masks, trials, symmetry_ops=None) -> dict:
    """Write one compressed archive with every trial's raw row predictions; return its summary."""

    if not trials:
        raise ValueError("no evaluated nuisance trials to serialize")
    rows = np.flatnonzero(np.asarray(masks["work"], dtype=bool) | np.asarray(masks["free"], dtype=bool))
    groups = np.asarray(trials[0]["arrays"]["groups"])[rows]
    for trial in trials[1:]:
        if not np.array_equal(np.asarray(trial["arrays"]["groups"])[rows], groups):
            raise ValueError("nuisance trials use different frame groups; held-out scoring needs one scale grouping")
    hkl = np.asarray(data.hkl, dtype=int)[rows]
    zcal = np.asarray(data.zcal, dtype=float)[rows]
    payload = {
        "row_index": rows.astype(int),
        "hkl": hkl,
        "orbit_key": np.asarray(orbit_keys(hkl, symmetry_ops, include_friedel=True), dtype=str),
        "d_A": np.asarray([data.cell.calculate_d(tuple(int(v) for v in h)) for h in hkl], dtype=float),
        "target": np.asarray(target, dtype=float)[rows],
        "sigma": np.asarray(base_sig, dtype=float)[rows],
        "group": groups.astype(int),
        "angle_deg": np.asarray([float(data.omega_from_z(z)) for z in zcal], dtype=float),
        "work": np.asarray(masks["work"], dtype=bool)[rows],
        "free": np.asarray(masks["free"], dtype=bool)[rows],
        "dyn_raw": np.stack([np.asarray(t["arrays"]["dyn_raw"], dtype=float)[rows] for t in trials]),
        "valid": np.stack([np.asarray(t["arrays"]["valid"], dtype=bool)[rows] for t in trials]),
        "nuisance_json": np.asarray([json.dumps(t["nuisance"], sort_keys=True, allow_nan=True) for t in trials], dtype=str),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return {"path": str(path), "trials": int(len(trials)), "rows": int(len(rows))}


def load_trial_rows_npz(path: Path) -> dict:
    """Read an archive written by write_trial_rows_npz with decoded nuisance blocks."""

    with np.load(Path(path), allow_pickle=False) as archive:
        out = {key: archive[key] for key in archive.files}
    out["nuisance"] = [json.loads(str(text)) for text in out.pop("nuisance_json")]
    return out
