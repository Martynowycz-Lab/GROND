#!/usr/bin/env python3
"""Quantify hydrogen omit visibility in CCP4 difference maps.

The Pilatanis/merged-style hydrogen claim is a map-quality claim, not only an
R-factor claim.  This script samples one or more Fo-Fc-like CCP4 maps at the
hydrogen positions from a reference full-H RES/CIF, searches a small sphere
around each H, and reports whether those H sites are actually stronger than the
background and stronger than a baseline map.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import gemmi
import numpy as np

if __package__:
    from .bloch_engine import parse_model
else:  # pragma: no cover - direct script execution
    from bloch_engine import parse_model


def parse_map_specs(specs: list[str]) -> list[tuple[str, Path]]:
    out = []
    for spec in specs:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            path = spec
            name = Path(path).parent.name
        out.append((name.strip(), Path(path).expanduser()))
    if not out:
        raise ValueError("pass at least one --map name=/path/to/fofc.map")
    return out


def read_map(path: Path):
    m = gemmi.read_ccp4_map(str(path))
    m.setup(float("nan"))
    grid = m.grid
    arr = np.asarray(grid, dtype=np.float32)
    return grid, np.asarray(arr, dtype=np.float64)


def frac_to_cart(cell: gemmi.UnitCell, frac) -> np.ndarray:
    p = cell.orthogonalize(gemmi.Fractional(float(frac[0]), float(frac[1]), float(frac[2])))
    return np.asarray([p.x, p.y, p.z], dtype=float)


def periodic_distance_A(cell: gemmi.UnitCell, frac_a, frac_b) -> float:
    d = np.asarray(frac_a, dtype=float) - np.asarray(frac_b, dtype=float)
    d -= np.round(d)
    p = cell.orthogonalize(gemmi.Fractional(float(d[0]), float(d[1]), float(d[2])))
    return float(math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z))


def hydrogen_sites(ref_model: dict) -> list[dict]:
    cell = gemmi.UnitCell(*[float(v) for v in ref_model["cell"]])
    sites = []
    for atom in ref_model["atoms"]:
        if str(atom.get("element", "")).upper() != "H":
            continue
        frac = np.asarray([float(atom["x"]), float(atom["y"]), float(atom["z"])], dtype=float) % 1.0
        sites.append({"name": atom["name"], "frac": frac, "cart": frac_to_cart(cell, frac)})
    return sites


def nearest_grid_value(arr: np.ndarray, frac) -> float:
    shape = np.asarray(arr.shape, dtype=int)
    # CCP4/Gemmi grid samples lie at fractional coordinates ijk/shape, not at
    # cell centres.  Round to the nearest periodic sample rather than taking
    # the lower voxel corner.
    ijk = np.rint(np.asarray(frac, dtype=float) * shape).astype(int) % shape
    return float(arr[tuple(ijk)])


def grid_points_near(arr: np.ndarray, cell: gemmi.UnitCell, frac, radius_A: float):
    shape = np.asarray(arr.shape, dtype=int)
    frac = np.asarray(frac, dtype=float)
    # Conservative index half-width from the shortest cell dimension.
    min_dim = max(min(cell.a / shape[0], cell.b / shape[1], cell.c / shape[2]), 1e-6)
    half = int(math.ceil(float(radius_A) / min_dim)) + 2
    center = np.rint(frac * shape).astype(int)
    for di in range(-half, half + 1):
        for dj in range(-half, half + 1):
            for dk in range(-half, half + 1):
                ijk = (center + np.asarray([di, dj, dk], dtype=int)) % shape
                grid_frac = ijk / shape
                dist = periodic_distance_A(cell, frac, grid_frac)
                if dist <= float(radius_A):
                    yield int(ijk[0]), int(ijk[1]), int(ijk[2]), grid_frac, dist


def audit_map(name: str, path: Path, sites: list[dict], ref_cell: gemmi.UnitCell, radius_A: float):
    grid, arr = read_map(path)
    cell = grid.unit_cell
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    zarr = (arr - mean) / max(std, 1e-12)
    rows = []
    for site in sites:
        best = {
            "z": -float("inf"),
            "value": float("nan"),
            "dist_A": float("nan"),
            "frac": [float(x) for x in site["frac"]],
        }
        for i, j, k, grid_frac, dist in grid_points_near(arr, cell, site["frac"], radius_A):
            z = float(zarr[i, j, k])
            if z > best["z"]:
                best = {
                    "z": z,
                    "value": float(arr[i, j, k]),
                    "dist_A": float(dist),
                    "frac": [float(x) for x in grid_frac],
                }
        direct_value = nearest_grid_value(arr, site["frac"])
        rows.append(
            {
                "map": name,
                "map_path": str(path),
                "hydrogen": site["name"],
                "search_radius_A": float(radius_A),
                "direct_value": float(direct_value),
                "direct_z": float((direct_value - mean) / max(std, 1e-12)),
                "local_peak_value": float(best["value"]),
                "local_peak_z": float(best["z"]),
                "local_peak_dist_A": float(best["dist_A"]),
                "local_peak_frac_x": float(best["frac"][0]),
                "local_peak_frac_y": float(best["frac"][1]),
                "local_peak_frac_z": float(best["frac"][2]),
            }
        )
    zvals = np.asarray([r["local_peak_z"] for r in rows], dtype=float)
    within = np.asarray([r["local_peak_dist_A"] for r in rows], dtype=float) <= float(radius_A)
    return {
        "name": name,
        "path": str(path),
        "grid_shape": list(arr.shape),
        "mean": mean,
        "std": std,
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "h_count": int(len(rows)),
        "h_peak_z_median": float(np.nanmedian(zvals)) if len(zvals) else float("nan"),
        "h_peak_z_p25": float(np.nanpercentile(zvals, 25)) if len(zvals) else float("nan"),
        "h_peak_z_p75": float(np.nanpercentile(zvals, 75)) if len(zvals) else float("nan"),
        "h_peak_z_max": float(np.nanmax(zvals)) if len(zvals) else float("nan"),
        "h_sites_over_2sigma": int(np.count_nonzero(zvals > 2.0)),
        "h_sites_over_3sigma": int(np.count_nonzero(zvals > 3.0)),
        "h_sites_over_4sigma": int(np.count_nonzero(zvals > 4.0)),
        "h_sites_with_peak_in_radius": int(np.count_nonzero(within)),
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-res", required=True, help="Full-H model used only for expected H positions.")
    ap.add_argument("--map", action="append", default=[], help="name=/path/to/fofc.map. May be repeated.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--radius-A", type=float, default=0.40)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref_model = parse_model(Path(args.reference_res))
    ref_cell = gemmi.UnitCell(*[float(v) for v in ref_model["cell"]])
    sites = hydrogen_sites(ref_model)
    if not sites:
        raise SystemExit("reference model has no hydrogens")
    summaries = []
    all_rows = []
    for name, path in parse_map_specs(args.map):
        summary = audit_map(name, path, sites, ref_cell, args.radius_A)
        summaries.append({k: v for k, v in summary.items() if k != "rows"})
        all_rows.extend(summary["rows"])
    write_csv(out / "hydrogen_omit_peaks.csv", all_rows)
    payload = {
        "reference_res": str(Path(args.reference_res).resolve()),
        "radius_A": float(args.radius_A),
        "hydrogen_count": int(len(sites)),
        "maps": summaries,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    lines = [
        "# Hydrogen Omit Map Audit",
        "",
        f"Reference hydrogens: `{len(sites)}`. Search radius: `{args.radius_A:.2f} A`.",
        "",
        "| map | H sites | median H peak z | p25/p75 H peak z | max H peak z | H >2σ | H >3σ | H >4σ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['name']} | {s['h_count']} | {s['h_peak_z_median']:.2f} | {s['h_peak_z_p25']:.2f}/{s['h_peak_z_p75']:.2f} | {s['h_peak_z_max']:.2f} | {s['h_sites_over_2sigma']} | {s['h_sites_over_3sigma']} | {s['h_sites_over_4sigma']} |"
        )
    lines.extend(["", f"Per-H rows: `{out / 'hydrogen_omit_peaks.csv'}`", f"JSON summary: `{out / 'summary.json'}`"])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str((out / "REPORT.md").resolve()), "rows": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
