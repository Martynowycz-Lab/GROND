#!/usr/bin/env python3
"""Extract raw rocking-profile dynrows from XDS spot predictions.

This is deliberately conservative.  It does not claim external/row contract integration
parity.  It uses XDS `INTEGRATE.HKL` as a spot-position contract, reads the raw
CBF frames, sums a small circular aperture over a frame window around ZCAL, and
subtracts an annular local background.

The primary output is a row-contract frame/observation JSONL contract preserving
the exact signed XDS row HKL.  The script also writes an XDS-like 21-column file
so the existing `bloch_engine.py` can score the same rows without changing
the frozen Bloch kernel.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import fabio
import numpy as np

from bloch_engine import XDSData, sha256


def circle_masks(radius, bg_inner, bg_outer):
    r = int(math.ceil(bg_outer))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    rr = np.sqrt(xx * xx + yy * yy)
    peak = rr <= float(radius)
    bg = (rr >= float(bg_inner)) & (rr <= float(bg_outer))
    return peak, bg, r


def frame_path(cbf_dir: Path, frame: int):
    return cbf_dir / f"image_{int(frame):05d}.cbf"


class FrameCache:
    def __init__(self, cbf_dir: Path, max_items=16):
        self.cbf_dir = Path(cbf_dir)
        self.max_items = int(max_items)
        self.cache = {}
        self.order = []

    def get(self, frame: int):
        frame = int(frame)
        if frame in self.cache:
            return self.cache[frame]
        p = frame_path(self.cbf_dir, frame)
        if not p.exists():
            return None
        arr = fabio.open(str(p)).data.astype(np.float32)
        self.cache[frame] = arr
        self.order.append(frame)
        while len(self.order) > self.max_items:
            old = self.order.pop(0)
            self.cache.pop(old, None)
        return arr


def extract_patch(arr, x, y, half):
    xi = int(round(float(x))) - 1
    yi = int(round(float(y))) - 1
    x0, x1 = xi - half, xi + half + 1
    y0, y1 = yi - half, yi + half + 1
    if x0 < 0 or y0 < 0 or x1 > arr.shape[1] or y1 > arr.shape[0]:
        return None
    return arr[y0:y1, x0:x1]


def integrate_row(cache, row, peak_mask, bg_mask, half, frame_radius):
    x = row["x"]
    y = row["y"]
    z = row["z"]
    center = int(round(float(z)))
    frames = range(center - int(frame_radius), center + int(frame_radius) + 1)
    prof = []
    for frame in frames:
        arr = cache.get(frame)
        if arr is None:
            continue
        patch = extract_patch(arr, x, y, half)
        if patch is None:
            continue
        peak_vals = patch[peak_mask]
        bg_vals = patch[bg_mask]
        if len(bg_vals) < 8:
            continue
        bg_med = float(np.median(bg_vals))
        bg_mad = float(1.4826 * np.median(np.abs(bg_vals - bg_med)))
        signal = float(np.sum(peak_vals - bg_med))
        sigma = math.sqrt(max(float(np.sum(np.maximum(peak_vals, 0.0))) + len(peak_vals) * bg_mad * bg_mad, 1.0))
        prof.append({"frame": frame, "signal": signal, "sigma": sigma, "bg": bg_med, "bg_mad": bg_mad})
    if not prof:
        return None
    signal = float(sum(p["signal"] for p in prof))
    sigma = float(math.sqrt(sum(p["sigma"] * p["sigma"] for p in prof)))
    weights = np.asarray([max(p["signal"], 0.0) for p in prof], dtype=float)
    if float(np.sum(weights)) > 0.0:
        z_cent = float(np.sum(weights * np.asarray([p["frame"] for p in prof])) / np.sum(weights))
    else:
        z_cent = float(z)
    return signal, sigma, z_cent, prof


def write_xds_like(path: Path, data: XDSData, selected_rows, results, source_path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = []
    for line in source_path.read_text(errors="replace").splitlines():
        if not line.startswith("!"):
            break
        if line.startswith("!OUTPUT_FILE="):
            header_lines.append("!OUTPUT_FILE=RAW_PROFILE_XDSLIKE.HKL")
        elif line.startswith("!END_OF_HEADER"):
            continue
        else:
            header_lines.append(line)
    with open(path, "w") as f:
        for line in header_lines:
            f.write(line + "\n")
        f.write("!RAW_PROFILE_SOURCE= XDS spot centers plus raw CBF aperture/annulus integration\n")
        f.write("!END_OF_HEADER\n")
        for idx, (I, sig, zcent, _prof) in zip(selected_rows, results):
            row = [
                *data.hkl[idx].tolist(),
                I,
                sig,
                data.xcal[idx],
                data.ycal[idx],
                zcent,
                data.rlp[idx],
                data.peak[idx],
                data.corr[idx],
                data.maxc[idx],
                data.xobs[idx],
                data.yobs[idx],
                zcent,
                data.alf0[idx],
                data.bet0[idx],
                data.alf1[idx],
                data.bet1[idx],
                data.psi[idx],
                data.iseg[idx],
            ]
            f.write(" ".join(str(int(v)) if n in {0, 1, 2, 20} else f"{float(v):.6g}" for n, v in enumerate(row)) + "\n")


def group_id_from_z(data: XDSData, zcal: float, group_width_frames: float) -> int:
    omega0 = data.omega_from_z(data.zcal[0])
    omega = data.omega_from_z(zcal)
    width_deg = max(float(group_width_frames) * float(data.oscillation_range), 1e-12)
    return int(math.floor((float(omega) - float(omega0)) / width_deg))


def write_dynrows(out_dir: Path, data: XDSData, selected_rows, results, profiles_by_row, source_path: Path, args, manifest):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / "frames.jsonl"
    obs_path = out_dir / "observations.jsonl"

    frames = {}
    for idx in selected_rows:
        z = float(data.zcal[idx])
        gid = group_id_from_z(data, z, args.group_width_frames)
        center = int(round(z))
        rec = frames.setdefault(
            gid,
            {
                "dataset_id": "xds_bloch",
                "frame_id": gid,
                "source": "XDS_ZCAL_bin",
                "group_width_frames": float(args.group_width_frames),
                "frame_min": center - int(args.frame_radius),
                "frame_max": center + int(args.frame_radius),
                "zcal_min": z,
                "zcal_max": z,
                "n_observations": 0,
            },
        )
        rec["frame_min"] = min(rec["frame_min"], center - int(args.frame_radius))
        rec["frame_max"] = max(rec["frame_max"], center + int(args.frame_radius))
        rec["zcal_min"] = min(rec["zcal_min"], z)
        rec["zcal_max"] = max(rec["zcal_max"], z)
        rec["n_observations"] += 1

    with frames_path.open("w") as f:
        for rec in sorted(frames.values(), key=lambda r: r["frame_id"]):
            rec = dict(rec)
            rec["angle_deg"] = float(data.omega_from_z(0.5 * (rec["zcal_min"] + rec["zcal_max"])))
            rec["wavelength_A"] = float(data.wavelength)
            rec["s0_vector"] = data.s0.astype(float).tolist()
            rec["rotation_axis"] = data.axis.astype(float).tolist()
            rec["A_matrix"] = data.A.astype(float).tolist()
            rec["image_range"] = [int(rec["frame_min"]), int(rec["frame_max"])]
            rec["geometry_source"] = str(source_path.resolve())
            f.write(json.dumps(rec, allow_nan=True) + "\n")

    with obs_path.open("w") as f:
        for idx, (I, sig, zcent, prof) in zip(selected_rows, results):
            z = float(data.zcal[idx])
            rec = {
                "obs_id": int(idx),
                "dataset_id": "xds_bloch",
                "signed_h": int(data.hkl[idx, 0]),
                "signed_k": int(data.hkl[idx, 1]),
                "signed_l": int(data.hkl[idx, 2]),
                "hkl": [int(v) for v in data.hkl[idx]],
                "frame_id": group_id_from_z(data, z, args.group_width_frames),
                "target_alpha0_deg": float(data.omega_from_z(z)),
                "Iobs_raw": float(I),
                "sigma_Iobs_raw": float(sig),
                "Iobs_xds": float(data.iobs[idx]),
                "sigma_xds": float(data.sigma[idx]),
                "xcal": float(data.xcal[idx]),
                "ycal": float(data.ycal[idx]),
                "zcal": z,
                "xobs": float(data.xobs[idx]),
                "yobs": float(data.yobs[idx]),
                "zobs": float(data.zobs[idx]),
                "raw_z_centroid": float(zcent),
                "bbox_model": {
                    "type": "circular_aperture_annular_background",
                    "frame_radius": int(args.frame_radius),
                    "aperture_radius_px": float(args.aperture_radius),
                    "bg_inner_px": float(args.bg_inner),
                    "bg_outer_px": float(args.bg_outer),
                },
                "quality": {
                    "rlp": float(data.rlp[idx]),
                    "peak": float(data.peak[idx]),
                    "corr": float(data.corr[idx]),
                    "maxc": float(data.maxc[idx]),
                    "iseg": int(data.iseg[idx]),
                },
                "profile": profiles_by_row.get(int(idx), prof if args.save_all_profiles else []),
            }
            f.write(json.dumps(rec, allow_nan=True) + "\n")

    manifest.setdefault("outputs", {})["frames_jsonl"] = str(frames_path.resolve())
    manifest.setdefault("outputs", {})["observations_jsonl"] = str(obs_path.resolve())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--integrate-hkl", required=True)
    ap.add_argument("--cbf-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--dynrows-dir", default=None, help="Directory for row-contract frames.jsonl and observations.jsonl.")
    ap.add_argument("--top-isigma", type=int, default=0, help="Keep strongest XDS I/sigma rows after other filters. 0 means all rows.")
    ap.add_argument("--positive-only", action="store_true", help="Drop non-positive XDS IOBS rows before raw integration.")
    ap.add_argument("--frame-radius", type=int, default=2)
    ap.add_argument("--aperture-radius", type=float, default=3.0)
    ap.add_argument("--bg-inner", type=float, default=5.0)
    ap.add_argument("--bg-outer", type=float, default=8.0)
    ap.add_argument("--group-width-frames", type=float, default=3.0)
    ap.add_argument("--profile-examples", type=int, default=50)
    ap.add_argument("--save-all-profiles", action="store_true")
    args = ap.parse_args()

    data = XDSData(Path(args.integrate_hkl))
    mask = np.ones(len(data.iobs), dtype=bool)
    if args.positive_only:
        mask &= data.iobs > 0
    idx = np.where(mask)[0]
    if args.top_isigma and len(idx) > args.top_isigma:
        isigi = data.iobs[idx] / np.maximum(data.sigma[idx], 1e-12)
        idx = idx[np.argsort(isigi)[::-1][: args.top_isigma]]
    idx = np.asarray(sorted(idx, key=lambda i: (float(data.zcal[i]), int(i))), dtype=int)

    peak_mask, bg_mask, half = circle_masks(args.aperture_radius, args.bg_inner, args.bg_outer)
    cache = FrameCache(Path(args.cbf_dir), max_items=2 * args.frame_radius + 8)
    selected, results, rejected = [], [], []
    profiles = []
    profiles_by_row = {}
    for n, row_idx in enumerate(idx, start=1):
        row = {"x": data.xcal[row_idx], "y": data.ycal[row_idx], "z": data.zcal[row_idx]}
        res = integrate_row(cache, row, peak_mask, bg_mask, half, args.frame_radius)
        if res is None:
            rejected.append(int(row_idx))
            continue
        selected.append(int(row_idx))
        results.append(res)
        profile_record = {
            "row_index": int(row_idx),
            "hkl": data.hkl[row_idx].astype(int).tolist(),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z_xds": float(row["z"]),
            "z_centroid": float(res[2]),
            "xds_iobs": float(data.iobs[row_idx]),
            "raw_iobs": float(res[0]),
            "profile": res[3],
        }
        if len(profiles) < int(args.profile_examples):
            profiles.append(profile_record)
            profiles_by_row[int(row_idx)] = res[3]
        if n % 500 == 0:
            print(f"integrated {n}/{len(idx)} candidate rows", flush=True)

    order = np.argsort(np.asarray(selected, dtype=int))
    selected = [selected[int(i)] for i in order]
    results = [results[int(i)] for i in order]

    manifest = {
        "input": str(Path(args.integrate_hkl).resolve()),
        "input_sha256": sha256(Path(args.integrate_hkl)),
        "cbf_dir": str(Path(args.cbf_dir).resolve()),
        "outputs": {"xds_like_hkl": str(Path(args.out).resolve())},
        "top_isigma": args.top_isigma,
        "positive_only": args.positive_only,
        "frame_radius": args.frame_radius,
        "aperture_radius": args.aperture_radius,
        "bg_inner": args.bg_inner,
        "bg_outer": args.bg_outer,
        "group_width_frames": args.group_width_frames,
        "candidate_rows": int(len(idx)),
        "written_rows": int(len(selected)),
        "rejected_rows": int(len(rejected)),
        "raw_i_median": float(np.median([r[0] for r in results])) if results else None,
        "raw_i_positive_fraction": float(np.mean([r[0] > 0 for r in results])) if results else None,
        "profile_examples": profiles,
    }
    write_xds_like(Path(args.out), data, selected, results, Path(args.integrate_hkl))
    if args.dynrows_dir:
        write_dynrows(Path(args.dynrows_dir), data, selected, results, profiles_by_row, Path(args.integrate_hkl), args, manifest)
    mpath = Path(args.manifest) if args.manifest else Path(args.out).with_suffix(".manifest.json")
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2, allow_nan=True))
    print(json.dumps({k: v for k, v in manifest.items() if k != "profile_examples"}, indent=2))


if __name__ == "__main__":
    main()
