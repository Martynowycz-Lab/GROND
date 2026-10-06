#!/usr/bin/env python3
"""Count Sg-limited Bloch basis sizes without running matrix propagation."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

if __package__:
    from .bloch_engine import (
        DEFAULT_MAX_SUPPORT_SG,
        Config,
        XDSBlochPort,
        XDSData,
        alternate_hand_model,
        parse_model,
        sha256,
    )
    from .native_row_regularize import TARGET_MODES, row_selection, target_arrays
else:  # pragma: no cover - direct script execution
    from bloch_engine import (
        DEFAULT_MAX_SUPPORT_SG,
        Config,
        XDSBlochPort,
        XDSData,
        alternate_hand_model,
        parse_model,
        sha256,
    )
    from native_row_regularize import TARGET_MODES, row_selection, target_arrays


_WORKER_PORT = None


def _pool_context_names() -> list[str]:
    """Return the process start methods to try, in order; macOS skips fork."""

    import platform

    if platform.system() == "Darwin":
        return ["forkserver", "spawn"]
    return ["forkserver", "fork", "spawn"]


def _init_worker(port):
    """Store the Bloch port for this worker process."""

    global _WORKER_PORT
    _WORKER_PORT = port


def _basis_size_worker(idx):
    """Return (row index, basis size) for one row, with the basis the predictions would use.

    PETS exact rows use their common orientation basis, centre mode the row-centre basis, and other modes
    the basis over the rocking window and mosaic orientations.
    """

    if _WORKER_PORT is None:
        raise RuntimeError("support scan worker was not initialized")
    port = _WORKER_PORT
    row_idx = int(idx)
    hkl = port.data.hkl[row_idx]
    if getattr(port.data, "pets_exact_a_center", None) is not None:
        orientations, _offsets, _weights, _semiangle, _ordinal = (
            port.pets_exact_orientation_samples(row_idx)
        )
        basis, _metadata = port.pets_exact_select_common_basis(
            hkl, orientations, row_idx
        )
    elif _uses_center_basis(getattr(port, "cfg", None)):
        omega_center = port.omega_from_z(port.data.zcal[row_idx])
        basis = port.select_basis(hkl, omega_center, row_idx=row_idx)
    else:
        # Profile-dependent bases (common_profile, gaussian_cone) are selected over the same
        # rocking window and mosaic orientations the predictions integrate.
        omega_center = port.omega_from_z(port.data.zcal[row_idx])
        center_sg = float(port.sg_for_hkls(np.asarray([hkl], dtype=int), omega_center)[0])
        sg_window_i, _meta = port.row_sg_window(row_idx, hkl, omega_center)
        omegas, tilts, *_rest = port.row_profile_orientation_points(row_idx, hkl, omega_center, center_sg, sg_window_i)
        basis, _basis_meta = port.select_basis_for_profile(hkl, omega_center, sg_window_i, omegas, tilts, row_idx=row_idx)
    return row_idx, int(len(basis))


def _uses_center_basis(cfg) -> bool:
    """The row-centre basis does not depend on the rocking window or a re-centred cone."""

    return (
        str(getattr(cfg, "profile_basis_mode", "center")).lower() == "center"
        and str(getattr(cfg, "mosaicity_mode", "none")).lower() != "gaussian_cone"
    )


def scan_basis_sizes(port: XDSBlochPort, indices: np.ndarray, workers: int, chunksize: int) -> np.ndarray:
    """Return the Bloch basis size of each row, in a process pool when possible and serially otherwise."""

    indices = np.asarray(indices, dtype=int)
    sizes = np.zeros(len(indices), dtype=int)
    position = {int(idx): pos for pos, idx in enumerate(indices)}

    def accept(idx, size):
        """Store one row's basis size at its position in indices."""

        sizes[position[int(idx)]] = int(size)

    def run_serial():
        """Compute every row's basis size in this process."""

        _init_worker(port)
        for idx in indices:
            row_idx, size = _basis_size_worker(int(idx))
            accept(row_idx, size)

    if int(workers) > 1 and len(indices) > 1:
        ran_parallel = False
        for context_name in _pool_context_names():
            try:
                ctx = multiprocessing.get_context(context_name)
            except Exception:
                continue
            try:
                with ctx.Pool(processes=int(workers), initializer=_init_worker, initargs=(port,)) as pool:
                    for idx, size in pool.imap_unordered(
                        _basis_size_worker,
                        [int(v) for v in indices],
                        chunksize=max(1, int(chunksize)),
                    ):
                        accept(idx, size)
                ran_parallel = True
                break
            except Exception:
                ran_parallel = False
                continue
        if not ran_parallel:
            run_serial()
    else:
        run_serial()
    return sizes


def main() -> int:
    """Count the Bloch basis size of each selected row and write the n_basis recommendation to --out; return 0.

    recommended_n_basis = min(max_final, max(min_final, largest basis + margin)).
    """

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hkl", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--hkl-transform", default="identity")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-basis-limit", type=int, default=512)
    ap.add_argument("--margin", type=int, default=10)
    ap.add_argument("--min-final", type=int, default=20)
    ap.add_argument("--max-final", type=int, default=500)
    ap.add_argument("--topup-d-min", type=float, default=0.8)
    ap.add_argument("--max-support-sg", type=float, default=DEFAULT_MAX_SUPPORT_SG)
    ap.add_argument("--min-d-spacing", type=float, default=0.70)
    ap.add_argument("--min-peak", type=float, default=90.0)
    ap.add_argument("--geometry-percentile", type=float, default=99.0)
    ap.add_argument("--target-mode", choices=list(TARGET_MODES), default="iobs")
    ap.add_argument("--positive-only", action="store_true", default=True)
    ap.add_argument("--include-nonpositive", dest="positive_only", action="store_false")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunksize", type=int, default=1)
    ap.add_argument("--alternate-hand", dest="alternate_hand", action="store_true")
    # Profile contract of the stages this basis serves.  The historical row-centre basis
    # depends only on --max-support-sg; profile-dependent bases use all of these.
    ap.add_argument("--n-points", type=int, default=1)
    ap.add_argument("--sg-window", type=float, default=Config.sg_window)
    ap.add_argument(
        "--sg-window-mode",
        choices=["fixed", "zobs_residual", "zobs_plus_floor", "rlp_peak", "dials_bbox"],
        default=Config.sg_window_mode,
    )
    ap.add_argument("--sg-window-min", type=float, default=Config.sg_window_min)
    ap.add_argument("--sg-window-max", type=float, default=Config.sg_window_max)
    ap.add_argument("--sg-window-scale", type=float, default=Config.sg_window_scale)
    ap.add_argument("--sg-window-frame-floor", type=float, default=Config.sg_window_frame_floor)
    ap.add_argument("--profile-basis-mode", choices=["center", "common_profile"], default="center")
    ap.add_argument(
        "--mosaicity-mode",
        choices=["none", "gaussian_omega", "gaussian_cone", "transverse_target"],
        default="none",
    )
    ap.add_argument("--mosaicity-sigma-deg", type=float, default=0.0, help="Largest mosaic width the basis must serve.")
    ap.add_argument("--mosaicity-points", type=int, default=3)
    args = ap.parse_args()
    effective_topup_d_min = min(float(args.topup_d_min), float(args.min_d_spacing)) if args.min_d_spacing > 0 else float(args.topup_d_min)

    t0 = time.perf_counter()
    data = XDSData(Path(args.hkl), hkl_transform=args.hkl_transform)
    target, _base_sig = target_arrays(data, args.target_mode)
    indices = row_selection(data, target, args.positive_only, args.min_d_spacing, args.min_peak, args.max_rows, args.geometry_percentile)
    model = parse_model(Path(args.res))
    if args.alternate_hand:
        model = alternate_hand_model(model)
    cone = str(args.mosaicity_mode) == "gaussian_cone"
    cfg = Config(
        n_basis=int(args.n_basis_limit),
        n_points=int(args.n_points),
        sg_window=float(args.sg_window),
        sg_window_mode=str(args.sg_window_mode),
        sg_window_min=float(args.sg_window_min),
        sg_window_max=float(args.sg_window_max),
        sg_window_scale=float(args.sg_window_scale),
        sg_window_frame_floor=float(args.sg_window_frame_floor),
        profile_basis_mode=str(args.profile_basis_mode),
        profile_coordinate="direct_angle" if cone else "target_sg",
        mosaicity_mode=str(args.mosaicity_mode),
        mosaicity_sigma_deg=float(args.mosaicity_sigma_deg),
        mosaicity_points=int(args.mosaicity_points),
        mosaicity_support_sigma_deg=float(args.mosaicity_sigma_deg) if cone else 0.0,
        topup_d_min=effective_topup_d_min,
        max_support_sg=float(args.max_support_sg),
    )
    port = XDSBlochPort(data, model, cfg, cache_dir=Path(args.out).resolve().parent / "support_cache", res_hash=sha256(Path(args.res)))
    sizes = scan_basis_sizes(port, indices, args.workers, args.chunksize)
    max_observed = int(np.max(sizes)) if len(sizes) else 0
    uncapped_recommended = max(int(args.min_final), max_observed + int(args.margin))
    recommended = min(int(args.max_final), uncapped_recommended)
    summary = {
        "rows": int(len(indices)),
        "n_basis_limit": int(args.n_basis_limit),
        "margin": int(args.margin),
        "min_final": int(args.min_final),
        "max_final": int(args.max_final),
        "selection_rule": "recommended_n_basis = min(max_final, max(min_final, observed_rowwise_max_basis + margin))",
        "observed_max_plus_margin": int(max_observed + int(args.margin)),
        "uncapped_recommended_n_basis": int(uncapped_recommended),
        "recommended_n_basis": int(recommended),
        "cap_hit": bool(recommended < uncapped_recommended),
        "basis_size_stats": {
            "min": int(np.min(sizes)) if len(sizes) else 0,
            "median": float(np.median(sizes)) if len(sizes) else 0.0,
            "p95": float(np.percentile(sizes, 95)) if len(sizes) else 0.0,
            "p99": float(np.percentile(sizes, 99)) if len(sizes) else 0.0,
            "max": max_observed,
        },
        "truncated_by_limit": bool(max_observed >= int(args.n_basis_limit)),
        "inputs": {
            "hkl": str(Path(args.hkl).resolve()),
            "res": str(Path(args.res).resolve()),
            "hkl_transform": data.hkl_transform_spec,
            "topup_d_min": float(effective_topup_d_min),
            "requested_topup_d_min": float(args.topup_d_min),
            "max_support_sg": float(args.max_support_sg),
            "n_points": int(args.n_points),
            "sg_window": float(args.sg_window),
            "sg_window_mode": str(args.sg_window_mode),
            "sg_window_max": float(args.sg_window_max),
            "profile_basis_mode": str(args.profile_basis_mode),
            "mosaicity_mode": str(args.mosaicity_mode),
            "mosaicity_sigma_deg": float(args.mosaicity_sigma_deg),
            "basis_rule": "row_center_max_support_sg" if _uses_center_basis(cfg) else "profile_support",
        },
        "wall_seconds": time.perf_counter() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
