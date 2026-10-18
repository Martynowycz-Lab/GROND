#!/usr/bin/env python3
"""Thickness response tables streamed from row spectra, without keeping the spectra.

The nested selection scores every thickness candidate on a table of each row's profile-summed intensity I(t) on
the support grid, one table per hand plus the Born term (grond.regularize.nested_selection.thickness_table).
Building that table from packed spectra (grond.legacy.engine.spectra_stage) holds every row's eigensystems in
memory at once, which for full-cone matrices of a large data set is hundreds of GB. Here each worker computes one
row's spectra, reduces them to the row's table entries with thickness_table itself and drops them, so memory is the
table plus one row's spectra per worker. The table values equal those thickness_table computes from the packed
spectra of the same rows, bit for bit.

A table is cached like spectra, in a directory named by a hash of the spectra identity, the support, the hands and
the table code, with one .npy file per array. It is read back into memory as read-only arrays (a memory map would
change NumPy's matrix-vector summation and so the last bits of the selection scores).

The exact characteristic-function evaluator (cf) needs the spectra, so a selection on tables scores and evaluates
every candidate on the table: it needs evaluator="table" and score_with="table".

  python -m grond.legacy.engine.response_stage --hkl INTEGRATE.HKL --res model.res --config-json cfg.json \
      --row-index-file rows.txt --out-dir DIR [--hands 1 -1] [--workers 8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402

from grond.bloch.config import Config  # noqa: E402
from grond.bloch.constants import JAX_BACKENDS, JAX_PADDED_BACKENDS  # noqa: E402
from grond.legacy.engine import spectra_stage as spectra  # noqa: E402
from grond.regularize import nested_selection  # noqa: E402
from grond.regularize.nested_selection import ThicknessSupport, thickness_table  # noqa: E402

RESPONSE_SCHEMA = "grond-response-table-v1"
ROW_KEYS = ("row_ids", "mu", "n_p", "n_b", "reused_points")


def hand_key(hand: int) -> str:
    """File stem of one hand's table: hand_plus or hand_minus."""
    return "hand_plus" if int(hand) > 0 else "hand_minus"


def table_sources_sha256() -> str:
    """Hash of the code that turns spectra into table rows (the physics sources are in the spectra identity)."""
    h = hashlib.sha256(RESPONSE_SCHEMA.encode())
    for path in (Path(nested_selection.__file__).resolve(), Path(__file__).resolve()):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def response_identity(hkl, res, cfg: Config, rows, support: ThicknessSupport, hands, *, hkl_transform: str = "identity",
                      alternate_hand: bool = False) -> dict:
    """Spectra identity of the rows plus the support, the hands and the table code."""
    ident = spectra.spectra_identity(hkl, res, cfg, rows, hkl_transform=hkl_transform, alternate_hand=alternate_hand)
    return {**ident, "response_schema": RESPONSE_SCHEMA, "table_sources_sha256": table_sources_sha256(),
            "support": support.to_dict(), "hands": [int(h) for h in hands]}


# ----------------------------------------------------------------------------- one row


def row_response(port, row: int, support: ThicknessSupport, hands, donor: dict | None = None) -> dict:
    """Table entries of one row: I(t) per hand and the Born term on the support grid, plus the row's spectra summary."""
    record = spectra.row_spectrum(port, int(row), donor)
    table = thickness_table(spectra.pack_spectra([record]), support, hands=hands, workers=1)
    out = {hand_key(h): table[h][0] for h in hands}
    out.update(born=table["born"][0], row_ids=record[0], mu=record[6], n_p=len(record[1]), n_b=record[2].shape[1],
               reused_points=record[8])
    return out


_SUPPORT: ThicknessSupport | None = None
_HANDS: tuple = (1,)


def _init_worker(hkl, hkl_transform, res, cfg, alternate_hand, ug_cache_dir, support, hands):
    """Build this worker's Bloch port once and keep the support and hands."""
    global _SUPPORT, _HANDS
    spectra._init_worker(hkl, hkl_transform, res, cfg, alternate_hand, ug_cache_dir)
    _SUPPORT, _HANDS = support, tuple(hands)


def _chunk_worker(rows):
    """Table entries of a chunk of rows with this worker's port."""
    if spectra._PORT is None or _SUPPORT is None:
        raise RuntimeError("response worker was not initialized")
    return [row_response(spectra._PORT, int(r), _SUPPORT, _HANDS) for r in rows]


# ----------------------------------------------------------------------------- storage


def _write_identity(directory: Path, ident: dict) -> None:
    (directory / "identity.json").write_text(spectra.identity_text(ident))


def _read_only(path: Path) -> np.ndarray:
    values = np.load(path)
    values.flags.writeable = False
    return values


def load_response_table(directory, ident: dict | None = None) -> dict:
    """Read a response table into read-only arrays; raise if its identity differs from ``ident``."""
    directory = Path(directory)
    stored = (directory / "identity.json").read_text()
    if ident is not None and stored != spectra.identity_text(ident):
        raise ValueError("response table identity differs from the request")
    meta = json.loads(stored)
    hands = tuple(int(h) for h in meta["hands"])
    table = {"t": _read_only(directory / "t.npy"), "support": meta["support"], "hands": hands,
             "born": _read_only(directory / "born.npy"), "identity": meta}
    for h in hands:
        table[h] = _read_only(directory / f"{hand_key(h)}.npy")
    for key in ROW_KEYS:
        table[key] = _read_only(directory / f"{key}.npy")
    validate_response_table(table)
    return table


def validate_response_table(table: dict, label: str = "response table") -> int:
    """Number of rows of a response table; inconsistent, negative or non-finite tables fail closed."""
    n = nested_selection.validate_table(table, ThicknessSupport(**table["support"]), table["hands"], label)
    for key in ROW_KEYS[2:]:
        if len(np.asarray(table[key])) != n:
            raise ValueError(f"{label}: {key} has {len(np.asarray(table[key]))} values, expected {n}")
    return n


# ----------------------------------------------------------------------------- driver


def _summary(table: dict) -> dict:
    """Row count, total profile points, median and maximum basis size and largest |mu| (as spectra_stage records)."""
    return {
        "n_rows": int(len(table["row_ids"])),
        "profile_points_total": int(np.sum(table["n_p"])),
        "basis_median": float(np.median(table["n_b"])),
        "basis_max": int(np.max(table["n_b"])),
        "mu_max": float(np.max(np.abs(table["mu"]))),
    }


def compute_response_table(hkl, res, cfg: Config, rows, support: ThicknessSupport, *, hands=(1,),
                           hkl_transform: str = "identity", alternate_hand: bool = False, workers: int = 1,
                           cache_dir=None, ug_cache_dir=None) -> tuple[dict, dict]:
    """Response table of ``rows`` (in the given order) for one Config and support, and a provenance record.

    Rows must be unique and nonnegative. A cached table is used only when its stored identity equals the request; an
    unreadable or mismatched one is recomputed and replaced. Rows are written to the table files as they arrive,
    so the parent never holds more than the table; without ``cache_dir`` the files go to a temporary directory that
    is removed after the table is read back. Device backends use one spawned worker, as in
    spectra_stage.compute_spectra.
    """
    rows = np.asarray(rows, dtype=int)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("response tables need a non-empty one-dimensional row list")
    if np.any(rows < 0):
        raise ValueError("response table row indices must be nonnegative")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("response table row indices must be unique")
    hands = tuple(dict.fromkeys(int(h) for h in hands))
    if not hands or any(h not in (1, -1) for h in hands):
        raise ValueError("hands must be +1 (model) and/or -1 (inverse)")
    backend = str(getattr(cfg, "compute_backend", "scipy") or "scipy").lower()
    if backend in spectra.UNREPRESENTED_BACKENDS:
        raise ValueError(f"response tables need the exact row eigensystems; compute_backend {backend!r} is a reduction")
    from grond.bloch import shared_grid

    grid_port = None
    if shared_grid.mode() != "off" and backend not in JAX_BACKENDS | JAX_PADDED_BACKENDS:
        if ug_cache_dir is None and cache_dir is not None:
            # One U_g cube per model for every variant of a selection (dose and cone do not change it).
            ug_cache_dir = Path(cache_dir) / "ug_cache"
        grid_port = spectra.build_port(str(hkl), str(hkl_transform), str(res), cfg, bool(alternate_hand), ug_cache_dir)
        path, reason = shared_grid.choose_path(grid_port, rows, workers, purpose="table")
        print(f"response table evaluator for {len(rows)} rows: {path} ({reason})", flush=True)
        if path != "shared_grid":
            grid_port = None
    ident = response_identity(hkl, res, cfg, rows, support, hands, hkl_transform=hkl_transform,
                              alternate_hand=alternate_hand)
    if grid_port is not None:
        ident["evaluator"] = {"shared_grid": True, "per_degree": shared_grid.per_degree(),
                              "precision": shared_grid.precision(), "margin": shared_grid.MARGIN,
                              "source_sha256": spectra.file_sha256(shared_grid.__file__)}
    key = spectra.identity_key(ident)
    final = None if cache_dir is None else Path(cache_dir) / f"response_{key}"
    info = {"key": key, "cache": None if final is None else str(final), "identity": ident}
    if final is not None and final.exists():
        try:
            table = load_response_table(final, ident)
            if not np.array_equal(table["row_ids"], rows):
                raise ValueError("cached response table rows differ from the request")
            return table, {**info, "cache_status": "hit", "seconds": 0.0, **_summary(table)}
        except Exception as exc:  # unreadable or foreign table under our key: recompute and replace it
            info["cache_rebuild_reason"] = f"{type(exc).__name__}: {str(exc)[:240]}"
            print(f"warning: response table {final} not usable ({info['cache_rebuild_reason']}); recomputing", flush=True)

    t0 = time.perf_counter()
    parent = Path(cache_dir) if final is not None else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="response_partial_", dir=parent))
    try:
        t = support.grid()
        np.save(work / "t.npy", t)
        arrays = {k: np.lib.format.open_memmap(work / f"{k}.npy", mode="w+", dtype=float, shape=(len(rows), len(t)))
                  for k in [*(hand_key(h) for h in hands), "born"]}
        per_row = {"row_ids": np.empty(len(rows), int), "mu": np.empty(len(rows)), "n_p": np.empty(len(rows), int),
                   "n_b": np.empty(len(rows), int), "reused_points": np.empty(len(rows), int)}
        device = backend in JAX_BACKENDS | JAX_PADDED_BACKENDS
        workers = 1 if device else max(1, int(workers))

        def store(position: int, entry: dict) -> None:
            for name, values in arrays.items():
                values[position] = entry[name]
            for name, values in per_row.items():
                values[position] = entry[name]

        if grid_port is not None:
            arrays_grid, _diag = shared_grid.response_table(grid_port, rows, support, hands=hands, workers=workers)
            for name, values in arrays.items():
                values[:] = arrays_grid[1 if name == "hand_plus" else -1 if name == "hand_minus" else name]
            for name, values in per_row.items():
                values[:] = arrays_grid[name]
            del arrays_grid
        elif device or workers > 1:
            # One or a few rows per task keeps every worker busy to the end; results arrive in row order.
            size = max(1, min(8, len(rows) // (4 * workers)))
            chunks = [rows[i:i + size] for i in range(0, len(rows), size)]
            context = mp.get_context("spawn") if device else spectra._pool_context()
            with ProcessPoolExecutor(max_workers=min(workers, len(chunks)), mp_context=context, initializer=_init_worker,
                                     initargs=(str(hkl), str(hkl_transform), str(res), cfg, bool(alternate_hand),
                                               None if ug_cache_dir is None else str(ug_cache_dir), support,
                                               hands)) as pool:
                position = 0
                for part in pool.map(_chunk_worker, chunks):
                    for entry in part:
                        store(position, entry)
                        position += 1
        else:
            port = spectra.build_port(str(hkl), str(hkl_transform), str(res), cfg, bool(alternate_hand), ug_cache_dir)
            for position, row in enumerate(rows):
                store(position, row_response(port, int(row), support, hands))
        if not np.array_equal(per_row["row_ids"], rows):
            raise RuntimeError("response workers returned rows out of order")
        for values in arrays.values():
            values.flush()
        arrays.clear()  # release the write maps before the table is read back
        for name, values in per_row.items():
            np.save(work / f"{name}.npy", values)
        _write_identity(work, ident)
        validate_response_table(load_response_table(work, ident))
        seconds = time.perf_counter() - t0
        if final is not None:
            if final.exists():
                shutil.rmtree(final)
            os.replace(work, final)
            table = load_response_table(final, ident)
            status = "rebuilt" if "cache_rebuild_reason" in info else "built"
        else:
            table = load_response_table(work, ident)
            shutil.rmtree(work)
            status = "disabled"
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return table, {**info, "cache_status": status, "seconds": seconds, "workers": workers,
                   "spectral_backend": backend if device else "scipy", "device_process": "spawn" if device else None,
                   "reused_points": int(np.sum(table["reused_points"])), **_summary(table)}


def main(argv=None) -> None:
    """Compute the response table of the listed rows into --out-dir (refuses an existing directory)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hkl", required=True, help="XDS INTEGRATE.HKL or a GROND DIALS/PETS row bridge")
    ap.add_argument("--hkl-transform", default="identity")
    ap.add_argument("--res", required=True)
    ap.add_argument("--alternate-hand", action="store_true", help="Invert the model coordinates first")
    ap.add_argument("--config-json", required=True, help="JSON object of bloch Config fields (every other field default)")
    ap.add_argument("--row-index-file", required=True, help="Zero-based row indices, one per line")
    ap.add_argument("--hands", type=int, nargs="+", default=[1])
    ap.add_argument("--t-min-A", type=float, default=ThicknessSupport().t_min_A)
    ap.add_argument("--t-max-A", type=float, default=ThicknessSupport().t_max_A)
    ap.add_argument("--dt-A", type=float, default=ThicknessSupport().dt_A)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ug-cache-dir", default="")
    ap.add_argument("--out-dir", required=True, help="New directory; the table is written to response_<key> inside it")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    if out.exists():
        raise SystemExit(f"refusing to write into existing {out}; choose a new output directory")
    cfg = Config(**json.loads(Path(args.config_json).read_text()))
    rows = np.asarray([int(v) for v in Path(args.row_index_file).read_text().split()], dtype=int)
    support = ThicknessSupport(args.t_min_A, args.t_max_A, args.dt_A)
    table, info = compute_response_table(args.hkl, args.res, cfg, rows, support, hands=args.hands,
                                         hkl_transform=args.hkl_transform, alternate_hand=args.alternate_hand,
                                         workers=args.workers, cache_dir=out, ug_cache_dir=args.ug_cache_dir or None)
    (out / "response_summary.json").write_text(json.dumps(info, indent=1, default=str) + "\n")
    print(f"response table: {info['n_rows']} rows, {info['profile_points_total']} profile points, basis median "
          f"{info['basis_median']:g} max {info['basis_max']}, {info['seconds']:.1f} s -> {info['cache']}", flush=True)


if __name__ == "__main__":
    main()
