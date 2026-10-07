#!/usr/bin/env python3
"""Per-row Bloch spectra for nested nuisance selection.

For every requested row the production row plan (``XDSBlochPort.row_prediction_plan``) gives the rocking-profile
weights and one Hamiltonian per profile point, for any Config: fixed or per-row (``dials_bbox``) Sg windows,
``n_points``, the re-centred transverse cone (``transverse_target`` sigma and points), dose B, ``max_support_sg``,
``n_basis``, PETS exact rows. Each Hamiltonian is split exactly as H = H0 + i*mu*I (non-uniform absorption fails
closed) and H0 is diagonalised once. The packed record per row has the layout of
tests/reference/reference_lib.py pack_spectra: profile weights, eigenvalues, couplings
V[1, j] conj(V[0, j]), the Born terms |H10|^2 and Re(H11 - H00), and mu. Spectra do not depend on thickness,
thickness distribution, uniform absorption strength or hand, so one spectra set serves every thickness candidate
of grond.regularize.nested_selection.

CPU rows are computed in parallel (one port per worker process). Device spectra use one spawned process, which
exits before CPU nuisance evaluation begins; the calling process never initializes JAX through this stage.
Standalone scripts calling compute_spectra with a device backend must use an ``if __name__ == "__main__":`` guard.
Spectra are cached by a hash of the Config (thickness fields
excluded), the model file, the reflection file, the transform, the row list and the engine sources. Every profile
point also stores a digest of its matrix bytes, so a later Config can reuse an identical matrix's spectrum exactly
(every nonzero cone width solves the same untilted block).

  python -m grond.legacy.engine.spectra_stage --hkl INTEGRATE.HKL --res model.res --config-json cfg.json \
      --out spectra.npz [--row-index-file rows.txt] [--workers 8] [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import multiprocessing as mp
import os
import platform
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
from grond.bloch.constants import BETHE_BACKENDS, JAX_BACKENDS, JAX_PADDED_BACKENDS  # noqa: E402
from grond.bloch.model import alternate_hand_model, parse_model  # noqa: E402
from grond.bloch.numerics import hermitian_eigh  # noqa: E402
from grond.bloch.source import physics_source_files  # noqa: E402
from grond.io.xds import XDSData  # noqa: E402

SPECTRA_SCHEMA = "grond-spectra-v2"
UNIFORM_ABSORPTION_RTOL = 2e-6
# Backends whose rows are not the exact eigensystem of the row-plan Hamiltonians (the Bethe/Schur strong-beam
# reduction). Their spectra would describe other physics than every engine stage that uses the pick, so they are
# refused.
UNREPRESENTED_BACKENDS = frozenset(BETHE_BACKENDS)
# Config fields that change no spectrum: the thickness distribution and its evaluator.
THICKNESS_FIELDS = (
    "thickness_A",
    "thickness_points",
    "thickness_model",
    "thickness_grad_frac",
    "thickness_sigma_frac",
    "thickness_gamma_shape",
    "thickness_evaluator",
    "thickness_min_A",
    "thickness_max_A",
    "thickness_table_dt_A",
)


# ----------------------------------------------------------------------------- one matrix


def split_uniform_absorption(H, rtol: float = UNIFORM_ABSORPTION_RTOL) -> tuple[np.ndarray, float]:
    """Exact split H = H0 + i*mu*I with H0 Hermitian; anything else fails closed.

    Production matrices may be complex64, so the anti-Hermitian part is compared to i*mu*I at single-precision
    tolerance and H0 is the exact Hermitian part (H + H^H)/2.
    """
    H = np.asarray(H, dtype=np.complex128)
    herm = 0.5 * (H + H.conj().T)
    anti = 0.5 * (H - H.conj().T)
    mu_diag = np.imag(np.diag(anti))
    mu = float(np.mean(mu_diag))
    scale = max(float(np.max(np.abs(H))), 1e-30)
    off = anti - np.diag(np.diag(anti))
    if float(np.max(np.abs(off), initial=0.0)) > rtol * scale or float(np.max(np.abs(mu_diag - mu))) > rtol * scale:
        raise ValueError(
            "spectra need a Hermitian Hamiltonian plus a uniform absorptive potential i*mu*I; this absorption "
            "model is non-uniform or off-diagonal (use absorption_mode none, mean or midgley_like with absorption_frac 0)"
        )
    return herm, mu


def matrix_spectrum(H, absorptive: bool):
    """(eigenvalues, coupling V[1, j] conj(V[0, j]), |H10|^2, Re(H11 - H00), mu) of one profile-point matrix."""
    H = np.asarray(H, dtype=np.complex128)
    if absorptive:
        H0, mu = split_uniform_absorption(H)
    else:
        H0, mu = 0.5 * (H + H.conj().T), 0.0
    lam, V = hermitian_eigh(H0)
    return lam, V[1, :] * np.conj(V[0, :]), float(abs(H[1, 0]) ** 2), float(np.real(H[1, 1] - H[0, 0])), mu


def matrix_digests(matrices, absorptive: bool) -> list[bytes]:
    """32-character hex digest of every profile-point matrix; equal digests mean byte-identical matrices."""
    head = hashlib.sha256(f"{SPECTRA_SCHEMA}|absorptive={bool(absorptive)}".encode())
    static = getattr(matrices, "static", None)
    if static is None:  # a plain list of dense matrices
        pairs = [(np.ascontiguousarray(H), b"") for H in matrices]
    else:  # ProfileHamiltonians: one static matrix plus a real excitation diagonal per point
        head.update(f"{static.dtype}{static.shape}".encode())
        head.update(np.ascontiguousarray(static).tobytes())
        pairs = [(np.ascontiguousarray(e), b"diag") for e in matrices.excitation]
    out = []
    for arr, tag in pairs:
        h = head.copy()
        h.update(tag + f"{arr.dtype}{arr.shape}".encode())
        h.update(arr.tobytes())
        out.append(h.hexdigest()[:32].encode())
    return out


def row_spectrum(port, row: int, donor: dict | None = None):
    """Packed record (row, w, lam[P, N], c[P, N], h10[P], delta[P], mu, digest[P]) of one row.

    A profile point whose matrix digest is in ``donor`` takes that spectrum instead of a new eigensolve.
    """
    absorptive = bool(port.absorption_enabled())
    plan = port.row_prediction_plan(int(row))
    w = np.asarray(plan["_profile_weights"], dtype=float)
    matrices = plan["_dyn_matrices"]
    keys = matrix_digests(matrices, absorptive)
    donor = donor or {}
    backend = str(getattr(getattr(port, "cfg", None), "compute_backend", "scipy") or "scipy").lower()
    if backend in JAX_BACKENDS | JAX_PADDED_BACKENDS:
        parts = _device_matrix_spectra(matrices, keys, donor, absorptive, backend,
                                       int(getattr(port.cfg, "jax_batch_size", 0) or 0))
    else:
        parts = [donor.get(key) or matrix_spectrum(matrices[i], absorptive) for i, key in enumerate(keys)]
    return (int(row), w, np.asarray([p[0] for p in parts]), np.asarray([p[1] for p in parts]),
            np.asarray([p[2] for p in parts]), np.asarray([p[3] for p in parts]),
            float(np.mean([p[4] for p in parts])), keys, sum(key in donor for key in keys))


def _device_matrix_spectra(matrices, keys, donor, absorptive, backend, batch_size):
    """Solve missing spectra on the requested device, retaining exact donor entries.

    Dense host buffers are bounded independently of the total number of cone
    and rocking nodes. Nonuniform absorption still fails the v2 contract.
    """
    from grond.bloch.backends import _jax_host_buffer_bytes
    from grond.bloch.spectral_gpu import hermitian_spectra

    parts = [donor.get(key) for key in keys]
    missing = [i for i, value in enumerate(parts) if value is None]
    if not missing:
        return parts
    n = np.asarray(matrices[missing[0]]).shape[0]
    # Include room for the source, Hermitian split, and contiguous device input.
    count = _jax_host_buffer_bytes() // (3 * n * n * np.dtype(complex).itemsize)
    if count < 1:
        raise MemoryError("one spectral matrix exceeds GROND_JAX_HOST_BUFFER_GIB; increase the explicit host budget")
    for start in range(0, len(missing), count):
        indices = missing[start:start + count]
        H0, details = [], []
        for i in indices:
            H = np.asarray(matrices[i], dtype=complex)
            hermitian, mu = split_uniform_absorption(H) if absorptive else (0.5 * (H + H.conj().T), 0.0)
            H0.append(hermitian)
            details.append((float(abs(H[1, 0]) ** 2), float(np.real(H[1, 1] - H[0, 0])), mu))
        lam, coupling = hermitian_spectra(np.asarray(H0), backend, batch_size)
        for j, i in enumerate(indices):
            parts[i] = (lam[j], coupling[j], *details[j])
    return parts


def load_donor(paths) -> dict:
    """Digest -> (lam, c, h10, delta, mu) of every profile point in earlier spectra files of this engine."""
    donor = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            identity = json.loads(str(z["identity_json"]))
            if identity.get("schema") != SPECTRA_SCHEMA or identity.get("engine_sha256") != engine_source_sha256():
                raise ValueError(f"donor spectra {path} have a different or missing schema/engine identity")
            n_p, n_b, lam, c, h10, delta, mu, digest = (z[k] for k in ("n_p", "n_b", "lam", "c", "h10", "delta", "mu",
                                                                        "digest"))
        a = p = 0
        for i, (points, beams) in enumerate(zip(n_p, n_b, strict=True)):
            for _ in range(int(points)):
                # mu is the same at every point of a row: the excitation errors only add real diagonal terms
                donor[bytes(digest[p])] = (lam[a:a + beams], c[a:a + beams], float(h10[p]), float(delta[p]), float(mu[i]))
                a += int(beams)
                p += 1
    return donor


def pack_spectra(records: list) -> dict:
    """Packed spectra in record order (the layout of the reference pack_spectra)."""
    n_p = np.asarray([len(t[1]) for t in records], dtype=int)
    n_b = np.asarray([t[2].shape[1] for t in records], dtype=int)
    return {
        "row_ids": np.asarray([t[0] for t in records], dtype=int),
        "n_p": n_p,
        "n_b": n_b,
        "w": np.concatenate([t[1] for t in records]),
        "lam": np.concatenate([t[2].ravel() for t in records]),
        "c": np.concatenate([t[3].ravel() for t in records]),
        "h10": np.concatenate([t[4] for t in records]),
        "delta": np.concatenate([t[5] for t in records]),
        "mu": np.asarray([t[6] for t in records], dtype=float),
        "digest": np.asarray([key for t in records for key in t[7]], dtype="S32"),
    }


# ----------------------------------------------------------------------------- workers

_PORT = None
_DONOR: dict = {}


def build_port(hkl: str, hkl_transform: str, res: str, cfg: Config, alternate_hand: bool = False, ug_cache_dir=None):
    """Build the Bloch port for one reflection file and model, optionally with the model hand inverted."""

    from grond.legacy.engine.bloch_engine import XDSBlochPort

    data = XDSData(Path(hkl), hkl_transform=hkl_transform)
    model = parse_model(Path(res))
    if alternate_hand:
        model = alternate_hand_model(model)
    return XDSBlochPort(data, model, cfg, cache_dir=None if ug_cache_dir is None else Path(ug_cache_dir))


def _init_worker(hkl, hkl_transform, res, cfg, alternate_hand, ug_cache_dir, reuse=()):
    """Build this worker process's Bloch port and donor spectra once."""

    global _PORT, _DONOR
    _PORT = build_port(hkl, hkl_transform, res, cfg, alternate_hand, ug_cache_dir)
    _DONOR = load_donor(reuse)


def _chunk_worker(rows):
    """Return the spectra of a chunk of rows with this worker's port."""

    if _PORT is None:
        raise RuntimeError("spectra worker was not initialized")
    return [row_spectrum(_PORT, int(r), _DONOR) for r in rows]


def _pool_context():
    """Return the process start method: fork on Linux, spawn on macOS or where fork is unavailable."""

    # Linux forks the parent copy-on-write; macOS starts clean interpreters, as the engine prediction pool does
    # (callers of compute_spectra with workers > 1 then need the usual `if __name__ == "__main__":` guard).
    if platform.system() != "Darwin" and "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context("spawn")


# ----------------------------------------------------------------------------- cache identity


def file_sha256(path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def engine_source_sha256() -> str:
    """Hash of every forward-model source file and of this stage."""
    h = hashlib.sha256(SPECTRA_SCHEMA.encode())
    for path in (*physics_source_files(), Path(__file__).resolve()):
        h.update(path.name.encode())
        h.update(Path(path).read_bytes())
    return h.hexdigest()


def spectral_config(cfg: Config) -> dict:
    """Every Config field that can change a spectrum (thickness fields excluded)."""
    fields = dataclasses.asdict(cfg)
    for name in THICKNESS_FIELDS:
        fields.pop(name, None)
    return fields


def spectra_identity(hkl, res, cfg: Config, rows, *, hkl_transform: str = "identity", alternate_hand: bool = False) -> dict:
    """Return the record that keys a spectra cache: engine, input files, transform, hand, spectral config, rows."""

    return {
        "schema": SPECTRA_SCHEMA,
        "engine_sha256": engine_source_sha256(),
        "hkl_sha256": file_sha256(hkl),
        "res_sha256": file_sha256(res),
        "hkl_transform": str(hkl_transform),
        "model_hand": "alternate_fractional_xyz" if alternate_hand else "original",
        "config": spectral_config(cfg),
        "rows_sha256": hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest(),
        "n_rows": int(len(rows)),
    }


def identity_text(ident: dict) -> str:
    """Canonical JSON of an identity (NaN-safe: compared as text, never as parsed floats)."""
    return json.dumps(ident, sort_keys=True, default=str)


def identity_key(ident: dict) -> str:
    """Return the 24-character cache key of an identity record."""

    return hashlib.sha256(identity_text(ident).encode()).hexdigest()[:24]


def _read_cache(path: Path, ident: dict) -> dict:
    """Load a cached spectra file; raise if its stored identity differs from ident."""

    with np.load(path, allow_pickle=False) as z:
        if str(z["identity_json"]) != identity_text(ident):
            raise ValueError("cached spectra identity differs from the request")
        return {k: z[k] for k in z.files if k != "identity_json"}


def write_spectra(path: Path, spec: dict, ident: dict) -> None:
    """Atomic npz write of spectra plus their identity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".partial")
    try:
        with os.fdopen(fd, "wb") as fh:  # write through the handle: np.savez appends .npz to bare names
            np.savez(fh, identity_json=np.asarray(identity_text(ident)), **spec)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ----------------------------------------------------------------------------- driver


def compute_spectra(hkl, res, cfg: Config, rows, *, hkl_transform: str = "identity", alternate_hand: bool = False,
                    workers: int = 1, cache_dir=None, ug_cache_dir=None, reuse=()) -> tuple[dict, dict]:
    """Packed spectra of ``rows`` (in the given order) for one Config, and a provenance record.

    Rows must be unique and nonnegative. A cached file is used only when its stored identity equals the request; an unreadable
    or mismatched cache file is recomputed and replaced (recorded in the provenance). ``reuse`` lists spectra files of
    this engine whose byte-identical matrices are taken over instead of solved again.

    Device backends always use one spawned worker, even when workers=1. This keeps JAX and its threads out of the
    caller so later CPU stages can safely create process pools. Standalone callers need the usual main guard.
    """
    rows = np.asarray(rows, dtype=int)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("spectra need a non-empty one-dimensional row list")
    if np.any(rows < 0):
        raise ValueError("spectra row indices must be nonnegative")
    backend = str(getattr(cfg, "compute_backend", "scipy") or "scipy").lower()
    if backend in UNREPRESENTED_BACKENDS:
        raise ValueError(
            f"spectra are the exact eigensystems of the full row Hamiltonians; compute_backend {backend!r} predicts "
            "rows from another Hamiltonian (Bethe/Schur reduction or the shared-grid predictor), so a selection on "
            "these spectra would not describe its rows; use compute_backend scipy"
        )
    if len(np.unique(rows)) != len(rows):
        raise ValueError("spectra row indices must be unique")
    ident = spectra_identity(hkl, res, cfg, rows, hkl_transform=hkl_transform, alternate_hand=alternate_hand)
    key = identity_key(ident)
    path = None if cache_dir is None else Path(cache_dir) / f"spectra_{key}.npz"
    info = {"key": key, "cache": None if path is None else str(path), "identity": ident}
    if path is not None and path.exists():
        try:
            spec = _read_cache(path, ident)
            if not np.array_equal(spec["row_ids"], rows):
                raise ValueError("cached spectra rows differ from the request")
            return spec, {**info, "cache_status": "hit", "seconds": 0.0, **_summary(spec)}
        except Exception as exc:  # unreadable or foreign file under our key: recompute and replace it
            info["cache_rebuild_reason"] = f"{type(exc).__name__}: {str(exc)[:240]}"
            print(f"warning: spectra cache {path} not usable ({info['cache_rebuild_reason']}); recomputing", flush=True)
    t0 = time.perf_counter()
    workers = max(1, int(workers))
    device_spectra = backend in JAX_BACKENDS | JAX_PADDED_BACKENDS
    if device_spectra:
        # JAX is initialized only in the spawned child, which closes before CPU nuisance evaluation can fork.
        workers = 1
    if device_spectra or (workers > 1 and len(rows) > 1):
        if device_spectra:
            context = mp.get_context("spawn")
            chunks = [rows[start:start + 32] for start in range(0, len(rows), 32)]
        else:
            context = _pool_context()
            chunks = [c for c in np.array_split(rows, max(1, min(len(rows), 4 * workers))) if len(c)]
        # A worker that dies (or cannot start, e.g. a spawned child re-running an unguarded caller script) raises
        # BrokenProcessPool here instead of hanging the stage.
        with ProcessPoolExecutor(max_workers=min(workers, len(chunks)), mp_context=context,
                                 initializer=_init_worker,
                                 initargs=(str(hkl), str(hkl_transform), str(res), cfg, bool(alternate_hand),
                                           None if ug_cache_dir is None else str(ug_cache_dir),
                                           tuple(str(p) for p in reuse))) as pool:
            parts = list(pool.map(_chunk_worker, chunks))
        records = [r for part in parts for r in part]
    else:
        port = build_port(str(hkl), str(hkl_transform), str(res), cfg, bool(alternate_hand), ug_cache_dir)
        donor = load_donor(reuse)
        records = [row_spectrum(port, int(r), donor) for r in rows]
    if [r[0] for r in records] != [int(r) for r in rows]:
        raise RuntimeError("spectra workers returned rows out of order")
    spec = pack_spectra(records)
    seconds = time.perf_counter() - t0
    status = "built"
    if path is not None:
        write_spectra(path, spec, ident)
        status = "rebuilt" if "cache_rebuild_reason" in info else "built"
    return spec, {**info, "cache_status": status if path is not None else "disabled", "seconds": seconds,
                  "workers": workers, "spectral_backend": backend if device_spectra else "scipy",
                  "device_process": "spawn" if device_spectra else None,
                  "reused_points": int(sum(r[8] for r in records)), **_summary(spec)}


def _summary(spec: dict) -> dict:
    """Return the row count, total profile points, median and maximum basis size and largest |mu| of spectra."""

    return {
        "n_rows": int(len(spec["row_ids"])),
        "profile_points_total": int(np.sum(spec["n_p"])),
        "basis_median": float(np.median(spec["n_b"])),
        "basis_max": int(np.max(spec["n_b"])),
        "mu_max": float(np.max(np.abs(spec["mu"]))),
    }


def main(argv=None) -> None:
    """Compute the spectra of the selected rows and write them with their identity to --out.

    Also writes a JSON summary beside the .npz; refuses to overwrite an existing output.
    """

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hkl", required=True, help="XDS INTEGRATE.HKL or a GROND DIALS/PETS row bridge")
    ap.add_argument("--hkl-transform", default="identity")
    ap.add_argument("--res", required=True)
    ap.add_argument("--alternate-hand", action="store_true", help="Invert the model coordinates first")
    ap.add_argument("--config-json", required=True, help="JSON object of bloch Config fields (every other field default)")
    ap.add_argument("--row-index-file", default="", help="Zero-based row indices, one per line; default: row filters below")
    ap.add_argument("--target-mode", default="iobs")
    ap.add_argument("--include-nonpositive", action="store_true")
    ap.add_argument("--min-d-spacing", type=float, default=0.8)
    ap.add_argument("--min-peak", type=float, default=90.0)
    ap.add_argument("--geometry-percentile", type=float, default=99.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--ug-cache-dir", default="")
    ap.add_argument("--out", required=True, help="Output .npz (spectra plus identity); a JSON summary is written beside it")
    args = ap.parse_args(argv)

    cfg = Config(**json.loads(Path(args.config_json).read_text()))
    if args.row_index_file:
        rows = np.asarray([int(v) for v in Path(args.row_index_file).read_text().split()], dtype=int)
    else:
        from grond.regularize.observations import row_selection, target_arrays

        data = XDSData(Path(args.hkl), hkl_transform=args.hkl_transform)
        target, _sig = target_arrays(data, args.target_mode)
        rows = np.asarray(row_selection(data, target, not args.include_nonpositive, args.min_d_spacing, args.min_peak,
                                        0, args.geometry_percentile), dtype=int)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}; choose a new output path")
    spec, info = compute_spectra(args.hkl, args.res, cfg, rows, hkl_transform=args.hkl_transform,
                                 alternate_hand=args.alternate_hand, workers=args.workers,
                                 cache_dir=args.cache_dir or None, ug_cache_dir=args.ug_cache_dir or None)
    write_spectra(out, spec, info["identity"])
    out.with_suffix(".json").write_text(json.dumps({k: v for k, v in info.items()}, indent=1, default=str) + "\n")
    print(f"spectra: {info['n_rows']} rows, {info['profile_points_total']} profile points, basis median "
          f"{info['basis_median']:g} max {info['basis_max']}, cache {info['cache_status']}, {info['seconds']:.1f} s -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
