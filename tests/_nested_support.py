"""Synthetic inputs for the nested-selection, spectra-stage and gamma-thickness engine tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tests._support import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))
import reference_lib as G  # noqa: E402

HEADER = (
    "!X-RAY_WAVELENGTH= 0.025079\n"
    "!STARTING_FRAME= 1.0\n"
    "!STARTING_ANGLE= 0.0\n"
    "!OSCILLATION_RANGE= 0.5\n"
    "!ROTATION_AXIS= 0.0 1.0 0.0\n"
    "!INCIDENT_BEAM_DIRECTION= 0.0 0.0 1.0\n"
    "!UNIT_CELL_CONSTANTS= 8.0 9.0 10.0 90.0 90.0 90.0\n"
    "!UNIT_CELL_A-AXIS= 8.0 0.0 0.0\n"
    "!UNIT_CELL_B-AXIS= 0.0 9.0 0.0\n"
    "!UNIT_CELL_C-AXIS= 0.0 0.0 10.0\n"
    "!SPACE_GROUP_NUMBER= 19\n"
)
MODEL = (
    "TITL nested selection fixture\n"
    "CELL 0.025079 8.0 9.0 10.0 90.0 90.0 90.0\n"
    "LATT -1\n"
    "SYMM 0.5-X, -Y, 0.5+Z\n"
    "SYMM -X, 0.5+Y, 0.5-Z\n"
    "SYMM 0.5+X, 0.5-Y, -Z\n"
    "SFAC C 1 1 1 1 1 1 1 1 0\n"
    "SFAC O 1 1 1 1 1 1 1 1 0\n"
    "C1 1 0.13 0.27 0.39 11.0 0.02\n"
    "C2 1 0.31 0.12 0.21 11.0 0.03\n"
    "O1 2 0.42 0.37 0.08 11.0 0.025\n"
    "HKLF 4\n"
    "END\n"
)
MAX_INDEX = 4
MIN_D_A = 1.2


def _row(h, z, iobs, *, bbox=None) -> str:
    alf0, bet0 = bbox if bbox is not None else (0.0, 0.0)
    return (f" {h[0]} {h[1]} {h[2]} {iobs:.3f} {0.05 * iobs + 1.0:.3f} 0.0 0.0 {z:.5f} 0.05 100.0 95.0 0.0"
            f" 0.0 0.0 0.0 {alf0:.4f} {bet0:.4f} 0.0 0.0 0.0 0\n")


def write_dataset(tmp_path: Path, *, dials_bbox: bool = False, seed: int = 0, max_rows: int = 24) -> tuple[Path, Path]:
    """INTEGRATE.HKL-style rows at their Bragg crossings (P2_12_12_1 cell 8 x 9 x 10 A) plus a three-atom model."""
    from grond.engine.bloch_engine import Config, XDSBlochPort, parse_model
    from grond.io.xds import XDSData

    tmp_path.mkdir(parents=True, exist_ok=True)
    res = tmp_path / "model.res"
    res.write_text(MODEL)
    probe = tmp_path / "probe.HKL"
    probe.write_text(HEADER + _row((1, 0, 0), 1.0, 100.0))
    port = XDSBlochPort(XDSData(probe), parse_model(res), Config(n_basis=4, n_points=1, topup_d_min=2.0, ug_d_min=2.0))
    span = range(-MAX_INDEX, MAX_INDEX + 1)
    candidates = np.asarray([(h, k, m) for h in span for k in span for m in range(0, MAX_INDEX + 1) if (h, k, m) != (0, 0, 0)
                             and np.sqrt((h / 8.0) ** 2 + (k / 9.0) ** 2 + (m / 10.0) ** 2) <= 1.0 / MIN_D_A], dtype=int)
    omegas = np.linspace(-45.0, 45.0, 901)
    sg = np.asarray([port.sg_for_hkls(candidates, om) for om in omegas])  # (omega, hkl)
    rng = np.random.default_rng(seed)
    rows = []
    for j, hkl in enumerate(candidates):
        crossings = np.where(np.diff(np.sign(sg[:, j])) != 0)[0]
        for i in crossings[:1]:
            lo, hi = omegas[i], omegas[i + 1]
            for _ in range(40):  # bisect the Bragg crossing
                mid = 0.5 * (lo + hi)
                s_lo = port.sg_for_hkls(np.asarray([hkl]), lo)[0]
                s_mid = port.sg_for_hkls(np.asarray([hkl]), mid)[0]
                lo, hi = (lo, mid) if np.sign(s_mid) != np.sign(s_lo) else (mid, hi)
            z = 1.0 + 2.0 * 0.5 * (lo + hi)
            rows.append((tuple(int(v) for v in hkl), z, float(rng.uniform(50.0, 500.0))))
    rows = rows[:: max(1, len(rows) // max_rows)][:max_rows]
    header = HEADER + ("!DIALS_BBOX_Z_RANGE_AS_ALF0_BET0= TRUE\n" if dials_bbox else "")
    hkl_path = tmp_path / "INTEGRATE.HKL"
    hkl_path.write_text(header + "".join(
        _row(h, z, i, bbox=(z - 1.2 - 0.3 * (k % 3), z + 1.1 + 0.2 * (k % 2)) if dials_bbox else None)
        for k, (h, z, i) in enumerate(rows)))
    probe.unlink()
    return hkl_path, res


# ----------------------------------------------------------------------------- random spectra


def random_hermitian(n: int, seed: int, coupling: float = 2e-3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    off = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) * coupling
    H0 = np.diag(rng.normal(scale=1e-2, size=n)).astype(complex) + off + off.conj().T
    H0[0, 0] = 0.0
    return H0


def random_spectra(n_rows: int, n_beams: int = 6, n_points: int = 3, seed: int = 1, shift: float = 0.0,
                   mu: float = 0.0) -> dict:
    """Packed reference-format spectra of random rows: a rocking profile of n_points shifted target excitations."""
    records = []
    for r in range(n_rows):
        base = random_hermitian(n_beams, seed * 1000 + r)
        mats = []
        for p in range(n_points):
            H = base.copy()
            H[1, 1] += (p - n_points // 2) * 1e-3 + shift
            mats.append(H + 1j * mu * np.eye(n_beams))
        parts = [G.matrix_spectrum(H, mu > 0.0) for H in mats]
        records.append((r, np.full(n_points, 1.0 / n_points), *[np.asarray([q[i] for q in parts]) for i in range(4)],
                        float(np.mean([q[4] for q in parts]))))
    return G.pack_spectra(records)


def synthetic_rows(n_rows: int, seed: int = 5):
    """(keys, ang) for random reflections of mmm: orbit keys |h|,|k|,|l| and a sorted scan angle."""
    rng = np.random.default_rng(seed)
    hkl = rng.integers(-7, 8, size=(n_rows, 3))
    keys = np.asarray([G.orbit_key(*h) for h in hkl], dtype=object)
    ang = np.sort(rng.uniform(-35.0, 35.0, n_rows))
    return keys, ang
