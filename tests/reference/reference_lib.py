"""Independent reference implementation of the held-out definitions, used by the tests to check grond.validation.

It is written separately from the engine so that the engine's nested selection, scales and statistics are
checked against code that does not share their implementation.

Spectra       Per-row Bloch spectra. A Hamiltonian with uniform absorption is split exactly as
              H = H0 + i*mu*I (H0 Hermitian), and H0 is diagonalised once per profile point with eigh.
              Spectra do not depend on thickness, absorption strength or hand.
Intensities   Row intensities for any thickness quadrature, uniform absorption mu, hand and per-row
              path-length factor, from the spectra:
                  I(t) = |sum_j c_j exp(2*pi*i*s*lam_j*t)|^2 * exp(-4*pi*mu*t),  s = +1 (model), -1 (inverse)
              The inverse hand is exact because H_inv = conj(H0) + i*mu*I (real form factors, uniform mu).
Statistics    P-spline frame scales fitted on training orbits only (scale extrapolation clamped), held-out
              predictions, row R1 metrics, paired merged-orbit bootstrap and hand statistics.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from grond.regularize.spline_scales import PSplineScaleFitter  # noqa: E402

# Salt of the three development folds the tests compare against.
DEV_SALT = "GROND-DEV-20260925"


# ----------------------------------------------------------------------------- keys and partitions

def orbit_key(h: int, k: int, ell: int) -> str:
    """Crystal-independent Laue-orbit key for mmm (P2_12_12_1)."""
    return f"{abs(int(h))}|{abs(int(k))}|{abs(int(ell))}"


def hash_mod(salt: str, key: str, n: int) -> int:
    return int(hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()[:12], 16) % n


# ----------------------------------------------------------------------------- spectra

def split_uniform_absorption(H: np.ndarray, rtol: float = 2e-6) -> tuple[np.ndarray, float]:
    """Exact split H = H0 + i*mu*I with H0 Hermitian; fail closed for anything else.

    The production matrices are stored in complex64, so the anti-Hermitian part is compared to
    i*mu*I at single-precision tolerance.
    """
    H = np.asarray(H, dtype=np.complex128)
    herm = 0.5 * (H + H.conj().T)
    anti = 0.5 * (H - H.conj().T)          # = i*mu*I for uniform absorption
    mu_diag = np.imag(np.diag(anti))
    mu = float(np.mean(mu_diag))
    scale = max(float(np.max(np.abs(H))), 1e-30)
    off = anti - np.diag(np.diag(anti))
    if float(np.max(np.abs(off), initial=0.0)) > rtol * scale or float(np.max(np.abs(mu_diag - mu))) > rtol * scale:
        raise ValueError(
            "Hamiltonian is not Hermitian + uniform i*mu*I; the reference does not support this absorption model"
        )
    return herm, mu


def matrix_spectrum(H: np.ndarray, absorptive: bool):
    """(lam, coupling, |H10|^2, Re(H11-H00), mu) for one profile-point Hamiltonian."""
    H = np.asarray(H, dtype=np.complex128)
    if absorptive:
        H0, mu = split_uniform_absorption(H)
    else:
        H0, mu = 0.5 * (H + H.conj().T), 0.0
    lam, V = eigh(H0)
    return lam, V[1, :] * np.conj(V[0, :]), float(abs(H[1, 0]) ** 2), float(np.real(H[1, 1] - H[0, 0])), mu


def pack_spectra(records: list) -> dict:
    records = sorted(records, key=lambda t: t[0])
    n_p = np.asarray([len(t[1]) for t in records], dtype=int)
    n_b = np.asarray([t[2].shape[1] for t in records], dtype=int)
    return {
        "row_ids": np.asarray([t[0] for t in records], dtype=int),
        "n_p": n_p, "n_b": n_b,
        "w": np.concatenate([t[1] for t in records]),
        "lam": np.concatenate([t[2].ravel() for t in records]),
        "c": np.concatenate([t[3].ravel() for t in records]),
        "h10": np.concatenate([t[4] for t in records]),
        "delta": np.concatenate([t[5] for t in records]),
        "mu": np.asarray([t[6] for t in records], dtype=float),
    }


def spectra_views(spec: dict):
    """Yield per-row (w, lam[n_p, N], c[n_p, N], h10[n_p], delta[n_p], mu)."""
    op = ob = 0
    for i in range(len(spec["row_ids"])):
        p, n = int(spec["n_p"][i]), int(spec["n_b"][i])
        yield (spec["w"][op:op + p], spec["lam"][ob:ob + p * n].reshape(p, n),
               spec["c"][ob:ob + p * n].reshape(p, n), spec["h10"][op:op + p], spec["delta"][op:op + p],
               float(spec["mu"][i]))
        op += p
        ob += p * n


# ----------------------------------------------------------------------------- row intensities

def intensities(spec: dict, t_nodes, t_w, hand: int = 1, mu: float | None = None, path_factor=None):
    """Dynamical and Born row intensities for one thickness quadrature.

    hand=+1 is the loaded model, hand=-1 its exact enantiomer. mu=None uses the value stored with the spectra.
    path_factor (per row) multiplies every thickness node of that row.
    """
    t_nodes = np.asarray(t_nodes, dtype=float)
    t_w = np.asarray(t_w, dtype=float)
    n_rows = len(spec["row_ids"])
    dyn = np.empty(n_rows)
    born = np.empty(n_rows)
    s = 2j * np.pi * float(hand)
    for i, (w, lam, c, h10, delta, mu_a) in enumerate(spectra_views(spec)):
        m = mu_a if mu is None else float(mu)
        t = t_nodes * (1.0 if path_factor is None else float(path_factor[i]))
        amp = np.einsum("pj,pjt->pt", c, np.exp(s * lam[:, :, None] * t[None, None, :]))
        per_t = (np.abs(amp) ** 2) * np.exp(-4.0 * np.pi * m * t)[None, :]
        dyn[i] = max(float(w @ (per_t @ t_w)), 1e-30)
        arg = np.pi * delta[:, None] * t[None, :]
        sinc = np.where(np.abs(arg) >= 1e-12, np.sin(arg) / np.where(arg == 0, 1.0, arg), 1.0)
        born_pt = (2.0 * np.pi) ** 2 * h10[:, None] * (t[None, :] ** 2) * sinc ** 2
        born[i] = max(float(w @ (born_pt @ t_w)), 1e-30)
    return dyn, born


def node_intensities(spec: dict, t_nodes, path_factor=None, hands=(1, -1)):
    """Profile-summed per-node intensities, shared by every mu value and both hands.

    Returns {"t": (rows, n_t) actual thicknesses, "born": (rows, n_t), hand: (rows, n_t)} where
    dyn(mu) = sum_t t_w * exp(-4*pi*mu*t) * node[hand]; the inverse hand reuses conj(exp(2*pi*i*lam*t)).
    """
    t_nodes = np.asarray(t_nodes, dtype=float)
    n_rows = len(spec["row_ids"])
    out = {h: np.empty((n_rows, len(t_nodes))) for h in hands}
    out["born"] = np.empty((n_rows, len(t_nodes)))
    out["t"] = np.empty((n_rows, len(t_nodes)))
    for i, (w, lam, c, h10, delta, _mu) in enumerate(spectra_views(spec)):
        t = t_nodes * (1.0 if path_factor is None else float(path_factor[i]))
        E = np.exp(2j * np.pi * lam[:, :, None] * t[None, None, :])
        for h in hands:
            amp = np.einsum("pj,pjt->pt", c, E if h > 0 else E.conj())
            out[h][i] = w @ (np.abs(amp) ** 2)
        arg = np.pi * delta[:, None] * t[None, :]
        sinc = np.where(np.abs(arg) >= 1e-12, np.sin(arg) / np.where(arg == 0, 1.0, arg), 1.0)
        out["born"][i] = w @ ((2.0 * np.pi) ** 2 * h10[:, None] * (t[None, :] ** 2) * sinc ** 2)
        out["t"][i] = t
    return out


def collapse(node: np.ndarray, t_rows: np.ndarray, t_w, mu: float) -> np.ndarray:
    """Row intensities from node intensities for one uniform absorption mu (exact)."""
    return np.maximum(np.sum(node * (np.asarray(t_w)[None, :] * np.exp(-4.0 * np.pi * float(mu) * t_rows)), axis=1), 1e-30)


# ----------------------------------------------------------------------------- scales and statistics

def fit_scales_cv(y, sig, raw, ang, group, folds, n_folds: int, scale_cfg: dict, clamp: bool = True):
    """Held-out predictions: P-spline scales fitted on training orbits of each fold.

    folds < 0 marks rows outside the development set (lockbox); they are never fitted or predicted here.
    Returns (pred_heldout [nan where not predicted], pred_insample_dev, info).
    """
    fitter = PSplineScaleFitter(n_basis=int(scale_cfg.get("n_basis", 32)), objective=scale_cfg.get("objective", "row_mr"),
                                floor_factor=float(scale_cfg.get("floor_factor", 1e-12)))
    lam = scale_cfg.get("lambda", 1.0)
    pred = np.full(len(y), np.nan)
    info = []
    dev = folds >= 0
    for f in range(n_folds):
        train = dev & (folds != f)
        val = folds == f
        a = np.clip(ang, np.min(ang[train]), np.max(ang[train])) if clamp else ang
        sol = fitter.fit(y, sig, raw, a, group=group, fit_mask=train, lambda_smooth=lam)
        pred[val] = sol["prediction"][val]
        info.append({"fold": f, "edf": float(sol["edf"]), "lambda": float(sol["lambda"]),
                     "n_train": int(train.sum()), "n_val": int(val.sum())})
    a = np.clip(ang, np.min(ang[dev]), np.max(ang[dev])) if clamp else ang
    sol = fitter.fit(y, sig, raw, a, group=group, fit_mask=dev, lambda_smooth=lam)
    insample = np.where(dev, sol["prediction"], np.nan)
    return pred, insample, {"folds": info, "insample_edf": float(sol["edf"])}


def r1(fo, fc, mask) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.sum(np.abs(fo[mask] - fc[mask])) / max(float(np.sum(fo[mask])), 1e-12) * 100.0)


def metrics(y, sig, d, pred, sel=None) -> dict:
    """Amplitude R1 metrics on the rows in sel."""
    sel = np.ones(len(y), dtype=bool) if sel is None else np.asarray(sel, dtype=bool)
    sel = sel & np.isfinite(pred)
    fo = np.sqrt(np.maximum(y, 0.0))
    fc = np.sqrt(np.maximum(np.nan_to_num(pred), 0.0))
    obs = y > 3.0 * sig
    return {
        "r_all": r1(fo, fc, sel), "r_obs": r1(fo, fc, sel & obs), "r_strong": r1(fo, fc, sel & (y > 50.0 * sig)),
        "r_low": r1(fo, fc, sel & obs & (d > 2.0)), "r_mid": r1(fo, fc, sel & obs & (d <= 2.0) & (d > 1.2)),
        "r_high": r1(fo, fc, sel & obs & (d <= 1.2)),
        "n_all": int(sel.sum()), "n_obs": int((sel & obs).sum()),
    }


def paired_orbit_bootstrap(y, sig, pred_a, pred_b, orbit, shell, sel, n_boot: int = 2000, seed: int = 20260925):
    """Paired, shell-stratified merged-orbit bootstrap of R(a) - R(b) (pp) for R_all and R_obs.

    pred_b may be None for a single-arm CI of R(a).
    """
    fo = np.sqrt(np.maximum(y, 0.0))
    fa = np.sqrt(np.maximum(np.nan_to_num(pred_a), 0.0))
    fb = None if pred_b is None else np.sqrt(np.maximum(np.nan_to_num(pred_b), 0.0))
    sel = np.asarray(sel, dtype=bool) & np.isfinite(pred_a)
    if pred_b is not None:
        sel &= np.isfinite(pred_b)
    uniq, inv = np.unique(np.asarray(orbit)[sel], return_inverse=True)
    shells_o = np.zeros(len(uniq), dtype=int)
    shells_o[inv] = np.asarray(shell)[sel]
    rng = np.random.default_rng(seed)
    counts = np.zeros((n_boot, len(uniq)))
    for s in np.unique(shells_o):
        members = np.where(shells_o == s)[0]
        draws = rng.integers(0, len(members), size=(n_boot, len(members)))
        for b in range(n_boot):
            counts[b] += np.bincount(members[draws[b]], minlength=len(uniq))
    out = {}
    obs = (y > 3.0 * sig)[sel]
    for name, m in (("r_all", np.ones(sel.sum(), dtype=bool)), ("r_obs", obs)):
        den = np.bincount(inv, weights=np.where(m, fo[sel], 0.0), minlength=len(uniq))
        na = np.bincount(inv, weights=np.where(m, np.abs(fo[sel] - fa[sel]), 0.0), minlength=len(uniq))
        ra = 100.0 * (counts @ na) / np.maximum(counts @ den, 1e-12)
        point_a = 100.0 * na.sum() / max(den.sum(), 1e-12)
        if fb is None:
            out[name] = {"value": point_a, "ci95": [float(np.percentile(ra, 2.5)), float(np.percentile(ra, 97.5))]}
            continue
        nb = np.bincount(inv, weights=np.where(m, np.abs(fo[sel] - fb[sel]), 0.0), minlength=len(uniq))
        rb = 100.0 * (counts @ nb) / np.maximum(counts @ den, 1e-12)
        point_b = 100.0 * nb.sum() / max(den.sum(), 1e-12)
        dd = ra - rb
        out[name] = {"delta": point_a - point_b, "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                     "a": point_a, "b": point_b}
    out["n_rows"] = int(sel.sum())
    out["n_orbits"] = int(len(uniq))
    return out


def orbit_sign_test(y, pred_model, pred_inverse, orbit, sel) -> dict:
    """Per-orbit sign of sum(|Fo - Fc_inverse| - |Fo - Fc_model|); z > 0 favours the model hand."""
    sel = np.asarray(sel, dtype=bool) & np.isfinite(pred_model) & np.isfinite(pred_inverse)
    fo = np.sqrt(np.maximum(y, 0.0))
    fm = np.sqrt(np.maximum(np.nan_to_num(pred_model), 0.0))
    fi = np.sqrt(np.maximum(np.nan_to_num(pred_inverse), 0.0))
    uniq, inv = np.unique(np.asarray(orbit)[sel], return_inverse=True)
    s = np.bincount(inv, weights=(np.abs(fo - fi) - np.abs(fo - fm))[sel], minlength=len(uniq))
    n_pos, n_neg = int((s > 0).sum()), int((s < 0).sum())
    z = (n_pos - n_neg) / math.sqrt(max(n_pos + n_neg, 1))
    rows = (np.abs(fo - fi) - np.abs(fo - fm))[sel]
    zr = (int((rows > 0).sum()) - int((rows < 0).sum())) / math.sqrt(max(int((rows != 0).sum()), 1))
    return {"orbit_sign_z": z, "orbits_model_better": n_pos, "orbits_inverse_better": n_neg,
            "row_sign_z_uncalibrated": zr}


# ----------------------------------------------------------------------------- thickness distributions

def thickness_grid(t0: float, grad: float, points: int):
    """GROND's uniform-gradient quadrature: Gauss-Legendre nodes on t0*(1 +- grad), t > 0, weights summing to 1."""
    t0, grad = float(t0), float(grad)
    if not (math.isfinite(t0) and t0 > 0.0 and math.isfinite(grad) and 0.0 <= grad <= 1.0):
        raise ValueError(f"invalid thickness t={t0!r}, grad={grad!r}")
    if int(points) <= 1 or grad <= 0.0:
        return np.asarray([t0]), np.asarray([1.0])
    x, w = np.polynomial.legendre.leggauss(int(points))
    t = t0 * (1.0 + grad * x)
    ok = t > 0.0
    return t[ok], w[ok] / np.sum(w[ok])


def thickness_nodes(family: str, params: dict, n: int):
    """Quadrature (t, w) of a thickness distribution.

    wedge: uniform on t0*(1 +- grad) (GROND's uniform-gradient model, Gauss-Legendre).
    gamma: p(t) ~ t^(k-1) exp(-t/theta) with mean m = k*theta (generalised Gauss-Laguerre); k = 1 is an
           exponential distribution, large k a narrow slab.
    """
    if family == "wedge":
        return thickness_grid(params["t0"], params["grad"], n)
    if family == "gamma":
        from scipy.special import roots_genlaguerre
        k, m = float(params["shape"]), float(params["mean"])
        x, wx = roots_genlaguerre(int(n), k - 1.0)
        t = (m / k) * np.asarray(x, dtype=float)
        w = np.asarray(wx, dtype=float)
        return t, w / w.sum()
    raise ValueError(f"unknown thickness family {family!r}")


def slab_path_factor(ang_deg, phi0_deg: float, n_perp: float, max_factor: float = 5.0) -> np.ndarray:
    """Beam path / normal thickness for a parallel plate rotating about an axis perpendicular to the beam.

    n_perp is the component of the plate normal perpendicular to the rotation axis; phi0 is the scan angle
    at which that component is parallel to the beam. Capped at max_factor (grazing incidence).
    """
    c = float(n_perp) * np.abs(np.cos(np.deg2rad(np.asarray(ang_deg, dtype=float) - float(phi0_deg))))
    return 1.0 / np.maximum(c, 1.0 / float(max_factor))


def path_candidates(sel_cfg: dict, ang):
    """[(label_dict or None, per-row factor or None)]; always includes the no-path (constant) model."""
    pc = sel_cfg.get("path")
    out = [({"model": "none"}, None)]
    if not pc:
        return out
    for phi0 in pc["phi0_grid"]:
        for npp in pc["n_perp_grid"]:
            out.append(({"model": "slab", "phi0": float(phi0), "n_perp": float(npp)},
                        slab_path_factor(ang, phi0, npp, pc.get("max_factor", 5.0))))
    return out


def thickness_candidates(sel_cfg: dict, n_t_default: int):
    """[(label_dict, t, w)] for the grid in sel_cfg, plus the per-parameter grids for edge checks."""
    fam = sel_cfg.get("family", "wedge")
    n = int(sel_cfg.get("n_t", n_t_default))
    if fam == "wedge":
        grids = {"t0": sel_cfg["t0_grid"], "grad": sel_cfg["grad_grid"]}
    elif fam == "gamma":
        grids = {"mean": sel_cfg["mean_grid"], "shape": sel_cfg["shape_grid"]}
    else:
        raise ValueError(fam)
    names = list(grids)
    out = []
    for a in grids[names[0]]:
        for b in grids[names[1]]:
            prm = {names[0]: float(a), names[1]: float(b)}
            t, w = thickness_nodes(fam, prm, n)
            out.append(({"family": fam, "n_t": n, **prm}, t, w))
    return out, {k: [float(v) for v in g] for k, g in grids.items()}


# ----------------------------------------------------------------------------- thickness tables (exact grid weights)

_TAB: dict = {}


def _table_worker(bounds):
    lo, hi = bounds
    S = _TAB
    spec, t = S["spec"], S["t"]
    out = {h: np.empty((hi - lo, len(t))) for h in S["hands"]}
    out["born"] = np.empty((hi - lo, len(t)))
    for i, (w, lam, c, h10, delta, _mu) in enumerate(spectra_views(spec)):
        if i < lo:
            continue
        if i >= hi:
            break
        for t_lo in range(0, len(t), S["block"]):
            tb = t[t_lo:t_lo + S["block"]]
            E = np.exp(2j * np.pi * lam[:, :, None] * tb[None, None, :])
            for h in S["hands"]:
                amp = np.einsum("pj,pjt->pt", c, E if h > 0 else E.conj())
                out[h][i - lo, t_lo:t_lo + len(tb)] = w @ (np.abs(amp) ** 2)
        arg = np.pi * delta[:, None] * t[None, :]
        sinc = np.where(np.abs(arg) >= 1e-12, np.sin(arg) / np.where(arg == 0, 1.0, arg), 1.0)
        out["born"][i - lo] = w @ ((2.0 * np.pi) ** 2 * h10[:, None] * (t[None, :] ** 2) * sinc ** 2)
    return lo, out


def thickness_table(spec: dict, dt: float = 2.0, t_max: float = 12000.0, hands=(1, -1), workers: int = 1):
    """Profile-summed target intensity of every row on a uniform thickness grid t = 0, dt, ..., t_max (no absorption).

    Any thickness distribution (with uniform absorption and per-row path factors) is then a weighted sum over the
    grid with hat-function weights (distribution_weights), exact up to linear interpolation of I(t) at step dt.
    """
    import multiprocessing as mp
    t = np.arange(0.0, float(t_max) + 0.5 * dt, float(dt))
    n = len(spec["row_ids"])
    _TAB.clear()
    _TAB.update(spec=spec, t=t, hands=tuple(hands), block=256)
    edges = np.linspace(0, n, max(2, min(n, 4 * workers)) + 1).astype(int)
    bounds = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    if workers > 1:
        with ctx.Pool(processes=workers) as pool:
            parts = pool.map(_table_worker, bounds)
    else:
        parts = [_table_worker(b) for b in bounds]
    out = {"t": t}
    for key in list(hands) + ["born"]:
        out[key] = np.empty((n, len(t)))
        for lo, part in parts:
            out[key][lo:lo + len(part[key])] = part[key]
    return out


def _gamma_moments(a, b, k, theta):
    """Integral of p and of t*p over [a, b] for a gamma(k, theta) density (vectorised)."""
    from scipy.special import gammainc
    fa, fb = gammainc(k, np.maximum(a, 0) / theta), gammainc(k, np.maximum(b, 0) / theta)
    ga, gb = gammainc(k + 1.0, np.maximum(a, 0) / theta), gammainc(k + 1.0, np.maximum(b, 0) / theta)
    return fb - fa, k * theta * (gb - ga)


def _hat_weights_from_moments(t, m0, m1):
    """Hat-function weights w_j = int p(t) phi_j(t) dt from per-cell integrals of p (m0) and t*p (m1).

    Cell c spans [t_c, t_c+dt]; phi_c falls from 1 to 0 across it and phi_{c+1} rises.
    """
    dt = t[1] - t[0]
    right = (t[1:] * m0 - m1) / dt      # contribution to the left node of each cell
    left = (m1 - t[:-1] * m0) / dt      # contribution to the right node of each cell
    w = np.zeros(m0.shape[:-1] + (len(t),))
    w[..., :-1] += right
    w[..., 1:] += left
    return w


def distribution_weights(family: str, params: dict, t, mu: float = 0.0, path_factor=None):
    """Grid weights for a thickness distribution times exp(-4*pi*mu*t_beam); shape (M,) or (rows, M).

    gamma(mean, shape) and wedge(t0, grad) along the beam; a per-row path factor p scales the distribution
    (t_beam = p * t). Absorption is folded in analytically for gamma (exponential tilt) and numerically for wedge.
    Weights are not renormalised after absorption (the P-spline scale absorbs the overall factor).
    """
    t = np.asarray(t, dtype=float)
    a, b = t[:-1], t[1:]
    pf = np.ones(1) if path_factor is None else np.asarray(path_factor, dtype=float)
    alpha = 4.0 * np.pi * float(mu)
    if family == "gamma":
        k, mean = float(params["shape"]), float(params["mean"])
        theta = (mean / k) * pf[:, None]                     # beam-path scale per row
        theta_a = theta / (1.0 + alpha * theta)              # exponential tilt of the gamma density
        norm = (1.0 + alpha * theta) ** (-k)
        m0, m1 = _gamma_moments(a[None, :], b[None, :], k, theta_a)
        w = norm * _hat_weights_from_moments(t, m0, m1)
    elif family == "wedge":
        t0, grad = float(params["t0"]), float(params["grad"])
        lo = t0 * (1.0 - grad) * pf[:, None]
        hi = t0 * (1.0 + grad) * pf[:, None]
        x1 = np.clip(a[None, :], lo, hi)
        x2 = np.clip(b[None, :], lo, hi)
        width = np.maximum(hi - lo, 1e-12)
        if alpha == 0.0:
            m0 = (x2 - x1) / width
            m1 = 0.5 * (x2 ** 2 - x1 ** 2) / width
        else:
            e1, e2 = np.exp(-alpha * x1), np.exp(-alpha * x2)
            m0 = (e1 - e2) / (alpha * width)
            m1 = ((alpha * x1 + 1.0) * e1 - (alpha * x2 + 1.0) * e2) / (alpha ** 2 * width)
        w = _hat_weights_from_moments(t, m0, m1)
    else:
        raise ValueError(family)
    return w[0] if path_factor is None else w


def table_intensities(table: dict, hand: int, weights) -> np.ndarray:
    """Row intensities from a thickness table and grid weights ((M,) shared or (rows, M) per row)."""
    X = table[hand]
    y = X @ weights if np.ndim(weights) == 1 else np.einsum("rm,rm->r", X, weights)
    return np.maximum(y, 1e-30)


# ----------------------------------------------------------------------------- nested nuisance selection

INNER_SALT = "GROND-INNER-20260925"
_SEL: dict = {}  # shared with forked grid workers


def _inner_cv_r_all(dyn, fit_rows, scale=None) -> float:
    """Inner 3-fold orbit CV R_all (amplitude, pooled) of the scale fit restricted to fit_rows."""
    S = _SEL
    scale = S["scale"] if scale is None else scale
    fitter = PSplineScaleFitter(n_basis=int(scale.get("n_basis", 32)), objective=scale.get("objective", "row_mr"), floor_factor=float(scale.get("floor_factor", 1e-12)))
    oof = np.full(len(dyn), np.nan)
    for h in range(3):
        tr = fit_rows & (S["inner"] != h)
        va = fit_rows & (S["inner"] == h)
        if not tr.any() or not va.any():
            continue
        a = np.clip(S["ang"], np.min(S["ang"][tr]), np.max(S["ang"][tr])) if S["clamp"] else S["ang"]
        sol = fitter.fit(S["y"], S["sig"], dyn, a, group=S["group"], fit_mask=tr, lambda_smooth=scale.get("lambda", 1.0))
        oof[va] = sol["prediction"][va]
    m = fit_rows & np.isfinite(oof)
    fo = np.sqrt(np.maximum(S["y"], 0.0))
    return r1(fo, np.sqrt(np.maximum(np.nan_to_num(oof), 0.0)), m)


def _grid_worker(gi: int):
    """Inner-CV scores for one (variant, t0, grad): every mu and every requested hand."""
    S = _SEL
    vi, qi, pi = S["grid"][gi]
    _lab, t, w = S["quads"][qi]
    node = node_intensities(S["specs"][vi], t, path_factor=S["paths"][pi][1], hands=S["hands"])
    scores = {}
    for h in S["hands"]:
        for mi, mu in enumerate(S["mu_grid"]):
            dyn = collapse(node[h], node["t"], w, mu)
            sc = [_inner_cv_r_all(dyn, S["dev"] & (S["folds"] != f)) for f in range(S["n_folds"])]
            sc.append(_inner_cv_r_all(dyn, S["dev"]))
            scores[(h, mi)] = sc
    return gi, scores


def thickness_param_candidates(sel_cfg: dict):
    """[(family, params)] and per-parameter grids for table-mode selection."""
    fam = sel_cfg.get("family", "gamma")
    if fam == "wedge":
        grids = {"t0": sel_cfg["t0_grid"], "grad": sel_cfg["grad_grid"]}
    elif fam == "gamma":
        grids = {"mean": sel_cfg["mean_grid"], "shape": sel_cfg["shape_grid"]}
    else:
        raise ValueError(fam)
    names = list(grids)
    cands = [{"family": fam, names[0]: float(a), names[1]: float(b)} for a in grids[names[0]] for b in grids[names[1]]]
    return cands, {k: [float(v) for v in g] for k, g in grids.items()}


def _table_grid_worker(gi: int):
    S = _SEL
    vi, qi, pi = S["grid"][gi]
    prm = S["cands"][qi]
    pf = S["paths"][pi][1]
    t = S["tables"][vi]["t"]
    scores = {}
    for mi, mu in enumerate(S["mu_grid"]):
        w = distribution_weights(prm["family"], prm, t, mu=mu, path_factor=pf)
        for h in S["hands"]:
            dyn = table_intensities(S["tables"][vi], h, w)
            for si, sopt in enumerate(S["scale_grid"]):
                sc = [_inner_cv_r_all(dyn, S["dev"] & (S["folds"] != f), sopt) for f in range(S["n_folds"])]
                sc.append(_inner_cv_r_all(dyn, S["dev"], sopt))
                scores[(h, mi, si)] = sc
    return gi, scores


def _gamma_cf_worker(bounds):
    lo, hi = bounds
    S = _TAB
    k, theta, alpha, hand = S["k"], S["theta"], S["alpha"], S["hand"]
    out = np.empty(hi - lo)
    for i, (w, lam, c, _h10, _delta, _mu) in enumerate(spectra_views(S["spec"])):
        if i < lo:
            continue
        if i >= hi:
            break
        dl = lam[:, :, None] - lam[:, None, :]                          # (P, N, N) eigenvalue differences
        g = (1.0 + theta * (alpha - 2j * np.pi * hand * dl)) ** (-k)       # gamma characteristic function
        out[i - lo] = float(np.real(w @ np.einsum("pj,pk,pjk->p", c, c.conj(), g)))
    return lo, out


def gamma_cf_intensities(spec: dict, prm: dict, mu: float, hand: int, workers: int = 1) -> np.ndarray:
    """Exact gamma(mean, shape) thickness average with uniform absorption, no thickness grid.

    I(t) = |sum_j c_j exp(2 pi i h lam_j t)|^2 exp(-4 pi mu t) averaged over t ~ gamma(k, theta) is
    sum_jk c_j conj(c_k) (1 + theta (alpha - 2 pi i h (lam_j - lam_k)))^(-k), alpha = 4 pi mu: the gamma
    characteristic function. It equals the table mode (exact hat weights on a 2 A grid, truncated at t_max) up to
    the table's interpolation and truncation error, at a fraction of the cost. Not renormalised after absorption,
    like distribution_weights.
    """
    import multiprocessing as mp
    if prm.get("family") != "gamma" or (prm.get("path") or {}).get("model") not in (None, "none"):
        raise ValueError(f"gamma_cf_intensities needs a gamma thickness without a path model, got {prm}")
    k, mean = float(prm["shape"]), float(prm["mean"])
    n = len(spec["row_ids"])
    _TAB.clear()
    _TAB.update(spec=spec, k=k, theta=mean / k, alpha=4.0 * np.pi * float(mu), hand=int(hand))
    edges = np.linspace(0, n, max(2, min(n, 4 * workers)) + 1).astype(int)
    bounds = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    if workers > 1:
        with ctx.Pool(processes=workers) as pool:
            parts = pool.map(_gamma_cf_worker, bounds)
    else:
        parts = [_gamma_cf_worker(b) for b in bounds]
    out = np.empty(n)
    for lo, part in parts:
        out[lo:lo + len(part)] = part
    return np.maximum(out, 1e-30)


def select_thickness_table(specs, R, folds, n_folds, sel_cfg, scale_cfg, clamp, workers, variant_labels=None, hands=(1, -1)):
    """Nested selection (variant x thickness distribution x path x mu), exact-weight table mode.

    Same contract and return value as select_thickness; picks carry {"family", params, "path"} and no n_t.
    """
    import multiprocessing as mp
    specs = specs if isinstance(specs, list) else [specs]
    labels = variant_labels or [str(i) for i in range(len(specs))]
    dt, t_max = float(sel_cfg.get("dt", 2.0)), float(sel_cfg.get("t_max", 12000.0))
    tables = [thickness_table(sp, dt, t_max, tuple(hands), workers) for sp in specs]
    cands, param_grids = thickness_param_candidates(sel_cfg)
    paths = path_candidates(sel_cfg, R["ang"])
    mu_grid = [float(v) for v in sel_cfg.get("mu_grid", [0.0])]
    scale_grid = [dict(scale_cfg, **o) for o in sel_cfg.get("scale_grid", [{}])]
    grid = [(v, q, p) for v in range(len(specs)) for q in range(len(cands)) for p in range(len(paths))]
    dev = folds >= 0
    _SEL.clear()
    _SEL.update(tables=tables, grid=grid, cands=cands, paths=paths, mu_grid=mu_grid, scale_grid=scale_grid, hands=tuple(hands), folds=folds, dev=dev,
                n_folds=int(n_folds), y=R["y"], sig=R["sig"], ang=R["ang"], group=R["group"], scale=scale_cfg,
                clamp=bool(clamp), inner=np.asarray([hash_mod(INNER_SALT, k, 3) for k in R["keys"]]))
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    with ctx.Pool(processes=max(1, min(workers, len(grid)))) as pool:
        res = dict(pool.map(_table_grid_worker, range(len(grid)), chunksize=max(1, len(grid) // (8 * max(1, workers)))))
    bounds = sel_cfg.get("_variant_bounds") or [False] * len(specs)
    physical = {"grad": (0.0, 1.0)}

    def edge(vi, prm, mu):
        flags = []
        if len(specs) > 2 and vi in (0, len(specs) - 1) and not bounds[vi] and not sel_cfg.get("variant_edge_ok", False):
            flags.append("variant")
        for k, g in param_grids.items():
            if len(g) > 1 and prm[k] in (min(g), max(g)) and prm[k] not in physical.get(k, ()):
                flags.append(k)
        pth = prm.get("path") or {}
        pc = sel_cfg.get("path") or {}
        if pth.get("model") == "slab" and len(pc.get("n_perp_grid", [])) > 1 and pth["n_perp"] == min(float(v) for v in pc["n_perp_grid"]):
            flags.append("n_perp")
        if len(mu_grid) > 1 and (mu == max(mu_grid) or (mu == min(mu_grid) and mu > 0.0)):
            flags.append("mu")
        return flags

    out = {}
    scale_keys = sorted({k for o in sel_cfg.get("scale_grid", [{}]) for k in o})
    for h in hands:
        cand = [(gi, mi, si) for gi in range(len(grid)) for mi in range(len(mu_grid)) for si in range(len(scale_grid))]
        table = np.asarray([res[gi][(h, mi, si)] for gi, mi, si in cand])

        def pick_dyn(ci, cand=cand, h=h):
            gi, mi, si = cand[ci]
            vi, qi, pi = grid[gi]
            prm = {**cands[qi], "path": paths[pi][0]}
            w = distribution_weights(prm["family"], prm, tables[vi]["t"], mu=mu_grid[mi], path_factor=paths[pi][1])
            return table_intensities(tables[vi], h, w), vi, prm, mu_grid[mi], scale_grid[si]

        def scale_edge(sopt):
            flags = []
            for k in scale_keys:
                vals = sorted({float(o[k]) for o in sel_cfg["scale_grid"] if k in o and isinstance(o[k], (int, float))})
                if len(vals) > 1 and isinstance(sopt.get(k), (int, float)) and float(sopt[k]) in (vals[0], vals[-1]):
                    flags.append(f"scale.{k}")
            return flags

        pred = np.full(len(R["y"]), np.nan)
        chosen = []
        for f in range(n_folds):
            ci = int(np.nanargmin(table[:, f]))
            dyn, vi, prm, mus, sopt = pick_dyn(ci)
            train = dev & (folds != f)
            val = folds == f
            a = np.clip(R["ang"], np.min(R["ang"][train]), np.max(R["ang"][train])) if clamp else R["ang"]
            fitter = PSplineScaleFitter(n_basis=int(sopt.get("n_basis", 32)), objective=sopt.get("objective", "row_mr"), floor_factor=float(sopt.get("floor_factor", 1e-12)))
            sol = fitter.fit(R["y"], R["sig"], dyn, a, group=R["group"], fit_mask=train, lambda_smooth=sopt.get("lambda", 1.0))
            pred[val] = sol["prediction"][val]
            chosen.append({"fold": f, "variant": labels[vi], "thickness": prm, "mu": mus, "scale": sopt,
                           "inner_r_all": float(table[ci, f]), "edge": edge(vi, prm, mus) + scale_edge(sopt)})
        ci = int(np.nanargmin(table[:, n_folds]))
        dyn_all, vi, prm, mus, sopt = pick_dyn(ci)
        info = {"grid_size": len(cand), "mode": "table", "dt": dt, "t_max": t_max, "variants": labels, "per_fold": chosen,
                "all_dev": {"variant": labels[vi], "thickness": prm, "mu": mus, "scale": sopt,
                            "inner_r_all": float(table[ci, n_folds]), "edge": edge(vi, prm, mus) + scale_edge(sopt)}}
        info["any_edge"] = any(c["edge"] for c in chosen) or bool(info["all_dev"]["edge"])
        out[h] = (pred, dyn_all, info)
    return out


def select_thickness(specs, R, folds, n_folds, sel_cfg, n_t, scale_cfg, clamp, workers, path_factor=None,
                     variant_labels=None, hands=(1, -1)):
    """Nested selection of (physics variant, t0, grad, mu) per outer fold and per hand.

    For each outer fold the candidate with the lowest inner 3-fold orbit-CV R_all on that fold's
    training rows is chosen, independently for each hand, and predicts the held-out fold. specs is one
    spectra dict or a list (physics variants). Physical bounds (grad 0 and 1, mu 0) are not grid edges.
    Returns {hand: (pred_heldout, dyn_all_dev_selected, info)}.
    """
    import multiprocessing as mp
    quads, param_grids = thickness_candidates(sel_cfg, n_t)
    paths = path_candidates(sel_cfg, R["ang"]) if path_factor is None else [({"model": "fixed"}, path_factor)]
    mu_grid = [float(v) for v in sel_cfg["mu_grid"]]
    specs = specs if isinstance(specs, list) else [specs]
    labels = variant_labels or [str(i) for i in range(len(specs))]
    grid = [(v, q, p) for v in range(len(specs)) for q in range(len(quads)) for p in range(len(paths))]
    dev = folds >= 0
    _SEL.clear()
    _SEL.update(specs=specs, grid=grid, quads=quads, paths=paths, mu_grid=mu_grid, hands=tuple(hands),
                folds=folds, dev=dev, n_folds=int(n_folds), y=R["y"], sig=R["sig"], ang=R["ang"], group=R["group"],
                scale=scale_cfg, clamp=bool(clamp), inner=np.asarray([hash_mod(INNER_SALT, k, 3) for k in R["keys"]]))
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    with ctx.Pool(processes=max(1, min(workers, len(grid)))) as pool:
        res = pool.map(_grid_worker, range(len(grid)))
    physical = {"grad": (0.0, 1.0), "shape": (), "mean": (), "t0": ()}

    def edge(vi, prm, mu):
        flags = []
        bounds = sel_cfg.get("_variant_bounds") or [False] * len(specs)
        if len(specs) > 2 and vi in (0, len(specs) - 1) and not bounds[vi] and not sel_cfg.get("variant_edge_ok", False):
            flags.append("variant")
        for k, g in param_grids.items():
            if len(g) > 1 and prm[k] in (min(g), max(g)) and prm[k] not in physical.get(k, ()):
                flags.append(k)
        pth = prm.get("path") or {}
        pc = sel_cfg.get("path") or {}
        if pth.get("model") == "slab" and len(pc.get("n_perp_grid", [])) > 1:
            g = [float(v) for v in pc["n_perp_grid"]]
            if pth["n_perp"] == min(g):
                flags.append("n_perp")
        if len(mu_grid) > 1 and (mu == max(mu_grid) or (mu == min(mu_grid) and mu > 0.0)):
            flags.append("mu")
        return flags

    fitter = PSplineScaleFitter(n_basis=int(scale_cfg.get("n_basis", 32)), objective=scale_cfg.get("objective", "row_mr"))
    out = {}
    for h in hands:
        # table[candidate] = scores over folds (+ all-dev), candidate = (gi, mi)
        cand = [(gi, mi) for gi in range(len(grid)) for mi in range(len(mu_grid))]
        table = np.full((len(cand), n_folds + 1), np.nan)
        by_gi = dict(res)
        for ci, (gi, mi) in enumerate(cand):
            table[ci] = by_gi[gi][(h, mi)]

        def dyn_for(ci, cand=cand, h=h):
            gi, mi = cand[ci]
            vi, qi, pi = grid[gi]
            prm, t, w = quads[qi]
            node = node_intensities(specs[vi], t, path_factor=paths[pi][1], hands=(h,))
            return collapse(node[h], node["t"], w, mu_grid[mi]), (vi, {**prm, "path": paths[pi][0]}, mu_grid[mi])

        pred = np.full(len(R["y"]), np.nan)
        chosen = []
        for f in range(n_folds):
            ci = int(np.nanargmin(table[:, f]))
            dyn, (vi, prm, mus) = dyn_for(ci)
            train = dev & (folds != f)
            val = folds == f
            a = np.clip(R["ang"], np.min(R["ang"][train]), np.max(R["ang"][train])) if clamp else R["ang"]
            sol = fitter.fit(R["y"], R["sig"], dyn, a, group=R["group"], fit_mask=train, lambda_smooth=scale_cfg.get("lambda", 1.0))
            pred[val] = sol["prediction"][val]
            chosen.append({"fold": f, "variant": labels[vi], "thickness": prm, "mu": mus,
                           "inner_r_all": float(table[ci, f]), "edge": edge(vi, prm, mus)})
        ci = int(np.nanargmin(table[:, n_folds]))
        dyn_all, (vi, prm, mus) = dyn_for(ci)
        info = {"grid_size": len(cand), "variants": labels, "per_fold": chosen,
                "all_dev": {"variant": labels[vi], "thickness": prm, "mu": mus,
                            "inner_r_all": float(table[ci, n_folds]), "edge": edge(vi, prm, mus)}}
        info["any_edge"] = any(c["edge"] for c in chosen) or bool(info["all_dev"]["edge"])
        out[h] = (pred, dyn_all, info)
    return out
