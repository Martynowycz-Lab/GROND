"""Shared orientation grid (grond.bloch.shared_grid) against the per-row engine on the P2_12_12_1 fixture."""

import numpy as np

from grond.bloch import shared_grid as SG
from grond.bloch.config import Config
from grond.legacy.engine import spectra_stage as S
from grond.regularize.nested_selection import ThicknessSupport
from tests._nested_support import write_dataset


def _port(tmp_path, **overrides):
    hkl, res = write_dataset(tmp_path, max_rows=24)
    cfg = dict(n_basis=1000, n_points=9, topup_d_min=2.0, ug_d_min=1.0, max_support_sg=0.05, thickness_model="gamma",
               thickness_A=300.0, thickness_gamma_shape=0.5, thickness_evaluator="table", thickness_max_A=3000.0)
    cfg.update(overrides)
    return S.build_port(str(hkl), "identity", str(res), Config(**cfg))


def _adjacent_rows(port, count=2):
    omegas = np.asarray([port.omega_from_z(z) for z in port.data.zcal])
    order = np.argsort(omegas)
    gaps = np.diff(omegas[order])
    i = int(np.argmin([np.ptp(omegas[order[j:j + count]]) for j in range(len(order) - count + 1)]))
    assert gaps.size
    return [int(r) for r in order[i:i + count]]


def test_grid_solve_matches_a_dense_eigendecomposition(tmp_path):
    port = _port(tmp_path)
    rows = _adjacent_rows(port, 3)
    targets = np.asarray([port.data.hkl[r] for r in rows], dtype=int)
    omega = float(np.mean([port.omega_from_z(port.data.zcal[r]) for r in rows]))
    basis, pos, sg = SG.grid_basis(port, omega, 0.0, targets, 0.05)
    H = SG._static_matrix(port, basis, 0.0, np.complex128)
    H[np.diag_indices(len(H))] += sg
    assert np.max(np.abs(H.imag)) > 0.0  # non-centrosymmetric: the complex Hermitian path
    lam, V = np.linalg.eigh(H)
    ref = SG.target_intensities(lam, V[pos] * np.conj(V[0])[None, :], port.thickness_uniform_grid(), (1, -1),
                                weights=np.asarray(port.thick_w))
    lam_g, C, n = SG.solve_grid_point(port, omega, 0.0, targets, 0.05, "double")
    got = SG.target_intensities(lam_g, C, port.thickness_uniform_grid(), (1, -1), weights=np.asarray(port.thick_w))
    assert n == len(basis)
    for h in (1, -1):
        np.testing.assert_allclose(got[h], ref[h], rtol=1e-9)


def test_response_table_reduces_to_the_grid_predictions(tmp_path, monkeypatch):
    monkeypatch.setenv("GROND_SHARED_GRID_PER_DEGREE", "50")
    port = _port(tmp_path, paired_inverse_hand=True)
    rows = _adjacent_rows(port, 2)
    pred = SG.predict_rows(port, rows, workers=1, log=lambda *_: None)
    support = ThicknessSupport(100.0, 3000.0, 2.0)
    table, _diag = SG.response_table(port, rows, support, hands=(1, -1), workers=1, log=lambda *_: None)
    w = np.asarray(port.thick_w)
    assert np.allclose(np.asarray(port.thick), support.grid())
    np.testing.assert_allclose(table[1] @ w, [p["dyn"] for p in pred], rtol=1e-12)
    np.testing.assert_allclose(table[-1] @ w, [p["dyn_inverse"] for p in pred], rtol=1e-12)
    for r, p in zip(rows, pred):
        plan = port.row_prediction_plan(r)
        assert abs(p["born"] / float(np.sum(plan["_profile_weights"] * plan["_born_vals"])) - 1.0) < 1e-12


def test_whole_pool_basis_matches_the_per_row_engine(tmp_path, monkeypatch):
    # With every pool beam inside the cutoff, and one row at a time (the fixture's pool stops at 2 A while its targets
    # reach 1.2 A, so another row's target would be an extra beam), the centre and grid-angle rules give the same
    # beams and the grid differs from the per-row engine only by its interpolation.
    monkeypatch.setenv("GROND_SHARED_GRID_PER_DEGREE", "400")
    port = _port(tmp_path, max_support_sg=5.0, mosaicity_mode="transverse_target", mosaicity_sigma_deg=0.2,
                 mosaicity_points=3)
    for r in _adjacent_rows(port, 2):
        got = SG.predict_rows(port, [r], workers=1, log=lambda *_: None)[0]["dyn"]
        ref = port.row_prediction(r)["dyn"]
        assert abs(got / ref - 1.0) < 2e-3


def test_path_chooser_defaults_off_and_refuses_unsupported_physics(tmp_path, monkeypatch):
    monkeypatch.delenv("GROND_SHARED_GRID", raising=False)
    port = _port(tmp_path)
    rows = list(range(len(port.data.zcal)))
    assert SG.choose_path(port, rows)[0] == "per_row"
    monkeypatch.setenv("GROND_SHARED_GRID", "force")
    assert SG.choose_path(port, rows)[0] == "shared_grid"
    jitter = _port(tmp_path / "jitter", sg_jitter_mode="gaussian", sg_jitter_sigma0=0.001)
    path, reason = SG.choose_path(jitter, rows)
    assert path == "per_row" and "jitter" in reason


def test_run_options_set_the_environment_for_every_stage(monkeypatch):
    from grond.cli_options.parser import build_parser

    for name in ("GROND_SHARED_GRID", "GROND_SHARED_GRID_GPUS", "GROND_BASIS_CENTRING_COSET"):
        monkeypatch.delenv(name, raising=False)
    args = build_parser().parse_args(["auto", "--prep", "p.json", "--out-dir", "o"])
    assert (args.shared_grid, args.shared_grid_gpus, args.basis_centring_coset) == (None, None, None)
    assert SG.apply_cli_options(args) == {"mode": "off", "basis_centring_coset": False}
    args = build_parser().parse_args(["auto", "--prep", "p.json", "--out-dir", "o", "--shared-grid", "auto",
                                      "--shared-grid-gpus", "0,1", "--basis-centring-coset", "on"])
    record = SG.apply_cli_options(args)
    assert record["mode"] == "auto" and record["gpus"] == ["0", "1"] and record["basis_centring_coset"]
    assert SG.os.environ["GROND_SHARED_GRID"] == "auto" and SG.os.environ["GROND_BASIS_CENTRING_COSET"] == "1"


def test_prediction_cache_identity_binds_the_evaluator(monkeypatch):
    monkeypatch.setenv("GROND_SHARED_GRID", "off")
    assert SG.cache_identity() is None  # per-row cache keys are unchanged
    monkeypatch.setenv("GROND_SHARED_GRID", "auto")
    monkeypatch.setenv("GROND_SHARED_GRID_GPUS", "0,1,2")
    identity = SG.cache_identity()
    assert identity["mode"] == "auto" and identity["gpus"] is True and identity["per_degree"] == 100.0
