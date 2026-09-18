"""Strict H-free comparisons must preserve settings and recalculate the scattering."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from grond.bloch.model import fcalc_res, parse_model
from grond.cli_options.parser import build_parser
from grond.legacy.engine.bloch_engine import Config, XDSBlochPort, XDSData
from grond.legacy.engine.shelx_hkl import write_fixed_width_hkl
from grond.macromolecule import sha256_file
from grond.workflows.matched_maps import _native_fcf, coordinate_cif
from grond.workflows.matched_prediction import predict_dataset, prediction_port, update_predictions
from grond.workflows.matched_reference import (
    ReferencePaths,
    frozen_merge_contract,
    load_reference,
    merge_changed_predictions,
    save_json,
    write_rows,
)
from grond.workflows.model_alignment import align_heavy_model, reindex_hkl, require_no_hydrogen, write_aligned_res
from tests.physics.test_continuous_thickness import _write_single_row_inputs


@pytest.fixture
def reference(tmp_path):
    hkl, no_h = _write_single_row_inputs(tmp_path)
    full = tmp_path / "full_H.res"
    text = no_h.read_text().replace("C1 1", "SFAC H 0.037 0.561 0.127 3.791 0.236 13.556 0.129 37.723 0\nC1 1")
    full.write_text(text.replace("HKLF 4", "H1 2 0.23 0.28 0.38 11 0.048\nHKLF 4"))
    cfg = Config(n_basis=4, n_points=3, thickness_A=100, thickness_model="single", thickness_points=1,
                 sg_window=0.0001, sg_window_min=0.0001, sg_window_max=0.0001, topup_d_min=0.7)
    data = XDSData(hkl)
    pred = XDSBlochPort(data, parse_model(full), cfg).predict_rows([0])
    rows = [{"row_index": 0, "h": 1, "k": 0, "l": 0, "target_I": 100, "xds_sigma": 10,
             "source_sigma": 10, "sigma_eff": 10, "d_A": 10, "target_mode": "iobs", "source_to_target_factor": 1,
             "basis_size": int(pred["basis_size"][0]), "basis_support_hash64": format(int(pred["basis_support_hash64"][0]), "016x"),
             "sg_window_inv_A": float(pred["sg_window"][0]), "sg_jitter_sigma_inv_A": 0,
             "omega_bragg_deg": float(pred["omega_bragg"][0]), "split": "work", "row_resolution_factor": 1,
             "dyn_raw": float(pred["dyn"][0]), "born_raw": float(pred["born"][0]),
             "dyn_raw_unmodulated": float(pred["dyn"][0]), "q_dyn_over_born": float(pred["dyn"][0] / pred["born"][0]),
             "dyn_scaled": float(pred["dyn"][0] * 2), "born_scaled": float(pred["born"][0] * 3)}]
    datasets, runs = [], []
    for name in ("one", "two"):
        folder = tmp_path / name
        folder.mkdir()
        summary = {"config": {**asdict(cfg), "hkl": str(hkl), "res": str(full), "hkl_transform": "identity",
                              "min_d_spacing": 0.7, "target_mode": "iobs"},
                   "inputs": {"res_sha256": sha256_file(full)}, "best_native": {"nuisance": asdict(cfg)}}
        save_json(folder / "summary.json", summary)
        write_rows(folder / "native_dyn_calc.csv", rows)
        runs.append({"name": name, "endpoint_weight": 1, "raw_row_weight": 1,
                     "source_artifacts": {key: {"path": str(folder / f), "sha256": sha256_file(folder / f)}
                                          for key, f in (("summary", "summary.json"), ("row_table", "native_dyn_calc.csv"))}})
        datasets.append(folder)
    endpoint_hkl = tmp_path / "reference.hkl"
    write_fixed_width_hkl(endpoint_hkl, [(1, 0, 0, 100, 10)])
    combined = {"provenance": {"endpoint_source": "native", "structure": {"path": str(full), "sha256": sha256_file(full)}},
                "runs": runs, "interdataset_scale": {"scales": {"one": 1, "two": 1}},
                "combined_endpoint": {"lambda_q": 1, "keep_exti": False, "merge_key": "model_symmetry", "hkl_safe_factor": 1,
                                      "dataset_sigma": {"one": {"status": "not_requested"}, "two": {"status": "not_requested"}},
                                      "artifacts": {"hkl": str(endpoint_hkl)}}}
    save_json(tmp_path / "combined_summary.json", combined)
    save_json(tmp_path / "grond_summary.json", {"combined": str(tmp_path)})
    return load_reference(tmp_path, ReferencePaths([])), no_h


def test_full_H_self_replay_and_no_H_have_the_same_beams(reference, tmp_path):
    ref, no_h = reference
    ds = ref["datasets"][0]
    full = Path(ref["model"])
    replay = predict_dataset(ds, full, full, tmp_path / "replay", workers=1, smoke=True)
    changed = predict_dataset(ds, no_h, full, tmp_path / "changed", workers=1)
    assert replay[0]["basis_support_hash64"] == changed[0]["basis_support_hash64"]
    assert not np.isclose(replay[0]["dyn_raw"], changed[0]["dyn_raw"], rtol=1e-5)
    for kind, scale in (("dyn", 2), ("born", 3)):
        assert changed[0][kind + "_scaled"] / changed[0][kind + "_raw"] == pytest.approx(scale)
    assert changed[0]["sigma_eff"] == replay[0]["sigma_eff"]


def test_reference_cannot_silently_change_beams_or_predictions(reference, tmp_path):
    ref, _ = reference
    ds = ref["datasets"][0]
    port = prediction_port(ds, Path(ref["model"]), Path(ref["model"]), tmp_path / "cache")
    pred = port.predict_rows([0])
    altered = copy.deepcopy(pred)
    altered["basis_support_hash64"][0] ^= np.uint64(1)
    with pytest.raises(ValueError, match="beam membership"):
        update_predictions(ds, altered)
    altered = copy.deepcopy(pred)
    altered["dyn"] *= 1.01
    with pytest.raises(ValueError, match="replay differs"):
        update_predictions(ds, altered, check_reference=True)


def test_merge_replays_reference_and_freezes_empirical_factors(reference):
    ref, _ = reference
    contract = frozen_merge_contract(ref)
    original = {ds["name"]: ds["rows"] for ds in ref["datasets"]}
    assert merge_changed_predictions(ref, contract, original)[0][3:] == pytest.approx((100, 10))
    changed = copy.deepcopy(original)
    for rows in changed.values():
        rows[0]["q_dyn_over_born"] = float(rows[0]["q_dyn_over_born"]) * 1.1
    assert merge_changed_predictions(ref, contract, changed)[0][3:] == pytest.approx((100 / 1.1, 10 / 1.1))


def test_changed_reference_artifact_fails(reference):
    ref, _ = reference
    Path(ref["datasets"][0]["row_table"]).write_text("replaced")
    with pytest.raises(ValueError, match="Changed or unbound"):
        load_reference(Path(ref["root"]), ReferencePaths([]))


def test_H_and_HFIX_are_rejected(reference, tmp_path):
    ref, no_h = reference
    with pytest.raises(ValueError, match="hydrogen atoms"):
        require_no_hydrogen(Path(ref["model"]))
    bad = tmp_path / "hfix.res"
    bad.write_text(no_h.read_text().replace("HKLF", "HFIX 43 C1\nHKLF"))
    with pytest.raises(ValueError, match="HFIX"):
        require_no_hydrogen(bad)


def test_alignment_preserves_amplitudes_and_uses_no_reference_H(reference, tmp_path):
    ref, no_h = reference
    no_h.write_text(no_h.read_text().replace("LATT -1", "ZERR 1 0.01 0.02 0.03 0 0 0\nLATT -1"))
    output = tmp_path / "aligned.res"
    report = write_aligned_res(no_h, Path(ref["model"]), output)
    assert all(len(line) <= 80 for line in output.read_text().splitlines() if line.startswith("CELL"))
    assert "ZERR 1 " in output.read_text()
    aligned = require_no_hydrogen(output)
    source = parse_model(no_h)
    matrix = np.array(report["fractional_matrix"])
    for hkl in ((1, 0, 0), (2, -1, 3)):
        transformed = np.linalg.inv(matrix).T @ hkl
        assert abs(fcalc_res(source, hkl)) == pytest.approx(abs(fcalc_res(aligned, transformed)), rel=1e-6)
    assert report["coordinates_copied_from_reference"] is False
    baseline = tmp_path / "input.hkl"
    write_fixed_width_hkl(baseline, [(1, 2, 3, 123, 4)])
    reindex_hkl(baseline, tmp_path / "aligned.hkl", [[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    assert (tmp_path / "aligned.hkl").read_text().splitlines()[0].split()[:5] == ["2", "1", "-3", "123.00", "4.00"]


def test_alignment_rejects_wrong_compound(reference):
    ref, no_h = reference
    source = parse_model(no_h)
    source["atoms"][0]["element"] = "S"
    with pytest.raises(ValueError, match="compositions differ"):
        align_heavy_model(source, parse_model(Path(ref["model"])))


def test_native_FCF_and_coordinate_exports(reference, tmp_path):
    import gemmi
    ref, no_h = reference
    template = tmp_path / "template.fcf"
    template.write_text("data_maps\n_cell_length_a 10\n_cell_length_b 10\n_cell_length_c 10\n_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\nloop_\n_refln_index_h\n_refln_index_k\n_refln_index_l\n_refln_F_squared_meas\n_refln_F_squared_sigma\n_refln_F_calc\n_refln_phase_calc\n1 0 0 100 10 8 0\n")
    _native_fcf([Path(ds["row_table"]) for ds in ref["datasets"]], no_h, template, tmp_path / "native.fcf")
    assert gemmi.cif.read_file(str(tmp_path / "native.fcf")).sole_block().find_value("_shelx_F_calc_maximum")
    coordinate_cif(no_h, tmp_path / "model.cif")
    assert "GROND matched-map coordinate export" in (tmp_path / "model.cif").read_text()


def test_command_dry_run_is_auditable(reference, tmp_path):
    ref, no_h = reference
    args = build_parser().parse_args(["never-h", "--reference-run", ref["root"], "--structure", str(no_h),
                                      "--baseline-hkl", ref["endpoint_hkl"], "--out-dir", str(tmp_path / "run"), "--dry-run"])
    assert args.func(args) == 0
    manifest = json.loads((tmp_path / "run/RUN_MANIFEST.json").read_text())
    assert "absorption" in manifest["frozen"] and "jitter" in manifest["frozen"]
    assert not (tmp_path / "run/cycle_01").exists()


def test_nonorthogonal_alignment_preserves_anisotropic_structure_factors(reference):
    from grond.bloch.model import reciprocal_lengths, shelx_u_matrix
    ref, no_h = reference
    source = parse_model(no_h)
    source["cell"] = (10, 11, 12, 90, 105, 90)
    source["atoms"] = [copy.deepcopy(source["atoms"][0]) for _ in range(3)]
    for atom, element, position in zip(source["atoms"], ("C", "N", "O"), ((.12, .23, .34), (.31, .41, .29), (.57, .26, .68)), strict=True):
        atom.update(name=element + "1", element=element, x=position[0], y=position[1], z=position[2],
                    adp_type="anis", u11=.03, u22=.04, u33=.05, u12=.002, u13=.004, u23=.003)
    target = copy.deepcopy(source)
    matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    target["cell"] = (11, 10, 12, 75, 90, 90)
    transform = np.diag(1 / np.array(reciprocal_lengths(target["cell"]))) @ matrix @ np.diag(reciprocal_lengths(source["cell"]))
    for atom in target["atoms"]:
        xyz = matrix @ [atom["x"], atom["y"], atom["z"]] + [.13, .29, .41]
        atom.update(zip(("x", "y", "z"), xyz, strict=True))
        tensor = transform @ shelx_u_matrix(atom) @ transform.T
        for name, pair in zip(("u11", "u22", "u33", "u23", "u13", "u12"), ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)), strict=True):
            atom[name] = tensor[pair]
    aligned, report = align_heavy_model(source, target)
    assert report["heavy_atom_rms_A"] < 1e-8
    conversion = np.linalg.inv(report["fractional_matrix"]).T
    for hkl in [(1, 2, -3), (-2, 1, 4), (3, 0, 2)]:
        assert abs(fcalc_res(source, hkl)) == pytest.approx(abs(fcalc_res(aligned, conversion @ hkl)), rel=1e-11)


def test_screw_axis_origin_change_is_respected(reference):
    _, no_h = reference
    source = parse_model(no_h)
    source["atoms"] = [copy.deepcopy(source["atoms"][0]) for _ in range(3)]
    for atom, element, position in zip(source["atoms"], ("C", "N", "O"), ((.12, .23, .34), (.31, .41, .29), (.57, .26, .68)), strict=True):
        atom.update(name=element, element=element, x=position[0], y=position[1], z=position[2])
    rotation = np.diag([-1, 1, -1])
    source["symmetry_ops"].append({"rot": rotation, "trans": np.array([0, .5, .5])})
    target = copy.deepcopy(source)
    shift = np.array([.25, .31, .25])
    target["symmetry_ops"][1]["trans"] = np.array([.5, .5, 0])
    for atom in target["atoms"]:
        atom.update(zip(("x", "y", "z"), np.array([atom["x"], atom["y"], atom["z"]]) + shift, strict=True))
    aligned, report = align_heavy_model(source, target)
    assert report["heavy_atom_rms_A"] < 1e-8
    conversion = np.linalg.inv(report["fractional_matrix"]).T
    for hkl in [(1, 2, -3), (-2, 1, 4), (3, 0, 2)]:
        assert abs(fcalc_res(source, hkl)) == pytest.approx(abs(fcalc_res(aligned, conversion @ hkl)), rel=1e-11)


def test_never_h_retries_npd_step_without_changing_data(reference, tmp_path):
    from grond.commands.never_h import refine_step
    from tests.refine.test_run_shelxl_baseline import _fake_shelxl

    _, no_h = reference
    work = tmp_path / "damped_step"
    work.mkdir()
    original = no_h.read_text()
    (work / "model.ins").write_text(original)
    (work / "model.hkl").write_text("unchanged reflection data\n")
    executable = _fake_shelxl(tmp_path, npd=True)
    script = executable.read_text()
    # The undamped update fails; the same model with a smaller step succeeds.
    insertion = "    if 'DAMP ' in Path(stem + '.ins').read_text():\n        p = Path(stem + '.lst')\n        p.write_text(p.read_text().replace('** NON POSITIVE DEFINITE **', '').replace('1  atoms NPD', '0  atoms NPD'))\n"
    script = script.replace("raise SystemExit(returncode)", insertion + "raise SystemExit(returncode)")
    executable.write_text(script)
    health = refine_step(str(executable), work, 2)
    assert health["safe_for_feedback"]
    assert (work / "before_damping.ins").read_text() == original
    assert (work / "model.hkl").read_text() == "unchanged reflection data\n"
    assert "DAMP 100 15" in (work / "model.ins").read_text()
    recovery = json.loads((work / "damping_recovery.json").read_text())
    assert recovery["npd_recovery"]["accepted_damp"] == 100
    assert len(recovery["npd_recovery"]["attempts"]) == 2
