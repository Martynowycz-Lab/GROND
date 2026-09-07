import copy
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grond.cli import _configure_static_pets_auto, build_parser
from grond.io.config import load_prep
from grond.io.pets_import import import_pets_to_grond, read_pets_contract
from grond.legacy.engine import native_row_regularize
from grond.legacy.engine.bloch_engine import Config, XDSBlochPort, XDSData, parse_model
from grond.legacy.engine.native_row_regularize import group_indices, main as native_regularize_main
from grond.legacy.engine.shelx_hkl import write_fixed_width_hkl
from grond.legacy.engine.support_basis_scan import scan_basis_sizes
from tests._support import REPO_ROOT


def _pets_text(*, repeat: bool = False, bad_zone: bool = False) -> str:
    second = "1 0 0 20 2 2\n" if repeat else "0 1 0 20 2 2\n"
    v = "0.00000" if bad_zone else "1.00000"
    return f"""data_pets
_cell_length_a 5
_cell_length_b 6
_cell_length_c 7
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_diffrn_radiation_wavelength 0.025
_diffrn_measurement_details
;
rotation axis position: 0
;
_diffrn_measurement_method_precession no
_diffrn_measurement_rotation_mode rotation
_diffrn_orient_matrix_UB_11 0.2
_diffrn_orient_matrix_UB_12 0
_diffrn_orient_matrix_UB_13 0
_diffrn_orient_matrix_UB_21 0
_diffrn_orient_matrix_UB_22 0.1666666667
_diffrn_orient_matrix_UB_23 0
_diffrn_orient_matrix_UB_31 0
_diffrn_orient_matrix_UB_32 0
_diffrn_orient_matrix_UB_33 0.1428571429
loop_
_diffrn_zone_axis_id
_diffrn_zone_axis_u
_diffrn_zone_axis_v
_diffrn_zone_axis_w
_diffrn_zone_axis_precession_angle
_diffrn_zone_axis_alpha
_diffrn_zone_axis_beta
_diffrn_zone_axis_omega
_diffrn_zone_axis_scale
1 0 0 1 0.1 0 0 0 1
2 0 {v} 0 0.1 90 0 0 1
loop_
_refln_index_h
_refln_index_k
_refln_index_l
_refln_intensity_meas
_refln_intensity_sigma
_refln_zone_axis_id
1 0 0 10 1 1
{second}"""


def _real_369_geometry_excerpt() -> str:
    return """data_pets
_cell_length_a 5.97508
_cell_length_b 6.89478
_cell_length_c 21.21338
_cell_angle_alpha 90.00003
_cell_angle_beta 90.00002
_cell_angle_gamma 90.00005
_diffrn_radiation_wavelength 0.02508
_diffrn_orient_matrix_UB_11 0.09703
_diffrn_orient_matrix_UB_12 -0.10545
_diffrn_orient_matrix_UB_13 -0.01734
_diffrn_orient_matrix_UB_21 0.13038
_diffrn_orient_matrix_UB_22 0.05255
_diffrn_orient_matrix_UB_23 0.02412
_diffrn_orient_matrix_UB_31 -0.03997
_diffrn_orient_matrix_UB_32 -0.08458
_diffrn_orient_matrix_UB_33 0.03660
_diffrn_measurement_method_precession no
_diffrn_measurement_rotation_mode rotation
_diffrn_reflns_reduction_process
;
data collection geometry: continuous rotation
tilt axis position: -1.927
;
loop_
_diffrn_zone_axis_id
_diffrn_zone_axis_u
_diffrn_zone_axis_v
_diffrn_zone_axis_w
_diffrn_zone_axis_precession_angle
_diffrn_zone_axis_alpha
_diffrn_zone_axis_beta
_diffrn_zone_axis_omega
_diffrn_zone_axis_scale
1 -1.00000 -0.67261 -0.01230 1.352 -58.728 -0.056 -2.026 1.000
2 -1.00000 -0.68832 -0.00153 1.352 -56.880 0.021 -1.956 0.892
loop_
_refln_index_h
_refln_index_k
_refln_index_l
_refln_intensity_meas
_refln_intensity_sigma
_refln_zone_axis_id
-3 4 -19 100 10 1
-3 4 -18 90 9 2
"""


def test_native_pets_rows_write_one_grond_row_per_observation(tmp_path: Path) -> None:
    source = tmp_path / "test.cif_pets"
    source.write_text(_pets_text())
    source.with_suffix(".hkl").write_text("1 0 0 10 1 1 0 0 0 -1 0 0 -1\n0 1 0 20 2 2 0 0 0 -1 0 0 -1\n")
    output = import_pets_to_grond(source, tmp_path / "bridge", space_group_number=19)
    text = output.hkl.read_text()
    assert "!GROND_SOURCE=PETS2" in text
    assert "!PETS_VIRTUAL_FRAMES= FALSE" in text
    numeric = [line for line in text.splitlines() if line and not line.startswith("!")]
    assert len(numeric) == 2
    assert output.scan_points.read_text().count("\n") == 2
    loaded = XDSData(output.hkl, hkl_transform="identity")
    assert loaded.pets_exact_a_center.shape == (2, 3, 3)
    assert loaded.pets_integration_semiangle_deg.tolist() == [0.1, 0.1]
    port = object.__new__(XDSBlochPort)
    port.data = loaded
    port.cfg = SimpleNamespace(n_points=3, profile_accumulation="sum")
    matrices, offsets, weights, semiangle, ordinal = port.pets_exact_orientation_samples(0)
    assert offsets.tolist() == pytest.approx([-0.1, 0.0, 0.1])
    assert weights.tolist() == [1.0, 1.0, 1.0]
    assert semiangle == pytest.approx(0.1)
    assert ordinal == 0
    assert np.allclose(matrices[1], loaded.pets_exact_a_center[0])
    assert port.make_groups().tolist() == [0, 1]
    assert group_indices(loaded, group_width_deg=999.0, frame_offset=123.0).tolist() == [0, 1]


def test_ordinary_pets_static_hkl_enables_compact_angular_scale_groups(
    tmp_path: Path,
) -> None:
    source = tmp_path / "test.cif_pets"
    source.write_text(_pets_text())
    source.with_suffix(".hkl").write_text("1 0 0 10 1 1 0 0 0 -1 0 0 -1\n0 1 0 20 2 2 0 0 0 -1 0 0 -1\n")
    baseline = tmp_path / "test_shelx.hkl"
    write_fixed_width_hkl(
        baseline,
        [(1, 0, 0, 10.0, 1.0), (0, 1, 0, 20.0, 2.0)],
    )

    output = import_pets_to_grond(
        source,
        tmp_path / "bridge",
        space_group_number=19,
        baseline_hkl=baseline,
    )
    summary = json.loads(output.summary.read_text())
    assert summary["static_endpoint_contract"]["status"] == "validated"
    assert summary["static_endpoint_contract"]["rows"] == 2
    assert "!PETS_STATIC_HKLF4_VALIDATED= TRUE" in output.hkl.read_text()

    loaded = XDSData(output.hkl, hkl_transform="identity")
    assert group_indices(loaded, group_width_deg=999.0, frame_offset=0.0).tolist() == [0, 0]
    assert group_indices(loaded, group_width_deg=45.0, frame_offset=0.0).tolist() == [0, 2]


def test_pets_static_hkl_mismatch_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "test.cif_pets"
    source.write_text(_pets_text())
    source.with_suffix(".hkl").write_text("1 0 0 10 1 1 0 0 0 -1 0 0 -1\n0 1 0 20 2 2 0 0 0 -1 0 0 -1\n")
    baseline = tmp_path / "wrong.hkl"
    write_fixed_width_hkl(
        baseline,
        [(1, 0, 0, 10.0, 1.0), (0, 1, 0, 21.0, 2.0)],
    )

    with pytest.raises(ValueError, match="observation differs"):
        import_pets_to_grond(
            source,
            tmp_path / "bridge",
            space_group_number=19,
            baseline_hkl=baseline,
        )


def test_pets_static_hkl_accepts_pets_eof_without_terminator(tmp_path: Path) -> None:
    source = tmp_path / "test.cif_pets"
    source.write_text(_pets_text())
    source.with_suffix(".hkl").write_text("1 0 0 10 1 1 0 0 0 -1 0 0 -1\n0 1 0 20 2 2 0 0 0 -1 0 0 -1\n")
    baseline = tmp_path / "pets_shelx.hkl"
    baseline.write_text(f"{1:4d}{0:4d}{0:4d}{10.0:8.2f}{1.0:8.2f}\n{0:4d}{1:4d}{0:4d}{20.0:8.2f}{2.0:8.2f}\n")

    output = import_pets_to_grond(
        source,
        tmp_path / "bridge",
        space_group_number=19,
        baseline_hkl=baseline,
    )
    summary = json.loads(output.summary.read_text())
    assert summary["static_endpoint_contract"]["status"] == "validated"


def test_static_pets_auto_profile_is_source_safe_and_leaves_nq_automatic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "test.cif_pets"
    source.write_text(_pets_text())
    source.with_suffix(".hkl").write_text("1 0 0 10 1 1 0 0 0 -1 0 0 -1\n0 1 0 20 2 2 0 0 0 -1 0 0 -1\n")
    structure = tmp_path / "model.res"
    structure.write_text(
        "TITL test\nCELL 0.025 5 6 7 90 90 90\nLATT -1\n"
        "SFAC C 1 0 1 0 1 0 1 0\nFVAR 1\n"
        "C1 1 0.1 0.2 0.3 11.0 0.05\nHKLF 4\nEND\n"
    )
    baseline = tmp_path / "model.hkl"
    write_fixed_width_hkl(
        baseline,
        [(1, 0, 0, 10.0, 1.0), (0, 1, 0, 20.0, 2.0)],
    )
    bridge = import_pets_to_grond(
        source,
        tmp_path / "bridge",
        space_group_number=1,
        structure=structure,
        baseline_hkl=baseline,
    )
    out = tmp_path / "auto"
    args = build_parser().parse_args(["auto", "--prep", str(bridge.prep), "--out-dir", str(out), "--dry-run"])

    payload = _configure_static_pets_auto(load_prep(bridge.prep), args)

    assert payload["status"] == "validated"
    assert args.pets_static_auto is True
    assert args.n_basis == 0
    assert args.n_points == 20
    assert args.screen_n_basis == 24
    assert args.screen_n_points == 10
    assert args.trust_hkl_transform is True
    assert args.detector_scale_bins == "0"
    assert args.group_width_grid == "999,10,5,3,2,1"
    assert (out / "PETS_STATIC_AUTO_CONTRACT.json").is_file()


def test_pets_prep_keeps_exact_edge_and_nonpositive_rows(tmp_path: Path) -> None:
    source = tmp_path / "test.cif_pets"
    source.write_text(_pets_text().replace("1 0 0 10 1 1", "1 0 0 -10 1 1"))
    source.with_suffix(".hkl").write_text("1 0 0 -10 1 1 0 0 0 -1 0 0 -1\n0 1 0 20 2 2 0 0 0 -1 0 0 -1\n")
    structure = tmp_path / "model.res"
    structure.write_text(
        "TITL test\nCELL 0.025 5 6 7 90 90 90\nLATT -1\n"
        "SFAC C 1 0 1 0 1 0 1 0\nFVAR 1\n"
        "C1 1 0.1 0.2 0.3 11.0 0.05\n"
        "C2 1 0.4 0.5 0.6 11.0 0.05\nHKLF 4\nEND\n"
    )
    output = import_pets_to_grond(
        source,
        tmp_path / "bridge",
        space_group_number=1,
        structure=structure,
    )
    payload = json.loads(output.prep.read_text())
    dataset = payload["datasets"][0]
    summary = json.loads(output.summary.read_text())
    assert dataset["include_nonpositive"] is True
    assert dataset["geometry_percentile"] == 100.0
    assert dataset["min_d_spacing"] < summary["selected_min_d_spacing_A"]
    assert dataset["min_d_spacing"] == summary["engine_min_d_spacing_A"]
    run_args = build_parser().parse_args(
        [
            "run",
            "--prep",
            str(output.prep),
            "--out-dir",
            str(tmp_path / "unsafe_auto"),
        ]
    )
    with pytest.raises(SystemExit, match="row-for-row validated"):
        run_args.func(run_args)
    auto_args = build_parser().parse_args(
        [
            "auto",
            "--prep",
            str(output.prep),
            "--out-dir",
            str(tmp_path / "unsafe_auto_alias"),
            "--dry-run",
        ]
    )
    with pytest.raises(SystemExit, match="row-for-row validated"):
        auto_args.func(auto_args)
    pets_args = build_parser().parse_args(
        [
            "pets-evaluate",
            "--prep",
            str(output.prep),
            "--out-dir",
            str(tmp_path / "pets_native"),
            "--n-basis",
            "64",
            "--n-points",
            "9",
            "--topup-d-min",
            "0.5",
            "--max-support-sg",
            "0.01",
            "--thickness-grid",
            "500,1000",
            "--workers",
            "2",
            "--dry-run",
        ]
    )
    assert pets_args.func(pets_args) == 0
    commands = (tmp_path / "pets_native" / "logs" / "commands.jsonl").read_text()
    assert "--skip-endpoint-refinement" in commands
    assert "--paired-inverse-hand" in commands
    assert "absolute_structure_diagnostic.py" in commands


def test_pets_native_regularizer_refuses_static_endpoint_and_detector_grid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # configure_shelxl_executable intentionally mutates this worker-global;
    # register it with monkeypatch so the broader suite sees the original.
    monkeypatch.setattr(native_row_regularize, "SHELXL", native_row_regularize.SHELXL)
    source = tmp_path / "test.cif_pets"
    source.write_text(_pets_text())
    source.with_suffix(".hkl").write_text("1 0 0 10 1 1 0 0 0 -1 0 0 -1\n0 1 0 20 2 2 0 0 0 -1 0 0 -1\n")
    bridge = import_pets_to_grond(source, tmp_path / "bridge", space_group_number=19)
    common = [
        "native_row_regularize",
        "--hkl",
        str(bridge.hkl),
        "--res",
        str(tmp_path / "model.res"),
        "--baseline-hkl",
        str(tmp_path / "model.hkl"),
        "--out-dir",
        str(tmp_path / "native"),
        "--shelxl",
        "/usr/bin/true",
    ]
    monkeypatch.setattr(sys, "argv", common)
    with pytest.raises(ValueError, match="validated row-preserving static SHELXL"):
        native_regularize_main()

    monkeypatch.setattr(
        sys,
        "argv",
        [*common, "--skip-endpoint-refinement", "--detector-scale-bins", "2"],
    )
    with pytest.raises(ValueError, match="no detector coordinates"):
        native_regularize_main()


def test_structure_space_group_mismatch_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "test.cif_pets"
    source.write_text(_pets_text())
    source.with_suffix(".hkl").write_text("1 0 0 10 1 1 0 0 0 -1 0 0 -1\n0 1 0 20 2 2 0 0 0 -1 0 0 -1\n")
    structure = tmp_path / "p1.res"
    structure.write_text(
        "TITL p1\nCELL 0.025 5 6 7 90 90 90\nLATT -1\n"
        "SFAC C 1 0 1 0 1 0 1 0\nFVAR 1\n"
        "C1 1 0.1 0.2 0.3 11.0 0.05\nHKLF 4\nEND\n"
    )
    with pytest.raises(ValueError, match="structure symmetry does not match"):
        import_pets_to_grond(
            source,
            tmp_path / "bridge",
            space_group_number=19,
            structure=structure,
        )


def test_nonconsecutive_source_zone_ids_are_normalized_but_preserved(
    tmp_path: Path,
) -> None:
    text = _pets_text()
    text = text.replace("1 0 0 1 0.1 0 0 0 1", "10 0 0 1 0.1 0 0 0 1")
    text = text.replace("2 0 1.00000 0 0.1 90 0 0 1", "20 0 1.00000 0 0.1 90 0 0 1")
    text = text.replace("1 0 0 10 1 1", "1 0 0 10 1 10")
    text = text.replace("0 1 0 20 2 2", "0 1 0 20 2 20")
    source = tmp_path / "zones.cif_pets"
    source.write_text(text)
    source.with_suffix(".hkl").write_text("1 0 0 10 1 10 0 0 0 -1 0 0 -1\n0 1 0 20 2 20 0 0 0 -1 0 0 -1\n")
    output = import_pets_to_grond(source, tmp_path / "bridge", space_group_number=1)
    with np.load(output.out_dir / "PETS_EXACT_GEOMETRY.npz") as geometry:
        assert geometry["source_zone_id"].tolist() == [1, 2]
        assert geometry["source_zone_id_text"].tolist() == ["10", "20"]
    loaded = XDSData(output.hkl, hkl_transform="identity")
    assert loaded.zcal.tolist() == [1.0, 2.0]


def test_single_zone_uses_exact_geometry_without_fake_scan_requirement(
    tmp_path: Path,
) -> None:
    text = (
        "\n".join(
            line
            for line in _pets_text().splitlines()
            if not line.startswith("2 0 ") and not line.startswith("0 1 0 20 ")
        )
        + "\n"
    )
    source = tmp_path / "one_zone.cif_pets"
    source.write_text(text)
    source.with_suffix(".hkl").write_text("1 0 0 10 1 1 0 0 0 -1 0 0 -1\n")
    output = import_pets_to_grond(source, tmp_path / "bridge", space_group_number=1)
    summary = json.loads(output.summary.read_text())
    assert summary["zone_axes"] == 1
    assert summary["orientation"]["zone_step_source"] == "single_zone_placeholder_not_used_by_exact_geometry"


def test_nonidentity_pets_setting_is_rejected_until_symmetry_is_conjugated(
    tmp_path: Path,
) -> None:
    source = tmp_path / "test.cif_pets"
    source.write_text(_pets_text())
    source.with_suffix(".hkl").write_text("1 0 0 10 1 1 0 0 0 -1 0 0 -1\n0 1 0 20 2 2 0 0 0 -1 0 0 -1\n")
    with pytest.raises(ValueError, match="requires hkl_transform=identity"):
        import_pets_to_grond(
            source,
            tmp_path / "bridge",
            space_group_number=1,
            hkl_transform="k,h,-l",
        )


def test_real_nonzero_axis_nonorthogonal_ub_golden_geometry(tmp_path: Path) -> None:
    source = tmp_path / "real_excerpt_dyn.cif_pets"
    source.write_text(_real_369_geometry_excerpt())
    contract = read_pets_contract(source, allow_virtual_frames=True)
    expected = np.asarray(
        [
            [0.097118579067509, -0.105447283111434, -0.017297331859940947],
            [0.033442457081667, -0.044928298159747, 0.043823691864677876],
            [-0.132129276822517, -0.088876041917990, -0.0016269103589076146],
        ]
    )
    # The reciprocal B[2, 1] sign now agrees with Gemmi's independent metric;
    # near-orthogonal printed cell angles still change this column by 3.6e-8.
    assert contract["axis_position_deg"] == pytest.approx(-1.927)
    assert contract["a_matrices"][0] == pytest.approx(expected, abs=5e-13)

    # GROND's column-vector transform convention must preserve the laboratory
    # reciprocal vector exactly: h_model=T h_source and A_model=A_source T^T.
    transform = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=int)
    source_hkl = np.asarray([-3, 4, -19], dtype=int)
    model_hkl = transform @ source_hkl
    model_a = expected @ transform.T
    assert model_a @ model_hkl == pytest.approx(expected @ source_hkl, abs=1e-13)


def test_pets_row_plan_cache_matches_direct_and_updates_current_model_ug(
    tmp_path: Path,
) -> None:
    source = tmp_path / "test.cif_pets"
    source.write_text(_pets_text())
    source.with_suffix(".hkl").write_text("1 0 0 10 1 1 0 0 0 -1 0 0 -1\n0 1 0 20 2 2 0 0 0 -1 0 0 -1\n")
    bridge = import_pets_to_grond(source, tmp_path / "bridge", space_group_number=1)
    structure = tmp_path / "model.res"
    structure.write_text(
        "TITL test\nCELL 0.025 5 6 7 90 90 90\nLATT -1\n"
        "SFAC C 1 0 1 0 1 0 1 0\nFVAR 1\n"
        "C1 1 0.1 0.2 0.3 11.0 0.05\n"
        "C2 1 0.4 0.5 0.6 11.0 0.05\nHKLF 4\nEND\n"
    )
    data = XDSData(bridge.hkl, hkl_transform="identity")
    model = parse_model(structure)
    config = Config(
        n_basis=4,
        n_points=3,
        topup_d_min=0.5,
        max_support_sg=1.0,
        thickness_A=10.0,
        thickness_model="single",
        thickness_grad_frac=0.0,
        thickness_points=1,
        ug_d_min=0.5,
        row_plan_cache="off",
    )
    direct = XDSBlochPort(data, model, config, cache_dir=tmp_path / "direct_cache", res_hash="model0").predict_rows(
        workers=1
    )
    cached_config = dataclasses.replace(config, row_plan_cache="readwrite")
    cache_dir = tmp_path / "row_cache"
    cached = XDSBlochPort(data, model, cached_config, cache_dir=cache_dir, res_hash="model0").predict_rows(workers=1)
    assert cached["dyn"] == pytest.approx(direct["dyn"], rel=2e-12, abs=1e-15)
    assert cached["born"] == pytest.approx(direct["born"], rel=2e-12, abs=1e-15)
    assert cached["basis_size"].tolist() == direct["basis_size"].tolist()

    moved_model = copy.deepcopy(model)
    moved_model["atoms"][0]["x"] += 0.02
    moved = XDSBlochPort(
        data,
        moved_model,
        cached_config,
        cache_dir=cache_dir,
        res_hash="model1",
    ).predict_rows(workers=1)
    assert not np.allclose(moved["dyn"], cached["dyn"], rtol=1e-8, atol=1e-15)


def test_pets_support_scan_uses_full_recorded_wedge(tmp_path: Path) -> None:
    source = tmp_path / "test.cif_pets"
    source.write_text(_pets_text())
    source.with_suffix(".hkl").write_text("1 0 0 10 1 1 0 0 0 -1 0 0 -1\n0 1 0 20 2 2 0 0 0 -1 0 0 -1\n")
    bridge = import_pets_to_grond(source, tmp_path / "bridge", space_group_number=1)
    structure = tmp_path / "model.res"
    structure.write_text(
        "TITL test\nCELL 0.025 5 6 7 90 90 90\nLATT -1\n"
        "SFAC C 1 0 1 0 1 0 1 0\nFVAR 1\n"
        "C1 1 0.1 0.2 0.3 11.0 0.05\nHKLF 4\nEND\n"
    )
    data = XDSData(bridge.hkl, hkl_transform="identity")
    config = Config(
        n_basis=8,
        n_points=3,
        topup_d_min=0.5,
        max_support_sg=1.0,
    )
    port = XDSBlochPort(
        data,
        parse_model(structure),
        config,
        cache_dir=tmp_path / "cache",
        res_hash="model",
    )
    orientations, *_rest = port.pets_exact_orientation_samples(0)
    expected, _metadata = port.pets_exact_select_common_basis(data.hkl[0], orientations, 0)

    observed = scan_basis_sizes(port, np.asarray([0]), workers=1, chunksize=1)

    assert observed.tolist() == [len(expected)]


def test_sg19_dynamical_absence_is_retained_and_flagged_by_default(
    tmp_path: Path,
) -> None:
    source = tmp_path / "absence.cif_pets"
    source.write_text(_pets_text().replace("0 1 0 20 2 2", "1 1 1 20 2 2"))
    source.with_suffix(".hkl").write_text("1 0 0 10 1 1 0 0 0 -1 0 0 -1\n1 1 1 20 2 2 0 0 0 -1 0 0 -1\n")
    retained = import_pets_to_grond(source, tmp_path / "retained", space_group_number=19)
    retained_summary = json.loads(retained.summary.read_text())
    retained_rows = json.loads((retained.out_dir / "PETS_ROW_SELECTION.json").read_text())
    absent = next(row for row in retained_rows if (row["h"], row["k"], row["l"]) == (1, 0, 0))
    assert absent["systematic_absence"] is True
    assert absent["selected"] is True
    assert retained_summary["rows"] == 2

    filtered = import_pets_to_grond(
        source,
        tmp_path / "filtered",
        space_group_number=19,
        exclude_systematic_absences=True,
    )
    filtered_summary = json.loads(filtered.summary.read_text())
    assert filtered_summary["rows"] == 1


def test_pets_evaluate_tiny_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PYTHONPATH", str((REPO_ROOT / "src").resolve()))
    source = tmp_path / "test.cif_pets"
    source.write_text(_pets_text() + "1 1 0 15 1.5 1\n1 0 1 12 1.2 2\n")
    source.with_suffix(".hkl").write_text(
        "1 0 0 10 1 1 0 0 0 -1 0 0 -1\n"
        "0 1 0 20 2 2 0 0 0 -1 0 0 -1\n"
        "1 1 0 15 1.5 1 0 0 0 -1 0 0 -1\n"
        "1 0 1 12 1.2 2 0 0 0 -1 0 0 -1\n"
    )
    structure = tmp_path / "model.res"
    structure.write_text(
        "TITL test\nCELL 0.025 5 6 7 90 90 90\nLATT -1\n"
        "SFAC C 1 0 1 0 1 0 1 0\nFVAR 1\n"
        "C1 1 0.1 0.2 0.3 11.0 0.05\nHKLF 4\nEND\n"
    )
    bridge = import_pets_to_grond(
        source,
        tmp_path / "bridge",
        space_group_number=1,
        structure=structure,
    )
    args = build_parser().parse_args(
        [
            "pets-evaluate",
            "--prep",
            str(bridge.prep),
            "--out-dir",
            str(tmp_path / "evaluate"),
            "--n-basis",
            "4",
            "--n-points",
            "1",
            "--topup-d-min",
            "0.5",
            "--max-support-sg",
            "1.0",
            "--thickness-grid",
            "10",
            "--workers",
            "1",
            "--no-paired-inverse-hand",
        ]
    )
    assert args.func(args) == 0
    result = json.loads((tmp_path / "evaluate" / "PETS_NATIVE_EVALUATION.json").read_text())
    assert result["status"] == "complete"
    assert result["frame_scale_count"] == 0
    assert result["applied_frame_groups"] == 2
    assert result["detector_scale_count"] == 0
    assert result["hand_diagnostic"] is None


def test_virtual_frame_materialization_requires_explicit_acceptance(tmp_path: Path) -> None:
    source = tmp_path / "dyn.cif_pets"
    source.write_text(_pets_text(repeat=True))
    with pytest.raises(ValueError, match="companion 13-column PETS .hkl"):
        read_pets_contract(source)
    contract = read_pets_contract(source, allow_virtual_frames=True)
    assert contract["virtual_frames"] is True
    assert contract["repeated_signed_hkls"] == 1


def test_orientation_loop_must_agree_with_pets_rotation_convention(tmp_path: Path) -> None:
    source = tmp_path / "bad.cif_pets"
    source.write_text(_pets_text(bad_zone=True))
    with pytest.raises(ValueError, match="inconsistent with Rz"):
        read_pets_contract(source)


def test_precession_geometry_is_rejected_instead_of_modeled_as_rotation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "precession.cif_pets"
    source.write_text(
        _pets_text().replace(
            "_diffrn_measurement_method_precession no",
            "_diffrn_measurement_method_precession yes",
        )
    )
    with pytest.raises(ValueError, match="continuous-rotation data only"):
        read_pets_contract(source)


def test_import_pets_is_exposed_by_top_level_cli() -> None:
    args = build_parser().parse_args(
        [
            "import-pets",
            "--cif-pets",
            "sample_dyn.cif_pets",
            "--allow-virtual-frames",
            "--klar-rsg",
            "0.66",
            "--space-group-number",
            "19",
            "--out-dir",
            "pets_bridge",
        ]
    )
    assert args.func.__name__ == "cmd_import_pets"
    assert args.allow_virtual_frames is True
    assert args.klar_rsg == pytest.approx(0.66)
    assert args.space_group_number == 19
