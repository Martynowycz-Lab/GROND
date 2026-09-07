import csv
import hashlib
import json
import sys
from pathlib import Path

from grond.legacy.engine import combine_native_nuisance_runs as combine
from grond.legacy.engine.combine_native_nuisance_runs import (
    apply_hkl_weight,
    canonical_model_friedel,
    canonical_model_symmetry,
    endpoint_hkl_path,
    endpoint_key_function,
    endpoint_rows_from_native,
    estimate_overlap_scales,
    file_provenance,
    filter_hkl_rows_by_resolution,
    filter_native_rows_by_resolution,
    hkl_resolution_stats,
    merge_hkl_rows,
    metrics,
    run_shelxl,
    safe_fixed_width_factor,
    scale_contract_provenance,
    scale_hkl_rows,
    selected_native_wght,
    write_no_exti_ins,
)


def test_resolution_filter_uses_shared_model_cell_not_per_dataset_row_d():
    model = {"cell": (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)}
    rows = [
        {"h": "10", "k": "0", "l": "0", "d_A": "0.5"},
        {"h": "20", "k": "0", "l": "0", "d_A": "1.0"},
        {"h": "0", "k": "10", "l": "0", "d_A": ""},
    ]

    kept, audit = filter_native_rows_by_resolution(rows, model, 0.75)

    assert kept == [rows[0], rows[2]]
    assert audit == {
        "requested_min_d_spacing_A": 0.75,
        "input_rows": 3,
        "kept_rows": 2,
        "retained_rows": 2,
        "dropped_rows": 1,
        "retained_fraction": 2 / 3,
        "actual_input_min_d_spacing_A": 0.5,
        "actual_retained_min_d_spacing_A": 1.0,
        "actual_retained_max_d_spacing_A": 1.0,
        "d_spacings_calculated_from_model": 3,
        "stored_d_A_membership_disagreements": 2,
        "cell_A_deg": [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
    }


def test_resolution_filter_applies_to_corrected_hkl_source():
    model = {"cell": (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)}
    rows = [(10, 0, 0, 100.0, 5.0), (20, 0, 0, 50.0, 5.0)]

    assert filter_hkl_rows_by_resolution(rows, model, 0.75) == [rows[0]]
    assert hkl_resolution_stats(rows, model) == {
        "rows": 2,
        "actual_min_d_spacing_A": 0.5,
        "actual_max_d_spacing_A": 1.0,
        "cell_A_deg": [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
    }


def test_file_and_scale_contract_provenance_identify_materialized_sources(tmp_path: Path):
    source = tmp_path / "native_dyn_calc.csv"
    source.write_bytes(b"exact row source\n")
    provenance = file_provenance(source)

    assert provenance == {
        "path": str(source.resolve()),
        "exists": True,
        "size_bytes": len(b"exact row source\n"),
        "sha256": hashlib.sha256(b"exact row source\n").hexdigest(),
    }

    summary = {
        "best_native": {
            "scale_count": 92,
            "scale_fit_split": "work",
            "nuisance": {"scale_compression_model": "adaptive_28_det_8"},
        },
        "scale_compression": {"selected_output": {"name": "adaptive_28_det_8"}},
    }
    rows = [
        {
            "scale_compression_model": "adaptive_28_det_8",
            "scale_replay_model": "stride_6_det_8",
        }
    ]

    assert scale_contract_provenance(summary, rows) == {
        "model": "stride_6_det_8",
        "models": ["stride_6_det_8"],
        "model_source": "row_table.scale_replay_model",
        "parameter_count": 92,
        "fit_split": "work",
    }


def test_endpoint_cutoff_alias_writes_complete_summary_and_report(tmp_path: Path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    fields = [
        "h",
        "k",
        "l",
        "d_A",
        "target_I",
        "sigma_eff",
        "dyn_raw",
        "dyn_scaled",
        "born_scaled",
        "q_dyn_over_born",
        "basis_size",
        "split",
        "dataset_role",
        "raw_row_weight",
        "endpoint_weight",
        "scale_replay_model",
    ]
    source_rows = [
        {
            "h": "10",
            "k": "0",
            "l": "0",
            "d_A": "0.5",
            "target_I": "100",
            "sigma_eff": "5",
            "dyn_raw": "50",
            "dyn_scaled": "100",
            "born_scaled": "80",
            "q_dyn_over_born": "1",
            "basis_size": "20",
            "split": "work",
            "dataset_role": "primary",
            "raw_row_weight": "1",
            "endpoint_weight": "1",
            "scale_replay_model": "adaptive_28_det_8",
        },
        {
            "h": "20",
            "k": "0",
            "l": "0",
            "d_A": "1.0",
            "target_I": "50",
            "sigma_eff": "5",
            "dyn_raw": "25",
            "dyn_scaled": "50",
            "born_scaled": "40",
            "q_dyn_over_born": "1",
            "basis_size": "20",
            "split": "free",
            "dataset_role": "primary",
            "raw_row_weight": "1",
            "endpoint_weight": "1",
            "scale_replay_model": "adaptive_28_det_8",
        },
    ]
    with (run / "native_dyn_calc.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(source_rows)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "best_native": {
                    "scale_count": 92,
                    "scale_fit_split": "work",
                    "nuisance": {"scale_replay_model": "adaptive_28_det_8"},
                    "metrics": {
                        "all": {
                            "dyn": {"n": 2, "R1_all": 0.99, "wR2_all": 0.98},
                            "born": {"n": 2, "R1_all": 0.97, "wR2_all": 0.96},
                        }
                    },
                }
            }
        )
        + "\n"
    )
    structure = tmp_path / "model.res"
    structure.write_text(
        "\n".join(
            [
                "TITL cutoff provenance",
                "CELL 0.025079 10 10 10 90 90 90",
                "SFAC C 1 0 1 0 1 0 1 0",
                "FVAR 1.0",
                "C1 1 0.1 0.2 0.3 11.0 0.05",
                "L.S. 1",
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )
    out = tmp_path / "combined"
    model = {
        "cell": (10.0, 10.0, 10.0, 90.0, 90.0, 90.0),
        "symmetry_ops": [{"rot": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}],
    }
    monkeypatch.setattr(combine, "parse_model", lambda _path: model)
    monkeypatch.setattr(combine, "require_combined_endpoint_shelxl", lambda *_args, **_kwargs: "fake-shelxl")
    monkeypatch.setattr(
        combine,
        "run_shelxl",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "R1_gt": 0.05,
            "R1_all": 0.06,
            "wR2_all": 0.10,
            "GooF": 1.0,
            "active_wght_a": 0.0,
            "active_wght_b": 0.0,
        },
    )
    monkeypatch.setattr(combine, "fail_if_shelxl_became_unavailable", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "combine_native_nuisance_runs",
            "--run",
            f"biotin={run}",
            "--out-dir",
            str(out),
            "--res",
            str(structure),
            "--endpoint-min-d-spacing",
            "0.75",
            "--interdataset-scale",
            "none",
        ],
    )

    combine.main()

    result = json.loads((out / "combined_summary.json").read_text())
    contract = result["provenance"]["resolution_contract"]
    assert contract["requested_min_d_spacing_A"] == 0.75
    assert contract["input_native_rows"] == 2
    assert contract["retained_native_rows"] == 1
    assert contract["dropped_native_rows"] == 1
    assert contract["actual_retained_native_min_d_spacing_A"] == 1.0
    assert contract["actual_merged_endpoint_min_d_spacing_A"] == 1.0
    assert result["combined_endpoint"]["actual_min_d_spacing_A"] == 1.0
    assert result["provenance"]["scale_contract"]["total_parameter_count"] == 92
    assert result["runs"][0]["scale_contract"]["model"] == "adaptive_28_det_8"
    assert result["runs"][0]["source_artifacts"]["row_table"]["sha256"]
    assert result["provenance"]["structure"]["sha256"]
    run_summary = result["runs"][0]
    filtered = run_summary["resolution_filtered_native_metrics"]
    assert filtered["all"]["dyn"]["n"] == 1
    assert filtered["all"]["dyn"]["R1_all"] == 0.0
    assert filtered["all"]["dyn"]["wR2_all"] == 0.0
    assert filtered["work"]["dyn"]["n"] == 1
    assert filtered["free"]["dyn"]["n"] == 0
    assert filtered["all"]["born"]["n"] == 1
    assert abs(filtered["all"]["born"]["R1_all"] - (1.0 - 0.8**0.5)) < 1e-12
    assert run_summary["best_native_summary"]["metrics"] == filtered
    assert run_summary["best_native_summary"]["metrics_contract"] == {
        "scope": "combined_shared_cell_resolution_filtered_rows",
        "cell_source": "shared_structure_res",
        "requested_min_d_spacing_A": 0.75,
        "input_rows": 2,
        "retained_resolution_rows": 1,
        "rows_used": 1,
        "dataset_role": "primary",
        "raw_row_weight": 1.0,
        "raw_row_weight_applied_to_sigma": False,
        "native_wght_a": 0.126534,
        "native_wght_b": 0.0,
    }
    assert run_summary["source_best_native_summary"]["metrics"]["all"]["dyn"]["R1_all"] == 0.99
    assert result["timing"]["compute_wall_seconds"] >= 0.0
    assert result["timing"]["wall_seconds"] == result["timing"]["compute_wall_seconds"]
    report = (out / "REPORT.md").read_text()
    assert "d=1.000000 A" in report
    assert "adaptive_28_det_8" in report
    assert "Source Provenance" in report
    assert "compute wall time" in report
    assert "not copied from the unfiltered source-run summary" in report
    assert "| 0.00 | 0.00 | 10.56 |" in report
    assert "99.00" not in report


def test_zero_endpoint_weight_excludes_hkl_source_rows():
    rows = [(1, 0, 0, 100.0, 5.0)]

    assert apply_hkl_weight(rows, 0.0) == []
    assert apply_hkl_weight(rows, -1.0) == []
    assert apply_hkl_weight(rows, 1.0) == rows
    assert apply_hkl_weight(rows, 0.25) == [(1, 0, 0, 100.0, 10.0)]


def test_selected_native_weights_use_current_summary_fields():
    summary = {
        "shelx_weight": {
            "source_a": 0.1,
            "source_b": 0.2,
            "selected_native_a": 0.3,
            "selected_native_b": 0.4,
        }
    }

    assert selected_native_wght(summary) == (0.3, 0.4)


def test_joint_metrics_are_order_invariant_with_per_dataset_native_weights():
    rows = [
        {
            "target_I": "100",
            "sigma_eff": "5",
            "dyn_scaled": "90",
            "native_wght_a": "0",
            "native_wght_b": "0",
        },
        {
            "target_I": "50",
            "sigma_eff": "2",
            "dyn_scaled": "40",
            "native_wght_a": "0.3",
            "native_wght_b": "0.1",
        },
    ]

    forward = metrics(rows, "dyn_scaled", 0.126534, 0.0)
    reverse = metrics(list(reversed(rows)), "dyn_scaled", 0.126534, 0.0)

    assert forward["wR2_all"] == reverse["wR2_all"]
    assert forward["mw_abs"] == reverse["mw_abs"]


def test_native_endpoint_weight_is_applied_once_after_row_conversion():
    native = [
        {
            "h": "1",
            "k": "0",
            "l": "0",
            "target_I": "100",
            "sigma_eff": "2",
            "dyn_raw": "50",
            "dyn_scaled": "100",
            "q_dyn_over_born": "1.0",
        }
    ]

    unweighted = endpoint_rows_from_native(native, 1.0)
    weighted = apply_hkl_weight(unweighted, 0.25)

    assert unweighted == [(1, 0, 0, 100.0, 2.0)]
    assert weighted == [(1, 0, 0, 100.0, 4.0)]


def test_dataset_sigma_calibration_precedes_policy_weight(monkeypatch):
    rows = [(1, 0, 0, 100.0, 4.0)]
    monkeypatch.setattr(
        combine,
        "fit_uniform_z_rms_sigma_scale",
        lambda *_args, **_kwargs: {
            "policy": "uniform_z_rms",
            "status": "ok",
            "sigma_scale": 0.5,
        },
    )

    calibrated, meta = combine.calibrate_dataset_endpoint_rows(
        rows,
        model={},
        policy="uniform_z_rms",
        holdout_fraction=0.2,
        min_scale=0.1,
        max_scale=4.0,
        holdout_floor=False,
    )
    weighted = apply_hkl_weight(calibrated, 0.25)

    assert meta["sigma_scale"] == 0.5
    assert calibrated == [(1, 0, 0, 100.0, 2.0)]
    assert weighted == [(1, 0, 0, 100.0, 4.0)]


def test_overlap_scales_put_datasets_on_common_intensity_scale():
    rows_by_dataset = {
        "d1": [(1, 0, 0, 10.0, 1.0), (0, 1, 0, 20.0, 1.0)],
        "d2": [(1, 0, 0, 20.0, 1.0), (0, 1, 0, 40.0, 1.0)],
    }
    result = estimate_overlap_scales(rows_by_dataset, iterations=4)
    scales = result["scales"]
    d1_scaled = scale_hkl_rows(rows_by_dataset["d1"], scales["d1"])
    d2_scaled = scale_hkl_rows(rows_by_dataset["d2"], scales["d2"])
    assert abs(d1_scaled[0][3] - d2_scaled[0][3]) < 1e-6
    assert abs(d1_scaled[1][3] - d2_scaled[1][3]) < 1e-6
    assert result["overlap_counts"]["d1"] == 2
    assert result["overlap_counts"]["d2"] == 2


def test_overlap_scales_can_use_model_symmetry_key():
    rows_by_dataset = {
        "d1": [(1, 2, 0, 10.0, 1.0)],
        "d2": [(2, 1, 0, 20.0, 1.0)],
    }
    swap_hk = [
        {"rot": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        {"rot": [[0, 1, 0], [1, 0, 0], [0, 0, 1]]},
    ]
    def key_for(h_index: int, k_index: int, l_index: int):
        return canonical_model_friedel(h_index, k_index, l_index, swap_hk)

    default = estimate_overlap_scales(rows_by_dataset, iterations=1)
    symmetry_aware = estimate_overlap_scales(rows_by_dataset, iterations=2, canonicalizer=key_for)

    assert default["overlap_counts"]["d1"] == 0
    assert symmetry_aware["overlap_counts"]["d1"] == 1
    assert symmetry_aware["overlap_counts"]["d2"] == 1


def test_within_dataset_replicates_do_not_count_as_overlap():
    rows_by_dataset = {
        "d1": [(1, 0, 0, 100.0, 5.0), (1, 0, 0, 102.0, 5.0)],
        "d2": [(0, 1, 0, 200.0, 5.0), (0, 1, 0, 198.0, 5.0)],
    }

    result = estimate_overlap_scales(rows_by_dataset, iterations=4, canonicalizer=lambda h, k, ell: (h, k, ell))

    assert result["iterations"] == 0
    assert result["overlap_counts"] == {"d1": 0, "d2": 0}
    assert result["scales"] == {"d1": 1.0, "d2": 1.0}


def test_endpoint_hkl_path_finds_b_zero_noexti_directory(tmp_path: Path):
    endpoint = tmp_path / "endpoint_lam_0p9_B_0p0_noexti"
    endpoint.mkdir()
    hkl = endpoint / "dyn_nuisance_corrected_unmerged.hkl"
    hkl.write_text("   1   0   0  100.00   10.00\n   0   0   0    0.00    0.00\n")
    assert endpoint_hkl_path(tmp_path, "0.9") == hkl


def test_merge_hkl_rows_can_inflate_disagreeing_replicates():
    rows = [(1, 0, 0, 10.0, 1.0), (1, 0, 0, 14.0, 1.0)]

    formal, formal_meta = merge_hkl_rows(rows, sigma_policy="formal")
    overdispersed, overdispersed_meta = merge_hkl_rows(rows, sigma_policy="chisq_overdispersion")

    assert formal[0][:4] == overdispersed[0][:4]
    assert abs(formal[0][4] - (0.5**0.5)) < 1e-12
    assert abs(overdispersed[0][4] - 2.0) < 1e-12
    assert formal_meta["inflated_rows"] == 0
    assert overdispersed_meta["inflated_rows"] == 1
    assert overdispersed_meta["rows_with_multiple_observations"] == 1


def test_merge_hkl_rows_can_merge_model_symmetry_before_endpoint_refinement():
    swap_hk = [
        {"rot": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        {"rot": [[0, 1, 0], [1, 0, 0], [0, 0, 1]]},
    ]
    rows = [(1, 2, 0, 10.0, 1.0), (2, 1, 0, 14.0, 1.0)]
    key_for = endpoint_key_function("model_symmetry", swap_hk)

    exact, _exact_meta = merge_hkl_rows(rows, sigma_policy="formal")
    merged, meta = merge_hkl_rows(rows, sigma_policy="formal", canonicalizer=key_for)

    assert len(exact) == 2
    assert len(merged) == 1
    assert merged[0][:3] == canonical_model_symmetry(1, 2, 0, swap_hk)
    assert abs(merged[0][3] - 12.0) < 1e-12
    assert abs(merged[0][4] - (0.5**0.5)) < 1e-12
    assert meta["rows_with_multiple_observations"] == 1


def test_model_symmetry_merge_preserves_friedel_opposites_until_requested():
    identity = [{"rot": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]
    rows = [(1, 0, 0, 10.0, 1.0), (-1, 0, 0, 14.0, 1.0)]

    symmetry_rows, _ = merge_hkl_rows(rows, sigma_policy="formal", canonicalizer=endpoint_key_function("model_symmetry", identity))
    friedel_rows, _ = merge_hkl_rows(rows, sigma_policy="formal", canonicalizer=endpoint_key_function("model_symmetry_friedel", identity))

    assert len(symmetry_rows) == 2
    assert len(friedel_rows) == 1
    assert friedel_rows[0][:3] == canonical_model_friedel(1, 0, 0, identity)


def test_safe_fixed_width_factor_keeps_shelxl_fields_in_range():
    assert safe_fixed_width_factor([(1, 0, 0, 100.0, 5.0)]) == 1.0
    positive_factor = safe_fixed_width_factor([(1, 0, 0, 120000.0, 5.0)])
    negative_factor = safe_fixed_width_factor([(1, 0, 0, -12000.0, 5.0)])
    assert abs(120000.0 * positive_factor - 90000.0) < 1e-9
    assert abs(-12000.0 * negative_factor - -9000.0) < 1e-9


def test_combined_endpoint_writer_forces_zero_wght_and_ls1(tmp_path: Path):
    src = tmp_path / "model.res"
    dst = tmp_path / "multi_dyn_corrected.ins"
    src.write_text(
        "\n".join(
            [
                "TITL combined",
                "CELL 1.0 5.0 6.0 7.0 90 90 90",
                "SFAC C 1 0 1 0 1 0 1 0",
                "WGHT 0.200000 1.000000",
                "EXTI 0.123",
                "CGLS 10",
                "FVAR 1.0",
                "HKLF 3",
                "END",
                "WGHT 0.1111 0.2222",
            ]
        )
        + "\n"
    )

    write_no_exti_ins(src, dst, "multi_dyn_corrected.hkl", wght=(0.0, 0.0), ls_cycles=1)

    text = dst.read_text()
    assert "WGHT      0.0000      0.0000" in text
    assert "WGHT 0.200000" not in text
    assert "WGHT 0.1111" not in text
    assert "EXTI" not in text
    assert "CGLS 1" in text
    assert "CGLS 10" not in text
    assert "HKLF 4" in text


def test_combined_run_shelxl_freeze_zero_survives_multiple_passes(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    stem = "multi_dyn_corrected"
    src = tmp_path / "model.res"
    src.write_text(
        "\n".join(
            [
                "TITL combined",
                "CELL 0.025079 5.0 6.0 7.0 90 90 90",
                "SFAC C 1 0 1 0 1 0 1 0",
                    "WGHT 0.200000 1.000000",
                    "FVAR 1.0",
                    "C1 1 0.10 0.20 0.30 11.0 0.05",
                    "L.S. 10",
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )
    write_no_exti_ins(src, work / f"{stem}.ins", f"{stem}.hkl", wght=(0.0, 0.0), ls_cycles=1)
    (work / f"{stem}.hkl").write_text("   1   0   0  100.00   10.00\n   0   0   0    0.00    0.00\n")
    fake_shelxl = tmp_path / "fake_shelxl"
    fake_shelxl.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import sys
stem = sys.argv[1]
ins = Path(stem + ".ins").read_text(errors="replace")
wght = next((line for line in ins.splitlines() if line.strip().upper().startswith("WGHT")), "WGHT 9.0000 9.0000")
Path(stem + ".lst").write_text("\\n".join([
    wght,
    " 1 Unique reflections, of which 0 suppressed",
    " wR2 = 0.1234, GooF = S = 0.987, Restrained GooF = 0.987 for all data",
    " R1 = 0.0500 for 1 Fo > 4sig(Fo) and 0.0600 for all 1 data",
    " Recommended weighting scheme:  WGHT    0.3210    0.6540",
    " 0 atoms may be split and 0 atoms NPD",
]) + "\\n")
Path(stem + ".res").write_text("\\n".join([
    "TITL combined output",
    "CELL 0.025079 5.0 6.0 7.0 90 90 90",
    "SFAC C 1 0 1 0 1 0 1 0",
    wght,
    "FVAR 1.0",
    "C1 1 0.10 0.20 0.30 11.0 0.05",
    "L.S. 10",
    "HKLF 4",
    "END",
    "WGHT 0.3210 0.6540",
]) + "\\n")
"""
    )
    fake_shelxl.chmod(0o755)

    result = run_shelxl(work, stem, str(fake_shelxl), iterations=2, wght_mode="freeze-zero", endpoint_ls_cycles=1)

    assert result["status"] == "ok"
    assert result["wght_mode"] == "freeze-zero"
    assert result["active_wght_a"] == 0.0
    assert result["active_wght_b"] == 0.0
    text = (work / f"{stem}.res").read_text()
    assert "WGHT      0.0000      0.0000" in text
    assert "WGHT 0.3210" not in text
    assert "L.S. 1" in text
    assert "L.S. 10" not in text


def test_combined_run_shelxl_rejects_stale_outputs_after_failure(tmp_path: Path):
    work = tmp_path / "work_failed"
    work.mkdir()
    stem = "multi_dyn_corrected"
    stale = "TITL stale\nCELL 0.025079 5 6 7 90 90 90\nSFAC C\nEND\n"
    (work / f"{stem}.ins").write_text(stale)
    (work / f"{stem}.res").write_text(stale)
    (work / f"{stem}.lst").write_text("wR2 = 0.1\n")
    failing = tmp_path / "failing_shelxl"
    failing.write_text("#!/bin/sh\nexit 9\n")
    failing.chmod(0o755)

    result = run_shelxl(work, stem, str(failing), iterations=1)

    assert result["status"] == "shelxl_failed"
    assert result["returncode"] == 9
    assert result.get("res_normalized_for_reopen") is not True
    assert not (work / f"{stem}.res").exists()
    assert not (work / f"{stem}.lst").exists()
