"""CLI model health regression tests."""


import json
from pathlib import Path

import pytest

import grond.cli as cli
from grond.cli import (
    build_parser,
    resolve_structure_update_strategy,
)
from tests.cli._support import bound_shelx_health, valid_shelx_res_text


def test_selected_feedback_endpoint_uses_one_hash_bound_cycle(tmp_path: Path):
    cycle = tmp_path / "cycle_02" / "combined"
    endpoint = cycle / "combined_endpoint_lam_1p0"
    endpoint.mkdir(parents=True)
    model = endpoint / "multi_dyn_corrected.res"
    model.write_text("TITL clean\nEND\n")
    lst = model.with_suffix(".lst")
    lst.write_text(
        "wR2 = 0.0874, GooF = S = 0.980 for all data\n"
        "R1 = 0.0731 for 813 Fo > 4sig(Fo) and 0.1752 for all 1917 data\n"
        "     0  atoms may be split and     0  atoms NPD\n"
    )
    digest = cli.sha256_file(model)
    (cycle / "combined_summary.json").write_text(
        json.dumps(
            {
                "combined_endpoint": {
                    "shelxl": {
                        "R1_gt": 0.0741,
                        "R1_all": 0.1773,
                        "wR2_all": 0.0738,
                        "GooF": 0.99,
                        "res_health": {"path": str(model), "sha256": digest},
                    },
                    "artifacts": {"res": str(model), "res_sha256": digest},
                }
            }
        )
        + "\n"
    )
    result = {
        "model": str(model),
        "structure_feedback": {
            "current_model": str(model),
            "cycles": [
                {
                    "cycle": 2,
                    "output_model": str(model),
                    "combined": str(cycle),
                    "adopted_for_feedback": True,
                }
            ],
        },
    }

    details = cli._quick_selected_feedback_shelxl(
        result,
        model,
        {"R1_gt": 0.01, "R1_all": 0.02, "wR2_all": 0.03, "npd_count": 2},
    )

    assert details["R1_gt"] == pytest.approx(0.0741)
    assert details["R1_all"] == pytest.approx(0.1773)
    assert details["wR2_all"] == pytest.approx(0.0738)
    assert details["GooF"] == pytest.approx(0.990)
    assert details["npd_count"] == 0
    assert details["split_count"] == 0


def test_partial_lst_cannot_erase_recorded_npd_pathology(tmp_path: Path):
    lst = tmp_path / "partial.lst"
    lst.write_text(
        "wR2 = 0.0874, GooF = S = 0.980 for all data\n"
        "R1 = 0.0731 for 813 Fo > 4sig(Fo) and 0.1752 for all 1917 data\n"
    )

    details = cli._quick_shelxl_details(
        {"npd_count": 2, "npd_atoms": ["C6", "C7"], "has_npd": True, "split_count": 0},
        lst,
    )

    assert details["R1_all"] == pytest.approx(0.1752)
    assert details["npd_count"] == 2
    assert details["npd_atoms"] == ["C6", "C7"]
    assert details["has_npd"] is True


def test_zero_pathology_summary_cannot_erase_recorded_npd_or_split_warnings(tmp_path: Path):
    lst = tmp_path / "replacement.lst"
    lst.write_text(
        "wR2 = 0.0500, GooF = S = 1.000 for all data\n"
        "R1 = 0.0400 for 100 Fo > 4sig(Fo) and 0.0600 for all 200 data\n"
        "     0  atoms may be split and     0  atoms NPD\n"
    )
    recorded = {
        "R1_all": 0.20,
        "wR2_all": 0.25,
        "npd_count": 2,
        "npd_atoms": ["C6", "C7"],
        "split_count": 1,
        "has_npd": True,
        "has_split_warnings": True,
    }

    details = cli._quick_shelxl_details(recorded, lst)

    assert details["R1_all"] == pytest.approx(0.20)
    assert details["wR2_all"] == pytest.approx(0.25)
    assert details["npd_count"] == 2
    assert details["npd_atoms"] == ["C6", "C7"]
    assert details["split_count"] == 1
    assert details["has_npd"] is True
    assert details["has_split_warnings"] is True


def test_structure_update_strategy_can_force_shelxl_feedback():
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--structure-update",
            "shelxl_feedback",
        ]
    )
    assert resolve_structure_update_strategy(args) == "shelxl_feedback"
    assert resolve_structure_update_strategy(args, multi_dataset=True) == "shelxl_feedback"


def test_structure_update_strategy_keeps_native_gradient_as_explicit_multi_dataset_diagnostic():
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--structure-update",
            "native_gradient",
        ]
    )
    assert resolve_structure_update_strategy(args, multi_dataset=True) == "native_gradient"


def test_shelxl_feedback_current_model_skips_empty_res(tmp_path: Path):
    run_root = tmp_path / "feedback"
    good_run = run_root / "cycle_04_N132Q20"
    bad_run = run_root / "cycle_05_N132Q20"
    good_res = good_run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    bad_res = bad_run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    good_res.parent.mkdir(parents=True)
    bad_res.parent.mkdir(parents=True)
    good_res.write_text(valid_shelx_res_text())
    bad_res.write_text("")
    (run_root / "shelxl_feedback_loop_summary.json").write_text(
        json.dumps(
            {
                    "current_model": str(bad_res),
                    "runs": [
                        {
                            "run_dir": str(good_run),
                            "returncode": 0,
                            "output_model": str(good_res),
                            "endpoint_shelxl_status": "ok",
                            "endpoint_res_health": bound_shelx_health(good_res),
                        },
                    {
                        "run_dir": str(bad_run),
                        "returncode": 3,
                        "output_model": str(bad_res),
                        "endpoint_res_health": {"ok": False, "reason": "empty_res"},
                    },
                ],
            }
        )
        + "\n"
    )

    assert cli.shelxl_feedback_current_model(run_root) == good_res
    assert cli.shelxl_feedback_current_run(run_root) == good_run


def test_shelxl_feedback_current_model_skips_npd_res(tmp_path: Path):
    run_root = tmp_path / "feedback"
    good_run = run_root / "cycle_04_N132Q20"
    npd_run = run_root / "cycle_05_N132Q20"
    good_res = good_run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    npd_res = npd_run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    good_res.parent.mkdir(parents=True)
    npd_res.parent.mkdir(parents=True)
    good_res.write_text(valid_shelx_res_text())
    npd_res.write_text(valid_shelx_res_text("npd"))
    (run_root / "shelxl_feedback_loop_summary.json").write_text(
        json.dumps(
            {
                    "current_model": str(npd_res),
                    "runs": [
                        {
                            "run_dir": str(good_run),
                            "returncode": 0,
                            "output_model": str(good_res),
                            "endpoint_shelxl_status": "ok",
                            "endpoint_res_health": bound_shelx_health(good_res),
                        },
                        {
                            "run_dir": str(npd_run),
                            "returncode": 0,
                            "output_model": str(npd_res),
                            "endpoint_shelxl_status": "adp_pathology",
                            "endpoint_res_health": bound_shelx_health(
                                npd_res,
                                safe_for_feedback=False,
                                npd_count=1,
                                split_count=0,
                                npd_atoms=["O001"],
                                has_npd=True,
                            ),
                    },
                ],
            }
        )
        + "\n"
    )

    assert cli.shelxl_feedback_current_model(run_root) == good_res
    assert cli.shelxl_feedback_current_run(run_root) == good_run


def test_shelxl_feedback_record_rejects_model_replaced_after_summary(tmp_path: Path):
    model = tmp_path / "endpoint.res"
    model.write_text(valid_shelx_res_text("recorded"))
    record = {
        "output_model": str(model),
        "endpoint_shelxl_status": "ok",
        "endpoint_res_health": bound_shelx_health(model),
    }
    assert cli.shelxl_feedback_record_safe_for_reuse(record) is True

    model.write_text(valid_shelx_res_text("replaced"))

    assert cli.shelxl_feedback_record_safe_for_reuse(record) is False


def test_multi_combined_endpoint_model_binds_current_lambda_and_artifact(tmp_path: Path):
    combined = tmp_path / "combined"
    stale = combined / "combined_endpoint_lam_1p0" / "multi_dyn_corrected.res"
    selected = combined / "combined_endpoint_lam_0p8" / "multi_dyn_corrected.res"
    stale.parent.mkdir(parents=True)
    selected.parent.mkdir(parents=True)
    stale.write_text(valid_shelx_res_text("stale_lambda_one"))
    selected.write_text(valid_shelx_res_text("selected_lambda_point_eight"))
    selected_health = bound_shelx_health(selected)
    summary = {
        "combined_endpoint": {
            "lambda_q": "0.8",
            "artifacts": {
                "res": str(selected),
                "res_sha256": selected_health["sha256"],
            },
            "shelxl": {"status": "ok", "res_health": selected_health},
        }
    }
    (combined / "combined_summary.json").write_text(json.dumps(summary) + "\n")

    assert cli.multi_combined_endpoint_model(combined) == selected

    summary["combined_endpoint"]["artifacts"]["res"] = str(stale)
    summary["combined_endpoint"]["artifacts"]["res_sha256"] = bound_shelx_health(stale)["sha256"]
    (combined / "combined_summary.json").write_text(json.dumps(summary) + "\n")
    assert cli.multi_combined_endpoint_model(combined) is None


def test_shelxl_feedback_current_run_prefers_final_row_recompute(tmp_path: Path):
    run_root = tmp_path / "feedback"
    cycle_run = run_root / "cycle_05_N108Q20"
    final_run = run_root / "final_row_recompute_N108Q20"
    final_res = cycle_run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    final_res.parent.mkdir(parents=True)
    final_run.mkdir(parents=True)
    final_res.write_text(valid_shelx_res_text())
    (run_root / "shelxl_feedback_loop_summary.json").write_text(
        json.dumps(
            {
                "current_model": str(final_res),
                "final_row_stats_run": str(final_run),
                "final_row_stats_model": str(final_res),
                    "runs": [
                        {
                            "run_dir": str(cycle_run),
                            "returncode": 0,
                            "output_model": str(final_res),
                            "endpoint_shelxl_status": "ok",
                            "endpoint_res_health": bound_shelx_health(final_res),
                        },
                        {
                        "label": "final_row_recompute",
                        "run_dir": str(final_run),
                        "returncode": 0,
                        "output_model": str(final_res),
                            "row_stats_model": str(final_res),
                            "skip_endpoint_refinement": True,
                            "endpoint_shelxl_status": "reused_validated_input_model",
                            "endpoint_res_health": bound_shelx_health(final_res),
                    },
                ],
            }
        )
        + "\n"
    )

    assert cli.shelxl_feedback_current_model(run_root) == final_res
    assert cli.shelxl_feedback_current_run(run_root) == final_run


def test_best_endpoint_model_rejects_empty_res(tmp_path: Path):
    run = tmp_path / "native"
    endpoint_dir = run / "endpoint_lam_1p0_B_0p0_noexti"
    endpoint_dir.mkdir(parents=True)
    lst = endpoint_dir / "dyn_nuisance_corrected.lst"
    res = endpoint_dir / "dyn_nuisance_corrected.res"
    lst.write_text("dummy\n")
    res.write_text("")
    (run / "summary.json").write_text(json.dumps({"best_endpoint": {"shelxl": {"lst": str(lst)}}}) + "\n")

    assert cli.best_endpoint_model(run) is None
