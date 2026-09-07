"""CLI report components regression tests."""


import json
from pathlib import Path

import pytest

import grond.cli as cli
from grond.io.config import DatasetSpec, PrepConfig


def test_quick_progress_info_keeps_saved_metadata_but_recomputes_counts_and_stage_sum(tmp_path: Path):
    (tmp_path / "grond_progress.json").write_text(
        json.dumps(
            {
                "completed": {
                    "old": {"returncode": 0, "elapsed_seconds": 40.0},
                    "new": {"returncode": 0, "elapsed_seconds": 30.0},
                    "failed": {"returncode": 1, "elapsed_seconds": 99.0},
                },
                "finished": {
                    "status": "ok",
                    "snapshot_kind": "completed_invocation",
                    "invocation_elapsed_seconds": 20.0,
                    "completed_stages": 1,
                    "completed_this_invocation": 1,
                    "reused_completed_stages": 0,
                    "failed_stages": 0,
                    "cumulative_stage_elapsed_seconds": 40.0,
                    "stage_elapsed_seconds": 40.0,
                },
            }
        )
        + "\n"
    )

    info = cli._quick_progress_finish_info(tmp_path)

    assert info["status"] == "ok"
    assert info["snapshot_kind"] == "completed_invocation"
    assert info["invocation_elapsed_seconds"] == 20.0
    assert info["completed_stages"] == 2
    assert info["completed_this_invocation"] == 1
    assert info["reused_completed_stages"] == 1
    assert info["failed_stages"] == 1
    assert info["cumulative_stage_elapsed_seconds"] == 70.0
    assert info["stage_elapsed_seconds"] == 70.0
    assert info["cumulative_stage_elapsed"] == "1m10s"
    assert info["stage_elapsed"] == "1m10s"


def test_quick_baseline_prefers_fresh_final_run_summary(tmp_path: Path):
    structure = tmp_path / "model.res"
    structure.write_text("TITL stale\n")
    structure.with_suffix(".log").write_text(
        "R1 = 0.1000 for 1 Fo > 4sig(Fo) and 0.1537 for all 2 data\n"
    )
    baseline_hkl = tmp_path / "baseline.hkl"
    baseline_hkl.write_text("")
    prep = PrepConfig(
        structure=str(structure),
        datasets=[
            DatasetSpec(
                name="d",
                hkl=str(tmp_path / "INTEGRATE.HKL"),
                baseline_hkl=str(baseline_hkl),
            )
        ],
    )
    summary = {
        "baseline_exti": {
            "status": "ok",
            "R1_gt": 0.1051,
            "R1_all": 0.1377,
            "wR2_all": 0.2684,
            "GooF": 1.166,
            "npd_count": 0,
            "split_count": 0,
            "lst": str(tmp_path / "fresh.lst"),
            "res_health": {
                "safe_for_feedback": True,
                "path": str(tmp_path / "fresh.res"),
            },
        }
    }

    details = cli._quick_baseline_shelxl(prep, summary)

    assert details["R1_all"] == pytest.approx(0.1377)
    assert details["fresh_rerun"] is True
    assert details["baseline_source"] == "final_run_summary.baseline_exti"
    assert details["res"] == str(tmp_path / "fresh.res")


def test_quick_source_shelx_controls_reads_fixed_width_exti_without_space(
    tmp_path: Path,
):
    model = tmp_path / "fixed_width_exti.res"
    model.write_text(
        "\n".join(
            [
                "TITL fixed-width EXTI",
                "WGHT 0.1622 1.034",
                "EXTI33651.195312",
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )

    controls = cli._quick_source_shelx_controls(model)

    assert controls["exti"] == "EXTI 33651.195312"
    assert "EXTI 33651.195312" in cli._quick_source_shelx_control_text(controls)


def test_quick_source_shelx_controls_recovers_exti_from_sibling_ins(
    tmp_path: Path,
):
    model = tmp_path / "refined.res"
    model.write_text("WGHT 0.1622 1.034\nHKLF 4\nEND\n")
    model.with_suffix(".ins").write_text(
        "WGHT 0.1622 1.034\nEXTI33651.195312\nHKLF 4\nEND\n"
    )

    controls = cli._quick_source_shelx_controls(model)

    assert controls["exti"] == "EXTI 33651.195312"
    assert controls["wght"] == "WGHT 0.162200 1.034000"


def test_effective_thickness_text_reports_mean_and_support():
    text = cli._quick_thickness_text(
        {"thickness_A": 1200.0, "thickness_model": "uniform_gradient", "thickness_grad_frac": 1.0}
    )

    assert text == "120.00 nm mean (1200.0 A); uniform-gradient support 0.00-240.00 nm"


def test_multi_thickness_text_and_table_keep_dataset_contracts_separate():
    contracts = [
        {
            "dataset": "a",
            "scale_count": 17,
            "native_weight": {"a": 0.0, "b": 0.0},
            "nuisance": {
                "thickness_A": 1200.0,
                "thickness_model": "uniform_gradient",
                "thickness_grad_frac": 0.5,
                "frame_offset": 0.0,
                "group_width_deg": 1.0,
                "detector_scale_bins": 8,
                "absorption_mode": "none",
                "absorption_frac": 0.0,
                "sg_jitter_mode": "none",
                "row_resolution_b": 0.0,
                "native_wght_a": 0.0,
                "native_wght_b": 0.0,
            },
        },
        {
            "dataset": "b",
            "scale_count": 23,
            "native_weight": {"a": 0.1, "b": 0.2},
            "nuisance": {
                "thickness_A": 600.0,
                "thickness_model": "uniform_gradient",
                "thickness_grad_frac": 1.0,
                "frame_offset": 0.25,
                "group_width_deg": 0.5,
                "detector_scale_bins": 4,
                "absorption_mode": "midgley_like",
                "absorption_frac": 0.02,
                "sg_jitter_mode": "none",
                "row_resolution_b": 0.1,
                "native_wght_a": 0.1,
                "native_wght_b": 0.2,
            },
        },
    ]

    text = cli._quick_thickness_text({}, {"thickness_A_mean": 900.0}, contracts)
    table = "\n".join(cli._quick_dataset_nuisance_table_lines(contracts))

    assert "a 120.00 nm (uniform_gradient, g=0.500)" in text
    assert "b 60.00 nm (uniform_gradient, g=1.000)" in text
    assert "center range 60.00-120.00 nm" in text
    assert "90.00-90.00 nm" not in text
    assert "120.00 nm; support 60.00-180.00 nm" in table
    assert "60.00 nm; support 0.00-120.00 nm" in table


def test_quick_lst_diagnostics_one_term_wght_before_exti(tmp_path: Path):
    lst = tmp_path / "trial.lst"
    lst.write_text("WGHT    0.100000\nEXTI 1320.486328\n")

    payload = cli._quick_lst_diagnostics(lst)

    assert payload["active_wght_a"] == pytest.approx(0.1)
    assert payload["active_wght_b"] == pytest.approx(0.0)
    assert payload["active_wght_terms"] == [pytest.approx(0.1)]
