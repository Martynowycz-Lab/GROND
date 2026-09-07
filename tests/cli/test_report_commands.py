"""CLI report commands regression tests."""


import json
from pathlib import Path

import grond.cli as cli
from grond.cli import (
    publication_report_cmd,
)
from grond.io.config import DatasetSpec, PrepConfig


def test_publication_report_includes_scale_polish_and_resolution_summaries(tmp_path: Path):
    prep = PrepConfig(structure=str(tmp_path / "model.res"), datasets=[DatasetSpec(name="111", hkl="INTEGRATE.HKL")])
    result = {
        "primary_run": str(tmp_path / "runs" / "111" / "final"),
        "model": prep.structure,
        "scale_polish": {"status": "accepted", "out_dir": str(tmp_path / "scale_polish" / "111")},
        "resolution_limits": {"111": {"artifacts": {"summary": str(tmp_path / "resolution_limits" / "111" / "summary.json")}}},
    }

    cmd = publication_report_cmd(
        "python",
        prep,
        result,
        tmp_path / "tables",
        "table",
        instability_u=0.02,
    )

    scale_index = cmd.index("--scale-polish")
    assert cmd[scale_index + 1].endswith("/scale_polish/111/scale_compression_summary.json")
    assert cmd[scale_index + 1].startswith("111=")
    resolution_index = cmd.index("--resolution-limit")
    assert cmd[resolution_index + 1].endswith("/resolution_limits/111/summary.json")
    assert cmd[resolution_index + 1].startswith("111=")
    assert cmd[cmd.index("--instability-u") + 1] == "0.02"


def test_multi_publication_report_separates_selected_and_full_scale_rows(tmp_path: Path):
    prep = PrepConfig(
        structure=str(tmp_path / "model.res"),
        datasets=[DatasetSpec(name="a", hkl="a.hkl"), DatasetSpec(name="b", hkl="b.hkl")],
    )
    combined = tmp_path / "runs" / "combined"
    combined.mkdir(parents=True)
    (combined / "combined_summary.json").write_text(
        json.dumps({"combined_endpoint": {"requested_min_d_spacing_A": 0.75}}) + "\n"
    )
    result = {
        "combined": str(combined),
        "model": prep.structure,
        "publication_runs": {"a": "/selected/a", "b": "/selected/b"},
        "publication_full_runs": {"a": "/full/a", "b": "/full/b"},
        "scale_contract_selection": {
            "selected_contract": "compact_min_strict",
            "publication_scale_polish_summaries": {},
        },
    }

    cmd = publication_report_cmd("python", prep, result, tmp_path / "tables", "table")

    selected_values = [cmd[index + 1] for index, value in enumerate(cmd) if value == "--run"]
    full_values = [cmd[index + 1] for index, value in enumerate(cmd) if value == "--full-run"]
    assert selected_values == ["a=/selected/a", "b=/selected/b"]
    assert full_values == ["a=/full/a", "b=/full/b"]
    assert cmd[cmd.index("--selected-scale-contract") + 1] == "compact_min_strict"
    assert cmd[cmd.index("--requested-min-d-spacing") + 1] == "0.75"


def test_combined_summary_exposes_joint_native_metrics(tmp_path: Path):
    run_dir = tmp_path / "combined"
    run_dir.mkdir()
    (run_dir / "combined_summary.json").write_text(
        json.dumps(
            {
                "wght": {"a": 0.12, "b": 0.0},
                "native_joint": {
                    "dyn": {
                        "all": {"R1_all": 0.13, "wR2_all": 0.24},
                        "work": {"R1_all": 0.131, "wR2_all": 0.241},
                        "free": {"R1_all": 0.132, "wR2_all": 0.242},
                    },
                    "born": {
                        "all": {"R1_all": 0.17, "wR2_all": 0.30},
                        "work": {"R1_all": 0.171, "wR2_all": 0.301},
                        "free": {"R1_all": 0.172, "wR2_all": 0.302},
                    },
                },
            }
        )
        + "\n"
    )

    summary = cli._quick_load_run_summary(run_dir)
    assert cli._quick_dyn_metric(summary, "all", "R1_all") == 0.13
    assert cli._quick_born_metric(summary, "free", "wR2_all") == 0.302
    native = cli._quick_best_native(summary)
    assert native["native_weight"]["a"] == 0.12
