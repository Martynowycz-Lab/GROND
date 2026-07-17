import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "grond" / "legacy" / "engine"))

from grond.legacy.engine.resolution_limit_audit import matched_metrics_or_empty
from grond.legacy.engine.row_mr_report import (
    canonical_hkl,
    compute_per_dataset_row_mr,
    compute_row_mr_summary,
    jana_table_rows,
    row_mr_r_factors,
    scale_param_count,
    table_rows,
)


def test_compact_scale_count_reports_materialized_selected_output():
    payloads = [
        {"selected": {"scale_params": 80}, "selected_output": {"scale_params": 96}},
        {"selected": {"scale_params": 85}, "selected_output": {"scale_params": 91}},
    ]

    assert scale_param_count(payloads, compact=True) == 187


def test_row_mr_mr_merges_model_symmetry_equivalents():
    model = {
        "symmetry_ops": [
            {"rot": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            {"rot": [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]},
        ]
    }
    rows = [
        {"h": "1", "k": "0", "l": "0", "target_I": "100", "sigma_eff": "2", "dyn_scaled": "100", "born_scaled": "90"},
        {"h": "-1", "k": "0", "l": "0", "target_I": "121", "sigma_eff": "2", "dyn_scaled": "120", "born_scaled": "95"},
    ]

    summary = compute_row_mr_summary(rows, model)

    assert canonical_hkl(1, 0, 0, model["symmetry_ops"]) == (-1, 0, 0)
    assert summary["dynamic"]["Nall"] == 2
    assert summary["dynamic"]["MNall"] == 1
    assert summary["dynamic"]["MRall"] < summary["kinematic"]["MRall"]
    assert summary["dynamic"]["Robs"] < summary["kinematic"]["Robs"]
    assert summary["born"] is summary["kinematic"]


def test_jana_pets_row_factors_follow_documented_formulas_and_observed_cutoff():
    metrics = row_mr_r_factors(
        obs=[100.0, 4.0],
        sigma=[2.0, 2.0],
        calc=[81.0, 9.0],
        instability_u=0.01,
    )

    weight_100 = 1.0 / math.sqrt((2.0 / (2.0 * 10.0)) ** 2 + (0.01 * 10.0) ** 2)
    weight_4 = 1.0 / math.sqrt((2.0 / (2.0 * 2.0)) ** 2 + (0.01 * 2.0) ** 2)
    expected_wr_all = math.sqrt(
        ((weight_100 * 19.0) ** 2 + (weight_4 * -5.0) ** 2) / ((weight_100 * 100.0) ** 2 + (weight_4 * 4.0) ** 2)
    )

    assert metrics["Nall"] == 2
    assert metrics["Nobs"] == 1
    assert metrics["Robs"] == pytest.approx(0.1)
    assert metrics["Rall"] == pytest.approx(1.0 / 6.0)
    assert metrics["wRobs"] == pytest.approx(0.19)
    assert metrics["wRall"] == pytest.approx(expected_wr_all)


def test_jana_pets_merge_keeps_friedel_opposites_separate_unless_model_symmetry_relates_them():
    rows = [
        {"h": "1", "k": "0", "l": "0", "target_I": "100", "sigma_eff": "2", "dyn_scaled": "90", "born_scaled": "80"},
        {"h": "-1", "k": "0", "l": "0", "target_I": "81", "sigma_eff": "2", "dyn_scaled": "85", "born_scaled": "75"},
    ]
    identity = {"symmetry_ops": [{"rot": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]}
    centric = {
        "symmetry_ops": [
            {"rot": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            {"rot": [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]},
        ]
    }

    assert compute_row_mr_summary(rows, identity)["dynamic"]["MNall"] == 2
    assert compute_row_mr_summary(rows, centric)["dynamic"]["MNall"] == 1


def test_jana_pets_summary_reports_invalid_rows_and_per_dataset_values():
    model = {"symmetry_ops": [{"rot": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]}
    rows = [
        {
            "dataset": "A",
            "h": "1",
            "k": "0",
            "l": "0",
            "target_I": "100",
            "sigma_eff": "2",
            "dyn_scaled": "100",
            "born_scaled": "90",
        },
        {
            "dataset": "A",
            "h": "2",
            "k": "0",
            "l": "0",
            "target_I": "nan",
            "sigma_eff": "2",
            "dyn_scaled": "10",
            "born_scaled": "10",
        },
        {
            "dataset": "B",
            "h": "3",
            "k": "0",
            "l": "0",
            "target_I": "64",
            "sigma_eff": "2",
            "dyn_scaled": "60",
            "born_scaled": "50",
        },
    ]

    summary = compute_row_mr_summary(rows, model)
    per_dataset = compute_per_dataset_row_mr(rows, model)

    assert summary["dynamic"]["Ninput"] == 3
    assert summary["dynamic"]["Nall"] == 2
    assert summary["dynamic"]["Nexcluded"] == 1
    assert sorted(per_dataset) == ["A", "B"]
    assert per_dataset["A"]["dynamic"]["Ninput"] == 2
    assert per_dataset["A"]["dynamic"]["Nall"] == 1
    assert per_dataset["B"]["dynamic"]["Nall"] == 1

    invalid_hkl = compute_row_mr_summary(
        [
            {
                "h": "nan",
                "k": "0",
                "l": "0",
                "target_I": "100",
                "sigma_eff": "2",
                "dyn_scaled": "100",
                "born_scaled": "90",
            }
        ],
        model,
    )
    assert invalid_hkl["dynamic"]["Nall"] == 1
    assert invalid_hkl["dynamic"]["MNall"] == 0
    assert invalid_hkl["dynamic"]["Nmerge_input"] == 1
    assert invalid_hkl["dynamic"]["Nmerge_used"] == 0
    assert invalid_hkl["dynamic"]["Nmerge_excluded"] == 1


def test_jana_pets_born_and_bloch_use_one_common_finite_support():
    model = {"symmetry_ops": [{"rot": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]}
    rows = [
        {
            "h": "1",
            "k": "0",
            "l": "0",
            "target_I": "100",
            "sigma_eff": "2",
            "dyn_scaled": "nan",
            "born_scaled": "90",
        },
        {
            "h": "2",
            "k": "0",
            "l": "0",
            "target_I": "64",
            "sigma_eff": "2",
            "dyn_scaled": "60",
            "born_scaled": "50",
        },
    ]

    summary = compute_row_mr_summary(rows, model)

    assert summary["support"] == {
        "Ninput": 2,
        "Nmatched": 1,
        "Nexcluded": 1,
        "status": "matched",
    }
    for model_name in ("born", "dynamic"):
        assert summary[model_name]["Ninput"] == 2
        assert summary[model_name]["Nmatched"] == 1
        assert summary[model_name]["Nexcluded"] == 1
        assert summary[model_name]["Nall"] == 1
        assert summary[model_name]["Nmerge_used"] == 1
    assert summary["born"]["Nobs"] == summary["dynamic"]["Nobs"]
    assert summary["born"]["MNall"] == summary["dynamic"]["MNall"]

    resolution_dynamic, resolution_born, resolution_support = matched_metrics_or_empty(rows, model)
    assert resolution_support["Nmatched"] == 1
    assert resolution_dynamic["Nall"] == resolution_born["Nall"] == 1


def test_legacy_and_canonical_table_shapes_are_both_available():
    model = {"symmetry_ops": [{"rot": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]}
    rows = [
        {"h": "1", "k": "0", "l": "0", "target_I": "100", "sigma_eff": "2", "dyn_scaled": "100", "born_scaled": "90"},
    ]
    summary = compute_row_mr_summary(rows, model)

    assert list(table_rows(summary)[0]) == [
        "Regularization",
        "Nobs, Nall",
        "MNobs, MNall",
        "Robs",
        "Rall",
        "wRall",
        "MRobs",
        "MRall",
        "MwRall",
    ]
    assert list(jana_table_rows(summary)[0]) == [
        "Row model",
        "Nobs, Nall",
        "Nmerge used, excluded",
        "MNobs, MNall",
        "Robs",
        "Rall",
        "wRobs",
        "wRall",
        "MRobs",
        "MRall",
        "MwRobs",
        "MwRall",
    ]


def test_row_mr_report_cli_writes_markdown_tex_and_json(tmp_path: Path):
    res = tmp_path / "model.res"
    res.write_text(
        "\n".join(
            [
                "TITL test",
                "CELL 0.02508 10 11 12 90 90 90",
                "LATT 1",
                "SFAC C 1 0 0 0 0 0 0 0 0",
                "UNIT 1",
                "C1 1 0.1 0.2 0.3 11.0 0.05",
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / "native_dyn_calc.csv").write_text(
        "\n".join(
            [
                "h,k,l,d_A,target_I,sigma_eff,dyn_scaled,born_scaled",
                "1,0,0,2.0,100,2,100,90",
                "-1,0,0,2.0,121,2,120,95",
                "2,0,0,1.0,25,2,30,20",
            ]
        )
        + "\n"
    )
    (run / "summary.json").write_text(
        json.dumps(
            {
                "config": {"n_basis": 20, "n_points": 5},
                "best_native": {"nuisance": {"thickness_A": 1000, "row_resolution_b": 0}},
            }
        )
    )
    weak_rows = tmp_path / "weak_rows.csv"
    weak_rows.write_text("h,k,l,d_A,target_I,sigma_eff,dyn_scaled,born_scaled\n3,0,0,2.0,1,10,1,1\n")
    scale_summary = tmp_path / "scale_compression_summary.json"
    scale_summary.write_text(
        json.dumps(
            {
                "baseline": {
                    "scale_params": 10,
                    "total_params": 14,
                    "Rall": 0.200,
                    "wRall": 0.300,
                    "native": {"R1_all": 0.220, "wR2_all": 0.330},
                },
                "selected": {
                    "name": "adaptive_4_det_0",
                    "scale_params": 4,
                    "total_params": 8,
                    "Nall": 3,
                    "Nobs": 3,
                    "delta_vs_baseline": {
                        "Rall": 0.001,
                        "wRall": 0.002,
                        "native_R1_all": 0.003,
                        "native_wR2_all": 0.004,
                    },
                },
            }
        )
        + "\n"
    )
    compact_run = tmp_path / "selected_compact_run"
    compact_run.mkdir()
    (compact_run / "native_dyn_calc.csv").write_text(
        "\n".join(
            [
                "h,k,l,d_A,target_I,sigma_eff,dyn_scaled,born_scaled",
                "1,0,0,2.0,100,2,101,90",
                "-1,0,0,2.0,121,2,122,95",
                "2,0,0,1.0,25,2,32,20",
            ]
        )
        + "\n"
    )
    scale_payload = json.loads(scale_summary.read_text())
    scale_payload["selected_run_dir"] = str(compact_run)
    scale_summary.write_text(json.dumps(scale_payload) + "\n")
    resolution_summary = tmp_path / "resolution_summary.json"
    resolution_summary.write_text(json.dumps({"decision": {"promoted_min_d_spacing": 2.0}}) + "\n")
    out = tmp_path / "tables"

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "src" / "grond" / "legacy" / "engine" / "row_mr_report.py"),
            "--run",
            f"test={run}",
            "--row-table",
            f"weak={weak_rows}",
            "--res",
            str(res),
            "--out-dir",
            str(out),
            "--scale-polish",
            f"test={scale_summary}",
            "--resolution-limit",
            f"test={resolution_summary}",
            "--instability-u",
            "0.02",
            "--no-pdf",
        ],
        check=True,
    )

    assert (out / "row_mr_metrics.json").exists()
    assert (out / "publication_table.md").exists()
    assert (out / "publication_table.tex").exists()
    assert (out / "JANA_ROW_STATISTICS.json").exists()
    assert (out / "JANA_ROW_STATISTICS.md").exists()
    assert (out / "JANA_ROW_STATISTICS.tex").exists()
    assert (out / "JANA_ROW_STATISTICS.csv").exists()
    assert (out / "JANA_ROW_STATISTICS_PER_DATASET.json").exists()
    assert (out / "JANA_ROW_STATISTICS_PER_DATASET.md").exists()
    assert (out / "JANA_ROW_STATISTICS_PER_DATASET.csv").exists()
    table_md = (out / "publication_table.md").read_text()
    assert "MRobs" in table_md
    assert "## GROND Row/MR R And MR Factors" in table_md
    assert "Resolution And Scale Endpoint Summary" in table_md
    assert "requested_resolution_full_scales" in table_md
    assert "requested_resolution_compact_scales" in table_md
    assert "suggested_resolution_compact_scales" in table_md
    assert "Compact Scale-Manifold Accounting" in table_md
    assert "10 -> 4 (6 saved)" in table_md
    artifacts = json.loads((out / "publication_artifacts.json").read_text())
    endpoints = artifacts["resolution_scale_endpoints"]["table_rows"]
    assert endpoints[0]["Endpoint"] == "requested_resolution_full_scales"
    assert endpoints[1]["Endpoint"] == "requested_resolution_compact_scales"
    assert endpoints[2]["Endpoint"] == "suggested_resolution_compact_scales"
    assert endpoints[0]["dmin A"] == "1.000"
    assert endpoints[2]["dmin A"] == "2.000"
    assert endpoints[2]["Nobs, Nall"] == "2, 2"
    assert artifacts["scale_polish"]["table_rows"][0]["Penalty dwR2all pp"] == "+0.40"
    assert artifacts["born"] == artifacts["kinematic"]
    canonical_md = (out / "JANA_ROW_STATISTICS.md").read_text()
    assert "## JANA/PETS-Style Row And MR Factors" in canonical_md
    assert "## Resolution And Scale Row-Contract Summary" in canonical_md
    assert "not SHELXL endpoint statistics" in canonical_md
    assert "instability factor is `u = 0.02`" in canonical_md
    assert "Nmerge excluded" in canonical_md
    assert "u=0.02" in (out / "JANA_ROW_STATISTICS_PER_DATASET.md").read_text()
    assert "u=0.02" in (out / "JANA_ROW_STATISTICS.tex").read_text()
    canonical_json_text = (out / "JANA_ROW_STATISTICS.json").read_text()
    assert "NaN" not in canonical_json_text
    assert "Infinity" not in canonical_json_text
    row_payload = json.loads(canonical_json_text)
    assert row_payload["schema"] == "grond.jana_row_statistics.v1"
    assert row_payload["numeric_units"] == "fractions"
    assert row_payload["reporting_contract"]["not_endpoint_statistics"] is True
    assert row_payload["row_mr"]["definition"]["instability_u"] == pytest.approx(0.02)
    assert row_payload["row_mr"]["support"]["status"] == "matched"
    assert row_payload["row_mr"]["born"] == row_payload["row_mr"]["kinematic"]
    assert row_payload["row_resolution_scale_contracts"]["table_rows"][0]["Row contract"] == (
        "requested_resolution_full_scales"
    )
    per_dataset_text = (out / "JANA_ROW_STATISTICS_PER_DATASET.json").read_text()
    assert "NaN" not in per_dataset_text
    assert "Infinity" not in per_dataset_text
    per_dataset = json.loads(per_dataset_text)
    assert per_dataset["schema"] == "grond.jana_row_statistics_per_dataset.v1"
    assert list(per_dataset["datasets"]) == ["test", "weak"]
    assert {row["Row model"] for row in per_dataset["table_rows"]} == {
        "Bloch dynamical",
        "matched Born (kinematical control)",
    }
    assert per_dataset["datasets"]["weak"]["dynamic"]["Nobs"] == 0
    assert per_dataset["datasets"]["weak"]["dynamic"]["Robs"] is None
    assert per_dataset["datasets"]["weak"]["dynamic"]["wRobs"] is None
    assert per_dataset["datasets"]["weak"]["dynamic"]["MRobs"] is None
    assert per_dataset["datasets"]["weak"]["dynamic"]["MwRobs"] is None
    combined_csv = (out / "JANA_ROW_STATISTICS.csv").read_text()
    assert combined_csv.startswith('Row model,"Nobs, Nall"')
    assert "Nmerge used, excluded" in combined_csv.splitlines()[0]
    assert "Bloch dynamical" in combined_csv


def test_multi_report_keeps_selected_compact_and_full_control_rows_distinct(tmp_path: Path):
    res = tmp_path / "model.res"
    res.write_text(
        "TITL test\n"
        "CELL 0.02508 10 10 10 90 90 90\n"
        "LATT 1\n"
        "SFAC C 1 0 0 0 0 0 0 0 0\n"
        "UNIT 1\n"
        "C1 1 0.1 0.2 0.3 11.0 0.05\n"
        "HKLF 4\n"
        "END\n"
    )
    selected = tmp_path / "selected_compact"
    full = tmp_path / "full_control"
    selected.mkdir()
    full.mkdir()
    header = "h,k,l,d_A,target_I,sigma_eff,dyn_scaled,born_scaled\n"
    (selected / "native_dyn_calc.csv").write_text(
        header + "1,0,0,99.0,100,2,100,80\n" + "2,0,0,99.0,100,2,100,80\n" + "3,0,0,99.0,100,2,100,80\n"
    )
    (full / "native_dyn_calc.csv").write_text(
        header + "1,0,0,99.0,100,2,50,80\n" + "2,0,0,99.0,100,2,50,80\n" + "3,0,0,99.0,100,2,50,80\n"
    )
    for run in (selected, full):
        (run / "summary.json").write_text(json.dumps({"config": {}, "best_native": {}}) + "\n")
    scale_summary = tmp_path / "projected_scale_summary.json"
    scale_summary.write_text(
        json.dumps(
            {
                "baseline": {"name": "full10", "scale_params": 10, "total_params": 14},
                "selected_output": {
                    "name": "bic4",
                    "scale_params": 4,
                    "total_params": 8,
                    "delta_vs_baseline": {"native_wR2_all": 0.004},
                },
                "selected": {
                    "name": "strict2",
                    "scale_params": 2,
                    "total_params": 6,
                    "delta_vs_baseline": {"native_wR2_all": 0.020},
                },
                "selected_run_dir": str(selected),
            }
        )
        + "\n"
    )
    out = tmp_path / "tables"

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "src" / "grond" / "legacy" / "engine" / "row_mr_report.py"),
            "--run",
            f"test={selected}",
            "--full-run",
            f"test={full}",
            "--res",
            str(res),
            "--out-dir",
            str(out),
            "--scale-polish",
            f"test={scale_summary}",
            "--requested-min-d-spacing",
            "4.0",
            "--selected-scale-contract",
            "compact_bic",
            "--no-pdf",
        ],
        check=True,
    )

    payload = json.loads((out / "row_mr_metrics.json").read_text())
    assert payload["metadata"]["row_count"] == 2
    assert payload["metadata"]["dmin_A"] == 5.0
    contract = payload["metadata"]["reporting_resolution_contract"]
    assert contract["selected_input_rows"] == 3
    assert contract["selected_retained_rows"] == 2
    assert contract["full_control_retained_rows"] == 2
    endpoints = payload["resolution_scale_endpoints"]["endpoints"]
    assert endpoints[0]["label"] == "requested_resolution_full_scales"
    assert endpoints[1]["label"] == "requested_resolution_compact_scales"
    assert endpoints[0]["row_mr"]["dynamic"]["Rall"] > 0.25
    assert endpoints[1]["row_mr"]["dynamic"]["Rall"] == 0.0
    assert payload["row_mr"]["dynamic"]["Rall"] == 0.0
    scale_row = payload["scale_polish"]["table_rows"][0]
    assert scale_row["Model"] == "bic4"
    assert scale_row["Scale params (full -> compact)"] == "10 -> 4 (6 saved)"
    assert scale_row["Penalty dwR2all pp"] == "+0.40"
