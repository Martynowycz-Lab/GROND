import json
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "grond" / "legacy" / "engine"))

from grond.legacy.engine.row_mr_report import canonical_hkl, compute_row_mr_summary


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


def test_row_mr_report_cli_writes_markdown_tex_and_json(tmp_path: Path):
    res = tmp_path / "model.res"
    res.write_text(
        "\n".join(
            [
                "TITL test",
                "CELL 0.02508 10 11 12 90 90 90",
                "LATT 1",
                "SFAC C 0 0 0 0 0 0 0 0 0",
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
    (run / "summary.json").write_text(json.dumps({"config": {"n_basis": 20, "n_points": 5}, "best_native": {"nuisance": {"thickness_A": 1000, "row_resolution_b": 0}}}))
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
            "--res",
            str(res),
            "--out-dir",
            str(out),
            "--scale-polish",
            f"test={scale_summary}",
            "--resolution-limit",
            f"test={resolution_summary}",
            "--no-pdf",
        ],
        check=True,
    )

    assert (out / "row_mr_metrics.json").exists()
    assert (out / "publication_table.md").exists()
    assert (out / "publication_table.tex").exists()
    table_md = (out / "publication_table.md").read_text()
    assert "MRobs" in table_md
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
