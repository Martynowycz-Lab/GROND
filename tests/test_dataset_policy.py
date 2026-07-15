import json
import subprocess
import sys
from pathlib import Path

from grond.workflows.dataset_policy import decide_dataset_policy, write_policy_weighted_runs


def test_dataset_policy_downweights_marginal_and_keeps_unique_endpoint_only():
    stats = [
        {
            "name": "good",
            "rows": 1000,
            "unique_hkl": 3,
            "hkl_keys": {(1, 0, 0), (2, 0, 0), (3, 0, 0)},
            "R1_all": 0.10,
            "wR2_all": 0.12,
            "CC": 0.98,
        },
        {
            "name": "marginal",
            "rows": 1000,
            "unique_hkl": 3,
            "hkl_keys": {(1, 0, 0), (2, 0, 0), (4, 0, 0)},
            "R1_all": 0.15,
            "wR2_all": 0.18,
            "CC": 0.94,
        },
        {
            "name": "weak_unique",
            "rows": 1000,
            "unique_hkl": 2,
            "hkl_keys": {(5, 0, 0), (6, 0, 0)},
            "R1_all": 0.50,
            "wR2_all": 0.80,
            "CC": 0.55,
        },
    ]

    policy = decide_dataset_policy(stats, unique_gain_fraction=0.01)
    by_name = {row["name"]: row for row in policy["datasets"]}

    assert by_name["good"]["action"] == "primary"
    assert by_name["good"]["raw_row_weight"] == 1.0
    assert by_name["marginal"]["action"] == "downweight"
    assert 0 < by_name["marginal"]["raw_row_weight"] < 1.0
    assert by_name["weak_unique"]["action"] == "completeness_only"
    assert by_name["weak_unique"]["include_in_row_regularization"] is False
    assert by_name["weak_unique"]["include_in_endpoint"] is True


def test_weighted_run_copy_preserves_sigma_and_marks_deferred_policy(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "native_dyn_calc.csv").write_text(
        "h,k,l,target_I,sigma_eff,dyn_scaled,born_scaled,dyn_raw,born_raw,q_dyn_over_born\n"
        "1,0,0,100,2,90,80,45,40,1.125\n"
    )
    (run / "summary.json").write_text("{}\n")
    policy = {
        "datasets": [
            {
                "name": "marginal",
                "action": "downweight",
                "raw_row_weight": 0.25,
                "endpoint_weight": 0.25,
            }
        ]
    }

    copied = write_policy_weighted_runs({"marginal": run}, policy, tmp_path / "weighted")
    text = (copied["marginal"] / "native_dyn_calc.csv").read_text()

    assert "dataset_role" in text
    assert "downweight" in text
    assert "sigma_eff_original" in text
    assert "policy_sigma_weight_applied" in text
    assert ",2," in text or ",2.0," in text


def test_combiner_uses_endpoint_only_rows_for_endpoint_not_native_metrics(tmp_path: Path):
    res = tmp_path / "model.res"
    res.write_text(
        "TITL test\n"
        "CELL 0.025 10 10 10 90 90 90\n"
        "LATT 1\n"
        "SFAC C 1 0 1 0 1 0 1 0\n"
        "UNIT 1\n"
        "C1 1 0.1 0.2 0.3 11.0 0.05\n"
        "HKLF 4\n"
        "END\n"
    )
    for name, role in (("good", "primary"), ("weak", "completeness_only")):
        run = tmp_path / name
        run.mkdir()
        run.joinpath("summary.json").write_text("{}\n")
        run.joinpath("native_dyn_calc.csv").write_text(
            "h,k,l,target_I,sigma_eff,dyn_scaled,born_scaled,dyn_raw,born_raw,q_dyn_over_born,dataset_role,raw_row_weight,endpoint_weight\n"
                f"1,0,0,100,2,100,90,50,45,1.111,{role},{1.0 if role == 'primary' else 0.0},{1.0 if role == 'primary' else 0.25}\n"
            )
    fake_shelxl = tmp_path / "fake_shelxl"
    fake_shelxl.write_text("#!/bin/sh\nexit 1\n")
    fake_shelxl.chmod(0o755)
    out = tmp_path / "combined"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "src" / "grond" / "legacy" / "engine" / "combine_native_nuisance_runs.py"),
            "--run",
            f"good={tmp_path / 'good'}",
            "--run",
            f"weak={tmp_path / 'weak'}",
            "--res",
            str(res),
            "--out-dir",
            str(out),
            "--shelxl",
            str(fake_shelxl),
        ],
        check=True,
    )
    summary = json.loads((out / "combined_summary.json").read_text())

    assert summary["coverage"]["rows_total"] == 1
    assert summary["coverage"]["endpoint_rows_total"] == 2
    assert summary["runs"][1]["dataset_role"] == "completeness_only"
    assert summary["runs"][1]["native_rows"] == 0
