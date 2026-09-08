"""CLI endpoint selection regression tests."""

import argparse
import json
from pathlib import Path

import pytest

import grond.cli as cli
import grond.workflows.endpoint_candidates as owner_workflows_endpoint_candidates
from grond.io.config import DatasetSpec, PrepConfig


def test_publication_endpoint_chooser_includes_calibrated_replay(tmp_path: Path):
    run = tmp_path / "grond"
    native_run = run / "runs" / "dataset" / "final"
    endpoint = run / "endpoint_calibration" / "dataset" / "adaptive_80_det_8" / "endpoint_freeze-zero"
    native_run.mkdir(parents=True)
    endpoint.mkdir(parents=True)
    endpoint_res = endpoint / "dyn_scale_replay.res"
    endpoint_hkl = endpoint / "dyn_scale_replay.hkl"
    endpoint_lst = endpoint / "dyn_scale_replay.lst"
    endpoint_res.write_text("TITL calibrated\nEND\n")
    endpoint_hkl.write_text("   1   0   0   10.00    1.00\n   0   1   0    5.00    1.00\n   0   0   0    0.00    0.00\n")
    endpoint_lst.write_text(
        "wR2 = 0.1115, GooF = S = 0.906 for all data\n"
        "R1 = 0.05 for 1 Fo > 4sig(Fo) and 0.1105 for all 2 data\n"
        "0 atoms may be split and 0 atoms NPD\n"
    )
    endpoint_manifest = endpoint / "manifest.json"
    endpoint_manifest.write_text(
        json.dumps(
            {
                "mod_p95": 1.2,
                "modulation_metric": "scale_invariant_symmetric_fold_change_p95",
                "endpoint_b": 0.0,
                "keep_exti": False,
                "artifacts": {
                    "res_sha256": cli.sha256_file(endpoint_res),
                    "hkl_sha256": cli.sha256_file(endpoint_hkl),
                    "lst_sha256": cli.sha256_file(endpoint_lst),
                },
            }
        )
        + "\n"
    )
    result = {
        "primary_run": str(native_run),
        "endpoint_calibration": {
            "lambda_q": 1.0,
            "publication_selection": {
                "status": "passed",
                "publication_endpoint": {
                    "spec": "adaptive_80_det_8",
                    "wght_mode": "freeze-zero",
                    "scale_params": 144,
                    "shelxl_R1gt": 0.05,
                    "shelxl_R1all": 0.1105,
                    "shelxl_wR2all": 0.1115,
                    "shelxl_GooF": 0.906,
                    "shelxl_Ngt": 1,
                    "shelxl_Nall": 2,
                    "shelxl_npd_count": 0,
                    "shelxl_split_count": 0,
                    "hkl_safe_factor": 1.0,
                    "mod_p95": 1.2,
                    "endpoint_b": 0.0,
                    "keep_exti": False,
                    "endpoint_dir": str(endpoint),
                    "endpoint_res": str(endpoint_res),
                    "endpoint_hkl": str(endpoint_hkl),
                    "endpoint_lst": str(endpoint_lst),
                    "endpoint_manifest": str(endpoint_manifest),
                }
            },
        },
    }
    args = argparse.Namespace(
        out_dir=str(run),
        dry_run=False,
        publication_endpoint_chooser=True,
        publication_endpoint_min_hkl_safe_factor=0.25,
        publication_endpoint_max_modulation_p95=20.0,
        endpoint_calibration_goof_min=0.85,
        endpoint_calibration_goof_max=1.15,
        publication_allow_npd=False,
    )
    prep = PrepConfig(structure=str(tmp_path / "model.res"), datasets=[DatasetSpec(name="dataset", hkl="input.hkl")])

    choice = cli.run_publication_endpoint_chooser(prep, result, args)

    assert choice["status"] == "ok"
    assert choice["best_physical_endpoint"]["source_kind"] == "endpoint_calibration"
    assert choice["best_physical_endpoint"]["R1_all"] == pytest.approx(0.1105)
    assert choice["best_physical_endpoint"]["res"] == str(endpoint_res)
    assert choice["best_physical_endpoint"]["modulation_p95"] == pytest.approx(1.2)
    assert choice["best_physical_endpoint"]["res_sha256"] == cli.sha256_file(endpoint_res)


def test_publication_endpoint_chooser_treats_combined_calibration_as_combined(tmp_path: Path):
    run = tmp_path / "grond"
    combined_run = run / "runs" / "combined"
    endpoint = run / "endpoint_calibration" / "combined" / "adaptive_80_det_8" / "endpoint_freeze-zero"
    combined_run.mkdir(parents=True)
    endpoint.mkdir(parents=True)
    result = {
        "combined": str(combined_run),
        "publication_runs": {},
        "endpoint_calibration": {
            "label": "combined",
            "lambda_q": 1.0,
            "publication_selection": {
                "status": "passed",
                "publication_endpoint": {
                    "spec": "adaptive_80_det_8",
                    "shelxl_R1gt": 0.06,
                    "shelxl_R1all": 0.11,
                    "shelxl_wR2all": 0.12,
                    "shelxl_GooF": 0.95,
                    "hkl_safe_factor": 1.0,
                    "endpoint_dir": str(endpoint),
                },
            },
        },
    }
    args = argparse.Namespace(
        out_dir=str(run),
        dry_run=False,
        publication_endpoint_chooser=True,
        publication_endpoint_min_hkl_safe_factor=0.25,
        publication_endpoint_max_modulation_p95=20.0,
        endpoint_calibration_goof_min=0.85,
        endpoint_calibration_goof_max=1.15,
        publication_allow_npd=False,
    )
    prep = PrepConfig(
        structure=str(tmp_path / "model.res"),
        datasets=[DatasetSpec(name="a", hkl="a.hkl"), DatasetSpec(name="b", hkl="b.hkl")],
    )

    choice = cli.run_publication_endpoint_chooser(prep, result, args)

    assert choice["status"] == "diagnostic_only"
    assert choice["best_physical_endpoint"] == {}
    diagnostic = choice["best_diagnostic_endpoint"]
    assert diagnostic["source_kind"] == "combined_endpoint_calibration"
    assert diagnostic["combined_baseline_gate"]["status"] == "warning_baseline_unavailable"
    assert diagnostic["combined_baseline_gate"]["passed"] is True
    assert choice["best_single_dataset_physical_endpoint"] == {}


def test_publication_endpoint_chooser_rejects_unpassed_or_unsafe_calibration(tmp_path: Path):
    endpoint = {
        "spec": "adaptive_80_det_8",
        "shelxl_R1gt": 0.06,
        "shelxl_R1all": 0.11,
        "shelxl_wR2all": 0.12,
        "shelxl_GooF": 0.95,
    }
    failed = {
        "endpoint_calibration": {
            "publication_selection": {"status": "no_candidate_passed_all_gates", "publication_endpoint": endpoint}
        }
    }
    passed_without_safety = {
        "endpoint_calibration": {"publication_selection": {"status": "passed", "publication_endpoint": endpoint}}
    }

    assert cli._endpoint_calibration_chooser_candidate(failed) == {}
    assert cli._endpoint_calibration_chooser_candidate(passed_without_safety)["hkl_safe_factor"] == 0.0


def test_publication_model_candidate_preserves_hash_bound_hkl_safety(tmp_path: Path):
    import hashlib

    combined = tmp_path / "cycle" / "combined"
    endpoint_dir = combined / "combined_endpoint_lam_1p0"
    endpoint_dir.mkdir(parents=True)
    res = endpoint_dir / "multi_dyn_corrected.res"
    lst = endpoint_dir / "multi_dyn_corrected.lst"
    hkl = endpoint_dir / "multi_dyn_corrected.hkl"
    res.write_text("TITL endpoint\nCELL 0.025079 5 6 7 90 90 90\nSFAC C\nC1 1 0 0 0 1 0.05\nEND\n")
    lst.write_text(
        "WGHT 0 0\n"
        "wR2 = 0.1000, GooF = S = 1.000 for all data\n"
        "R1 = 0.0500 for 1 Fo > 4sig(Fo) and 0.0800 for all 2 data\n"
        "0 atoms may be split and 0 atoms NPD\n"
    )
    hkl.write_text("0 0 0 0 0\n")
    digest = hashlib.sha256(res.read_bytes()).hexdigest()
    hkl_digest = hashlib.sha256(hkl.read_bytes()).hexdigest()
    lst_digest = hashlib.sha256(lst.read_bytes()).hexdigest()
    (combined / "combined_summary.json").write_text(
        json.dumps(
            {
                "combined_endpoint": {
                    "hkl_safe_factor": 0.10,
                    "endpoint_b": 0.0,
                    "keep_exti": False,
                    "artifacts": {
                        "res": str(res),
                        "res_sha256": digest,
                        "hkl": str(hkl),
                        "hkl_sha256": hkl_digest,
                        "lst": str(lst),
                        "lst_sha256": lst_digest,
                    },
                    "shelxl": {
                        "R1_gt": 0.05,
                        "R1_all": 0.08,
                        "wR2_all": 0.10,
                        "GooF": 1.0,
                        "lst": str(lst),
                        "res_health": {"path": str(res), "sha256": digest, "lst": str(lst)},
                    },
                }
            }
        )
    )

    candidate = cli._publication_model_endpoint_candidate(
        {"combined": str(combined), "model": str(res)}
    )

    assert candidate["res_provenance_ok"] is True
    assert candidate["hkl_safe_factor"] == pytest.approx(0.10)
    assert candidate["wght_mode"] is None
    assert candidate["pathology_status"] == "complete"
    assert candidate["shelxl"]["pathology_status"] == "complete"
    assert candidate["shelxl"]["active_wght_a"] == 0.0
    assert candidate["shelxl"]["active_wght_b"] == 0.0
    normalized = cli._normalize_publication_endpoint_block(
        candidate,
        selection_source="endpoint_chooser",
    )
    assert normalized["wght_mode"] == "freeze-zero"
    assert normalized["pathology_status"] == "complete"
    assert normalized["shelxl_npd_count"] == 0
    assert normalized["shelxl_split_count"] == 0

    res.write_text(res.read_text() + "REM mutated\n")
    stale = cli._publication_model_endpoint_candidate(
        {"combined": str(combined), "model": str(res)}
    )
    assert stale["res_provenance_ok"] is False
    assert stale["hkl_safe_factor"] == 0.0


def test_multi_feedback_endpoint_candidates_include_adopted_cycles(tmp_path: Path, monkeypatch):
    run = tmp_path / "grond"
    summary = run / "runs" / "shared_shelxl_feedback" / "multi_shelxl_feedback_loop_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "cycles": [
                    {
                        "cycle": 1,
                        "output_model": str(tmp_path / "cycle_01.res"),
                        "combined": str(tmp_path / "cycle_01_combined"),
                        "adopted_for_feedback": True,
                        "endpoint_modulation_p95": 1.25,
                        "endpoint_modulation_metric": "fold_change_p95",
                    },
                    {
                        "cycle": 2,
                        "output_model": str(tmp_path / "cycle_02.res"),
                        "combined": str(tmp_path / "cycle_02_combined"),
                        "adopted_for_feedback": False,
                    },
                ]
            }
        )
    )

    monkeypatch.setattr(owner_workflows_endpoint_candidates, "_publication_model_endpoint_candidate", lambda result: {
            "id": "final_ls1_corrected_hkl",
            "model": result["model"],
            "combined": result["combined"],
        })

    candidates = cli._multi_feedback_endpoint_candidates(
        {"combined": str(tmp_path / "final_combined")},
        run,
    )

    assert len(candidates) == 1
    assert candidates[0]["id"] == "shared_ls1_cycle_01"
    assert candidates[0]["source_kind"] == "combined_feedback_cycle"
    assert candidates[0]["modulation_p95"] == pytest.approx(1.25)
    assert candidates[0]["model"] == str(tmp_path / "cycle_01.res")


def test_multi_feedback_endpoint_candidates_ignore_single_dataset_runs(tmp_path: Path):
    assert cli._multi_feedback_endpoint_candidates({"primary_run": "run"}, tmp_path) == []


def test_publication_endpoint_recovers_hkl_from_res(tmp_path: Path):
    endpoint_res = tmp_path / "multi_dyn_corrected.res"
    endpoint_hkl = tmp_path / "multi_dyn_corrected.hkl"
    endpoint_res.write_text("TITL test\nEND\n")
    endpoint_hkl.write_text("   0   0   0    0.00    0.00\n")

    endpoint = cli._normalize_publication_endpoint_block(
        {"res": str(endpoint_res), "shelxl": {"R1_all": 0.1}},
        selection_source="endpoint_chooser",
    )

    assert endpoint["endpoint_hkl"] == str(endpoint_hkl)
