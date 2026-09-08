import hashlib
import json
from pathlib import Path

import pytest

from grond.workflows.endpoint_chooser import (
    choose_publication_endpoints,
    endpoint_artifact_provenance,
    write_publication_endpoint_choice,
)


def endpoint(lam: float, endpoint_b: float, wr2: float, r1: float, *, exti: bool = False, safe: float = 1.0) -> dict:
    return {
        "lambda_q": lam,
        "endpoint_b": endpoint_b,
        "keep_exti": exti,
        "hkl_safe_factor": safe,
        "mod_p95": 1.5,
        "shelxl": {
            "R1_gt": r1 - 0.02,
            "R1_all": r1,
            "wR2_all": wr2,
            "GooF": 1.1,
            "N_gt": 50,
            "N_all": 100,
        },
    }


def materialize_endpoint_artifacts(root: Path, name: str, candidate: dict, *, pathology: str = "complete") -> dict:
    endpoint_dir = root / name
    endpoint_dir.mkdir(parents=True)
    res = endpoint_dir / "endpoint.res"
    hkl = endpoint_dir / "endpoint.hkl"
    lst = endpoint_dir / "endpoint.lst"
    res.write_text("TITL endpoint\nEND\n")
    hkl.write_text("   1   0   0   10.00    1.00\n   0   0   0    0.00    0.00\n")
    pathology_line = "0 atoms may be split and 0 atoms NPD\n" if pathology == "complete" else ""
    shelxl = candidate.get("shelxl") or {}
    lst.write_text(
        f"wR2 = {float(shelxl['wR2_all']):.4f}, GooF = S = {float(shelxl['GooF']):.3f} for all data\n"
        f"R1 = {float(shelxl['R1_gt']):.4f} for {int(shelxl['N_gt'])} Fo > 4sig(Fo) and "
        f"{float(shelxl['R1_all']):.4f} for all {int(shelxl['N_all'])} data\n"
        + pathology_line
    )
    artifacts = {
        "res": str(res),
        "res_sha256": hashlib.sha256(res.read_bytes()).hexdigest(),
        "hkl": str(hkl),
        "hkl_sha256": hashlib.sha256(hkl.read_bytes()).hexdigest(),
        "lst": str(lst),
        "lst_sha256": hashlib.sha256(lst.read_bytes()).hexdigest(),
    }
    out = dict(candidate)
    out["artifacts"] = artifacts
    out["shelxl"] = {**dict(candidate.get("shelxl") or {}), "lst": str(lst)}
    return out


def test_endpoint_artifact_provenance_requires_current_res_hkl_lst_and_complete_pathology(tmp_path: Path):
    candidate = materialize_endpoint_artifacts(tmp_path, "current", endpoint(1.0, 0.0, 0.20, 0.12))

    current = endpoint_artifact_provenance(candidate)
    assert current["passed"] is True
    assert current["status"] == "ok"
    assert current["lst_pathology"]["pathology_status"] == "complete"

    missing_hash = dict(candidate)
    missing_hash["artifacts"] = {**candidate["artifacts"], "hkl_sha256": ""}
    missing = endpoint_artifact_provenance(missing_hash)
    assert missing["passed"] is False
    assert missing["status"] == "missing_hkl_sha256"

    Path(candidate["artifacts"]["lst"]).write_text("mutated\n")
    stale = endpoint_artifact_provenance(candidate)
    assert stale["passed"] is False
    assert stale["status"] == "lst_sha256_mismatch"
    assert stale["lst_pathology"]["pathology_status"] == "unverified"


def test_real_collector_fails_closed_when_hash_bound_lst_pathology_is_incomplete(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    candidate = materialize_endpoint_artifacts(
        tmp_path,
        "incomplete",
        endpoint(1.0, 0.0, 0.20, 0.12),
        pathology="incomplete",
    )
    (run / "summary.json").write_text(json.dumps({"best_endpoint": candidate}) + "\n")

    decision = write_publication_endpoint_choice({"dataset": run}, tmp_path / "chooser")

    assert decision["status"] == "diagnostic_only"
    diagnostic = decision["best_diagnostic_endpoint"]
    assert diagnostic["artifact_provenance_ok"] is True
    assert diagnostic["pathology_status"] == "incomplete"
    assert any("pathology audit is missing or incomplete" in reason for reason in diagnostic["rejection_reasons"])


def test_real_collector_uses_hash_verified_lst_metrics_and_rejects_stored_disagreement(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    candidate = materialize_endpoint_artifacts(tmp_path, "metric_mismatch", endpoint(1.0, 0.0, 0.20, 0.12))
    candidate["shelxl"] = {**candidate["shelxl"], "R1_all": 0.88, "wR2_all": 0.77}
    (run / "summary.json").write_text(json.dumps({"best_endpoint": candidate}) + "\n")

    decision = write_publication_endpoint_choice({"dataset": run}, tmp_path / "chooser")

    diagnostic = decision["best_diagnostic_endpoint"]
    assert diagnostic["R1_all"] == pytest.approx(0.12)
    assert diagnostic["wR2_all"] == pytest.approx(0.20)
    assert diagnostic["metric_consistency_ok"] is False
    assert any("stored/live SHELXL metric audit failed" in reason for reason in diagnostic["rejection_reasons"])


def test_real_collector_requires_direct_manifest_consistent_promotion_fields(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    candidate = materialize_endpoint_artifacts(tmp_path, "manifest_only", endpoint(1.0, 0.0, 0.20, 0.12))
    manifest = tmp_path / "manifest_only" / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "hkl_safe_factor": 1.0,
                "mod_p95": 1.5,
                "endpoint_b": 0.0,
                "keep_exti": False,
                "artifacts": candidate["artifacts"],
            }
        )
        + "\n"
    )
    for key in ("hkl_safe_factor", "mod_p95", "endpoint_b", "keep_exti"):
        candidate.pop(key)
    candidate["endpoint_manifest"] = str(manifest)
    (run / "summary.json").write_text(json.dumps({"best_endpoint": candidate}) + "\n")

    decision = write_publication_endpoint_choice({"dataset": run}, tmp_path / "chooser")

    diagnostic = decision["best_diagnostic_endpoint"]
    assert diagnostic["promotion_fields_ok"] is False
    assert any("producer promotion-field contract failed" in reason for reason in diagnostic["rejection_reasons"])


def test_real_collector_rejects_manifest_disagreement_and_canonicalizes_endpoint_aliases(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    candidate = materialize_endpoint_artifacts(tmp_path, "aliases", endpoint(1.0, 0.0, 0.20, 0.12))
    artifacts = candidate.pop("artifacts")
    candidate.update(
        {
            "endpoint_res": artifacts["res"],
            "endpoint_res_sha256": artifacts["res_sha256"],
            "endpoint_hkl": artifacts["hkl"],
            "endpoint_hkl_sha256": artifacts["hkl_sha256"],
            "endpoint_lst": artifacts["lst"],
            "endpoint_lst_sha256": artifacts["lst_sha256"],
        }
    )
    manifest = tmp_path / "aliases" / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "hkl_safe_factor": 1.0,
                "mod_p95": 2.5,
                "endpoint_b": 0.0,
                "keep_exti": False,
                "artifacts": artifacts,
            }
        )
        + "\n"
    )
    candidate["endpoint_manifest"] = str(manifest)
    (run / "summary.json").write_text(json.dumps({"best_endpoint": candidate}) + "\n")

    decision = write_publication_endpoint_choice({"dataset": run}, tmp_path / "chooser")

    physical = decision["best_physical_endpoint"]
    assert physical["res"] == str(Path(artifacts["res"]).resolve())
    assert physical["hkl"] == str(Path(artifacts["hkl"]).resolve())
    assert physical["lst"] == str(Path(artifacts["lst"]).resolve())
    assert physical["promotion_fields_ok"] is True
    assert any(
        "modulation_p95 disagrees with manifest" in warning
        for warning in physical["promotion_field_contract"]["warnings"]
    )


def test_global_publication_blocker_keeps_otherwise_physical_endpoint_diagnostic():
    candidate = {
        "id": "combined",
        "source_kind": "combined_corrected_hkl",
        "lambda_q": 1.0,
        "endpoint_b": 0.0,
        "keep_exti": False,
        "hkl_safe_factor": 1.0,
        "modulation_p95": 1.2,
        "R1_gt": 0.08,
        "R1_all": 0.10,
        "wR2_all": 0.18,
        "GooF": 1.0,
    }

    decision = choose_publication_endpoints(
        [candidate],
        global_publication_blockers=["post-feedback hand diagnostic materially favors the opposite hand"],
    )

    assert decision["status"] == "diagnostic_only"
    assert decision["best_physical_endpoint"] == {}
    assert any("global publication blocker" in reason for reason in decision["best_diagnostic_endpoint"]["rejection_reasons"])


def test_endpoint_chooser_keeps_b_zero_physical_and_reports_b_diagnostic():
    candidates = [
        {
            "id": "physical",
            "label": "dataset",
            "source_kind": "native_corrected_hkl",
            "lambda_q": 1.0,
            "endpoint_b": 0.0,
            "keep_exti": False,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 1.2,
            "R1_gt": 0.11,
            "R1_all": 0.14,
            "wR2_all": 0.24,
            "GooF": 1.02,
        },
        {
            "id": "diagnostic_b",
            "label": "dataset",
            "source_kind": "native_corrected_hkl",
            "lambda_q": 1.0,
            "endpoint_b": 18.0,
            "keep_exti": False,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 1.2,
            "R1_gt": 0.09,
            "R1_all": 0.11,
            "wR2_all": 0.19,
            "GooF": 1.02,
        },
    ]
    decision = choose_publication_endpoints(candidates)
    assert decision["best_physical_endpoint"]["id"] == "physical"
    assert decision["best_diagnostic_endpoint"]["id"] == "diagnostic_b"
    assert any("endpoint-B" in reason for reason in decision["best_diagnostic_endpoint"]["rejection_reasons"])


def test_endpoint_chooser_rejects_nonunit_q_exponent_for_publication():
    candidates = [
        {
            "id": "diagnostic_qexp",
            "label": "dataset",
            "source_kind": "native_corrected_hkl",
            "lambda_q": 0.95,
            "endpoint_b": 0.0,
            "keep_exti": False,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 1.2,
            "R1_gt": 0.08,
            "R1_all": 0.10,
            "wR2_all": 0.18,
            "GooF": 1.02,
        },
        {
            "id": "physical",
            "label": "dataset",
            "source_kind": "native_corrected_hkl",
            "lambda_q": 1.0,
            "endpoint_b": 0.0,
            "keep_exti": False,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 1.2,
            "R1_gt": 0.11,
            "R1_all": 0.13,
            "wR2_all": 0.22,
            "GooF": 1.02,
        },
    ]
    decision = choose_publication_endpoints(candidates)
    assert decision["best_physical_endpoint"]["id"] == "physical"
    rejected_qexp = [c for c in decision["rejected_corrections"] if c["id"] == "diagnostic_qexp"][0]
    assert any("non-unit q exponent" in reason for reason in rejected_qexp["rejection_reasons"])


def test_endpoint_chooser_warns_on_missing_modulation_measurement():
    candidate = {
        "id": "missing_modulation",
        "label": "dataset",
        "source_kind": "native_corrected_hkl",
        "lambda_q": 1.0,
        "endpoint_b": 0.0,
        "keep_exti": False,
        "hkl_safe_factor": 1.0,
        "R1_gt": 0.10,
        "R1_all": 0.12,
        "wR2_all": 0.20,
        "GooF": 1.0,
    }

    decision = choose_publication_endpoints([candidate])

    assert decision["status"] == "ok"
    assert any(
        "modulation is unavailable" in reason
        for reason in decision["best_physical_endpoint"]["warnings"]
    )


def test_endpoint_summary_normalization_does_not_invent_modulation(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    candidate = endpoint(1.0, 0.0, 0.20, 0.12)
    candidate.pop("mod_p95")
    (run / "summary.json").write_text(json.dumps({"best_endpoint": candidate}) + "\n")

    decision = write_publication_endpoint_choice({"dataset": run}, tmp_path / "chooser")

    assert decision["status"] == "diagnostic_only"
    normalized = decision["best_diagnostic_endpoint"]
    assert normalized["modulation_p95"] != normalized["modulation_p95"]
    assert any("modulation is unavailable" in reason for reason in normalized["warnings"])


def test_endpoint_chooser_warns_on_high_goof_but_keeps_model_available():
    candidates = [
        {
            "id": "low_r_bad_goof",
            "label": "dataset:observation_polish",
            "source_kind": "observation_polish",
            "lambda_q": 1.0,
            "endpoint_b": 0.0,
            "keep_exti": False,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 1.2,
            "observation_polish_accepted": True,
            "R1_gt": 0.10,
            "R1_all": 0.12,
            "wR2_all": 0.18,
            "GooF": 1.90,
        },
        {
            "id": "boring_goof",
            "label": "dataset",
            "source_kind": "native_corrected_hkl",
            "lambda_q": 1.0,
            "endpoint_b": 0.0,
            "keep_exti": False,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 1.2,
            "R1_gt": 0.12,
            "R1_all": 0.16,
            "wR2_all": 0.22,
            "GooF": 0.98,
        },
    ]

    decision = choose_publication_endpoints(candidates)

    assert decision["best_physical_endpoint"]["id"] == "low_r_bad_goof"
    selected = decision["best_physical_endpoint"]
    assert selected["rejection_reasons"] == []
    assert any("GooF" in warning and "outside" in warning for warning in selected["warnings"])


def test_endpoint_chooser_requires_no_npd_by_default():
    candidates = [
        {
            "id": "low_r_npd",
            "label": "combined",
            "source_kind": "combined_corrected_hkl",
            "lambda_q": 1.0,
            "endpoint_b": 0.0,
            "keep_exti": False,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 1.2,
            "R1_gt": 0.08,
            "R1_all": 0.09,
            "wR2_all": 0.15,
            "GooF": 1.02,
            "npd_count": 2,
            "npd_atoms": ["C005", "O001"],
        },
        {
            "id": "clean_slightly_worse",
            "label": "combined",
            "source_kind": "combined_corrected_hkl",
            "lambda_q": 1.0,
            "endpoint_b": 0.0,
            "keep_exti": False,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 1.2,
            "R1_gt": 0.10,
            "R1_all": 0.11,
            "wR2_all": 0.18,
            "GooF": 1.02,
            "npd_count": 0,
        },
    ]

    decision = choose_publication_endpoints(candidates)

    assert decision["best_physical_endpoint"]["id"] == "clean_slightly_worse"
    assert decision["best_diagnostic_endpoint"]["id"] == "low_r_npd"
    rejected = [c for c in decision["rejected_corrections"] if c["id"] == "low_r_npd"][0]
    assert any("NPD" in reason and "NPD == 0" in reason for reason in rejected["rejection_reasons"])


def test_endpoint_chooser_reports_split_but_does_not_reject_publication_candidate():
    candidate = {
        "id": "split_only",
        "label": "combined",
        "source_kind": "combined_corrected_hkl",
        "lambda_q": 1.0,
        "endpoint_b": 0.0,
        "keep_exti": False,
        "hkl_safe_factor": 1.0,
        "modulation_p95": 1.2,
        "R1_gt": 0.08,
        "R1_all": 0.09,
        "wR2_all": 0.15,
        "GooF": 1.02,
        "npd_count": 0,
        "split_count": 3,
        "has_split_warnings": True,
    }

    decision = choose_publication_endpoints([candidate])

    assert decision["best_physical_endpoint"]["id"] == "split_only"
    assert decision["best_physical_endpoint"]["split_count"] == 3
    assert not decision["best_physical_endpoint"]["rejection_reasons"]
    assert any("split warnings are reported" in item for item in decision["policy"]["physical_endpoint_requires"])


def test_endpoint_chooser_can_explicitly_allow_npd_diagnostic_override():
    candidate = {
        "id": "npd_override",
        "label": "combined",
        "source_kind": "combined_corrected_hkl",
        "lambda_q": 1.0,
        "endpoint_b": 0.0,
        "keep_exti": False,
        "hkl_safe_factor": 1.0,
        "modulation_p95": 1.2,
        "R1_gt": 0.08,
        "R1_all": 0.09,
        "wR2_all": 0.15,
        "GooF": 1.02,
        "npd_count": 1,
    }

    decision = choose_publication_endpoints([candidate], allow_npd=True)

    assert decision["best_physical_endpoint"]["id"] == "npd_override"
    assert decision["policy"]["allow_npd_override"] is True


def test_write_publication_endpoint_choice_collects_outputs(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    candidates = [
        materialize_endpoint_artifacts(tmp_path, "b_diagnostic", endpoint(1.0, 16.0, 0.18, 0.10)),
        materialize_endpoint_artifacts(tmp_path, "physical", endpoint(1.0, 0.0, 0.22, 0.13)),
        materialize_endpoint_artifacts(tmp_path, "exti_diagnostic", endpoint(1.00, 0.0, 0.23, 0.14, exti=True)),
    ]
    (run / "summary.json").write_text(
        json.dumps(
            {
                "endpoint_results_sorted": candidates
            }
        )
    )
    out = tmp_path / "chooser"
    decision = write_publication_endpoint_choice({"dataset": run}, out)
    assert decision["status"] == "ok"
    assert decision["best_physical_endpoint"]["endpoint_b"] == 0.0
    assert decision["best_physical_endpoint"]["keep_exti"] is False
    assert decision["best_diagnostic_endpoint"]["endpoint_b"] == 16.0
    assert (out / "publication_endpoint_choice.json").exists()
    assert (out / "PUBLICATION_ENDPOINT_CHOOSER.md").exists()
    report = (out / "PUBLICATION_ENDPOINT_CHOOSER.md").read_text()
    assert "Rejected Corrections" in report
    assert "| label | source | endpoint B | EXTI | class | R1(F)gt | R1(F)all | wR2(F^2)all | GooF | modulation p95 | pathology |" in report
    assert "| label | source | lambda |" not in report


def test_endpoint_chooser_prefers_combined_endpoint_when_multi_dataset():
    candidates = [
        {
            "id": "single_best",
            "label": "d07",
            "source_kind": "native_corrected_hkl",
            "lambda_q": 1.0,
            "endpoint_b": 0.0,
            "keep_exti": False,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 1.2,
            "R1_gt": 0.09,
            "R1_all": 0.10,
            "wR2_all": 0.20,
            "GooF": 1.02,
        },
        {
            "id": "combined",
            "label": "combined",
            "source_kind": "combined_corrected_hkl",
            "lambda_q": 1.0,
            "endpoint_b": 0.0,
            "keep_exti": False,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 1.2,
            "R1_gt": 0.12,
            "R1_all": 0.13,
            "wR2_all": 0.24,
            "GooF": 1.02,
        },
    ]
    decision = choose_publication_endpoints(candidates, prefer_combined=True)
    assert decision["best_physical_endpoint"]["id"] == "combined"
    assert decision["best_single_dataset_physical_endpoint"]["id"] == "single_best"
    rejected_single = [c for c in decision["rejected_corrections"] if c["id"] == "single_best"][0]
    assert "single-dataset endpoint" in rejected_single["rejection_reasons"][0]


def test_missing_required_combined_baseline_warns_without_withholding_endpoint():
    candidates = [
        {
            "id": "single_best",
            "label": "d07",
            "source_kind": "native_corrected_hkl",
            "lambda_q": 1.0,
            "endpoint_b": 0.0,
            "keep_exti": False,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 1.2,
            "R1_gt": 0.07,
            "R1_all": 0.10,
            "wR2_all": 0.18,
            "GooF": 1.02,
        },
        {
            "id": "combined",
            "label": "combined",
            "source_kind": "combined_corrected_hkl",
            "lambda_q": 1.0,
            "endpoint_b": 0.0,
            "keep_exti": False,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 1.2,
            "R1_gt": 0.08,
            "R1_all": 0.11,
            "wR2_all": 0.08,
            "GooF": 1.01,
        },
    ]

    decision = choose_publication_endpoints(
        candidates,
        prefer_combined=True,
        require_combined_baseline=True,
    )

    assert decision["status"] == "ok"
    assert decision["best_physical_endpoint"]["id"] == "combined"
    assert decision["best_single_dataset_physical_endpoint"]["id"] == "single_best"
    assert decision["best_single_dataset_diagnostic_endpoint"]["id"] == "single_best"
    assert decision["best_diagnostic_endpoint"]["id"] == "single_best"
    single = next(candidate for candidate in decision["candidates"] if candidate["id"] == "single_best")
    assert single["publication_class"] == "single_dataset_diagnostic"
    assert single["promotable"] is False
    combined = next(candidate for candidate in decision["candidates"] if candidate["id"] == "combined")
    assert combined["combined_baseline_gate"] == {
        "status": "warning_baseline_unavailable",
        "passed": True,
        "required": True,
        "available": False,
        "baseline_R1_all": None,
        "baseline_source": "",
        "endpoint_R1_all": 0.11,
        "delta_R1_all": None,
        "reasons": [],
        "warnings": combined["combined_baseline_gate"]["warnings"],
    }
    assert "remains available" in combined["combined_baseline_gate"]["warnings"][0]


def test_combined_endpoint_worse_than_validated_baseline_warns_but_remains_available():
    candidate = {
        "id": "combined_worse",
        "label": "combined",
        "source_kind": "combined_corrected_hkl",
        "lambda_q": 1.0,
        "endpoint_b": 0.0,
        "keep_exti": False,
        "hkl_safe_factor": 1.0,
        "modulation_p95": 1.2,
        "R1_gt": 0.09,
        "R1_all": 0.16,
        "wR2_all": 0.20,
        "GooF": 1.00,
    }

    decision = choose_publication_endpoints(
        [candidate],
        prefer_combined=True,
        combined_baseline_r1_all=0.145,
        combined_baseline_source="/validated/combined_baseline.lst",
    )

    accepted = decision["best_physical_endpoint"]
    assert accepted["id"] == "combined_worse"
    gate = accepted["combined_baseline_gate"]
    assert gate["status"] == "warning_worse_than_baseline"
    assert gate["passed"] is True
    assert gate["baseline_R1_all"] == 0.145
    assert gate["baseline_source"] == "/validated/combined_baseline.lst"
    assert gate["endpoint_R1_all"] == 0.16
    assert gate["delta_R1_all"] == pytest.approx(0.015)
    assert gate["reasons"] == []
    assert any("warning only" in warning for warning in gate["warnings"])


def test_combined_endpoint_not_worse_than_validated_baseline_passes():
    candidate = {
        "id": "combined_better",
        "label": "combined",
        "source_kind": "combined_corrected_hkl",
        "lambda_q": 1.0,
        "endpoint_b": 0.0,
        "keep_exti": False,
        "hkl_safe_factor": 1.0,
        "modulation_p95": 1.2,
        "R1_gt": 0.08,
        "R1_all": 0.13,
        "wR2_all": 0.19,
        "GooF": 1.01,
    }

    decision = choose_publication_endpoints(
        [candidate],
        prefer_combined=True,
        combined_baseline_r1_all=0.13,
        combined_baseline_source="fresh combined baseline",
        require_combined_baseline=True,
    )

    selected = decision["best_physical_endpoint"]
    assert selected["id"] == "combined_better"
    assert selected["combined_baseline_gate"]["status"] == "passed"
    assert selected["combined_baseline_gate"]["passed"] is True
    assert selected["combined_baseline_gate"]["delta_R1_all"] == 0.0


def test_write_combined_baseline_gate_records_policy_and_report(tmp_path: Path):
    single = tmp_path / "single"
    single.mkdir()
    (single / "summary.json").write_text(json.dumps({"best_endpoint": endpoint(1.0, 0.0, 0.18, 0.10)}))
    combined = tmp_path / "combined"
    combined.mkdir()
    (combined / "combined_summary.json").write_text(
        json.dumps({"combined_endpoint": endpoint(1.0, 0.0, 0.20, 0.16)})
    )
    out = tmp_path / "chooser"

    decision = write_publication_endpoint_choice(
        {"single": single},
        out,
        combined_run=combined,
        combined_baseline_r1_all=0.145,
        combined_baseline_source="/validated/combined_baseline.lst",
        require_combined_baseline=True,
    )

    assert decision["status"] == "diagnostic_only"
    assert decision["best_physical_endpoint"] == {}
    assert decision["best_single_dataset_diagnostic_endpoint"]["label"] == "single"
    assert decision["policy"]["combined_baseline"] == {
        "required": True,
        "available": True,
        "R1_all": 0.145,
        "source": "/validated/combined_baseline.lst",
    }
    report = (out / "PUBLICATION_ENDPOINT_CHOOSER.md").read_text()
    assert "missing conventional comparison" in report
    assert "Multi-dataset runs never substitute a single-dataset endpoint" in report
    assert "warning_worse_than_baseline" in report
    assert "/validated/combined_baseline.lst" in report


def test_combined_collection_preserves_small_fixed_width_factor_as_warning(tmp_path: Path):
    combined = tmp_path / "combined"
    combined.mkdir()
    unsafe = materialize_endpoint_artifacts(tmp_path, "small_scale", endpoint(1.0, 0.0, 0.18, 0.12, safe=0.10))
    unsafe.update(unsafe["artifacts"])
    (combined / "combined_summary.json").write_text(
        json.dumps({"combined_endpoint": unsafe})
    )

    decision = write_publication_endpoint_choice(
        {},
        tmp_path / "chooser",
        combined_run=combined,
        combined_baseline_r1_all=0.13,
        combined_baseline_source="fresh combined baseline",
        require_combined_baseline=True,
    )

    assert decision["status"] == "ok"
    candidate = decision["best_physical_endpoint"]
    assert candidate["hkl_safe_factor"] == pytest.approx(0.10)
    assert any("HKL rescaling factor 0.1" in warning for warning in candidate["warnings"])


def test_identical_hkl_bytes_have_identical_admission_under_different_units(tmp_path):
    from grond.legacy.engine.shelx_hkl import safe_fixed_width_factor, write_fixed_width_hkl
    from grond.workflows.endpoint_chooser import classify_publication_endpoint

    rows = [(1, 0, 0, 500_000.0, 100.0), (0, 1, 0, 1000.0, 20.0)]
    decisions = []
    payloads = []
    for units in (1.0, 0.5):
        source = [(h, k, ell, intensity * units, sigma * units) for h, k, ell, intensity, sigma in rows]
        factor = safe_fixed_width_factor(source)
        path = tmp_path / f"units_{units}.hkl"
        write_fixed_width_hkl(path, [(h, k, ell, intensity * factor, sigma * factor) for h, k, ell, intensity, sigma in source])
        payloads.append(path.read_bytes())
        candidate = {"lambda_q": 1.0, "endpoint_b": 0.0, "keep_exti": False,
                     "hkl_safe_factor": factor, "hkl": str(path), "modulation_p95": 1.0,
                     "R1_all": 0.10, "R1_gt": 0.08, "wR2_all": 0.12, "GooF": 1.0}
        decisions.append(classify_publication_endpoint(candidate))
    assert payloads[0] == payloads[1]
    assert all(result["promotable"] for result in decisions)
    assert decisions[0]["warnings"] and not decisions[1]["warnings"]
    assert all(result["hkl_precision"]["status"] == "ok" for result in decisions)


@pytest.mark.parametrize("factor", [1.0, 0.01])
def test_unusable_written_sigma_blocks_endpoint_regardless_of_scale(tmp_path, factor):
    from grond.legacy.engine.shelx_hkl import write_fixed_width_hkl
    from grond.workflows.endpoint_chooser import classify_publication_endpoint

    path = tmp_path / "zero_sigma.hkl"
    write_fixed_width_hkl(path, [(1, 0, 0, 100.0, 0.0)])
    result = classify_publication_endpoint({"hkl": str(path), "hkl_safe_factor": factor,
        "lambda_q": 1.0, "endpoint_b": 0.0, "keep_exti": False,
        "R1_all": 0.10, "R1_gt": 0.08, "wR2_all": 0.12, "GooF": 1.0})
    assert not result["promotable"]
    assert any("sigma is nonpositive" in message for message in result["rejection_reasons"])
