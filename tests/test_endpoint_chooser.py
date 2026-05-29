import json
from pathlib import Path

from grond.workflows.endpoint_chooser import choose_publication_endpoints, write_publication_endpoint_choice


def endpoint(lam: float, endpoint_b: float, wr2: float, r1: float, *, exti: bool = False, safe: float = 1.0) -> dict:
    return {
        "lambda_q": lam,
        "endpoint_b": endpoint_b,
        "keep_exti": exti,
        "hkl_safe_factor": safe,
        "mod_p95": 1.5,
        "shelxl": {"R1_gt": r1 - 0.02, "R1_all": r1, "wR2_all": wr2, "GooF": 1.1, "N_all": 100},
    }


def test_endpoint_chooser_keeps_b_zero_physical_and_reports_b_diagnostic():
    candidates = [
        {
            "id": "physical",
            "label": "dataset",
            "source_kind": "native_corrected_hkl",
            "lambda_q": 0.95,
            "endpoint_b": 0.0,
            "keep_exti": False,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 1.2,
            "R1_gt": 0.11,
            "R1_all": 0.14,
            "wR2_all": 0.24,
        },
        {
            "id": "diagnostic_b",
            "label": "dataset",
            "source_kind": "native_corrected_hkl",
            "lambda_q": 0.95,
            "endpoint_b": 18.0,
            "keep_exti": False,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 1.2,
            "R1_gt": 0.09,
            "R1_all": 0.11,
            "wR2_all": 0.19,
        },
    ]
    decision = choose_publication_endpoints(candidates)
    assert decision["best_physical_endpoint"]["id"] == "physical"
    assert decision["best_diagnostic_endpoint"]["id"] == "diagnostic_b"
    assert any("endpoint-B" in reason for reason in decision["best_diagnostic_endpoint"]["rejection_reasons"])


def test_write_publication_endpoint_choice_collects_outputs(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps(
            {
                "endpoint_results_sorted": [
                    endpoint(0.95, 16.0, 0.18, 0.10),
                    endpoint(0.95, 0.0, 0.22, 0.13),
                    endpoint(1.00, 0.0, 0.23, 0.14, exti=True),
                ]
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
    assert "Rejected Corrections" in (out / "PUBLICATION_ENDPOINT_CHOOSER.md").read_text()


def test_endpoint_chooser_prefers_combined_endpoint_when_multi_dataset():
    candidates = [
        {
            "id": "single_best",
            "label": "d07",
            "source_kind": "native_corrected_hkl",
            "lambda_q": 0.95,
            "endpoint_b": 0.0,
            "keep_exti": False,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 1.2,
            "R1_gt": 0.09,
            "R1_all": 0.10,
            "wR2_all": 0.20,
        },
        {
            "id": "combined",
            "label": "combined",
            "source_kind": "combined_corrected_hkl",
            "lambda_q": 0.95,
            "endpoint_b": 0.0,
            "keep_exti": False,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 1.2,
            "R1_gt": 0.12,
            "R1_all": 0.13,
            "wR2_all": 0.24,
        },
    ]
    decision = choose_publication_endpoints(candidates, prefer_combined=True)
    assert decision["best_physical_endpoint"]["id"] == "combined"
    assert decision["best_single_dataset_physical_endpoint"]["id"] == "single_best"
    rejected_single = [c for c in decision["rejected_corrections"] if c["id"] == "single_best"][0]
    assert "single-dataset endpoint" in rejected_single["rejection_reasons"][0]
