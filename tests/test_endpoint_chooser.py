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


def test_endpoint_chooser_rejects_high_goof_for_publication():
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

    assert decision["best_physical_endpoint"]["id"] == "boring_goof"
    assert decision["best_diagnostic_endpoint"]["id"] == "low_r_bad_goof"
    rejected = [c for c in decision["rejected_corrections"] if c["id"] == "low_r_bad_goof"][0]
    assert any("GooF" in reason and "outside" in reason for reason in rejected["rejection_reasons"])


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
    (run / "summary.json").write_text(
        json.dumps(
            {
                "endpoint_results_sorted": [
                    endpoint(1.0, 16.0, 0.18, 0.10),
                    endpoint(1.0, 0.0, 0.22, 0.13),
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
    report = (out / "PUBLICATION_ENDPOINT_CHOOSER.md").read_text()
    assert "Rejected Corrections" in report
    assert "| label | source | endpoint B | EXTI | class | R1(F)gt | R1(F)all | wR2(F^2)all | GooF | pathology |" in report
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
