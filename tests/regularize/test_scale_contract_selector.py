from __future__ import annotations

from copy import deepcopy

import pytest

from grond.workflows.scale_contract_selector import select_multi_scale_contract


def _contract(
    label: str,
    *,
    scale_params: int,
    aggregate_bic: float,
    input_endpoint_rows: int = 100_000,
    row_all_rall: float = 0.180,
    row_work_rall: float = 0.175,
    row_free_rall: float = 0.205,
    endpoint_r1_all: float = 0.084,
    endpoint_r1_gt: float = 0.059,
    endpoint_wr2_all: float = 0.124,
    endpoint_goof: float = 1.001,
    endpoint_reflections: int = 2_832,
    endpoint_npd: int = 0,
    endpoint_splits: int = 0,
) -> dict[str, object]:
    return {
        "label": label,
        "scale_params": scale_params,
        "aggregate_bic": aggregate_bic,
        "input_endpoint_rows": input_endpoint_rows,
        "row_rall": {
            "all": row_all_rall,
            "work": row_work_rall,
            "free": row_free_rall,
        },
        "bloch_gain_rall": {"all": 0.08, "work": 0.07, "free": 0.09},
        "endpoint": {
            "R1_all": endpoint_r1_all,
            "R1_gt": endpoint_r1_gt,
            "wR2_all": endpoint_wr2_all,
            "GooF": endpoint_goof,
            "n_reflections": endpoint_reflections,
            "npd_count": endpoint_npd,
            "split_count": endpoint_splits,
            "returncode": 0,
            "status": "ok",
            "safe_for_feedback": True,
            "artifacts_exist": True,
            "hkl_safe_factor": 1.0,
            "modulation_p95": 10.0,
        },
    }


def _passing_pair() -> tuple[dict[str, object], dict[str, object]]:
    full = _contract("full", scale_params=1_000, aggregate_bic=12_000.0)
    compact = _contract(
        "compact",
        scale_params=850,
        aggregate_bic=11_500.0,
        input_endpoint_rows=99_950,
        row_all_rall=0.185,
        row_work_rall=0.181,
        row_free_rall=0.212,
        endpoint_r1_all=0.087,
        endpoint_r1_gt=0.061,
        endpoint_wr2_all=0.130,
        endpoint_goof=1.02,
    )
    return full, compact


def _select(full: dict[str, object], compact: dict[str, object]) -> dict[str, object]:
    return select_multi_scale_contract(full, compact)


def test_selector_promotes_a_strictly_gated_compact_contract_without_mutating_inputs() -> None:
    full, compact = _passing_pair()
    original_full = deepcopy(full)
    original_compact = deepcopy(compact)

    result = _select(full, compact)

    assert result["accepted"] is True
    assert result["selected_contract"] == "compact"
    assert result["reasons"] == []
    assert result["diagnostics"]["scale_reduction_fraction"] == pytest.approx(0.15)
    assert result["diagnostics"]["input_endpoint_row_loss_fraction"] == pytest.approx(0.0005)
    assert full == original_full
    assert compact == original_compact


def test_all_numeric_gates_are_inclusive_at_the_documented_boundaries() -> None:
    full, compact = _passing_pair()
    compact.update(
        scale_params=900,
        aggregate_bic=11_999.0,
        input_endpoint_rows=99_900,
        row_rall={"all": 0.190, "work": 0.185, "free": 0.215},
        endpoint={
            "R1_all": 0.089,
            "R1_gt": 0.064,
            "wR2_all": 0.134,
            "GooF": 0.85,
            "n_reflections": 2_832,
            "npd_count": 0,
            "split_count": 0,
            "returncode": 0,
            "status": "ok",
            "safe_for_feedback": True,
            "artifacts_exist": True,
            "hkl_safe_factor": 0.25,
            "modulation_p95": 20.0,
        },
    )

    lower = _select(full, compact)
    assert lower["accepted"] is True
    assert lower["selected_contract"] == "compact"

    compact["endpoint"]["GooF"] = 1.15
    upper = _select(full, compact)
    assert upper["accepted"] is True
    assert upper["selected_contract"] == "compact"


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ({"scale_params": 901}, "scale_reduction_below_minimum"),
        ({"input_endpoint_rows": 99_899}, "input_endpoint_row_loss_exceeded"),
    ],
)
def test_contract_level_gates_fail_closed(
    mutation: dict[str, object], expected_reason: str
) -> None:
    full, compact = _passing_pair()
    compact.update(mutation)

    result = _select(full, compact)

    assert result["accepted"] is False
    assert result["selected_contract"] == "full"
    assert expected_reason in result["reasons"]


def test_bic_is_diagnostic_only_for_scale_topology() -> None:
    full, compact = _passing_pair()
    compact["aggregate_bic"] = full["aggregate_bic"] + 500.0

    result = select_multi_scale_contract(
        full,
        compact,
        require_bic_improvement=True,
    )

    assert result["accepted"] is True
    assert "aggregate_bic_not_improved_diagnostic_only" in result["warnings"]


@pytest.mark.parametrize(
    ("split", "value", "expected_reason"),
    [
        ("all", 0.190_001, "row_rall_all_cost_exceeded"),
        ("work", 0.185_001, "row_rall_work_cost_exceeded"),
        ("free", 0.215_001, "row_rall_free_cost_exceeded"),
    ],
)
def test_each_row_split_is_independently_protected(
    split: str, value: float, expected_reason: str
) -> None:
    full, compact = _passing_pair()
    compact["row_rall"][split] = value

    result = _select(full, compact)

    assert result["accepted"] is False
    assert result["selected_contract"] == "full"
    assert expected_reason in result["reasons"]


@pytest.mark.parametrize(
    ("metric", "value", "expected_reason"),
    [
        ("R1_all", 0.089_001, "endpoint_r1_all_cost_exceeded"),
        ("R1_gt", 0.064_001, "endpoint_r1_gt_cost_exceeded"),
        ("wR2_all", 0.134_001, "endpoint_wr2_all_cost_exceeded"),
    ],
)
def test_each_endpoint_residual_is_diagnostic_only(
    metric: str, value: float, expected_reason: str
) -> None:
    full, compact = _passing_pair()
    compact["endpoint"][metric] = value

    result = _select(full, compact)

    assert result["accepted"] is True
    assert result["selected_contract"] == "compact"
    assert f"{expected_reason}_diagnostic_only" in result["warnings"]


@pytest.mark.parametrize("goof", [0.849_999, 1.150_001, float("nan")])
def test_compact_endpoint_goof_is_warning_only(goof: float) -> None:
    full, compact = _passing_pair()
    compact["endpoint"]["GooF"] = goof

    result = _select(full, compact)

    assert result["accepted"] is True
    assert result["selected_contract"] == "compact"
    assert result["reasons"] == []
    assert any("goof" in warning for warning in result["warnings"])


@pytest.mark.parametrize(
    ("endpoint_changes", "expected_reason"),
    [
        ({"npd_count": 1}, "endpoint_npd_present"),
        ({"split_count": 1}, "endpoint_new_split_pathology"),
        ({"n_reflections": 2_831}, "merged_reflection_count_mismatch"),
    ],
)
def test_endpoint_pathology_and_reflection_identity_gates_fail_closed(
    endpoint_changes: dict[str, object], expected_reason: str
) -> None:
    full, compact = _passing_pair()
    compact["endpoint"].update(endpoint_changes)

    result = _select(full, compact)

    assert result["accepted"] is False
    assert result["selected_contract"] == "full"
    assert expected_reason in result["reasons"]


def test_existing_split_warning_is_not_counted_as_a_new_compact_pathology() -> None:
    full, compact = _passing_pair()
    full["endpoint"]["split_count"] = 2
    compact["endpoint"]["split_count"] = 2

    result = _select(full, compact)

    assert result["accepted"] is True
    assert "endpoint_new_split_pathology" not in result["reasons"]


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ({"returncode": 1}, "endpoint_shelxl_failed"),
        ({"safe_for_feedback": False}, "endpoint_res_unsafe_for_feedback"),
        ({"artifacts_exist": False}, "endpoint_artifacts_missing"),
        ({"hkl_safe_factor": 0.0}, "endpoint_hkl_scale_invalid"),
    ],
)
def test_compact_endpoint_health_and_physical_safety_fail_closed(
    mutation: dict[str, object], expected_reason: str
) -> None:
    full, compact = _passing_pair()
    compact["endpoint"].update(mutation)

    result = _select(full, compact)

    assert result["accepted"] is False
    assert expected_reason in result["reasons"]


def test_compact_endpoint_modulation_is_warning_only() -> None:
    full, compact = _passing_pair()
    compact["endpoint"]["modulation_p95"] = 20.001

    result = _select(full, compact)

    assert result["accepted"] is True
    assert any("exceeds warning level" in warning for warning in result["warnings"])


@pytest.mark.parametrize("split", ["all", "work", "free"])
def test_pooled_bloch_gain_is_required_on_every_split(split: str) -> None:
    full, compact = _passing_pair()
    compact["bloch_gain_rall"][split] = 0.0019

    result = _select(full, compact)

    assert result["accepted"] is False
    assert f"pooled_bloch_rall_gain_{split}_below_minimum" in result["reasons"]


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ((), "scale_params"),
        ((), "aggregate_bic"),
        (("row_rall",), "free"),
        (("endpoint",), "R1_all"),
        (("endpoint",), "GooF"),
        (("endpoint",), "n_reflections"),
    ],
)
def test_missing_required_values_fail_closed_as_an_invalid_contract(
    path: tuple[str, ...], key: str
) -> None:
    full, compact = _passing_pair()
    target = compact
    for part in path:
        target = target[part]
    target.pop(key)

    result = _select(full, compact)

    assert result["accepted"] is False
    assert result["selected_contract"] == "full"
    assert "invalid_candidate_contract" in result["reasons"]


def test_compact_selection_is_not_rejected_by_intensity_units():
    full, compact = _passing_pair()
    compact["endpoint"]["hkl_safe_factor"] = 0.01
    result = _select(full, compact)
    assert result["accepted"] is True
    assert any("HKL rescaling" in item for item in result["warnings"])
