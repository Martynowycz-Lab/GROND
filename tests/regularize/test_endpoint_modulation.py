import numpy as np
import pytest

from grond.legacy.engine.combine_native_nuisance_runs import (
    audit_combined_endpoint_modulation,
    endpoint_rows_and_corrections_from_native,
)
from grond.legacy.engine.endpoint_modulation import modulation_manifest_fields, symmetric_fold_change_audit


def test_symmetric_fold_change_is_scale_invariant_and_direction_symmetric() -> None:
    corrections = np.asarray([0.25, 0.5, 1.0, 2.0, 4.0], dtype=float)

    base = symmetric_fold_change_audit(corrections)
    globally_rescaled = symmetric_fold_change_audit(corrections * 1.0e6)
    reciprocals = symmetric_fold_change_audit(1.0 / corrections)

    assert base["status"] == "ok"
    assert base["fold_change_p95"] == pytest.approx(globally_rescaled["fold_change_p95"])
    assert base["fold_change_p95"] == pytest.approx(reciprocals["fold_change_p95"])
    assert base["fold_change_max"] == pytest.approx(4.0)


def test_modulation_audit_is_explicitly_unavailable_without_positive_factors() -> None:
    fields = modulation_manifest_fields([float("nan"), 0.0, -1.0])

    assert fields["mod_p95"] is None
    assert fields["modulation_p95"] is None
    assert fields["modulation_audit"]["status"] == "unavailable_no_positive_finite_corrections"
    assert fields["modulation_audit"]["rows_valid_positive"] == 0


def test_modulation_audit_rejects_mismatched_mask_shape() -> None:
    with pytest.raises(ValueError, match="mask shape"):
        symmetric_fold_change_audit([1.0, 2.0], [True])


def test_native_endpoint_corrections_include_scale_q_and_median_rescale() -> None:
    rows = [
        {
            "h": "1",
            "k": "0",
            "l": "0",
            "target_I": "100",
            "sigma_eff": "2",
            "dyn_raw": "50",
            "dyn_scaled": "50",
            "q_dyn_over_born": "1",
        },
        {
            "h": "0",
            "k": "1",
            "l": "0",
            "target_I": "100",
            "sigma_eff": "2",
            "dyn_raw": "50",
            "dyn_scaled": "200",
            "q_dyn_over_born": "1",
        },
    ]

    endpoint_rows, corrections = endpoint_rows_and_corrections_from_native(rows, 1.0)

    assert endpoint_rows == [(1, 0, 0, 160.0, 3.2), (0, 1, 0, 40.0, 0.8)]
    assert corrections == pytest.approx([1.6, 0.4])
    assert modulation_manifest_fields(corrections)["mod_p95"] == pytest.approx(2.0)


def test_combined_modulation_includes_relative_interdataset_scales() -> None:
    combined, per_dataset = audit_combined_endpoint_modulation(
        {
            "a": np.asarray([1.0, 1.0]),
            "b": np.asarray([1.0, 1.0]),
        },
        {"a": 1.0, "b": 100.0},
        endpoint_source="native",
    )

    assert per_dataset["a"]["mod_p95"] == pytest.approx(1.0)
    assert per_dataset["b"]["mod_p95"] == pytest.approx(1.0)
    assert combined["mod_p95"] == pytest.approx(10.0)


def test_combined_hkl_source_fails_closed_without_row_corrections() -> None:
    combined, per_dataset = audit_combined_endpoint_modulation(
        {"a": None},
        {"a": 1.0},
        endpoint_source="hkl",
    )

    assert combined["mod_p95"] is None
    assert combined["modulation_audit"]["status"] == "unavailable_endpoint_source_hkl"
    assert per_dataset["a"]["mod_p95"] is None
