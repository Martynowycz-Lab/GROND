import argparse
import csv
import json
import math
import sys
from types import SimpleNamespace

import pytest

from grond import cli, manual_workflow
from grond.legacy.engine import absolute_structure_diagnostic as hand
from grond.workflows import parity_evidence


def paired_rows(n=20, *, same=False, near=False):
    selected = []
    other = []
    for i in range(n):
        target = 100.0 + i
        left = target + 1.0
        right = left + 1e-13 if near else left if same else target
        common = {"row_index": str(i), "h": str(i + 1), "k": "0", "l": "0",
                  "target_I": str(target), "source_sigma": "1", "sigma_eff": "1",
                  "born_scaled": str(target + 5), "split": "work"}
        selected.append({**common, "dyn_scaled": str(left)})
        other.append({**common, "dyn_scaled": str(right)})
    return selected, other


def adjusted(left, right):
    return hand.klar_palatinus_adjusted_hand_z(
        list(zip(left, right, strict=True)), {"selected": [], "other": []}, "dyn_scaled"
    )


@pytest.mark.parametrize("near", [False, True])
def test_identical_or_numerically_tied_hands_never_flip(near):
    left, right = paired_rows(same=True, near=near)
    result = adjusted(left, right)
    swapped = adjusted(right, left)
    assert result["z"] == swapped["z"] == 0.0
    assert result["informative_nontied_rows"] == 0
    assert result["probability_is_calibrated"] is False
    assert not cli.hand_diagnostic_supports_other(
        {"all": {"klar_palatinus_adjusted_hand_z": result}},
        argparse.Namespace(hand_flip_zkp_all=3.5),
    )[0]
    preference = cli.hand_preference_summary(result["z"], active_hand="original", threshold=3.5)
    assert preference["preferred_hand"] == "no_preference"
    assert preference["hand_confidence"] == "no_sensitivity"


@pytest.mark.parametrize("free", [
    {}, {"n": 0}, {"n": 1, "klar_palatinus_adjusted_hand_z": {"status": "ok", "z": 0.2}},
    {"n": 5, "klar_palatinus_adjusted_hand_z": {"status": "ok", "z": 3.0}},
])
def test_all_row_hand_decision_survives_absent_small_or_conflicting_free(free):
    left, right = paired_rows()
    all_z = adjusted(left, right)
    assert all_z["z"] < -3.5
    stats = {"status": "ok", "all": {"klar_palatinus_adjusted_hand_z": all_z}, "free": free}
    supported, errors = cli.hand_diagnostic_supports_other(stats, argparse.Namespace(hand_flip_zkp_all=3.5))
    assert supported and not errors
    warnings = cli._hand_evidence_warnings(
        all_z=all_z["z"], free_z=float((free.get("klar_palatinus_adjusted_hand_z") or {}).get("z", "nan")),
        switched=True, threshold=3.5,
    )
    assert warnings
    assert any("not calibrated" in warning for warning in warnings)


def test_weak_inverse_preference_is_reported_without_forcing_switch():
    preference = cli.hand_preference_summary(-0.4, active_hand="original", threshold=3.5)
    assert preference["preferred_hand"] == "alternate"
    assert preference["hand_confidence"] == "weak_conditional"
    assert not cli.hand_diagnostic_supports_other(
        {"all": {"klar_palatinus_adjusted_hand_z": {"status": "ok", "z": -0.4}}},
        argparse.Namespace(hand_flip_zkp_all=3.5),
    )[0]


def test_source_observation_error_cannot_select_hand_even_with_large_partial_z():
    left, right = paired_rows(100)
    right[0]["target_I"] = "999"
    result = adjusted(left, right)
    assert result["status"] == "invalid_row_contract"
    assert result["calculation_errors"]
    assert math.isfinite(result["z"])  # Partial calculation is visible but not decision-valid.
    supported, errors = cli.hand_diagnostic_supports_other(
        {"all": {"klar_palatinus_adjusted_hand_z": result}},
        argparse.Namespace(hand_flip_zkp_all=3.5),
    )
    assert not supported and errors


def test_explicit_geometry_error_cannot_select_hand():
    stats = {"status": "calculation_error", "calculation_status": "error",
             "calculation_errors": ["center_sg_inv_A differs between original/inverse physical rows"],
             "all": {"klar_palatinus_adjusted_hand_z": {"status": "ok", "z": -9.0}}}
    supported, errors = cli.hand_diagnostic_supports_other(stats, argparse.Namespace(hand_flip_zkp_all=3.5))
    assert not supported
    assert any("center_sg_inv_A" in error for error in errors)


def test_physical_pair_audit_allows_no_free_but_rejects_changed_geometry(tmp_path):
    exact = ("split", "group", "detector_bin", "basis_size", "omega_bragg_root_available",
             "sg_window_clipped", "sg_jitter_kernel_points", "sg_profile_points_total")
    numeric = ("target_I", "sigma_eff", "xds_sigma", "d_A", "peak", "corr", "rlp",
               "row_resolution_factor", "center_sg_inv_A", "omega_bragg_deg",
               "omega_bragg_minus_center_deg", "sg_window_inv_A", "sg_window_raw_inv_A",
               "sg_window_half_frames", "sg_dSg_domega_inv_A_per_deg", "sg_dSg_dframe_inv_A",
               "sg_jitter_sigma_inv_A", "sg_jitter_d_inv_A", "sg_jitter_frame_fraction",
               "sg_profile_weight_sum", "max_sg_solve_residual")
    row = {**dict.fromkeys(exact, "1"), **dict.fromkeys(numeric, "1"),
           "row_index": "0", "h": "1", "k": "0", "l": "0", "split": "work"}
    left, right = tmp_path / "left.csv", tmp_path / "right.csv"

    def write(path, values):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values))
            writer.writeheader()
            writer.writerow(values)

    write(left, row)
    write(right, row)
    assert cli._paired_row_split_contract(left, right)["passed"]
    write(right, {**row, "center_sg_inv_A": "0.002"})
    mismatch = cli._paired_row_split_contract(left, right)
    assert not mismatch["passed"]
    assert mismatch["numeric_mismatch_counts"]["center_sg_inv_A"] == 1


def test_centrosymmetric_producer_and_endpoint_policy_agree(tmp_path):
    payload = cli.run_multi_final_elastic_hand_selection(
        SimpleNamespace(datasets=[]), argparse.Namespace(hand_control=True), "python",
        tmp_path / "logs", tmp_path, final_model="unused.res", selected_hand="original", final_runs={},
        shared_hand_decision={"symmetry_policy": {"run_hand_control": False, "reason": "centrosymmetric"}},
    )
    assert payload["preferred_hand"] == "not_applicable"
    assert cli._multi_hand_publication_blockers({"combined": "combined", "post_feedback_multi_hand_diagnostic": payload}) == []


def test_diagnostic_reports_all_row_bloch_and_born_both_hands(tmp_path, monkeypatch):
    left, right = paired_rows()
    paths = []
    for label, rows in (("selected", left), ("other", right)):
        path = tmp_path / f"{label}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    out = tmp_path / "diagnostic.json"
    monkeypatch.setattr(sys, "argv", ["hand", "--best", str(paths[0]), "--other", str(paths[1]), "--out", str(out)])
    hand.main()
    result = json.loads(out.read_text())
    assert result["status"] == "ok"
    assert result["calculation_status"] == "complete"
    dynamic = result["all"]["matched_prediction_metrics"]
    born = result["matched_born_control"]["all"]["matched_prediction_metrics"]
    assert dynamic["other"]["wR2_all"] == 0
    assert 0 < dynamic["selected"]["wR2_all"] < born["selected"]["wR2_all"]
    assert born["selected"] == born["other"]
    assert result["free"]["n"] == 0
    assert "not calibrated" in result["confidence_interpretation"]


def test_pooling_uses_neutral_tie_effective_count():
    left, right = paired_rows(same=True)
    result = adjusted(left, right)
    pooled = cli._combine_adjusted_hand_z(
        [{"all": {"klar_palatinus_adjusted_hand_z": result}}] * 2, "all"
    )
    assert pooled["z"] == 0
    manual = manual_workflow._hand_evidence({
        "one": {"all": {"n": 20, "klar_palatinus_adjusted_hand_z": result}, "free": {}}
    })
    assert manual["preferred_hand"] == "no_preference"
    assert manual["recommendation"] == "ambiguous"


def test_legacy_hand_replay_score_does_not_use_free_metrics():
    metrics = {"all": {"dyn": {"wR2_all": .2, "R1_all": .1, "R1_gt": .09, "CC": .9},
                       "born": {"R1_gt": .3, "CC": .8}}, "free": {"dyn": {"wR2_all": .01}}}
    score = parity_evidence._score_from_metrics(metrics)
    metrics["free"]["dyn"]["wR2_all"] = float("nan")
    assert parity_evidence._score_from_metrics(metrics)["score"] == score["score"]


def test_calibration_replay_allows_only_matching_unavailable_free_diagnostic():
    empty = {"status": "no_valid_rows", "n": 0}
    assert cli._adjusted_hand_blocks_match(empty, dict(empty), allow_unavailable=True)
    assert not cli._adjusted_hand_blocks_match(empty, dict(empty), allow_unavailable=False)
    assert not cli._adjusted_hand_blocks_match(empty, {"status": "ok", "n": 3, "z": 1.}, allow_unavailable=True)
    assert not cli._adjusted_hand_blocks_match({"status": "ok", "n": 0}, {"status": "ok", "n": 0}, allow_unavailable=True)


def test_legacy_hand_replay_uses_immutable_sigma_in_matching_target_units():
    left, _ = paired_rows(2)
    for row in left:
        row.update(xds_sigma="2", source_sigma="0.5", source_to_target_factor="0.25")
    arrays = parity_evidence._row_arrays(left)
    assert arrays["xds_sigma"].tolist() == [.5, .5]
