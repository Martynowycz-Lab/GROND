"""Manual reporting must preserve the hand statistic's neutral-tie convention."""

import json
import math

import pytest

from grond.manual.hand import _hand_evidence
from grond.manual.reporting import _selected_hand_summary


def _workflow(tmp_path, diagnostics, *, choice="input"):
    paths = {}
    for name, diagnostic in diagnostics.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(diagnostic))
        paths[name] = str(path)
    return {
        "stages": {
            "hand": {
                "outputs": {
                    "choice": choice,
                    "dataset_diagnostics": paths,
                }
            }
        }
    }


def _diagnostic(n, strict_wins, ties, *, w=0.0):
    effective_wins = strict_wins + ties / 2
    kp = {
        "status": "ok",
        "n": n,
        "w": w,
        "k_selected_lower_abs_intensity_residual": strict_wins,
        "k_effective_selected_wins_plus_half_ties": effective_wins,
        "z": (2 * effective_wins - n) / math.sqrt(n - w),
    }
    return {support: {"n": n, "klar_palatinus_adjusted_hand_z": dict(kp)} for support in ("all", "free")}


@pytest.mark.parametrize("choice", ["input", "inverse"])
def test_manual_report_all_ties_remain_zero_and_match_hand_decision(tmp_path, choice):
    diagnostics = {"a": _diagnostic(100, 0, 100), "b": _diagnostic(40, 0, 40)}
    evidence = _hand_evidence(diagnostics)
    report = _selected_hand_summary(_workflow(tmp_path, diagnostics, choice=choice))

    assert evidence["preferred_hand"] == "no_preference"
    for support in ("all", "free"):
        assert evidence["combined_adjusted_z_input_favored"][support] == 0.0
        assert report["combined"][support]["z_selected_coordinates"] == 0.0
        assert report["combined"][support]["dataset_count"] == 2
        assert all(row[support]["z_selected_coordinates"] == 0.0 for row in report["datasets"].values())


def test_manual_report_with_ties_is_antisymmetric_when_hands_swap(tmp_path):
    original = {"a": _diagnostic(100, 60, 20, w=4), "b": _diagnostic(40, 20, 10, w=2)}
    inverse = {"a": _diagnostic(100, 20, 20, w=4), "b": _diagnostic(40, 10, 10, w=2)}
    forward = _selected_hand_summary(_workflow(tmp_path, original))
    swapped = _selected_hand_summary(_workflow(tmp_path, inverse))
    reoriented = _selected_hand_summary(_workflow(tmp_path, original, choice="inverse"))
    evidence = _hand_evidence(original)

    for support in ("all", "free"):
        z = forward["combined"][support]["z_selected_coordinates"]
        assert z > 0.0
        assert z == pytest.approx(evidence["combined_adjusted_z_input_favored"][support])
        assert swapped["combined"][support]["z_selected_coordinates"] == pytest.approx(-z)
        assert reoriented["combined"][support]["z_selected_coordinates"] == pytest.approx(-z)


def test_manual_report_retains_legacy_strict_count_fallback(tmp_path):
    diagnostic = _diagnostic(20, 13, 0, w=4)
    for support in ("all", "free"):
        del diagnostic[support]["klar_palatinus_adjusted_hand_z"]["k_effective_selected_wins_plus_half_ties"]
    report = _selected_hand_summary(_workflow(tmp_path, {"legacy": diagnostic}))

    for support in ("all", "free"):
        assert report["combined"][support]["z_selected_coordinates"] == pytest.approx(1.5)
        assert report["datasets"]["legacy"][support]["z_selected_coordinates"] == pytest.approx(1.5)
