"""CLI hand contracts regression tests."""

import argparse

import pytest

import grond.cli as cli


def test_h_omit_mode_is_explicit_about_single_dataset_only():
    assert cli._validated_h_omit_mode("off", 10) == "off"
    assert cli._validated_h_omit_mode("on", 1) == "on"
    assert cli._validated_h_omit_mode("both", 1) == "on"
    with pytest.raises(ValueError, match="not implemented for multi-dataset"):
        cli._validated_h_omit_mode("on", 2)


def test_post_feedback_hand_diagnostic_gate_requires_signed_support():
    args = argparse.Namespace(hand_flip_zkp_all=3.5)
    ok, reasons = cli.hand_diagnostic_supports_other(
        {
            "all": {
                "n": 100,
                "paired_row_t_delta_chi2": -2.4,
                "klar_palatinus_adjusted_hand_z": {"status": "ok", "z": -10.0},
            },
            "free": {
                "n": 10,
                "paired_row_t_delta_chi2": 0.8,
                "klar_palatinus_adjusted_hand_z": {"status": "ok", "z": 1.5},
            },
        },
        args,
    )
    assert ok is True
    assert reasons == []

    weak, weak_reasons = cli.hand_diagnostic_supports_other(
        {
            "all": {
                "n": 100,
                "paired_row_t_delta_chi2": -10.0,
                "klar_palatinus_adjusted_hand_z": {"status": "ok", "z": -3.49},
            },
            "free": {
                "n": 10,
                "paired_row_t_delta_chi2": -10.0,
                "klar_palatinus_adjusted_hand_z": {"status": "ok", "z": -9.0},
            },
        },
        args,
    )
    assert weak is False
    assert "all-row Klar-Palatinus adjusted hand Z" in weak_reasons[0]

    missing, missing_reasons = cli.hand_diagnostic_supports_other(
        {
            "all": {"n": 100, "paired_row_t_delta_chi2": -4.0},
            "free": {"n": 9, "paired_row_t_delta_chi2": -2.0},
        },
        args,
    )
    assert missing is False
    assert any("unavailable" in reason for reason in missing_reasons)

    warnings = cli._hand_evidence_warnings(
        all_z=-10.0,
        free_z=-1.5,
        switched=True,
        threshold=3.5,
    )
    assert any("supports the selected hand only weakly" in warning for warning in warnings)
