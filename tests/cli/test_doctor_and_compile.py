"""CLI doctor and compile regression tests."""


import json

import grond.cli as cli
import grond.commands.doctor as owner_commands_doctor
import grond.commands.validation as owner_commands_validation
from grond.cli import (
    build_parser,
)


def test_doctor_reports_current_install_feature_flags(capsys):
    args = build_parser().parse_args(["doctor"])

    assert args.func(args) == 0

    payload = json.loads(capsys.readouterr().out)
    grond = payload["grond"]
    assert grond["install_status"] == "current_required_features_present"
    assert grond["required_features"] == {
        "shelxl_feedback_res_guard": True,
        "usable_feedback_model_selection": True,
        "final_feedback_row_recompute": True,
        "corrected_endpoint_hklf4_contract": True,
        "absolute_structure_no_pandas": True,
        "proper_hand_parity_contract": True,
        "canonical_hand_hklf4_gauge_repair": True,
        "separate_parity_calibration_commands": True,
    }
    assert payload["ready"] is True
    assert payload["ready_scope"] == "core"
    assert payload["failures"] == []
    assert payload["requirements"]["core_python"] == ["numpy", "scipy", "gemmi"]
    assert payload["requirements"]["default_small_molecule_endpoint"] == ["shelxl"]
    assert payload["requirements"]["native_dynamical_maps"] == []
    assert set(payload["tool_config"]["external_tools"]) == {"shelxl", "shelx2map"}
    assert "optional_packages" not in payload
    assert "jax" not in payload


def test_doctor_strict_checks_exact_requested_python(monkeypatch, capsys):
    requested = "/usr/bin/python-without-grond-deps"
    seen = []

    def fake_probe(python):
        seen.append(python)
        return {
            "python": python,
            "returncode": 0,
            "packages": {
                "numpy": {"available": True},
                "scipy": {"available": True},
                "gemmi": {"available": False, "error": "ModuleNotFoundError: gemmi"},
            },
        }

    monkeypatch.setattr(owner_commands_doctor, "probe_accelerators", fake_probe)
    args = build_parser().parse_args(["doctor", "--python", requested, "--strict"])

    assert args.func(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert seen == [requested]
    assert payload["ready"] is False
    assert any("required package gemmi is unavailable" in failure for failure in payload["failures"])


def test_doctor_non_strict_remains_report_only(monkeypatch, capsys):
    monkeypatch.setattr(
        owner_commands_doctor,
        "probe_accelerators",
        lambda python: {
            "python": python,
            "probe_error": "FileNotFoundError: requested interpreter",
        },
    )
    args = build_parser().parse_args(["doctor", "--python", "/missing/python"])

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert any("dependency probe failed" in failure for failure in payload["failures"])


def test_doctor_small_molecule_scope_requires_shelxl(monkeypatch, capsys):
    monkeypatch.setattr(
        owner_commands_doctor,
        "probe_accelerators",
        lambda python: {
            "python": python,
            "returncode": 0,
            "packages": {name: {"available": True} for name in ("numpy", "scipy", "gemmi")},
        },
    )
    monkeypatch.setattr(
        owner_commands_doctor,
        "external_tool_statuses",
        lambda args: {
            "shelxl": {"available": False, "error": "not found"},
            "shelx2map": {"available": False, "error": "not found"},
        },
    )
    args = build_parser().parse_args(["doctor", "--workflow", "small-molecule", "--strict"])

    assert args.func(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready_scope"] == "small-molecule"
    assert any("small-molecule workflow requires shelxl" in failure for failure in payload["failures"])


def test_doctor_strict_fails_requested_bloch_operation(monkeypatch, capsys):
    monkeypatch.setattr(
        owner_commands_doctor,
        "probe_accelerators",
        lambda python: {
            "python": python,
            "returncode": 0,
            "packages": {name: {"available": True} for name in ("numpy", "scipy", "gemmi")},
        },
    )
    monkeypatch.setattr(
        owner_commands_doctor,
        "probe_bloch_ops",
        lambda python: {
            "python": python,
            "returncode": 0,
            "scipy_cpu": {
                "eig": {"ok": False, "error": "LAPACK failure"},
                "eigh": {"ok": True},
            },
        },
    )
    args = build_parser().parse_args(["doctor", "--probe-bloch", "--strict"])

    assert args.func(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert any("SciPy eig Bloch operation failed" in failure for failure in payload["failures"])


def test_compile_command_uses_temporary_bytecode_cache(monkeypatch, capsys):
    args = build_parser().parse_args(["compile"])
    seen = {}

    def fake_compile_dir(root, quiet, force, workers):
        seen["compile_prefix"] = cli.sys.pycache_prefix
        return True

    def fake_run(cmd, check, env):
        seen["run_prefix"] = env.get("PYTHONPYCACHEPREFIX")
        seen["cmd"] = cmd

    monkeypatch.setattr(cli.compileall, "compile_dir", fake_compile_dir)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(owner_commands_validation, "choose_python", lambda requested: "python-x")
    monkeypatch.setattr(owner_commands_validation, "external_tool_statuses", lambda args: {})

    assert cli.cmd_compile(args) == 0

    output = capsys.readouterr().out
    assert "compile_check_passed" in output
    assert "temporary" in output
    assert seen["compile_prefix"]
    assert seen["compile_prefix"] == seen["run_prefix"]
    assert seen["cmd"][0] == "python-x"
