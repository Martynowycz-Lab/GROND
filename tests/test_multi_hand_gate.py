import argparse
import csv
import hashlib
import json
from pathlib import Path

import pytest

import grond.cli as cli
from grond.cli import build_parser
from grond.io.config import DatasetSpec, PrepConfig


@pytest.fixture(autouse=True)
def _paired_nuisance_selector(monkeypatch):
    """Keep orchestration tests focused on the four-stage hand contract."""

    monkeypatch.setattr(
        cli,
        "select_branch_symmetric_nuisance",
        lambda *_args, **_kwargs: {
            "schema_version": 1,
            "status": "selected",
            "selection_rule": (
                "min_paired_mean_work_dyn_wR2_then_nuisance_sha256_v1"
            ),
            "hand_label_invariant": True,
            "free_rows_used_for_selection": False,
            "known_configuration_used": False,
            "selected_nuisance": {
                "thickness_A": 1200.0,
                "frame_offset": 0.1,
            },
        },
    )


def _hand_decision(
    delta: float,
    *,
    selected: str = "alternate",
    gate_status: str = "passed",
    warning: str = "",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "selected": selected,
        "selected_hand": selected,
        "quality_gate_status": gate_status,
        "score_delta_original_minus_alternate": delta,
        "active_score": {"dyn_wR2_all": 0.22, "free_dyn_wR2_all": 0.25},
        "alternate_score": {"dyn_wR2_all": 0.18, "free_dyn_wR2_all": 0.21},
    }
    if warning:
        payload["warning"] = warning
    return payload


def _auto_args(tmp_path: Path, model: Path, hkl: Path) -> argparse.Namespace:
    return build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            str(tmp_path / "out"),
            "--hkl",
            str(hkl),
            "--structure",
            str(model),
        ]
    )


def _native_summary(
    dyn_wr2: float,
    free_wr2: float,
    *,
    row_selection: tuple[int, int, int] = (500, 450, 50),
) -> dict[str, object]:
    selected, work, free = row_selection
    return {
        "row_selection": {"selected_rows": selected, "work_rows": work, "free_rows": free},
        "best_native": {
            "nuisance": {"thickness_A": 1200.0, "frame_offset": 0.1},
            "metrics": {
                "all": {
                    "dyn": {"wR2_all": dyn_wr2, "R1_all": dyn_wr2 * 0.8, "R1_gt": dyn_wr2 * 0.6, "CC": 0.95},
                    "born": {"R1_gt": 0.30, "CC": 0.90},
                },
                "free": {"dyn": {"wR2_all": free_wr2}},
            },
        },
    }


@pytest.mark.parametrize(
    ("actual", "matches"),
    [
        ({"dose_b_ref_frame": float("nan")}, True),
        ({}, False),
        ({"dose_b_ref_frame": None}, True),
        ({"dose_b_ref_frame": "not-a-number"}, False),
        ({"dose_b_ref_frame": float("inf")}, False),
        ({"dose_b_ref_frame": 0.0}, False),
    ],
)
def test_pinned_hand_nuisance_contract_only_accepts_matching_nan_sentinels(
    actual: dict[str, object],
    matches: bool,
) -> None:
    mismatches = cli._pinned_nuisance_mismatches(
        {"dose_b_ref_frame": float("nan")},
        actual,
    )
    assert (not mismatches) is matches


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        (float("inf"), float("inf")),
        (float("inf"), float("-inf")),
        (float("nan"), float("inf")),
        (float("inf"), float("nan")),
    ],
)
def test_pinned_hand_nuisance_contract_rejects_all_infinite_combinations(
    expected: float,
    actual: float,
) -> None:
    mismatches = cli._pinned_nuisance_mismatches(
        {"dose_b_ref_frame": expected},
        {"dose_b_ref_frame": actual},
    )
    assert mismatches


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ({"selected_rows": 500, "work_rows": 450, "free_rows": 50}, (500, 450, 50)),
        ({"free_rows": 50}, None),
        ({"selected_rows": "500", "work_rows": "450", "free_rows": "50"}, None),
        ({"selected_rows": 500.0, "work_rows": 450, "free_rows": 50}, None),
        ({"selected_rows": True, "work_rows": 0, "free_rows": 1}, None),
        ({"selected_rows": -1, "work_rows": -1, "free_rows": 0}, None),
        ({"selected_rows": 500, "work_rows": 450, "free_rows": 49}, None),
    ],
)
def test_hand_row_selection_contract_requires_exact_consistent_counts(
    rows: dict[str, object],
    expected: tuple[int, int, int] | None,
) -> None:
    assert cli._validated_row_selection_counts(rows) == expected


def test_choose_hand_uses_symmetric_grid_then_two_matched_scout_commands(
    tmp_path: Path, monkeypatch
):
    model = tmp_path / "model.res"
    hkl = tmp_path / "rows.hkl"
    baseline = tmp_path / "baseline.hkl"
    model.write_text("TITL test\nCELL 0.025079 5 6 7 90 90 90\nLATT -1\nEND\n")
    hkl.write_text("! rows\n")
    baseline.write_text("0 0 1 1.0 1.0\n")
    ds = DatasetSpec(name="a", hkl=str(hkl), hkl_transform="-h,k,l")
    prep = PrepConfig(structure=str(model), datasets=[ds], baseline_hkl=str(baseline))
    args = _auto_args(tmp_path, model, hkl)
    args.split_seed = 731
    args.production_scales = True
    active_run = tmp_path / "out" / "runs" / "a" / "screen"
    active_run.mkdir(parents=True)
    (active_run / "summary.json").write_text(json.dumps(_native_summary(0.09, 0.11, row_selection=(4000, 3600, 400))) + "\n")
    commands: list[list[str]] = []

    def fake_run_logged(cmd, _logs, _label, **_kwargs):
        commands.append(list(cmd))
        out_dir = Path(cmd[cmd.index("--out-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        alternate = "--alternate-hand" in cmd
        summary = _native_summary(0.15, 0.20) if alternate else _native_summary(0.22, 0.27)
        table = out_dir / "native_dyn_calc.csv"
        fields = [
            "row_index", "h", "k", "l", "target_I", "sigma_eff",
            "xds_sigma", "group", "detector_bin", "split", "dyn_raw",
            "born_raw", "dyn_scaled", "born_scaled",
        ]
        with table.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in range(20):
                target = 100.0 + index
                error = 0.01 if alternate else 0.25
                prediction = target * (
                    1.0 + error * (-1.0 if index % 2 else 1.0)
                )
                writer.writerow(
                    {
                        "row_index": index,
                        "h": index + 1,
                        "k": 0,
                        "l": 0,
                        "target_I": target,
                        "sigma_eff": 5.0,
                        "xds_sigma": 5.0,
                        "group": 0,
                        "detector_bin": 0,
                        "split": "free" if index >= 18 else "work",
                        "dyn_raw": prediction,
                        "born_raw": target,
                        "dyn_scaled": prediction,
                        "born_scaled": target,
                    }
                )
        summary["artifacts"] = {
            "native_row_table": str(table.resolve()),
            "native_row_table_sha256": hashlib.sha256(
                table.read_bytes()
            ).hexdigest(),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary) + "\n")

    monkeypatch.setattr(cli, "run_logged", fake_run_logged)
    monkeypatch.setattr(
        cli,
        "build_common_sigma_hand_evidence",
        lambda *_args, **_kwargs: {
            "status": "replayed",
            "original_score": {
                "score": 0.30,
                "dyn_wR2_all": 0.22,
                "free_dyn_wR2_all": 0.27,
            },
            "alternate_score": {
                "score": 0.16,
                "dyn_wR2_all": 0.15,
                "free_dyn_wR2_all": 0.20,
            },
            "score_delta_original_minus_alternate": 0.14,
            "adjusted_z_threshold_passed": True,
        },
    )
    selected, decision, alternate_run = cli.choose_hand_by_scout(
        prep,
        ds,
        args,
        "python",
        tmp_path / "logs",
        tmp_path / "out",
        active_run,
        scout_max_rows=500,
    )

    assert selected == "alternate"
    assert len(commands) == 4
    grid_original_cmd, grid_alternate_cmd, original_cmd, alternate_cmd = commands
    assert "--alternate-hand" not in grid_original_cmd
    assert "--alternate-hand" in grid_alternate_cmd
    assert "--alternate-hand" not in original_cmd
    assert "--alternate-hand" in alternate_cmd

    def normalized(cmd):
        result = list(cmd)
        result.remove("--alternate-hand") if "--alternate-hand" in result else None
        for flag in ("--out-dir", "--prediction-cache-dir"):
            index = result.index(flag)
            del result[index : index + 2]
        return result

    assert normalized(grid_original_cmd) == normalized(grid_alternate_cmd)
    assert normalized(original_cmd) == normalized(alternate_cmd)
    assert (
        grid_original_cmd[grid_original_cmd.index("--thickness-grid") + 1]
        == args.thickness_grid
    )
    assert (
        original_cmd[original_cmd.index("--thickness-grid") + 1]
        == "1200"
    )
    for flag, value in (("--n-basis", "24"), ("--n-points", "10"), ("--max-rows", "500"), ("--split-seed", "731")):
        assert original_cmd[original_cmd.index(flag) + 1] == value
        assert alternate_cmd[alternate_cmd.index(flag) + 1] == value
    assert "--hkl-transform=-h,k,l" in original_cmd
    assert "--hkl-transform=-h,k,l" in alternate_cmd
    assert original_cmd[original_cmd.index("--scale-fit-split") + 1] == "work"
    assert alternate_cmd[alternate_cmd.index("--scale-fit-split") + 1] == "work"
    assert original_cmd[original_cmd.index("--lambda-grid") + 1] == ""
    assert alternate_cmd[alternate_cmd.index("--lambda-grid") + 1] == ""
    assert "--make-maps" not in original_cmd
    assert "--make-maps" not in alternate_cmd
    assert decision["comparison_contract"]["row_selection_match"] is True
    assert decision["comparison_contract"]["max_rows"] == 500
    assert decision["comparison_contract"]["scale_fit_split"] == "work"
    assert decision["comparison_contract"]["production_scale_preference_ignored_for_holdout"] is True
    assert decision["production_active_run"] == str(active_run)
    assert decision["production_active_score"]["dyn_wR2_all"] == pytest.approx(0.09)
    evidence = decision["comparison_contract"]["common_sigma_hand_evidence"]
    assert evidence["status"] == "replayed"
    assert decision["active_score"] == evidence["original_score"]
    assert decision["original_run"].endswith("hand_scout_original")
    assert alternate_run == tmp_path / "out" / "decisions" / "a" / "hand_scout_alt"


def test_choose_hand_rejects_switch_when_reported_scout_rows_do_not_match(tmp_path: Path, monkeypatch):
    model = tmp_path / "model.res"
    hkl = tmp_path / "rows.hkl"
    model.write_text("TITL test\nCELL 0.025079 5 6 7 90 90 90\nLATT -1\nEND\n")
    hkl.write_text("! rows\n")
    ds = DatasetSpec(name="a", hkl=str(hkl))
    prep = PrepConfig(structure=str(model), datasets=[ds])
    args = _auto_args(tmp_path, model, hkl)
    active_run = tmp_path / "out" / "runs" / "a" / "screen"
    active_run.mkdir(parents=True)
    (active_run / "summary.json").write_text(json.dumps(_native_summary(0.20, 0.25)) + "\n")

    def fake_run_logged(cmd, _logs, _label, **_kwargs):
        out_dir = Path(cmd[cmd.index("--out-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        if "--alternate-hand" in cmd:
            summary = _native_summary(0.15, 0.20, row_selection=(499, 449, 50))
        else:
            summary = _native_summary(0.22, 0.27, row_selection=(500, 450, 50))
        (out_dir / "summary.json").write_text(json.dumps(summary) + "\n")

    monkeypatch.setattr(cli, "run_logged", fake_run_logged)
    selected, decision, _alternate_run = cli.choose_hand_by_scout(
        prep,
        ds,
        args,
        "python",
        tmp_path / "logs",
        tmp_path / "out",
        active_run,
        scout_max_rows=500,
    )

    assert decision["scout_selected_before_quality_gate"] == "alternate"
    assert selected == "original"
    assert decision["quality_gate_status"] == "rejected_switch"
    assert decision["comparison_contract"]["row_selection_match"] is False
    assert any("identical selected/work/free row counts" in reason for reason in decision["quality_gate_reasons"])


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("zero_free", "same positive number of held-out free rows"),
        ("missing_free_metric", "finite raw free-row dynamical wR2 metrics"),
        ("partial_rows", "identical selected/work/free row counts"),
        ("malformed_rows", "identical selected/work/free row counts"),
        ("inconsistent_rows", "identical selected/work/free row counts"),
    ],
)
def test_choose_hand_requires_positive_rows_and_raw_free_metrics(
    tmp_path: Path,
    monkeypatch,
    case: str,
    expected_reason: str,
):
    model = tmp_path / "model.res"
    hkl = tmp_path / "rows.hkl"
    model.write_text("TITL test\nCELL 0.025079 5 6 7 90 90 90\nLATT -1\nEND\n")
    hkl.write_text("! rows\n")
    ds = DatasetSpec(name="a", hkl=str(hkl))
    prep = PrepConfig(structure=str(model), datasets=[ds])
    args = _auto_args(tmp_path, model, hkl)
    active_run = tmp_path / "out" / "runs" / "a" / "screen"
    active_run.mkdir(parents=True)
    (active_run / "summary.json").write_text(json.dumps(_native_summary(0.20, 0.25)) + "\n")

    def fake_run_logged(cmd, _logs, _label, **_kwargs):
        out_dir = Path(cmd[cmd.index("--out-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = (500, 500, 0) if case == "zero_free" else (500, 450, 50)
        summary = _native_summary(0.15, 0.20, row_selection=rows) if "--alternate-hand" in cmd else _native_summary(0.22, 0.27, row_selection=rows)
        if case == "missing_free_metric":
            summary["best_native"]["metrics"].pop("free")
        elif case == "partial_rows":
            summary["row_selection"] = {"free_rows": 50}
        elif case == "malformed_rows":
            summary["row_selection"] = {
                "selected_rows": "garbage",
                "work_rows": "garbage",
                "free_rows": 50,
            }
        elif case == "inconsistent_rows":
            summary["row_selection"] = {
                "selected_rows": 500,
                "work_rows": 450,
                "free_rows": 49,
            }
        (out_dir / "summary.json").write_text(json.dumps(summary) + "\n")

    monkeypatch.setattr(cli, "run_logged", fake_run_logged)
    selected, decision, _alternate_run = cli.choose_hand_by_scout(
        prep,
        ds,
        args,
        "python",
        tmp_path / "logs",
        tmp_path / "out",
        active_run,
        scout_max_rows=500,
    )

    assert decision["scout_selected_before_quality_gate"] == "alternate"
    assert selected == "original"
    assert decision["quality_gate_status"] == "rejected_switch"
    assert any(expected_reason in reason for reason in decision["quality_gate_reasons"])


def test_step_estimator_counts_scout_and_final_hand_pairs_per_multi_dataset(tmp_path: Path):
    model = tmp_path / "model.res"
    hkl = tmp_path / "rows.hkl"
    model.write_text("TITL test\nCELL 0.025079 5 6 7 90 90 90\nLATT -1\nEND\n")
    hkl.write_text("! rows\n")
    prep = PrepConfig(
        structure=str(model),
        datasets=[DatasetSpec(name="a", hkl=str(hkl)), DatasetSpec(name="b", hkl=str(hkl))],
    )
    args = _auto_args(tmp_path, model, hkl)
    args.trust_hkl_transform = True
    with_hand = cli.estimate_run_steps(prep, args)
    args.hand_control = False
    without_hand = cli.estimate_run_steps(prep, args)

    assert with_hand - without_hand == 6 * len(prep.datasets) + 1


def test_shared_hand_aggregation_requires_unanimous_quality_gated_support():
    args = argparse.Namespace(hand_accept_delta=0.015)
    result = cli.aggregate_shared_hand_decisions(
        {
            "dataset_b": _hand_decision(0.031),
            "dataset_a": _hand_decision(0.022),
        },
        args,
    )

    assert result["selected_hand"] == "alternate"
    assert result["quality_gate_status"] == "passed"
    assert result["alternate_support_count"] == 2
    assert result["aggregate_score_delta_original_minus_alternate"] == pytest.approx(0.0265)
    assert [row["dataset"] for row in result["dataset_evidence"]] == ["dataset_a", "dataset_b"]


@pytest.mark.parametrize(
    "blocking_decision",
    [
        _hand_decision(0.010, selected="original"),
        _hand_decision(0.030, selected="original", gate_status="rejected_switch"),
        _hand_decision(float("nan"), selected="original", warning="alternate scout failed"),
    ],
)
def test_shared_hand_aggregation_keeps_original_for_weak_mixed_or_failed_evidence(blocking_decision):
    args = argparse.Namespace(hand_accept_delta=0.015)
    result = cli.aggregate_shared_hand_decisions(
        {
            "supports_alternate": _hand_decision(0.030),
            "blocks_switch": blocking_decision,
        },
        args,
    )

    assert result["selected_hand"] == "original"
    assert result["quality_gate_status"] == "retained_original"
    assert result["alternate_support_count"] == 1
    assert any(reason.startswith("blocks_switch:") for reason in result["quality_gate_reasons"])


def test_shared_hand_scout_persists_dataset_evidence_and_actual_shared_choice(tmp_path: Path, monkeypatch):
    model = tmp_path / "model.res"
    model.write_text("TITL test\nCELL 0.025079 5 6 7 90 90 90\nLATT -1\nEND\n")
    hkl = tmp_path / "rows.hkl"
    hkl.write_text("! rows\n")
    prep = PrepConfig(
        structure=str(model),
        datasets=[DatasetSpec(name="a", hkl=str(hkl)), DatasetSpec(name="b", hkl=str(hkl))],
    )
    args = _auto_args(tmp_path, model, hkl)
    evidence = {
        "a": _hand_decision(0.030),
        "b": _hand_decision(0.005, selected="original"),
    }

    def fake_choose(_prep, ds, *_args, **_kwargs):
        return str(evidence[ds.name]["selected_hand"]), evidence[ds.name], tmp_path / f"alt_{ds.name}"

    monkeypatch.setattr(cli, "choose_hand_by_scout", fake_choose)
    selected, shared, _alternate_runs = cli.choose_shared_hand_by_scout(
        prep,
        args,
        "python",
        tmp_path / "logs",
        tmp_path / "out",
        {"a": tmp_path / "screen_a", "b": tmp_path / "screen_b"},
        {"a": 500, "b": 600},
    )

    assert selected == "original"
    assert shared["selected_hand"] == "original"
    saved_shared = json.loads((tmp_path / "out" / "decisions" / "shared_absolute_structure_decision.json").read_text())
    assert saved_shared["selected_hand"] == "original"
    saved_a = json.loads((tmp_path / "out" / "decisions" / "a" / "absolute_structure_decision.json").read_text())
    saved_b = json.loads((tmp_path / "out" / "decisions" / "b" / "absolute_structure_decision.json").read_text())
    assert saved_a["dataset_scout_selected_hand"] == "alternate"
    assert saved_a["selected_hand"] == "original"
    assert saved_b["dataset_scout_selected_hand"] == "original"
    assert saved_b["selected_hand"] == "original"


def test_shared_hand_dry_run_persists_original_without_executing_scouts(tmp_path: Path, monkeypatch):
    model = tmp_path / "model.res"
    hkl = tmp_path / "rows.hkl"
    model.write_text("TITL test\nCELL 0.025079 5 6 7 90 90 90\nLATT -1\nEND\n")
    hkl.write_text("! rows\n")
    prep = PrepConfig(structure=str(model), datasets=[DatasetSpec(name="a", hkl=str(hkl))])
    args = _auto_args(tmp_path, model, hkl)
    args.dry_run = True
    monkeypatch.setattr(
        cli,
        "choose_hand_by_scout",
        lambda *args, **kwargs: pytest.fail("dry-run must not execute a hand scout"),
    )

    selected, shared, alternate_runs = cli.choose_shared_hand_by_scout(
        prep,
        args,
        "python",
        tmp_path / "logs",
        tmp_path / "out",
        {"a": tmp_path / "screen_a"},
        {"a": 500},
    )

    assert selected == "original"
    assert alternate_runs == {}
    assert shared["status"] == "skipped_dry_run"
    saved = json.loads((tmp_path / "out" / "decisions" / "shared_absolute_structure_decision.json").read_text())
    assert saved["selected_model"] == str(model)


def _stub_multi_tail(monkeypatch, observed: dict[str, object]) -> None:
    monkeypatch.setattr(cli, "best_nuisance", lambda _run: {"thickness_A": 1000.0})
    monkeypatch.setattr(cli, "resolved_frame_offset_candidates", lambda _run: [])
    monkeypatch.setattr(cli, "continuous_thickness_enabled", lambda _args: False)
    monkeypatch.setattr(
        cli,
        "discrete_thickness_promotion_record",
        lambda run, **_kwargs: {"family": "discrete_thickness_polish", "status": "accepted", "run": str(run)},
    )
    monkeypatch.setattr(cli, "run_dataset_policy_gate", lambda *args, **kwargs: {"status": "disabled"})
    monkeypatch.setattr(cli, "write_datasets_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "run_scale_window_polish", lambda *args, **kwargs: {})
    monkeypatch.setattr(cli, "run_resolution_limit_gate", lambda *args, **kwargs: {})
    monkeypatch.setattr(cli, "run_forensic_gate", lambda *args, **kwargs: {})
    monkeypatch.setattr(cli, "run_sg_window_scout", lambda *args, **kwargs: {})

    def fake_combine(_python, _runs, structure, _out, _args, **_kwargs):
        observed.setdefault("combine_models", []).append(structure)
        return ["combine"]

    monkeypatch.setattr(cli, "combine_cmd", fake_combine)
    monkeypatch.setattr(cli, "run_logged", lambda *args, **kwargs: None)


def test_multi_defers_hand_until_after_all_nonfast_model_stages(tmp_path: Path, monkeypatch):
    model = tmp_path / "model.res"
    alternate = tmp_path / "model_alternate.res"
    hkl = tmp_path / "rows.hkl"
    model.write_text("TITL input\nCELL 0.025079 5 6 7 90 90 90\nLATT -1\nEND\n")
    alternate.write_text("TITL alternate\nCELL 0.025079 5 6 7 90 90 90\nLATT -1\nEND\n")
    hkl.write_text("! rows\n")
    prep = PrepConfig(
        structure=str(model),
        datasets=[DatasetSpec(name="a", hkl=str(hkl)), DatasetSpec(name="b", hkl=str(hkl))],
    )
    args = _auto_args(tmp_path, model, hkl)
    args.auto_decide = False
    observed: dict[str, object] = {
        "native": [],
        "features": [],
        "feedback": [],
        "materializations": [],
        "thickness_contracts": [],
    }

    def fake_native(_python, ds, structure, _baseline, _out, _args, *, stage, **kwargs):
        observed["native"].append((ds.name, stage, structure))
        if stage == "multi_thickness_polish":
            observed["thickness_contracts"].append(kwargs)
        return ["native", stage]

    def fake_shared(*_args, **_kwargs):
        raise AssertionError("pre-refinement shared hand scout must not run")

    def fake_feature(_prep, ds, _args, _python, _logs, _root, *, family, active_model, active_run, active_nui):
        observed["features"].append((ds.name, family, active_model))
        return active_nui, active_run, {"family": family, "status": "rejected"}

    def fake_feedback(*_args, start_model, **_kwargs):
        observed["feedback"].append(start_model)
        return start_model, {"status": "ok"}

    def fake_materialize(_python, structure, _root, _logs, _dry_run):
        observed["materializations"].append(structure)
        return alternate

    monkeypatch.setattr(cli, "native_stage_cmd", fake_native)
    monkeypatch.setattr(cli, "choose_shared_hand_by_scout", fake_shared)
    monkeypatch.setattr(cli, "make_alternate_hand_model", fake_materialize)
    monkeypatch.setattr(cli, "run_feature_gate", fake_feature)
    monkeypatch.setattr(cli, "resolve_structure_update_strategy", lambda *args, **kwargs: "shelxl_feedback")
    monkeypatch.setattr(cli, "run_multi_shelxl_feedback_loop", fake_feedback)
    monkeypatch.setattr(cli, "resolve_final_n_basis", lambda *args, **kwargs: 24)
    _stub_multi_tail(monkeypatch, observed)

    result = cli.run_multi(prep, args, "python", tmp_path / "logs")

    input_screen = [row for row in observed["native"] if row[1] == "screen"]
    downstream = [row for row in observed["native"] if row[1] in {"multi_thickness_polish", "final"}]
    assert input_screen and all(row[2] == str(model) for row in input_screen)
    assert downstream and all(row[2] == str(model) for row in downstream)
    assert observed["features"] and all(row[2] == str(model) for row in observed["features"])
    assert observed["feedback"] == [str(model)]
    assert observed["materializations"] == []
    assert observed["combine_models"] == [str(model)]
    assert observed["thickness_contracts"]
    assert all(contract["scale_fit_split_override"] == "work" for contract in observed["thickness_contracts"])
    assert all(contract["selection_metric_override"] == "work_wR2" for contract in observed["thickness_contracts"])
    assert all(contract["promotion_baseline"]["thickness_A"] == 1000.0 for contract in observed["thickness_contracts"])
    assert result["selected_hand"] == "original"
    assert result["model"] == str(model)


def test_multi_fast_does_not_run_a_pre_refinement_hand_scout(tmp_path: Path, monkeypatch):
    model = tmp_path / "model.res"
    alternate = tmp_path / "model_alternate.res"
    hkl = tmp_path / "rows.hkl"
    model.write_text("TITL input\nCELL 0.025079 5 6 7 90 90 90\nLATT -1\nEND\n")
    alternate.write_text("TITL alternate\nCELL 0.025079 5 6 7 90 90 90\nLATT -1\nEND\n")
    hkl.write_text("! rows\n")
    prep = PrepConfig(
        structure=str(model),
        datasets=[DatasetSpec(name="a", hkl=str(hkl)), DatasetSpec(name="b", hkl=str(hkl))],
    )
    args = _auto_args(tmp_path, model, hkl)
    args.auto_decide = False
    args.fast = True
    observed: dict[str, object] = {"native": []}

    def fake_native(_python, ds, structure, _baseline, out, _args, *, stage, **_kwargs):
        observed["native"].append((ds.name, stage, structure, str(out)))
        return ["native", stage]

    monkeypatch.setattr(cli, "native_stage_cmd", fake_native)
    monkeypatch.setattr(
        cli,
        "choose_shared_hand_by_scout",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("pre-refinement shared hand scout must not run")
        ),
    )
    monkeypatch.setattr(cli, "make_alternate_hand_model", lambda *args, **kwargs: alternate)
    monkeypatch.setattr(cli, "resolve_structure_update_strategy", lambda *args, **kwargs: "off")
    _stub_multi_tail(monkeypatch, observed)

    result = cli.run_multi(prep, args, "python", tmp_path / "logs")

    selected_runs = [row for row in observed["native"] if row[1] == "multi_fast_selected_hand"]
    assert selected_runs == []
    assert observed["combine_models"] == [str(model)]
    assert result["runs"]["a"].endswith("fast_N20_Q5")


def _final_hand_summary(
    n_basis: int,
    n_points: int,
    dyn_wr2: float,
    free_wr2: float,
    model: Path,
    hkl: Path,
    row_table: Path | None,
) -> dict[str, object]:
    summary = {
        "artifacts": (
            {
                "native_row_table": str(row_table.resolve()),
                "native_row_table_sha256": cli.sha256_file(row_table),
            }
            if row_table is not None and row_table.is_file()
            else {}
        ),
        "inputs": {
            "res": str(model.resolve()),
            "res_sha256": cli.sha256_file(model),
            "model_tag": "original;hkl_transform=identity",
            "hkl": str(hkl.resolve()),
            "hkl_sha256": cli.sha256_file(hkl),
            "hkl_transform": "identity",
        },
        "config": {
            "res": str(model.resolve()),
            "alternate_hand": False,
            "n_basis": n_basis,
            "n_points": n_points,
            "thickness_points": 7,
            "scale_fit_split": "work",
            "sg_window": 0.004,
            "sg_window_mode": "fixed",
            "sg_jitter_mode_grid": "none",
            "sg_window_min": 0.003,
            "sg_window_max": 0.012,
            "sg_window_scale": 1.0,
            "sg_window_frame_floor": 0.5,
            "profile_quadrature": "gauss_legendre",
            "profile_accumulation": "sum",
            "topup_d_min": 0.7,
            "max_support_sg": 0.005,
            "free_fraction": 0.10,
            "split_seed": 42,
            "scale_smoothness": 0.0,
            "target_mode": "iobs",
            "positive_only": True,
            "hkl_transform": "identity",
            "min_d_spacing": 0.7,
            "min_peak": 90.0,
            "geometry_percentile": 99.0,
            "compute_backend": "numpy",
            "bethe_strong_count": 0,
            "bethe_variant": "full",
            "bethe_promote_denom": 0.0,
            "bethe_denom_eps": 1e-8,
            "matrix_convention": "j_minus_i",
            "max_rows": None,
        },
        "row_selection": {"selected_rows": 4, "work_rows": 2, "free_rows": 2},
        "best_native": {
            "nuisance": {
                "thickness_A": 1234.0,
                "thickness_model": "uniform_gradient",
                "thickness_grad_frac": 0.25,
                "frame_offset": 0.125,
                "group_width_deg": 1.0,
                "detector_scale_bins": 4,
                "sg_jitter_mode": "none",
                "sg_jitter_student_df": 5.5,
                "sg_jitter_width_factor": 2.5,
                "dose_b_ref_frame": 17.0,
                "native_wght_a": 0.1,
                "native_wght_b": 0.2,
                "matrix_convention": "j_minus_i",
            },
            "metrics": {
                "all": {
                    "dyn": {"wR2_all": dyn_wr2, "R1_all": dyn_wr2 * 0.8, "R1_gt": dyn_wr2 * 0.6, "CC": 0.95},
                    "born": {"R1_gt": 0.25, "CC": 0.90},
                },
                "work": {"dyn": {"wR2_all": dyn_wr2 * 0.9}},
                "free": {"dyn": {"wR2_all": free_wr2}},
            },
        },
    }
    return summary


def _write_final_hand_rows(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row_index",
        "h",
        "k",
        "l",
        "d_A",
        "target_I",
        "sigma_eff",
        "xds_sigma",
        "peak",
        "corr",
        "rlp",
        "group",
        "detector_bin",
        "basis_size",
        "center_sg_inv_A",
        "omega_bragg_deg",
        "omega_bragg_minus_center_deg",
        "sg_window_inv_A",
        "sg_window_raw_inv_A",
        "sg_window_half_frames",
        "sg_window_clipped",
        "sg_dSg_domega_inv_A_per_deg",
        "sg_dSg_dframe_inv_A",
        "sg_jitter_sigma_inv_A",
        "sg_jitter_d_inv_A",
        "sg_jitter_frame_fraction",
        "sg_jitter_kernel_points",
        "sg_profile_points_total",
        "sg_profile_weight_sum",
        "max_sg_solve_residual",
        "split",
        "row_resolution_factor",
        "dyn_residual_over_sigma",
    ]
    base = {
        "d_A": 0.8,
        "target_I": 100,
        "sigma_eff": 5,
        "xds_sigma": 4,
        "peak": 95,
        "corr": 80,
        "rlp": 1.2,
        "group": 0,
        "detector_bin": 0,
        "basis_size": 32,
        "center_sg_inv_A": 0.001,
        "omega_bragg_deg": 1.5,
        "omega_bragg_minus_center_deg": 0.1,
        "sg_window_inv_A": 0.004,
        "sg_window_raw_inv_A": 0.004,
        "sg_window_half_frames": 0.5,
        "sg_window_clipped": 0,
        "sg_dSg_domega_inv_A_per_deg": 0.002,
        "sg_dSg_dframe_inv_A": 0.0002,
        "sg_jitter_sigma_inv_A": 0.0,
        "sg_jitter_d_inv_A": 1.25,
        "sg_jitter_frame_fraction": 0.1,
        "sg_jitter_kernel_points": 1,
        "sg_profile_points_total": 14,
        "sg_profile_weight_sum": 1.0,
        "max_sg_solve_residual": 1e-8,
        "split": "work",
        "row_resolution_factor": 1.0,
        "dyn_residual_over_sigma": 0.5,
    }
    rows = []
    for row_index, hkl, updates in (
        (0, (1, 0, 0), {}),
        (1, (0, 1, 0), {"d_A": 0.9, "target_I": 120, "dyn_residual_over_sigma": -0.7, "detector_bin": 1}),
        (2, (0, 0, 1), {"d_A": 1.0, "target_I": 90, "split": "free", "group": 1, "dyn_residual_over_sigma": 0.8}),
        (3, (1, 1, 0), {"d_A": 1.1, "target_I": 80, "split": "free", "group": 1, "detector_bin": 1, "dyn_residual_over_sigma": -0.9}),
    ):
        rows.append({**base, **updates, "row_index": row_index, "h": hkl[0], "k": hkl[1], "l": hkl[2]})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_final_multi_hand_selection_uses_one_elastic_pair_and_switches_without_nuisance_grid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = tmp_path / "model.res"
    inverse = tmp_path / "model_alternate.res"
    hkl = tmp_path / "rows.hkl"
    model.write_text("TITL input\nCELL 0.025079 5 6 7 90 90 90\nLATT -1\nEND\n")
    inverse.write_text("TITL inverse\nCELL 0.025079 5 6 7 90 90 90\nLATT -1\nEND\n")
    hkl.write_text("! rows\n")
    source_run = tmp_path / "runs" / "a" / "final"
    source_rows = source_run / "native_dyn_calc.csv"
    _write_final_hand_rows(source_rows)
    source_summary = _final_hand_summary(32, 14, 0.30, 0.32, model, hkl, source_rows)
    source_summary["best_native"]["nuisance"].update(
        {
            "absorption_mode": "midgley_like",
            "absorption_mu": 0.25,
            "absorption_frac": 0.1,
        }
    )
    source_run.mkdir(parents=True, exist_ok=True)
    (source_run / "summary.json").write_text(json.dumps(source_summary) + "\n")
    prep = PrepConfig(
        structure=str(model),
        datasets=[DatasetSpec(name="a", hkl=str(hkl), min_peak=90.0)],
    )
    args = _auto_args(tmp_path, model, hkl)
    args.hand_min_z_all = 2.0
    args.hand_min_z_free = 0.5
    args.hand_min_free_rows = 2
    args.hand_accept_delta = 0.001
    commands: list[list[str]] = []

    def fake_logged(command, _logs, label, *args, **kwargs):
        commands.append(list(command))
        if label.endswith("elastic_hand_pair"):
            pair_dir = Path(command[command.index("--out-dir") + 1])
            pair_dir.mkdir(parents=True, exist_ok=True)
            _write_final_hand_rows(pair_dir / "native_dyn_calc.csv")
            _write_final_hand_rows(pair_dir / "native_inverse_hand_dyn_calc.csv")
            inverse_path = pair_dir / "native_inverse_hand_dyn_calc.csv"
            with inverse_path.open(newline="") as handle:
                inverse_rows = list(csv.DictReader(handle))
                fieldnames = list(inverse_rows[0])
            for row in inverse_rows:
                row["dyn_residual_over_sigma"] = "0.01"
            with inverse_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(inverse_rows)
            pair_summary = _final_hand_summary(
                32,
                14,
                0.30,
                0.32,
                model,
                hkl,
                pair_dir / "native_dyn_calc.csv",
            )
            pair_summary["paired_inverse_hand"] = {
                "status": "complete",
                "selected_metrics": pair_summary["best_native"]["metrics"],
                "inverse_metrics": _native_summary(0.20, 0.21)["best_native"]["metrics"],
            }
            (pair_dir / "summary.json").write_text(json.dumps(pair_summary) + "\n")
        elif label.endswith("elastic_hand_diagnostic"):
            out = Path(command[command.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "all": {"n": 4, "paired_row_t_delta_chi2": -4.0},
                        "work": {"n": 2, "paired_row_t_delta_chi2": -3.0},
                        "free": {"n": 2, "paired_row_t_delta_chi2": -2.0},
                    }
                )
                + "\n"
            )

    monkeypatch.setattr(cli, "run_logged", fake_logged)
    monkeypatch.setattr(cli, "make_alternate_hand_model", lambda *args, **kwargs: inverse)

    result = cli.run_multi_final_elastic_hand_selection(
        prep,
        args,
        "python",
        tmp_path / "logs",
        tmp_path,
        final_model=str(model),
        selected_hand="original",
        final_runs={"a": source_run},
        shared_hand_decision={"symmetry_policy": cli.absolute_structure_policy(str(model))},
    )

    pair_command = commands[0]
    assert "--paired-inverse-hand" in pair_command
    assert pair_command[pair_command.index("--absorption-mode-grid") + 1] == "none"
    assert float(pair_command[pair_command.index("--absorption-mu-grid") + 1]) == 0.0
    assert pair_command[pair_command.index("--sg-jitter-mode-grid") + 1] == "none"
    assert pair_command[pair_command.index("--n-basis") + 1] == "32"
    assert pair_command[pair_command.index("--n-points") + 1] == "14"
    assert pair_command[pair_command.index("--compute-backend") + 1] == "scipy"
    assert result["hand_switched"] is True
    assert result["selected_hand_after"] == "alternate"
    assert result["selected_model_after"] == str(inverse)
    assert result["nuisance_refit"] is False
    assert result["jitter_refit"] is False


def test_multi_post_feedback_hand_diagnostic_pins_final_contract_and_warns_without_switching(
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "out"
    model = tmp_path / "final.res"
    opposite_model = tmp_path / "final_alternate.res"
    hkl = tmp_path / "rows.hkl"
    model.write_text("TITL final\nCELL 0.025079 5 6 7 90 90 90\nLATT -1\nEND\n")
    opposite_model.write_text("TITL opposite\nCELL 0.025079 5 6 7 90 90 90\nLATT -1\nEND\n")
    hkl.write_text("! rows\n")
    prep = PrepConfig(
        structure=str(model),
        datasets=[DatasetSpec(name="a", hkl=str(hkl)), DatasetSpec(name="b", hkl=str(hkl))],
    )
    args = _auto_args(tmp_path, model, hkl)
    final_runs = {"a": root / "runs" / "a" / "final", "b": root / "runs" / "b" / "final"}
    args.hand_min_free_rows = 2
    specs = {"a": (32, 14, 0.10, 0.12), "b": (40, 18, 0.45, 0.48)}
    for name, run in final_runs.items():
        run.mkdir(parents=True)
        n_basis, n_points, dyn_wr2, free_wr2 = specs[name]
        _write_final_hand_rows(run / "native_dyn_calc.csv")
        (run / "summary.json").write_text(
            json.dumps(
                _final_hand_summary(
                    n_basis,
                    n_points,
                    dyn_wr2,
                    free_wr2,
                    model,
                    hkl,
                    run / "native_dyn_calc.csv",
                )
            )
            + "\n"
        )

    monkeypatch.setattr(cli, "make_alternate_hand_model", lambda *args, **kwargs: opposite_model)
    native_commands: dict[str, list[str]] = {}

    def fake_run_logged(cmd, _logs, label, **_kwargs):
        command = list(cmd)
        if "absolute_structure_diagnostic.py" in command[1]:
            name = label.split("_multi_post_feedback", 1)[0]
            out = Path(command[command.index("--out") + 1])
            sign = 1.0 if name == "a" else -1.0
            stats = {
                "status": "ok",
                "row_join": {"n_best": 4, "n_other": 4, "n_joined": 4},
                "all": {
                    "n": 4,
                    "delta_chi2_other_minus_best": sign * 25.0,
                    "paired_row_t_delta_chi2": sign * 5.0,
                    "klar_palatinus_adjusted_hand_z": {"z": -sign * 999.0},
                },
                "work": {"n": 2, "delta_chi2_other_minus_best": sign * 16.0, "paired_row_t_delta_chi2": sign * 4.0},
                "free": {
                    "n": 2,
                    "delta_chi2_other_minus_best": sign * 9.0,
                    "paired_row_t_delta_chi2": sign * 3.0,
                    "klar_palatinus_adjusted_hand_z": {"z": -sign * 999.0},
                },
            }
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(stats) + "\n")
            return
        name = label.split("_multi_post_feedback", 1)[0]
        native_commands[name] = command
        out = Path(command[command.index("--out-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        n_basis = int(command[command.index("--n-basis") + 1])
        n_points = int(command[command.index("--n-points") + 1])
        dyn_wr2, free_wr2 = (0.20, 0.22) if name == "a" else (0.10, 0.12)
        _write_final_hand_rows(out / "native_dyn_calc.csv")
        (out / "summary.json").write_text(
            json.dumps(
                _final_hand_summary(
                    n_basis,
                    n_points,
                    dyn_wr2,
                    free_wr2,
                    opposite_model,
                    hkl,
                    out / "native_dyn_calc.csv",
                )
            )
            + "\n"
        )

    monkeypatch.setattr(cli, "run_logged", fake_run_logged)

    payload = cli.run_multi_post_feedback_hand_diagnostic(
        prep,
        args,
        "python",
        root / "logs",
        root,
        final_model=str(model),
        selected_hand="original",
        final_runs=final_runs,
        publication_runs=final_runs,
        shared_hand_decision={"symmetry_policy": {"run_hand_control": True}},
    )

    assert payload["status"] == "warning_opposite_hand_support"
    assert payload["selected_hand_after_diagnostic"] == "original"
    assert payload["hand_switched"] is False
    assert payload["merged_z"] is None
    assert payload["datasets"]["a"]["status"] == "consistent_no_material_opposite_support"
    assert payload["datasets"]["b"]["status"] == "warning_material_opposite_support"
    assert payload["datasets"]["b"]["materially_favors_opposite"] is True
    assert payload["datasets"]["b"]["paired_t_style_z_positive_favors_selected"]["free"] == pytest.approx(-3.0)
    assert payload["datasets"]["b"]["paired_t_style_positive_favors_selected"]["free"] == {
        "t": pytest.approx(-3.0),
        "n": 2,
        "df": 1,
    }
    assert payload["datasets"]["a"]["row_counts"] == {
        "selected": 4,
        "opposite": 4,
        "joined": 4,
        "all": 4,
        "work": 2,
        "free": 2,
    }
    assert payload["datasets"]["a"]["comparison_contract"]["same_final_shared_model"] is True
    assert payload["datasets"]["a"]["comparison_contract"]["selected_model_binding"]["passed"] is True
    assert payload["datasets"]["a"]["comparison_contract"]["selected_hkl_binding"]["passed"] is True
    assert payload["datasets"]["a"]["comparison_contract"]["source_row_table_binding"]["passed"] is True
    assert payload["datasets"]["a"]["comparison_contract"]["policy_selected_table_binding"]["passed"] is True
    assert payload["datasets"]["a"]["comparison_contract"]["opposite_model_binding"]["passed"] is True
    assert payload["datasets"]["a"]["comparison_contract"]["opposite_hkl_binding"]["passed"] is True
    assert payload["datasets"]["a"]["comparison_contract"]["opposite_row_table_binding"]["passed"] is True
    assert payload["datasets"]["a"]["comparison_contract"]["physics_config_contract"]["passed"] is True
    assert payload["datasets"]["a"]["comparison_contract"]["model_hand_semantics_aligned"] is True
    assert payload["datasets"]["a"]["row_pair_contract"]["split_mismatch_count"] == 0
    assert payload["datasets"]["a"]["row_pair_contract"]["duplicate_keys_selected"] == 0
    for name, (n_basis, n_points, *_rest) in specs.items():
        command = native_commands[name]
        assert command[command.index("--n-basis") + 1] == str(n_basis)
        assert command[command.index("--n-points") + 1] == str(n_points)
        assert command[command.index("--thickness-grid") + 1] == "1234"
        assert command[command.index("--scale-fit-split") + 1] == "work"
        assert command[command.index("--lambda-grid") + 1] == ""
        assert command[command.index("--sg-jitter-student-df") + 1] == "5.5"
        assert command[command.index("--sg-jitter-width-factor") + 1] == "2.5"
        assert command[command.index("--dose-b-ref-frame") + 1] == "17"
        assert command[command.index("--wght-a-grid") + 1] == "0.1"
        assert command[command.index("--wght-b-grid") + 1] == "0.2"
        assert "--make-maps" not in command
    artifact = json.loads((root / "decisions" / "shared_post_feedback_hand_diagnostic.json").read_text())
    assert artifact["status"] == payload["status"]
    full_lines = "\n".join(cli._quick_multi_hand_evidence_lines(payload))
    compact_lines = "\n".join(cli._quick_multi_hand_evidence_lines(payload, compact=True))
    assert "no pooled statistic" in full_lines.lower()
    assert "a | consistent_no_material_opposite_support | 4/2/2 | 5.000/4.000/3.000" in compact_lines


def test_multi_post_feedback_hand_diagnostic_fails_closed_when_final_rows_are_missing(
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "out"
    model = tmp_path / "final.res"
    hkl = tmp_path / "rows.hkl"
    model.write_text("TITL final\nCELL 0.025079 5 6 7 90 90 90\nLATT -1\nEND\n")
    hkl.write_text("! rows\n")
    prep = PrepConfig(structure=str(model), datasets=[DatasetSpec(name="a", hkl=str(hkl))])
    args = _auto_args(tmp_path, model, hkl)
    final = root / "runs" / "a" / "final"
    final.mkdir(parents=True)
    (final / "summary.json").write_text(
        json.dumps(_final_hand_summary(32, 14, 0.1, 0.12, model, hkl, None)) + "\n"
    )
    monkeypatch.setattr(
        cli,
        "make_alternate_hand_model",
        lambda *args, **kwargs: pytest.fail("missing final rows must stop before model materialization"),
    )

    payload = cli.run_multi_post_feedback_hand_diagnostic(
        prep,
        args,
        "python",
        root / "logs",
        root,
        final_model=str(model),
        selected_hand="original",
        final_runs={"a": final},
        publication_runs={"a": final},
        shared_hand_decision={"symmetry_policy": {"run_hand_control": True}},
    )

    assert payload["status"] == "warning_incomplete"
    assert payload["consistency_gate"]["passed"] is False
    assert payload["datasets"]["a"]["status"] == "warning_incomplete"
    assert "selected row table is missing" in payload["datasets"]["a"]["warnings"][0]


def test_multi_post_feedback_hand_diagnostic_centrosymmetric_policy_precedes_disabled_flag(
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "out"
    model = tmp_path / "final.res"
    hkl = tmp_path / "rows.hkl"
    model.write_text("TITL centric\nCELL 0.025079 5 6 7 90 90 90\nLATT 1\nEND\n")
    hkl.write_text("! rows\n")
    prep = PrepConfig(structure=str(model), datasets=[DatasetSpec(name="a", hkl=str(hkl))])
    args = _auto_args(tmp_path, model, hkl)
    args.hand_control = False
    monkeypatch.setattr(
        cli,
        "make_alternate_hand_model",
        lambda *args, **kwargs: pytest.fail("centric models must not materialize an alternate hand"),
    )

    payload = cli.run_multi_post_feedback_hand_diagnostic(
        prep,
        args,
        "python",
        root / "logs",
        root,
        final_model=str(model),
        selected_hand="original",
        final_runs={},
        publication_runs={},
        shared_hand_decision={
            "symmetry_policy": {
                "run_hand_control": False,
                "reason": "SHELX LATT is centrosymmetric; alternate hand check is not meaningful",
            }
        },
    )

    assert payload["status"] == "skipped_not_applicable"
    assert payload["consistency_gate"]["status"] == "not_applicable"
    assert payload["consistency_gate"]["passed"] is None
    assert "centrosymmetric" in payload["consistency_gate"]["warnings"][0]


def test_multi_post_feedback_hand_diagnostic_rejects_unbound_selected_model(
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "out"
    model = tmp_path / "final.res"
    hkl = tmp_path / "rows.hkl"
    model.write_text("TITL final\nCELL 0.025079 5 6 7 90 90 90\nLATT -1\nEND\n")
    hkl.write_text("! rows\n")
    prep = PrepConfig(structure=str(model), datasets=[DatasetSpec(name="a", hkl=str(hkl))])
    args = _auto_args(tmp_path, model, hkl)
    final = root / "runs" / "a" / "final"
    final.mkdir(parents=True)
    _write_final_hand_rows(final / "native_dyn_calc.csv")
    summary = _final_hand_summary(
        32,
        14,
        0.1,
        0.12,
        model,
        hkl,
        final / "native_dyn_calc.csv",
    )
    summary["inputs"]["res_sha256"] = "not-the-final-model-hash"
    (final / "summary.json").write_text(json.dumps(summary) + "\n")
    monkeypatch.setattr(
        cli,
        "make_alternate_hand_model",
        lambda *args, **kwargs: pytest.fail("unbound selected model must stop before opposite materialization"),
    )

    payload = cli.run_multi_post_feedback_hand_diagnostic(
        prep,
        args,
        "python",
        root / "logs",
        root,
        final_model=str(model),
        selected_hand="original",
        final_runs={"a": final},
        publication_runs={"a": final},
        shared_hand_decision={"symmetry_policy": {"run_hand_control": True}},
    )

    record = payload["datasets"]["a"]
    assert payload["status"] == "warning_incomplete"
    assert record["status"] == "warning_incomplete"
    assert record["comparison_contract"]["same_final_shared_model"] is False
    assert any("res_sha256 does not match" in warning for warning in record["warnings"])


def test_model_and_row_pair_contracts_reject_hidden_inversion_split_drift_and_duplicates(tmp_path: Path):
    selected_model = tmp_path / "selected.res"
    opposite_model = tmp_path / "opposite.res"
    selected_model.write_text("selected bytes\n")
    opposite_model.write_text("opposite bytes\n")
    summary = {
        "inputs": {
            "res": str(opposite_model.resolve()),
            "res_sha256": cli.sha256_file(opposite_model),
            "model_tag": "alternate_fractional_xyz;hkl_transform=identity",
        },
        "config": {"alternate_hand": True},
    }

    binding = cli._summary_model_binding(summary, opposite_model, semantic_hand="alternate")

    assert binding["passed"] is False
    assert any("additional engine-side hand transformation" in error for error in binding["errors"])
    assert any("unexpected second hand inversion" in error for error in binding["errors"])

    selected_rows = tmp_path / "selected.csv"
    opposite_rows = tmp_path / "opposite.csv"
    _write_final_hand_rows(selected_rows)
    _write_final_hand_rows(opposite_rows)
    with opposite_rows.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        opposite_payload = list(reader)
    opposite_payload[0]["split"] = "free"
    opposite_payload[0]["target_I"] = "101"
    opposite_payload[1]["xds_sigma"] = "nan"
    opposite_payload.insert(1, dict(opposite_payload[0]))
    opposite_payload[1]["dyn_residual_over_sigma"] = "0.6"
    with opposite_rows.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(opposite_payload)

    row_contract = cli._paired_row_split_contract(selected_rows, opposite_rows)

    assert row_contract["passed"] is False
    assert row_contract["split_mismatch_count"] == 1
    assert row_contract["duplicate_keys_opposite"] == 1
    assert row_contract["numeric_mismatch_counts"]["target_I"] == 1
    assert row_contract["nonfinite_counts"]["xds_sigma"] == 1


def test_row_pair_contract_accepts_matching_optional_geometry_nan_sentinels(tmp_path: Path):
    selected_rows = tmp_path / "selected.csv"
    opposite_rows = tmp_path / "opposite.csv"
    nullable = {
        "sg_window_half_frames",
        "sg_jitter_d_inv_A",
        "sg_jitter_frame_fraction",
    }
    for path in (selected_rows, opposite_rows):
        _write_final_hand_rows(path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        for row in rows:
            for field in nullable:
                row[field] = "nan"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    row_contract = cli._paired_row_split_contract(
        selected_rows,
        opposite_rows,
        nullable_numeric_fields=nullable,
    )

    assert row_contract["passed"] is True
    assert set(row_contract["nullable_numeric_columns"]) == nullable
    assert all(row_contract["nonfinite_counts"][field] == 0 for field in nullable)
    assert all(row_contract["matched_nullable_nan_counts"][field] == 4 for field in nullable)


@pytest.mark.parametrize("bad_token", ["", "not-a-number", "inf", "-inf"])
def test_row_pair_contract_never_treats_corrupt_tokens_as_nullable_nan(
    tmp_path: Path,
    bad_token: str,
):
    selected_rows = tmp_path / "selected.csv"
    opposite_rows = tmp_path / "opposite.csv"
    for path in (selected_rows, opposite_rows):
        _write_final_hand_rows(path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        rows[0]["sg_window_half_frames"] = bad_token
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    row_contract = cli._paired_row_split_contract(
        selected_rows,
        opposite_rows,
        nullable_numeric_fields={"sg_window_half_frames"},
    )

    assert row_contract["passed"] is False
    assert row_contract["nonfinite_counts"]["sg_window_half_frames"] == 1


def test_row_pair_contract_requires_physics_mode_to_allow_nan(tmp_path: Path):
    selected_rows = tmp_path / "selected.csv"
    opposite_rows = tmp_path / "opposite.csv"
    for path in (selected_rows, opposite_rows):
        _write_final_hand_rows(path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        rows[0]["sg_window_half_frames"] = "nan"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    row_contract = cli._paired_row_split_contract(selected_rows, opposite_rows)

    assert row_contract["passed"] is False
    assert row_contract["nonfinite_counts"]["sg_window_half_frames"] == 1


def test_row_contract_nullable_fields_follow_recorded_physics_modes():
    assert cli._row_contract_nullable_numeric_fields(
        {"sg_window_mode": "fixed"},
        {"sg_jitter_mode": "none"},
    ) == {
        "sg_window_half_frames",
        "sg_jitter_d_inv_A",
        "sg_jitter_frame_fraction",
    }
    assert cli._row_contract_nullable_numeric_fields(
        {"sg_window_mode": "zobs_residual"},
        {"sg_jitter_mode": "gaussian"},
    ) == set()


def test_native_row_artifact_hkl_and_source_hash_bindings_reject_stale_bytes(tmp_path: Path):
    from grond.legacy.engine.native_row_regularize import native_row_table_artifact

    model = tmp_path / "model.res"
    hkl = tmp_path / "rows.hkl"
    row_table = tmp_path / "native_dyn_calc.csv"
    model.write_text("model bytes\n")
    hkl.write_text("hkl bytes\n")
    _write_final_hand_rows(row_table)
    producer_artifact = native_row_table_artifact(row_table)
    summary = _final_hand_summary(32, 14, 0.1, 0.12, model, hkl, row_table)

    assert producer_artifact == summary["artifacts"]
    assert cli._summary_hkl_binding(summary, hkl)["passed"] is True
    assert cli._summary_native_row_table_binding(summary, row_table)["passed"] is True

    hkl.write_text("mutated hkl bytes\n")
    row_table.write_text(row_table.read_text() + "\n")

    stale_hkl = cli._summary_hkl_binding(summary, hkl)
    stale_rows = cli._summary_native_row_table_binding(summary, row_table)
    assert stale_hkl["passed"] is False
    assert any("hkl_sha256" in error for error in stale_hkl["errors"])
    assert stale_rows["passed"] is False
    assert any("native_row_table_sha256" in error for error in stale_rows["errors"])


def test_policy_selected_table_binding_allows_only_policy_annotation_changes(tmp_path: Path):
    source = tmp_path / "source.csv"
    selected = tmp_path / "selected.csv"
    _write_final_hand_rows(source)
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        source_fields = list(reader.fieldnames or [])
        rows = list(reader)
    policy_fields = [
        "dataset",
        "dataset_role",
        "raw_row_weight",
        "endpoint_weight",
        "sigma_eff_original",
        "policy_sigma_weight_applied",
    ]
    for row in rows:
        row.update(
            {
                "dataset": "a",
                "dataset_role": "downweight",
                "raw_row_weight": "0.5",
                "endpoint_weight": "1.0",
                "sigma_eff_original": row["sigma_eff"],
                "policy_sigma_weight_applied": "false",
            }
        )
    with selected.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*source_fields, *policy_fields])
        writer.writeheader()
        writer.writerows(rows)

    accepted = cli._policy_selected_table_binding(source, selected)
    assert accepted["passed"] is True

    rows[0]["raw_row_weight"] = "0.25"
    rows[0]["dyn_residual_over_sigma"] = "999"
    with selected.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*source_fields, *policy_fields])
        writer.writeheader()
        writer.writerows(rows)

    rejected = cli._policy_selected_table_binding(source, selected)
    assert rejected["passed"] is False
    assert rejected["non_policy_mismatch_counts"] == {"dyn_residual_over_sigma": 1}


def test_native_physics_config_contract_rejects_hand_branch_drift(tmp_path: Path):
    model = tmp_path / "model.res"
    hkl = tmp_path / "rows.hkl"
    row_table = tmp_path / "native_dyn_calc.csv"
    model.write_text("model bytes\n")
    hkl.write_text("hkl bytes\n")
    _write_final_hand_rows(row_table)
    selected = _final_hand_summary(32, 14, 0.1, 0.12, model, hkl, row_table)
    opposite = json.loads(json.dumps(selected))

    assert cli._native_physics_config_contract(selected, opposite)["passed"] is True

    opposite["config"]["profile_quadrature"] = "uniform"
    opposite["config"]["sg_jitter_mode_grid"] = "gaussian"
    opposite["config"]["sg_window"] = 0.005
    drift = cli._native_physics_config_contract(selected, opposite)
    assert drift["passed"] is False
    assert "profile_quadrature" in drift["exact_mismatches"]
    assert "sg_jitter_mode_grid" in drift["exact_mismatches"]
    assert "sg_window" in drift["numeric_mismatches"]
