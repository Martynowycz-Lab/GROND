import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "src" / "grond" / "legacy" / "engine" / "absolute_structure_diagnostic.py"
)


def write_rows(path: Path, residuals: list[float]) -> None:
    fields = ["row_index", "h", "k", "l", "split", "dyn_residual_over_sigma"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, residual in enumerate(residuals):
            writer.writerow(
                {
                    "row_index": idx,
                    "h": 1,
                    "k": 0,
                    "l": idx,
                    "split": "free" if idx == 1 else "work",
                    "dyn_residual_over_sigma": residual,
                }
            )


def write_intensity_rows(
    path: Path,
    targets: list[float],
    predictions: list[float],
    sigmas: list[float],
    *,
    born_predictions: list[float] | None = None,
    splits: list[str] | None = None,
    datasets: list[str] | None = None,
    source_sigmas: list[float] | None = None,
) -> None:
    fields = [
        "row_index",
        "h",
        "k",
        "l",
        "split",
        "dyn_residual_over_sigma",
        "target_I",
        "dyn_scaled",
        "born_scaled",
        "sigma_eff",
        "dataset",
    ]
    if source_sigmas is not None:
        fields.append("source_sigma")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, (target, prediction, sigma) in enumerate(zip(targets, predictions, sigmas, strict=True)):
            row = {
                    "row_index": idx,
                    "h": 1,
                    "k": 0,
                    "l": idx,
                    "split": splits[idx] if splits is not None else "work",
                    "dyn_residual_over_sigma": (target - prediction) / sigma,
                    "target_I": target,
                    "dyn_scaled": prediction,
                    "born_scaled": born_predictions[idx] if born_predictions is not None else prediction,
                    "sigma_eff": sigma,
                    "dataset": datasets[idx] if datasets is not None else "crystal_1",
                }
            if source_sigmas is not None:
                row["source_sigma"] = source_sigmas[idx]
            writer.writerow(row)


def run_diagnostic(best: Path, other: Path, out: Path, *, prediction_column: str = "dyn_scaled") -> dict:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--best",
            str(best),
            "--other",
            str(other),
            "--out",
            str(out),
            "--prediction-column",
            prediction_column,
        ],
        check=True,
    )
    return json.loads(out.read_text())


def test_absolute_structure_diagnostic_runs_without_pandas(tmp_path: Path):
    best = tmp_path / "best.csv"
    other = tmp_path / "other.csv"
    out = tmp_path / "diagnostic.json"
    write_rows(best, [1.0, 2.0])
    write_rows(other, [2.0, 3.0])

    subprocess.run(
        [sys.executable, str(SCRIPT), "--best", str(best), "--other", str(other), "--out", str(out)],
        check=True,
    )

    payload = json.loads(out.read_text())
    assert payload["status"] == "ok"
    assert payload["row_join"]["n_joined"] == 2
    assert payload["all"]["delta_chi2_other_minus_best"] == 8.0
    assert payload["work"]["n"] == 1
    assert payload["free"]["n"] == 1
    adjusted = payload["all"]["klar_palatinus_adjusted_hand_z"]
    assert adjusted["status"] == "missing_required_columns"
    assert adjusted["n"] == 0


def test_absolute_structure_diagnostic_creates_nested_output_parent(tmp_path: Path):
    best = tmp_path / "best.csv"
    other = tmp_path / "other.csv"
    out = tmp_path / "new" / "nested" / "diagnostic.json"
    write_rows(best, [1.0, 2.0])
    write_rows(other, [2.0, 3.0])

    assert not out.parent.exists()
    subprocess.run(
        [sys.executable, str(SCRIPT), "--best", str(best), "--other", str(other), "--out", str(out)],
        check=True,
    )

    payload = json.loads(out.read_text())
    assert payload["status"] == "ok"
    assert payload["row_join"]["n_joined"] == 2


def test_absolute_structure_diagnostic_skips_missing_inputs(tmp_path: Path):
    best = tmp_path / "missing_best.csv"
    other = tmp_path / "other.csv"
    out = tmp_path / "diagnostic.json"
    write_rows(other, [2.0, 3.0])

    subprocess.run(
        [sys.executable, str(SCRIPT), "--best", str(best), "--other", str(other), "--out", str(out)],
        check=True,
    )

    payload = json.loads(out.read_text())
    assert payload["status"] == "missing_input_files"
    assert str(best) in payload["missing_files"]
    assert payload["row_join"]["n_joined"] == 0


def test_klar_palatinus_adjusted_hand_z_has_expected_sign_and_split_scope(tmp_path: Path):
    best = tmp_path / "best.csv"
    other = tmp_path / "other.csv"
    out = tmp_path / "diagnostic.json"
    targets = [0.0, 0.0, 0.0, 0.0]
    best_predictions = [0.1, 0.1, 0.1, 2.0]
    other_predictions = [1.0, 1.0, 1.0, 1.0]
    sigmas = [0.5, 0.5, 0.5, 0.5]
    splits = ["work", "work", "free", "free"]
    write_intensity_rows(best, targets, best_predictions, sigmas, splits=splits)
    write_intensity_rows(other, targets, other_predictions, sigmas, splits=splits)

    payload = run_diagnostic(best, other, out)
    all_z = payload["all"]["klar_palatinus_adjusted_hand_z"]
    work_z = payload["work"]["klar_palatinus_adjusted_hand_z"]
    free_z = payload["free"]["klar_palatinus_adjusted_hand_z"]
    expected_w = 3.0 * (1.0 - 0.5 * math.erfc(-0.9 / math.sqrt(2.0))) + (1.0 - 0.5 * math.erfc(-1.0 / math.sqrt(2.0)))
    expected_z = 2.0 / math.sqrt(4.0 - expected_w)

    assert all_z["status"] == "ok"
    assert all_z["n"] == 4
    assert all_z["method_version"] == "3"
    assert all_z["tie_policy"] == "strict_lower_residual_klar_2023"
    assert all_z["k_selected_lower_abs_intensity_residual"] == 3
    assert all_z["k_effective_selected_wins_plus_half_ties"] == 3.0
    assert math.isclose(all_z["w"], expected_w)
    assert math.isclose(all_z["z"], expected_z)
    assert math.isclose(all_z["z_neutral_half_tie_diagnostic"], expected_z)
    assert all_z["probability_phi_z"] > 0.5
    assert work_z["n"] == 2
    assert work_z["z"] > 0.0
    assert free_z["n"] == 2
    assert free_z["z"] == 0.0
    assert free_z["probability_phi_z"] == 0.5

    swapped = run_diagnostic(other, best, tmp_path / "swapped.json")
    swapped_all = swapped["all"]["klar_palatinus_adjusted_hand_z"]
    assert math.isclose(swapped_all["z"], -all_z["z"])
    assert math.isclose(swapped_all["probability_phi_z"], all_z["probability_one_sided_other_hand"])


def test_klar_palatinus_adjusted_hand_z_reports_exact_residual_ties(tmp_path: Path):
    best = tmp_path / "best.csv"
    other = tmp_path / "other.csv"
    write_intensity_rows(best, [0.0], [1.0], [1.0])
    write_intensity_rows(other, [0.0], [-1.0], [1.0])

    payload = run_diagnostic(best, other, tmp_path / "diagnostic.json")
    adjusted = payload["all"]["klar_palatinus_adjusted_hand_z"]
    assert adjusted["n"] == 1
    assert adjusted["k_selected_lower_abs_intensity_residual"] == 0
    assert adjusted["n_tied_abs_intensity_residual"] == 1
    assert adjusted["k_effective_selected_wins_plus_half_ties"] == 0.0
    assert adjusted["z"] < 0.0
    assert adjusted["probability_phi_z"] < 0.5
    assert adjusted["z_neutral_half_tie_diagnostic"] == 0.0


def test_klar_palatinus_exact_ties_are_neutral_and_swap_antisymmetric(tmp_path: Path):
    best = tmp_path / "best.csv"
    other = tmp_path / "other.csv"
    targets = [0.0, 0.0, 0.0, 0.0, 0.0]
    best_predictions = [0.0, 0.0, 2.0, 1.0, -1.0]
    other_predictions = [2.0, 2.0, 0.0, -1.0, 1.0]
    sigmas = [1.0] * 5
    write_intensity_rows(best, targets, best_predictions, sigmas)
    write_intensity_rows(other, targets, other_predictions, sigmas)

    selected = run_diagnostic(best, other, tmp_path / "selected.json")
    swapped = run_diagnostic(other, best, tmp_path / "swapped.json")
    selected_z = selected["all"]["klar_palatinus_adjusted_hand_z"]
    swapped_z = swapped["all"]["klar_palatinus_adjusted_hand_z"]

    assert selected_z["n_tied_abs_intensity_residual"] == 2
    assert selected_z["k_selected_lower_abs_intensity_residual"] == 2
    assert selected_z["k_effective_selected_wins_plus_half_ties"] == 2.0
    assert selected_z["z"] < 0.0
    assert swapped_z["z"] < selected_z["z"]
    assert math.isclose(
        swapped_z["z_neutral_half_tie_diagnostic"],
        -selected_z["z_neutral_half_tie_diagnostic"],
    )


def test_klar_palatinus_all_ties_and_born_ties_are_neutral(tmp_path: Path):
    best = tmp_path / "best.csv"
    other = tmp_path / "other.csv"
    targets = [0.0, 0.0, 0.0]
    sigmas = [1.0, 1.0, 1.0]
    write_intensity_rows(
        best,
        targets,
        [1.0, -1.0, 2.0],
        sigmas,
        born_predictions=[3.0, -3.0, 4.0],
    )
    write_intensity_rows(
        other,
        targets,
        [-1.0, 1.0, -2.0],
        sigmas,
        born_predictions=[-3.0, 3.0, -4.0],
    )

    for prediction_column in ("dyn_scaled", "born_scaled"):
        payload = run_diagnostic(
            best,
            other,
            tmp_path / f"{prediction_column}.json",
            prediction_column=prediction_column,
        )
        adjusted = payload["all"]["klar_palatinus_adjusted_hand_z"]
        assert adjusted["n_tied_abs_intensity_residual"] == 3
        assert adjusted["k_effective_selected_wins_plus_half_ties"] == 0.0
        assert adjusted["z"] < 0.0
        assert adjusted["probability_phi_z"] < 0.5
        assert adjusted["z_neutral_half_tie_diagnostic"] == 0.0


def test_klar_palatinus_adjusted_hand_z_sigma_adjustment_follows_declared_formula(tmp_path: Path):
    targets = [0.0, 0.0]
    selected_predictions = [0.0, 0.0]
    other_predictions = [1.0, 1.0]
    best_small = tmp_path / "best_small.csv"
    other_small = tmp_path / "other_small.csv"
    best_large = tmp_path / "best_large.csv"
    other_large = tmp_path / "other_large.csv"
    write_intensity_rows(best_small, targets, selected_predictions, [0.1, 0.1])
    write_intensity_rows(other_small, targets, other_predictions, [0.1, 0.1])
    write_intensity_rows(best_large, targets, selected_predictions, [10.0, 10.0])
    write_intensity_rows(other_large, targets, other_predictions, [10.0, 10.0])

    small = run_diagnostic(best_small, other_small, tmp_path / "small.json")
    large = run_diagnostic(best_large, other_large, tmp_path / "large.json")
    small_z = small["all"]["klar_palatinus_adjusted_hand_z"]
    large_z = large["all"]["klar_palatinus_adjusted_hand_z"]
    assert small_z["k_selected_lower_abs_intensity_residual"] == 2
    assert large_z["k_selected_lower_abs_intensity_residual"] == 2
    assert large_z["w"] > small_z["w"]
    assert large_z["z"] > small_z["z"]


def test_klar_palatinus_z_uses_immutable_source_sigma_not_sigma_eff(tmp_path: Path):
    best = tmp_path / "best.csv"
    other = tmp_path / "other.csv"
    write_intensity_rows(
        best,
        [0.0, 0.0],
        [0.0, 0.0],
        [100.0, 100.0],
        source_sigmas=[1.0, 1.0],
    )
    write_intensity_rows(
        other,
        [0.0, 0.0],
        [1.0, 1.0],
        [100.0, 100.0],
        source_sigmas=[1.0, 1.0],
    )

    adjusted = run_diagnostic(best, other, tmp_path / "diagnostic.json")["all"][
        "klar_palatinus_adjusted_hand_z"
    ]
    phi_half = 0.5 * math.erfc(-0.5 / math.sqrt(2.0))
    expected_w = 2.0 * (1.0 - phi_half)
    expected_z = 2.0 / math.sqrt(2.0 - expected_w)

    assert adjusted["sigma_column"] == "source_sigma"
    assert adjusted["z"] == pytest.approx(expected_z)


def test_klar_palatinus_adjusted_hand_z_excludes_mismatched_observation_contract(tmp_path: Path):
    best = tmp_path / "best.csv"
    other = tmp_path / "other.csv"
    write_intensity_rows(best, [0.0, 0.0], [0.0, 0.0], [1.0, 1.0])
    write_intensity_rows(other, [1.0, 0.0], [2.0, 1.0], [1.0, -1.0])

    payload = run_diagnostic(best, other, tmp_path / "diagnostic.json")
    adjusted = payload["all"]["klar_palatinus_adjusted_hand_z"]
    assert adjusted["status"] == "no_valid_rows"
    assert adjusted["n"] == 0
    assert adjusted["exclusions"]["target_I_mismatch_between_hands"] == 1
    assert adjusted["exclusions"]["nonpositive_sigma"] == 1


def test_klar_palatinus_adjusted_hand_z_does_not_silently_pool_datasets(tmp_path: Path):
    best = tmp_path / "best.csv"
    other = tmp_path / "other.csv"
    datasets = ["crystal_1", "crystal_2"]
    write_intensity_rows(best, [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], datasets=datasets)
    write_intensity_rows(other, [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], datasets=datasets)

    payload = run_diagnostic(best, other, tmp_path / "diagnostic.json")
    adjusted = payload["all"]["klar_palatinus_adjusted_hand_z"]
    assert adjusted["status"] == "multiple_datasets_not_pooled"
    assert adjusted["n"] == 0
    assert adjusted["dataset_labels"] == datasets


def test_absolute_structure_diagnostic_selects_matched_born_control(tmp_path: Path):
    best = tmp_path / "best.csv"
    other = tmp_path / "other.csv"
    targets = [0.0, 0.0]
    sigmas = [1.0, 1.0]
    write_intensity_rows(best, targets, [0.0, 0.0], sigmas, born_predictions=[1.0, 1.0])
    write_intensity_rows(other, targets, [1.0, 1.0], sigmas, born_predictions=[0.0, 0.0])

    dynamical = run_diagnostic(best, other, tmp_path / "dynamical.json")
    born = run_diagnostic(best, other, tmp_path / "born.json", prediction_column="born_scaled")

    assert dynamical["interpretation"]["prediction_column"] == "dyn_scaled"
    assert dynamical["all"]["klar_palatinus_adjusted_hand_z"]["z"] > 0.0
    assert dynamical["all"]["delta_chi2_other_minus_best"] > 0.0
    assert born["interpretation"]["prediction_column"] == "born_scaled"
    assert born["all"]["klar_palatinus_adjusted_hand_z"]["prediction_column"] == "born_scaled"
    assert born["all"]["klar_palatinus_adjusted_hand_z"]["z"] < 0.0
    assert born["all"]["delta_chi2_other_minus_best"] < 0.0
