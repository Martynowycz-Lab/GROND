import csv
import json
import math
import subprocess
import sys
from pathlib import Path

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
    splits: list[str] | None = None,
    datasets: list[str] | None = None,
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
        "sigma_eff",
        "dataset",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, (target, prediction, sigma) in enumerate(zip(targets, predictions, sigmas, strict=True)):
            writer.writerow(
                {
                    "row_index": idx,
                    "h": 1,
                    "k": 0,
                    "l": idx,
                    "split": splits[idx] if splits is not None else "work",
                    "dyn_residual_over_sigma": (target - prediction) / sigma,
                    "target_I": target,
                    "dyn_scaled": prediction,
                    "sigma_eff": sigma,
                    "dataset": datasets[idx] if datasets is not None else "crystal_1",
                }
            )


def run_diagnostic(best: Path, other: Path, out: Path) -> dict:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--best", str(best), "--other", str(other), "--out", str(out)],
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
    assert all_z["k_selected_lower_abs_intensity_residual"] == 3
    assert math.isclose(all_z["w"], expected_w)
    assert math.isclose(all_z["z"], expected_z)
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
    assert adjusted["z"] < 0.0


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
