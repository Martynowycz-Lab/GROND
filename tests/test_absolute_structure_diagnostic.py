import csv
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "grond" / "legacy" / "engine" / "absolute_structure_diagnostic.py"


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
