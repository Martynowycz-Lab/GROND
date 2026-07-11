import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ENGINE_DIR = Path(__file__).resolve().parents[1] / "src" / "grond" / "legacy" / "engine"
sys.path.insert(0, str(ENGINE_DIR))

import native_multi_structure_update as multi  # noqa: E402


def test_dataset_state_keeps_selection_metadata_in_report(tmp_path: Path):
    nuisance = multi.Nuisance(
        thickness_A=700.0,
        thickness_grad_frac=0.4,
        frame_offset=0.1,
        group_width_deg=0.169,
        detector_scale_bins=8,
        sigma_a=0.0,
        sigma_b_corr=0.0,
        sigma_c_peak=0.0,
    )
    state = multi.DatasetState(
        name="crystal_a",
        hkl=tmp_path / "INTEGRATE.HKL",
        data=SimpleNamespace(hkl_transform_spec="identity"),
        target=np.ones(3),
        base_sig=np.ones(3),
        sig=np.ones(3),
        target_indices=np.array([0, 1, 2]),
        objective_indices=np.array([0, 2]),
        masks={"work": np.array([True, False, True]), "free": np.array([False, True, False])},
        validation_fit_indices=np.array([0, 1, 2]),
        nuisance=nuisance,
        cfg=multi.Config(),
        objective_cfg=multi.Config(),
        wght_a=0.1,
        wght_b=0.0,
        detector_scale_bins=8,
        min_d_spacing=0.7,
        min_peak=90.0,
        target_mode="iobs",
        dataset_weight=0.5,
        dataset_role="downweight",
    )

    multi.write_dataset_rows(tmp_path, [state])
    rows = list(csv.DictReader((tmp_path / "datasets_used.csv").open()))
    assert rows[0]["min_d_spacing"] == "0.7"
    assert rows[0]["min_peak"] == "90.0"
    assert rows[0]["target_mode"] == "iobs"
    assert rows[0]["dataset_weight"] == "0.5"
    assert rows[0]["dataset_role"] == "downweight"


def test_joint_metrics_are_order_invariant_with_dataset_specific_weights(
    tmp_path: Path,
    monkeypatch,
):
    masks = {
        "all": np.ones(3, dtype=bool),
        "work": np.asarray([True, True, False]),
        "free": np.asarray([False, False, True]),
    }
    state_a = SimpleNamespace(
        name="a",
        target=np.asarray([100.0, 80.0, 60.0]),
        sig=np.asarray([5.0, 5.0, 5.0]),
        masks=masks,
        wght_a=0.0,
        wght_b=0.0,
    )
    state_b = SimpleNamespace(
        name="b",
        target=np.asarray([50.0, 40.0, 30.0]),
        sig=np.asarray([2.0, 2.0, 2.0]),
        masks=masks,
        wght_a=0.3,
        wght_b=0.1,
    )
    predictions = {
        "a": np.asarray([90.0, 86.0, 57.0]),
        "b": np.asarray([48.0, 35.0, 36.0]),
    }

    def fake_evaluate_state(state, *_args, **_kwargs):
        return {
            "name": state.name,
            "splits": {},
            "scaled": predictions[state.name],
            "valid": np.ones(3, dtype=bool),
            "scale_count": 1,
        }

    monkeypatch.setattr(multi, "evaluate_state", fake_evaluate_state)
    forward = multi.evaluate_all([state_a, state_b], None, tmp_path, "test", 1, 1)
    reverse = multi.evaluate_all([state_b, state_a], None, tmp_path, "test", 1, 1)

    for split in ("all", "work", "free"):
        for metric in ("R1_all", "R2_all", "wR2_all", "GooF"):
            assert forward["splits"][split][metric] == pytest.approx(
                reverse["splits"][split][metric]
            )


@pytest.mark.parametrize(
    "names",
    [
        ["../escape"],
        ["Crystal_A", "crystal_a"],
    ],
)
def test_multi_dataset_csv_rejects_unsafe_or_colliding_names(tmp_path: Path, names):
    path = tmp_path / "datasets.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["include", "name", "hkl"])
        writer.writeheader()
        for name in names:
            writer.writerow({"include": "true", "name": name, "hkl": "input.hkl"})

    with pytest.raises(ValueError, match="dataset name"):
        multi.read_datasets_csv(path)
