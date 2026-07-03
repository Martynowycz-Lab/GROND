import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


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
