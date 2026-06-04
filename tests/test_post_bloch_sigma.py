from pathlib import Path

import pytest

from grond.legacy.engine.bloch_engine import parse_model
from grond.legacy.engine.post_bloch_sigma import fit_uniform_z_rms_sigma_scale, scale_row_sigmas


def constant_fcalc_model(tmp_path: Path):
    res = tmp_path / "model.res"
    res.write_text(
        "\n".join(
            [
                "TITL constant fcalc",
                "CELL 0.025 10 10 10 90 90 90",
                "LATT -1",
                "SFAC C 1 0 1 0 1 0 1 0 0",
                "UNIT 1",
                "C1 1 0 0 0 1.0 0.0",
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )
    return parse_model(res)


def test_uniform_z_rms_sigma_scale_matches_known_residuals(tmp_path: Path):
    model = constant_fcalc_model(tmp_path)
    # For this model Fc = 4 for every HKL, so Fc^2 = 16.  The fitted model
    # scale is 10, and the residuals are exactly +/-5 on sigma=10.
    rows = []
    for idx in range(40):
        residual = 5.0 if idx % 2 == 0 else -5.0
        rows.append((idx + 1, 0, 0, 160.0 + residual, 10.0))

    result = fit_uniform_z_rms_sigma_scale(
        rows,
        model,
        holdout_fraction=0.0,
        min_scale=0.10,
        max_scale=2.0,
    )

    assert result["status"] == "ok"
    assert result["model_intensity_scale"] == pytest.approx(10.0)
    assert result["sigma_scale"] == pytest.approx(0.5)
    assert result["train_rms_z_before"] == pytest.approx(0.5)
    assert result["train_rms_z_after"] == pytest.approx(1.0)


def test_scale_row_sigmas_leaves_hkl_and_intensity_unchanged():
    rows = [(1, -2, 3, 123.4, 5.0), (0, 1, -1, 20.0, 2.5)]

    scaled = scale_row_sigmas(rows, 0.5)

    assert scaled == [(1, -2, 3, 123.4, 2.5), (0, 1, -1, 20.0, 1.25)]
