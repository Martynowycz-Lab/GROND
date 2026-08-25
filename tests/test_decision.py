from __future__ import annotations

import math
from pathlib import Path

import pytest

from grond.io.config import DatasetSpec
from grond.workflows.decision import (
    apply_preflight,
    expand_transform_candidates,
    preflight_dataset,
    score_native_summary,
    thickness_polish_nuisance,
)
from grond.workflows.transform_metadata import metadata_prefilter_transforms


def _write_minimal_xds(path: Path, n: int = 300, sigma: float = 10.0) -> None:
    lines = ["!UNIT_CELL_CONSTANTS= 5 6 7 90 90 90\n"]
    for i in range(n):
        h = i % 4 + 1
        k = (i // 4) % 4
        ell = (i // 16) % 4
        x = 100.0 + (i % 5)
        y = 120.0 + (i % 7)
        z = 10.0 + (i % 3)
        row = [
            h,
            k,
            ell,
            1000.0 + i,
            sigma,
            x,
            y,
            z,
            0.001,
            100.0,
            0.9,
            0.0,
            x + 0.1,
            y - 0.1,
            z + 0.02,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        lines.append(" ".join(str(v) for v in row) + "\n")
    path.write_text("".join(lines))


def test_expand_transform_candidates_deduplicates_current():
    transforms = expand_transform_candidates("l,-k,h", "auto;l,-k,h")
    assert transforms[0] == "l,-k,h"
    assert transforms.count("l,-k,h") == 1
    assert "identity" in transforms


def test_score_native_summary_tolerates_missing_values():
    score = score_native_summary({"best_native": {"metrics": {"all": {"dyn": {"wR2_all": None}}}}})
    assert math.isinf(score["score"])


def test_thickness_polish_never_proposes_negative_support():
    polished = thickness_polish_nuisance(
        {"thickness_A": 1400.0, "thickness_grad_frac": 1.0, "frame_offset": 1.4},
        frame_offset_candidates=[1.0, 1.4, 1.8],
    )

    assert polished["thickness_grad_frac"] == "0.8,1"
    assert polished["frame_offset"] == "1,1.3,1.4,1.5,1.8"


def test_preflight_keeps_healthy_minimal_xds(tmp_path: Path):
    hkl = tmp_path / "INTEGRATE.HKL"
    _write_minimal_xds(hkl)
    ds = DatasetSpec(name="tiny", hkl=str(hkl), min_peak=90.0, geometry_percentile=99.0)
    decision = preflight_dataset(ds, requested_min_peak=90.0, requested_geometry_percentile=99.0, requested_scout_rows=500)
    assert decision.status == "ok"
    assert decision.row_count == 300
    assert decision.scout_max_rows == 300
    assert decision.recommended_min_peak == 90.0


def test_preflight_excludes_all_zero_xds_centroids_from_geometry_diagnostics(tmp_path: Path):
    hkl = tmp_path / "INTEGRATE.HKL"
    _write_minimal_xds(hkl)
    lines = hkl.read_text().splitlines()
    for index in range(1, 61):
        fields = lines[index].split()
        fields[12:15] = ["0", "0", "0"]
        lines[index] = " ".join(fields)
    hkl.write_text("\n".join(lines) + "\n")
    ds = DatasetSpec(name="missing_centroids", hkl=str(hkl), min_peak=90.0)

    decision = preflight_dataset(
        ds,
        requested_min_peak=90.0,
        requested_geometry_percentile=99.0,
        requested_scout_rows=500,
    )

    assert decision.status == "ok"
    assert decision.observed_geometry_fraction == pytest.approx(0.8)
    assert decision.geometry_dr_p99_px < 1.0
    assert decision.geometry_abs_dz_p99_frame < 1.0
    assert decision.recommended_geometry_percentile == 99.0
    assert any(
        "60 rows have absent XDS observed centroids" in value
        for value in decision.warnings
    )


def test_preflight_rejects_rows_with_negative_sigmas(tmp_path: Path):
    hkl = tmp_path / "INTEGRATE.HKL"
    _write_minimal_xds(hkl, sigma=-5.0)
    ds = DatasetSpec(name="bad_sigma", hkl=str(hkl), min_peak=90.0, geometry_percentile=99.0)

    decision = preflight_dataset(
        ds,
        requested_min_peak=90.0,
        requested_geometry_percentile=99.0,
        requested_scout_rows=500,
    )

    assert decision.status == "fatal"
    assert decision.sigma_positive_fraction == 0.0
    assert decision.scout_max_rows == 0
    assert "few rows have finite intensity and positive sigma" in decision.warnings


def test_apply_preflight_preserves_dataset_baseline_and_paths(tmp_path: Path):
    hkl = tmp_path / "INTEGRATE.HKL"
    _write_minimal_xds(hkl)
    ds = DatasetSpec(
        name="dataset",
        hkl=str(hkl),
        xds_dir=str(tmp_path),
        baseline_hkl=str(tmp_path / "baseline.hkl"),
        hkl_transform="-h,k,l",
    )
    decision = preflight_dataset(
        ds,
        requested_min_peak=75.0,
        requested_geometry_percentile=97.5,
        requested_scout_rows=500,
    )
    updated = apply_preflight(ds, decision)

    assert updated.baseline_hkl == ds.baseline_hkl
    assert updated.xds_dir == ds.xds_dir
    assert updated.hkl_transform == ds.hkl_transform
    assert updated.min_peak == ds.min_peak
    assert updated.geometry_percentile == ds.geometry_percentile


def test_apply_preflight_does_not_apply_relaxed_support_suggestion(tmp_path: Path):
    hkl = tmp_path / "INTEGRATE.HKL"
    _write_minimal_xds(hkl)
    lines = hkl.read_text().splitlines()
    for index in range(1, 251):
        fields = lines[index].split()
        fields[9] = "75"
        lines[index] = " ".join(fields)
    hkl.write_text("\n".join(lines) + "\n")
    ds = DatasetSpec(name="sparse_p90", hkl=str(hkl), min_peak=90.0, geometry_percentile=99.0)

    decision = preflight_dataset(
        ds,
        requested_min_peak=90.0,
        requested_geometry_percentile=99.0,
        requested_scout_rows=500,
    )
    updated = apply_preflight(ds, decision)

    assert decision.recommended_min_peak < 90.0
    assert updated.min_peak == 90.0
    assert updated.geometry_percentile == 99.0
    assert any("suggestion only" in warning for warning in decision.warnings)


def test_metadata_transform_prefilter_promotes_hkl_overlap(tmp_path: Path):
    moving = tmp_path / "INTEGRATE.HKL"
    baseline = tmp_path / "model.hkl"
    moving_lines = []
    baseline_lines = []
    for i in range(1, 220):
        h = i % 7 + 1
        k = (i // 7) % 5 - 2
        ell = (i // 35) % 9 + 1
        intensity = float(500 + 11 * h + 23 * k + 37 * ell + i)
        sigma = 10.0
        moving_lines.append(f"{h:4d}{k:4d}{ell:4d}{intensity:10.2f}{sigma:8.2f}\n")
        baseline_lines.append(f"{ell:4d}{-k:4d}{h:4d}{(2.0 * intensity):10.2f}{sigma:8.2f}\n")
    moving.write_text("".join(moving_lines))
    baseline.write_text("".join(baseline_lines))

    selected, payload = metadata_prefilter_transforms(
        moving,
        baseline,
        ["identity", "l,-k,h", "-l,k,-h"],
        min_pairs=50,
        min_correlation=0.70,
        min_correlation_gap=0.05,
    )

    assert payload["status"] == "promoted"
    assert selected == ["l,-k,h"]
    assert payload["records"][0]["transform"] == "l,-k,h"
    assert payload["records"][0]["paired_rows"] >= 200
