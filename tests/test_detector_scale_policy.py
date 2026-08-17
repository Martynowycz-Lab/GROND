import numpy as np

from grond.legacy.engine.detector_scale_regularization import (
    solve_regularized_detector_field,
)
from grond.legacy.engine.scale_compression_polish import (
    detector_scale_policy_diagnostics,
)
from grond.legacy.engine.scale_contract_replay import choose_publication_endpoint


def test_detector_regularization_shrinks_sparse_tiles_and_smooths_neighbors():
    diag = np.full(16, 100.0)
    rhs = diag * np.linspace(0.75, 1.25, 16)
    counts = np.full(16, 40)
    counts[5] = 2
    rhs[5] = 10_000.0

    field = solve_regularized_detector_field(
        diag,
        rhs,
        counts,
        n_bins=4,
        ridge=0.10,
        smoothness=0.20,
        min_rows=20,
    )

    assert len(field) == 16
    assert 0.8 < field[5] < 1.2
    assert np.max(field) < 1.25
    assert np.min(field) > 0.75


def _synthetic_detector_rows(seed=7):
    rng = np.random.default_rng(seed)
    n_per_tile = 120
    source_tile = np.repeat(np.arange(64), n_per_tile)
    x = source_tile % 8
    y = source_tile // 8
    coarse = (y // 2) * 4 + (x // 2)
    true_field = np.asarray(
        [
            0.88, 0.94, 1.05, 1.12,
            0.91, 0.97, 1.04, 1.10,
            0.90, 0.98, 1.03, 1.09,
            0.89, 0.95, 1.06, 1.11,
        ]
    )
    n = len(source_tile)
    groups = np.arange(n) % 24
    frame = 0.92 + 0.16 * (groups / 23.0)
    raw_dyn = rng.uniform(20.0, 200.0, n)
    # Born differs by row, not by detector position, so a genuine detector
    # field should be recovered consistently from either physical model.
    raw_born = raw_dyn * (1.0 + 0.18 * np.sin(np.arange(n) * 0.173))
    obs = raw_dyn * frame * true_field[coarse]
    obs *= 1.0 + rng.normal(0.0, 0.015, n)
    sig = np.sqrt(np.maximum(obs, 1.0)) * 0.25
    rows = [
        {
            "row_index": str(i),
            "h": str(i % 17),
            "k": str((i // 17) % 19),
            "l": str((i // 323) % 23),
            "group": str(groups[i]),
            "detector_bin": str(source_tile[i]),
        }
        for i in range(n)
    ]
    return rows, obs, sig, raw_dyn, raw_born, groups, source_tile


def test_four_by_four_detector_field_earns_validation_but_eight_is_diagnostic():
    rows, obs, sig, raw_dyn, raw_born, groups, source_tile = (
        _synthetic_detector_rows()
    )
    common = dict(
        rows=rows,
        obs=obs,
        sig=sig,
        raw_dyn=raw_dyn,
        raw_born=raw_born,
        groups=groups,
        source_detector_bins=source_tile,
        detector_base=8,
        fit_mask=np.ones(len(rows), dtype=bool),
        objective="native",
        native_wght_a=0.0,
        native_wght_b=0.0,
        instability_u=0.01,
        detector_ridge=0.05,
        detector_smoothness=0.10,
        detector_min_rows=20,
    )

    four = detector_scale_policy_diagnostics(detector_target=4, **common)
    eight = detector_scale_policy_diagnostics(detector_target=8, **common)

    assert four["eligible"] is True
    assert four["improved_split_count"] >= 2
    assert four["support"]["all_tiles_adequate"] is True
    assert eight["eligible"] is False
    assert "eight_by_eight_requires_independent_calibration" in eight["reasons"]


def test_detector_policy_preserves_predeclared_free_rows():
    rows, obs, sig, raw_dyn, raw_born, groups, source_tile = (
        _synthetic_detector_rows()
    )
    fit_mask = np.ones(len(rows), dtype=bool)
    fit_mask[-800:] = False

    result = detector_scale_policy_diagnostics(
        rows=rows,
        obs=obs,
        sig=sig,
        raw_dyn=raw_dyn,
        raw_born=raw_born,
        groups=groups,
        source_detector_bins=source_tile,
        detector_base=8,
        detector_target=4,
        fit_mask=fit_mask,
        objective="native",
        native_wght_a=0.0,
        native_wght_b=0.0,
        instability_u=0.01,
    )

    assert result["holdout_splits"][0]["split"] == "predeclared_free"


def test_publication_selector_rejects_unvalidated_detector_and_split_warning():
    base = {
        "wght_mode": "freeze-zero",
        "shelxl_R1all": 0.10,
        "shelxl_GooF": 1.0,
        "native_free_wR2_all": 0.20,
        "mod_p95": 1.1,
        "shelxl_npd_count": 0,
        "scale_params": 100,
    }
    rows = [
        {
            **base,
            "spec": "detector_free",
            "detector_bins": 0,
            "scale_params": 120,
        },
        {
            **base,
            "spec": "detector_4_unvalidated",
            "detector_bins": 4,
            "detector_policy_eligible": False,
            "shelxl_R1all": 0.05,
            "native_free_wR2_all": 0.01,
            "scale_params": 80,
        },
        {
            **base,
            "spec": "split_pathology",
            "detector_bins": 0,
            "shelxl_split_count": 1,
            "scale_params": 40,
        },
    ]

    result = choose_publication_endpoint(
        rows,
        goof_min=0.85,
        goof_max=1.15,
        max_r1_cost=0.01,
        free_wr2_tolerance=0.04,
        prefer_wght_mode="freeze-zero",
    )

    assert result["status"] == "passed"
    assert result["publication_endpoint"]["spec"] == "detector_free"
    rejected = {row["spec"]: row["reasons"] for row in result["rejections"]}
    assert any("detector field failed" in reason for reason in rejected["detector_4_unvalidated"])
    assert any("split warnings" in reason for reason in rejected["split_pathology"])
