from grond.legacy.engine.scale_contract_replay import choose_publication_endpoint


def test_publication_endpoint_prefers_boring_goof_over_low_r_diagnostic():
    rows = [
        {
            "spec": "baseline",
            "wght_mode": "freeze-zero",
            "scale_params": 700,
            "shelxl_R1all": 0.102,
            "shelxl_GooF": 0.40,
            "native_free_wR2_all": 0.14,
            "bic_fitted_variance": 100.0,
            "endpoint_dir": "baseline",
        },
        {
            "spec": "adaptive_120_det_8",
            "wght_mode": "freeze-zero",
            "scale_params": 184,
            "shelxl_R1all": 0.109,
            "shelxl_GooF": 0.91,
            "native_free_wR2_all": 0.15,
            "bic_fitted_variance": 200.0,
            "endpoint_dir": "adaptive_120",
        },
        {
            "spec": "window_5deg_det_4",
            "wght_mode": "freeze-zero",
            "scale_params": 38,
            "shelxl_R1all": 0.131,
            "shelxl_GooF": 0.90,
            "native_free_wR2_all": 0.20,
            "bic_fitted_variance": 500.0,
            "endpoint_dir": "too_coarse",
        },
    ]
    for row in rows:
        row["mod_p95"] = 1.2

    selected = choose_publication_endpoint(
        rows,
        goof_min=0.85,
        goof_max=1.15,
        max_r1_cost=0.010,
        free_wr2_tolerance=0.030,
        prefer_wght_mode="freeze-zero",
    )

    assert selected["status"] == "passed"
    assert selected["low_r_diagnostic"]["spec"] == "baseline"
    assert selected["publication_endpoint"]["spec"] == "adaptive_120_det_8"


def test_publication_endpoint_default_uses_bic_after_sanity_gates():
    rows = [
        {
            "spec": "few_scales",
            "wght_mode": "freeze-zero",
            "scale_params": 50,
            "shelxl_R1all": 0.130,
            "shelxl_GooF": 1.02,
            "native_free_wR2_all": 0.22,
            "bic_fitted_variance": 2000.0,
            "endpoint_dir": "few",
        },
        {
            "spec": "bic_best",
            "wght_mode": "freeze-zero",
            "scale_params": 150,
            "shelxl_R1all": 0.115,
            "shelxl_GooF": 1.03,
            "native_free_wR2_all": 0.18,
            "bic_fitted_variance": 1500.0,
            "endpoint_dir": "bic_best",
        },
        {
            "spec": "low_r_bad_goof",
            "wght_mode": "freeze-zero",
            "scale_params": 500,
            "shelxl_R1all": 0.095,
            "shelxl_GooF": 0.55,
            "native_free_wR2_all": 0.15,
            "bic_fitted_variance": 1000.0,
            "endpoint_dir": "low_r",
        },
    ]
    for row in rows:
        row["mod_p95"] = 1.2

    selected = choose_publication_endpoint(
        rows,
        goof_min=0.85,
        goof_max=1.15,
        max_r1_cost=0.010,
        free_wr2_tolerance=0.030,
        prefer_wght_mode="freeze-zero",
    )

    assert selected["status"] == "passed"
    assert selected["selection_policy"] == "bic"
    assert selected["publication_endpoint"]["spec"] == "bic_best"
    assert selected["selected_information_criterion"] == 1500.0


def test_publication_endpoint_legacy_policy_keeps_smallest_passing_scale_count():
    rows = [
        {
            "spec": "few_scales",
            "wght_mode": "freeze-zero",
            "scale_params": 50,
            "shelxl_R1all": 0.101,
            "shelxl_GooF": 1.02,
            "native_free_wR2_all": 0.151,
            "bic_fitted_variance": 2000.0,
            "endpoint_dir": "few",
        },
        {
            "spec": "bic_best",
            "wght_mode": "freeze-zero",
            "scale_params": 150,
            "shelxl_R1all": 0.100,
            "shelxl_GooF": 1.03,
            "native_free_wR2_all": 0.150,
            "bic_fitted_variance": 1500.0,
            "endpoint_dir": "bic_best",
        },
    ]
    for row in rows:
        row["mod_p95"] = 1.2

    selected = choose_publication_endpoint(
        rows,
        goof_min=0.85,
        goof_max=1.15,
        max_r1_cost=0.010,
        free_wr2_tolerance=0.030,
        prefer_wght_mode="freeze-zero",
        selection_policy="gated-smallest",
    )

    assert selected["status"] == "passed"
    assert selected["selection_policy"] == "gated-smallest"
    assert selected["publication_endpoint"]["spec"] == "few_scales"


def test_publication_endpoint_modulation_gate_rejects_missing_and_excessive_values():
    rows = [
        {
            "spec": "missing",
            "wght_mode": "freeze-zero",
            "scale_params": 10,
            "shelxl_R1all": 0.10,
            "shelxl_GooF": 1.0,
            "native_free_wR2_all": 0.15,
            "bic_fitted_variance": 100.0,
        },
        {
            "spec": "excessive",
            "wght_mode": "freeze-zero",
            "scale_params": 10,
            "shelxl_R1all": 0.10,
            "shelxl_GooF": 1.0,
            "native_free_wR2_all": 0.15,
            "bic_fitted_variance": 100.0,
            "mod_p95": 20.1,
        },
    ]

    selected = choose_publication_endpoint(
        rows,
        goof_min=0.85,
        goof_max=1.15,
        max_r1_cost=0.010,
        free_wr2_tolerance=0.030,
        prefer_wght_mode="freeze-zero",
        max_modulation_p95=20.0,
    )

    assert selected["status"] == "no_candidate_passed_all_gates"
    reasons = {row["spec"]: row["reasons"] for row in selected["rejections"]}
    assert any("missing finite" in reason for reason in reasons["missing"])
    assert any("exceeds" in reason for reason in reasons["excessive"])
