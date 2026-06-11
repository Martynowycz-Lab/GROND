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
            "endpoint_dir": "baseline",
        },
        {
            "spec": "adaptive_120_det_8",
            "wght_mode": "freeze-zero",
            "scale_params": 184,
            "shelxl_R1all": 0.109,
            "shelxl_GooF": 0.91,
            "native_free_wR2_all": 0.15,
            "endpoint_dir": "adaptive_120",
        },
        {
            "spec": "window_5deg_det_4",
            "wght_mode": "freeze-zero",
            "scale_params": 38,
            "shelxl_R1all": 0.131,
            "shelxl_GooF": 0.90,
            "native_free_wR2_all": 0.20,
            "endpoint_dir": "too_coarse",
        },
    ]

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

