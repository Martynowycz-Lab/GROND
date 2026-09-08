from grond.forensics.resolution import classify_endpoint_candidate, promote_resolution_limit, suggest_resolution_limit


def test_classify_endpoint_candidate_separates_promoted_from_diagnostic():
    best = 0.2249
    promoted = classify_endpoint_candidate(
        {"hkl_safe_factor": 0.323, "mod_p95": 14.6, "shelxl": {"wR2_all": 0.2260, "R1_all": 0.125, "R1_gt": 0.0698}},
        best_wR2_all=best,
    )
    diagnostic = classify_endpoint_candidate(
        {"hkl_safe_factor": 0.005, "mod_p95": 417.0, "shelxl": {"wR2_all": 0.2305, "R1_all": 0.0935, "R1_gt": 0.0714}},
        best_wR2_all=best,
    )
    assert promoted.status == "promote"
    assert diagnostic.status == "diagnostic"
    assert any("wR2 penalty" in reason for reason in diagnostic.reasons)
    assert any("HKL rescaling factor" in warning for warning in diagnostic.warnings)


def test_suggest_resolution_limit_uses_first_failed_shell():
    shells = [
        {"shell": 1, "d_low": 2.09, "d_high": 9.43, "n_rows": 1000, "n_strong": 600, "mean_I_over_sigma": 10.0, "split_cc": 0.99, "split_common": 250},
        {"shell": 2, "d_low": 1.66, "d_high": 2.09, "n_rows": 1000, "n_strong": 240, "mean_I_over_sigma": 3.1, "split_cc": 0.98, "split_common": 240},
        {"shell": 3, "d_low": 1.45, "d_high": 1.66, "n_rows": 1000, "n_strong": 60, "mean_I_over_sigma": 1.1, "split_cc": 0.89, "split_common": 230},
        {"shell": 4, "d_low": 1.31, "d_high": 1.45, "n_rows": 1000, "n_strong": 8, "mean_I_over_sigma": 0.69, "split_cc": 0.67, "split_common": 240},
        {"shell": 5, "d_low": 1.21, "d_high": 1.31, "n_rows": 1000, "n_strong": 11, "mean_I_over_sigma": 0.59, "split_cc": 0.89, "split_common": 245},
        {"shell": 6, "d_low": 1.14, "d_high": 1.21, "n_rows": 1080, "n_strong": 5, "mean_I_over_sigma": 0.48, "split_cc": 0.82, "split_common": 236},
        {"shell": 7, "d_low": 1.07, "d_high": 1.14, "n_rows": 1000, "n_strong": 1, "mean_I_over_sigma": 0.40, "split_cc": 0.72, "split_common": 247},
        {"shell": 8, "d_low": 1.02, "d_high": 1.07, "n_rows": 1000, "n_strong": 0, "mean_I_over_sigma": 0.24, "split_cc": 0.43, "split_common": 223},
    ]
    decision = suggest_resolution_limit(shells)
    assert round(decision.strict_min_d_spacing, 2) == 1.21
    assert round(decision.exploratory_min_d_spacing, 2) == 1.07
    assert decision.first_strict_rejection["shell"] == 6


def test_promote_resolution_limit_uses_row_mr_and_holdout_stability():
    shells = [
        {
            "shell": 1,
            "d_low": 1.8,
            "d_high": 3.0,
            "n_rows": 200,
            "n_strong": 80,
            "split_common": 50,
            "split_cc": 0.95,
            "mean_I_over_sigma": 4.0,
            "dynamic_MwRall": 0.11,
            "cumulative_dynamic_MwRall": 0.11,
            "work_Nall": 150,
            "free_Nall": 50,
            "free_work_MwRall_gap": 0.02,
        },
        {
            "shell": 2,
            "d_low": 1.3,
            "d_high": 1.8,
            "n_rows": 200,
            "n_strong": 20,
            "split_common": 40,
            "split_cc": 0.75,
            "mean_I_over_sigma": 1.1,
            "dynamic_MwRall": 0.18,
            "cumulative_dynamic_MwRall": 0.13,
            "work_Nall": 150,
            "free_Nall": 50,
            "free_work_MwRall_gap": 0.05,
        },
        {
            "shell": 3,
            "d_low": 1.1,
            "d_high": 1.3,
            "n_rows": 200,
            "n_strong": 4,
            "split_common": 40,
            "split_cc": 0.20,
            "mean_I_over_sigma": 0.4,
            "dynamic_MwRall": 0.44,
            "cumulative_dynamic_MwRall": 0.18,
            "work_Nall": 150,
            "free_Nall": 50,
            "free_work_MwRall_gap": 0.18,
        },
    ]
    decision = promote_resolution_limit(shells)
    assert round(decision.promoted_min_d_spacing, 2) == 1.30
    assert decision.promoted_last_shell == 2
    assert decision.first_rejection["shell"] == 3
    assert any("split CC" in reason or "work/free" in reason for reason in decision.reasons)
