"""CLI terminal reports regression tests."""







import grond.cli as cli


def test_terminal_finish_summary_is_human_readable():
    payload = {
        "status": "ok",
        "out_dir": "/tmp/grond",
        "final_summary": "/tmp/grond/FINAL_SUMMARY.md",
        "quick_stats": "/tmp/grond/QUICK_STATS.md",
        "full_json": "/tmp/grond/grond_summary.json",
        "finished": {
            "completed_stages": 25,
            "completed_this_invocation": 20,
            "reused_completed_stages": 5,
            "estimated_stages": 30,
            "invocation_elapsed_seconds": 300,
            "cumulative_stage_elapsed_seconds": 822,
            "stage_elapsed_seconds": 822,
        },
        "final_hand": "alternate",
        "final_hand_t_style_all": 1.505,
        "final_hand_t_style_all_n": 40,
        "final_hand_t_style_all_df": 39,
        "final_hand_t_style_free": 0.876,
        "final_hand_t_style_free_n": 10,
        "final_hand_t_style_free_df": 9,
        "final_hand_t_style_scope": "single_dataset",
        "final_hand_zkp_all": 10.0,
        "final_hand_zkp_free": 1.5,
        "final_hand_warnings": ["held-out ZKP 1.5 supports the selected hand only weakly (< 3.5)"],
        "selected_transform": "identity",
        "input_shelxl": {
            "R1gt": 0.1382,
            "R1all": 0.1714,
            "wR2all": 0.3845,
            "GooF": 1.406,
            "parameters": 123,
            "restraints": 75,
            "npd_count": 0,
            "split_count": 1,
        },
        "input_model": {"row_R1all": 0.22128, "row_wR2all": 0.43413, "born_R1all": 0.28, "born_wR2all": 0.42},
        "final_row_model": {
            "row_R1all": 0.21300,
            "row_wR2all": 0.33100,
            "row_free_wR2all": 0.33448,
            "born_R1all": 0.27,
            "born_wR2all": 0.41,
            "born_free_wR2all": 0.40,
            "row_table": "/tmp/rows.csv",
        },
        "final_feedback_endpoint": {
            "R1gt": 0.1100,
            "R1all": 0.1600,
            "wR2all": 0.2200,
            "GooF": 0.950,
            "npd_count": 0,
            "split_count": 0,
        },
        "publication_endpoint": {
            "R1gt": 0.1446,
            "R1all": 0.1785,
            "wR2all": 0.2572,
            "GooF": 1.036,
            "scale_spec": "adaptive_120_det_8",
            "scale_params": 184,
            "wght_mode": "freeze-zero",
            "npd_count": 0,
            "split_count": 0,
        },
        "files": {"publication_res": "/tmp/endpoint.res", "publication_hkl": "/tmp/endpoint.hkl"},
    }

    text = cli._format_terminal_finish_summary(payload)

    assert "GROND finished successfully in 5m00s wall time for this invocation." in text
    assert "successfully in 13m42s" not in text
    assert (
        "Progress: 25 stages completed; initial estimate was 30; 20 ran in this invocation; "
        "5 reused from prior invocations." in text
    )
    assert "Timing: cumulative completed-stage runtime 13m42s." in text
    assert "concurrent stages overlap" in text
    assert "not workflow wall time" in text
    assert "Hand: alternate (adjusted ZKP all/free = 10.000 / 1.500; positive favors the selected hand" in text
    assert "paired t-style all/free = t 1.505 (n 40, df 39) / t 0.876 (n 10, df 9)" in text
    assert "Hand warning: held-out ZKP 1.5 supports the selected hand only weakly" in text
    assert "not a calibrated normal Z score" in text
    assert "normal-equivalent confidence" not in text
    assert "Starting SHELXL reference" in text
    assert "R1(F)gt 13.82%, R1(F)all 17.14%, wR2(F^2)all 38.45%, GooF 1.406" in text
    assert "Input dynamical rows: R(I)all 22.13%, wR(I)all 43.41%" in text
    assert "Input matched Born rows: R(I)all 28.00%, wR(I)all 42.00%" in text
    assert "Final dynamical rows: R(I)all 21.30%, wR(I)all 33.10%, free wR(I)all 33.45%" in text
    assert "Final matched Born rows: R(I)all 27.00%, wR(I)all 41.00%, free wR(I)all 40.00%" in text
    assert "Change: R(I)all -0.83 pp, wR(I)all -10.31 pp" in text
    assert "Direct LS1 corrected-HKL candidate: R1(F)gt 11.00%, R1(F)all 16.00%, wR2(F^2)all 22.00%, GooF 0.950" in text
    assert "Recommended model-handoff endpoint: R1(F)gt 14.46%, R1(F)all 17.85%, wR2(F^2)all 25.72%, GooF 1.036" in text


def test_terminal_finish_summary_prefers_jana_pets_rows_and_links_table1():
    payload = {
        "status": "ok",
        "out_dir": "/tmp/grond",
        "table1": "/tmp/grond/TABLE1_FINAL_STATISTICS.md",
        "table1_json": "/tmp/grond/TABLE1_FINAL_STATISTICS.json",
        "input_model": {"row_R1all": 0.99, "row_wR2all": 0.98},
        "final_row_model": {"row_R1all": 0.97, "row_wR2all": 0.96},
        "jana_pets_row_statistics": {
            "born": {
                "Nobs": 20,
                "Nall": 30,
                "Robs": 0.12,
                "Rall": 0.14,
                "wRall": 0.16,
                "MNobs": 18,
                "MNall": 25,
                "MRobs": 0.10,
                "MRall": 0.11,
                "MwRall": 0.13,
            },
            "dynamic": {
                "Nobs": 20,
                "Nall": 30,
                "Robs": 0.07,
                "Rall": 0.08,
                "wRall": 0.09,
                "MNobs": 18,
                "MNall": 25,
                "MRobs": 0.05,
                "MRall": 0.06,
                "MwRall": 0.07,
            },
        },
        "publication_endpoint": {},
        "input_shelxl": {},
        "finished": {"completed_stages": 0, "cumulative_stage_elapsed_seconds": 0},
        "files": {},
    }

    text = cli._format_terminal_finish_summary(payload)

    assert "Final JANA/PETS-style row statistics (computed by GROND)" in text
    assert "Matched Born: Nobs/Nall 20/30, Robs 12.00%, Rall 14.00%, wRall 16.00%" in text
    assert "Bloch dynamical: Nobs/Nall 20/30, Robs 7.00%, Rall 8.00%, wRall 9.00%" in text
    assert "Legacy native-objective row diagnostics" not in text
    assert "No corrected endpoint passed the hard artifact" in text
    assert "Canonical Table 1: /tmp/grond/TABLE1_FINAL_STATISTICS.md" in text


def test_jana_legacy_fallback_requires_complete_core_and_does_not_invent_counts():
    assert cli._quick_final_jana_row_summary({"row_mr": {"dynamic": {"Rall": 0.1}}}) == {}

    legacy = {
        "Nobs": 4,
        "Nall": 10,
        "MNobs": 3,
        "MNall": 7,
        "Robs": 0.09,
        "Rall": 0.10,
        "wRall": 0.11,
        "MRobs": 0.07,
        "MRall": 0.08,
        "MwRall": 0.09,
    }
    summary = cli._quick_final_jana_row_summary(
        {"row_mr": {"definition": {"instability_u": 0.01}, "dynamic": legacy}}
    )
    record = cli._table1_jana_record("legacy", "row", summary["dynamic"], "legacy row_mr")

    assert record["status"] == "available"
    assert record["Ninput"] is None
    assert record["Nexcluded"] is None
    assert record["Nmatched"] is None
    assert record["Nmerge_used"] is None
    assert record["Nmerge_excluded"] is None

    incomplete = cli._table1_jana_record("partial", "row", {"Rall": 0.1}, "legacy row_mr")
    assert incomplete["status"] == "not_available"
    assert "incomplete core JANA/PETS factor family" in incomplete["status_reason"]


def test_terminal_finish_summary_does_not_treat_legacy_stage_sum_as_wall_time():
    text = cli._format_terminal_finish_summary(
        {
            "status": "ok",
            "out_dir": "/tmp/legacy",
            "finished": {
                "completed_stages": 4,
                "estimated_stages": 4,
                "stage_elapsed_seconds": 600,
            },
        }
    )

    assert "GROND finished successfully." in text
    assert "GROND finished successfully in 10m00s" not in text
    assert "Timing: cumulative completed-stage runtime 10m00s." in text
    assert "not workflow wall time" in text


def test_terminal_finish_summary_labels_recorded_workflow_timing_for_finish():
    text = cli._format_terminal_finish_summary(
        {
            "status": "ok",
            "out_dir": "/tmp/grond",
            "finished": {
                "completed_stages": 25,
                "estimated_stages": 25,
                "invocation_elapsed_seconds": 300,
                "cumulative_stage_elapsed_seconds": 822,
            },
            "timing_context": {
                "kind": "finish_replay",
                "finish_invocation_elapsed_seconds": 2,
            },
        }
    )

    assert "GROND finish completed successfully in 0m02s wall time." in text
    assert "recorded workflow wall time 5m00s" in text
    assert "wall time for this invocation" not in text
    assert "cumulative completed-stage runtime 13m42s" in text
