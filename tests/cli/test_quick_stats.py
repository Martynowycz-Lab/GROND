"""CLI quick stats regression tests."""

import argparse
import json

import pytest

import grond.cli as cli
from grond.io.config import DatasetSpec, PrepConfig


def test_write_quick_stats_summarizes_row_and_endpoint_tables(tmp_path):
    out_dir = tmp_path / "grond_out"
    tables = out_dir / "tables"
    tables.mkdir(parents=True)
    (tables / "row_mr_metrics.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "row_count": 10,
                    "n_basis": "24",
                    "n_points": "20",
                    "thickness_A_mean": 1000.0,
                    "sg_window": "0.00400",
                    "absorption_mode": "midgley_like",
                    "sg_jitter_mode": "gaussian",
                },
                "row_mr": {
                    "definition": {"instability_u": 0.02},
                    "dynamic": {
                        "Ninput": 10,
                        "Nmatched": 10,
                        "Nexcluded": 0,
                        "Nobs": 4,
                        "Nall": 10,
                        "Nmerge_input": 10,
                        "Nmerge_used": 9,
                        "Nmerge_excluded": 1,
                        "MNobs": 3,
                        "MNall": 7,
                        "Robs": 0.09,
                        "Rall": 0.10,
                        "wRall": 0.11,
                        "MRobs": 0.07,
                        "MRall": 0.08,
                        "MwRall": 0.09,
                    },
                    "born": {
                        "Ninput": 10,
                        "Nmatched": 10,
                        "Nexcluded": 0,
                        "Nobs": 4,
                        "Nall": 10,
                        "Nmerge_input": 10,
                        "Nmerge_used": 9,
                        "Nmerge_excluded": 1,
                        "MNobs": 3,
                        "MNall": 7,
                        "Robs": 0.19,
                        "Rall": 0.20,
                        "wRall": 0.21,
                        "MRobs": 0.17,
                        "MRall": 0.18,
                        "MwRall": 0.19,
                    },
                },
                "resolution_scale_endpoints": {
                    "endpoints": [
                        {
                            "label": "requested_resolution_full_scales",
                            "dmin_A": 0.7,
                            "scale_params": "100",
                            "row_mr": {
                                "dynamic": {
                                    "Nobs": 4,
                                    "Nall": 10,
                                    "Rall": 0.10,
                                    "wRall": 0.11,
                                    "MRall": 0.08,
                                    "MwRall": 0.09,
                                },
                                "kinematic": {"Rall": 0.20, "wRall": 0.21},
                            },
                        }
                    ]
                },
            }
        )
        + "\n"
    )
    prep = PrepConfig(structure="model.res", datasets=[DatasetSpec(name="dataset", hkl="INTEGRATE.HKL")])
    final_run = out_dir / "runs" / "dataset" / "adaptive_promoted_hand_final"
    final_run.mkdir(parents=True)
    (final_run / "native_dyn_calc.csv").write_text(
        "\n".join(
            [
                ",".join(
                    [
                        "row_index",
                        "h",
                        "k",
                        "l",
                        "d_A",
                        "group",
                        "detector_bin",
                        "basis_size",
                        "split",
                        "q_dyn_over_born",
                        "center_sg_inv_A",
                        "sg_window_inv_A",
                        "sg_window_raw_inv_A",
                        "sg_window_half_frames",
                        "sg_window_clipped",
                        "sg_dSg_domega_inv_A_per_deg",
                        "sg_profile_points_total",
                        "sg_profile_weight_sum",
                        "max_sg_solve_residual",
                        "dyn_residual_over_sigma",
                        "born_residual_over_sigma",
                    ]
                ),
                "0,1,0,0,0.80,0,0,32,work,0.90,0.001,0.004,0.005,0.5,0,0.002,60,1.0,0.00001,0.5,1.5",
                "1,0,1,0,0.90,0,1,36,work,1.10,-0.002,0.004,0.005,0.5,0,0.002,60,1.0,0.00002,-1.5,-2.0",
                "2,0,0,1,1.00,1,0,40,free,1.25,0.003,0.004,0.006,0.5,1,0.002,60,1.0,0.00003,2.0,2.5",
                "3,1,1,0,1.10,1,1,44,free,0.75,-0.004,0.004,0.006,0.5,0,0.002,60,1.0,0.00004,-1.0,-1.2",
            ]
        )
        + "\n"
    )
    (final_run / "summary.json").write_text(
        json.dumps(
            {
                "best_native": {
                    "nuisance": {
                        "thickness_A": 1100.0,
                        "thickness_model": "uniform_gradient",
                        "thickness_grad_frac": 1.0,
                        "frame_offset": 0.2,
                        "absorption_mode": "midgley_like",
                        "absorption_frac": 0.0,
                        "sg_jitter_mode": "gaussian",
                        "matrix_convention": "j_minus_i",
                        "native_wght_a": 0.12,
                        "native_wght_b": 0.56,
                    },
                    "metrics": {
                        "all": {
                            "dyn": {"R1_all": 0.11, "wR2_all": 0.21},
                            "born": {"R1_all": 0.17, "wR2_all": 0.31},
                        },
                        "free": {
                            "dyn": {"wR2_all": 0.22},
                            "born": {"wR2_all": 0.32},
                        },
                    },
                },
                "best_endpoint": {"q_p05": 0.2, "q_median": 0.4, "q_p95": 1.9},
            }
        )
        + "\n"
    )
    decisions = out_dir / "decisions" / "dataset"
    decisions.mkdir(parents=True)
    diagnostic = final_run.parent / "adaptive_absolute_structure_diagnostic.json"
    diagnostic.write_text(
        json.dumps(
            {
                "all": {"n": 4, "delta_chi2_other_minus_best": -25.0, "paired_row_t_delta_chi2": -5.0},
                "work": {"n": 2, "delta_chi2_other_minus_best": -16.0, "paired_row_t_delta_chi2": -4.0},
                "free": {"n": 2, "delta_chi2_other_minus_best": -9.0, "paired_row_t_delta_chi2": -3.0},
            }
        )
        + "\n"
    )
    decision_tree = decisions / "decision_tree.json"
    decision_tree.write_text(
        json.dumps(
            {
                "selected_transform": "identity",
                "selected_hand": "alternate",
                "transform_decision": {
                    "selected": {"transform": "identity", "status": "ok"},
                    "metadata_transform_decision": {
                        "status": "promoted",
                        "correlation_gap": 0.0876,
                        "records": [{"log_intensity_corr": 0.9658, "paired_rows": 3931, "unique_overlap": 4039}],
                    },
                },
                "absolute_structure_decision": {
                    "selected_hand": "original",
                    "score_delta_original_minus_alternate": -0.01,
                },
                "post_feedback_hand_promotion": {
                    "status": "promoted",
                    "active_hand": "original",
                    "opposite_hand": "alternate",
                    "diagnostic": str(diagnostic),
                    "score_delta_final_minus_opposite": 0.08,
                },
            }
        )
        + "\n"
    )
    (out_dir / "grond_progress.json").write_text(
        json.dumps(
            {
                "completed": {
                    "old": {
                        "returncode": 0,
                        "elapsed_seconds": 1.0,
                        "log": str(out_dir / "logs" / "old.log"),
                        "command": [
                            "python",
                            "/old/GROND/src/grond/legacy/engine/native_row_regularize.py",
                        ],
                    },
                    "new": {
                        "returncode": 0,
                        "elapsed_seconds": 2.0,
                        "log": str(out_dir / "logs" / "new.log"),
                        "command": [
                            "python",
                            "/new/GROND/src/grond/legacy/engine/native_row_regularize.py",
                        ],
                    },
                    "failed": {
                        "returncode": 1,
                        "elapsed_seconds": 99.0,
                        "log": str(out_dir / "logs" / "failed.log"),
                        "command": ["false"],
                    },
                },
                "finished": {
                    "status": "ok",
                    "invocation_elapsed_seconds": 1.25,
                    "cumulative_stage_elapsed_seconds": 3.0,
                    "stage_elapsed_seconds": 3.0,
                    "completed_stages": 2,
                    "failed_stages": 1,
                    "estimated_stages": 2,
                },
            }
        )
        + "\n"
    )
    (out_dir / "sfac_check.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "cell": {"endpoint_wavelength_A": 0.025079},
                "sfac_rows": [{"kind": "numeric_coefficients", "element": "C"}],
            }
        )
        + "\n"
    )
    result = {
        "model": "final.res",
        "primary_run": str(final_run),
        "decision_tree": str(decision_tree),
        "endpoint_calibration": {
            "report": str(out_dir / "endpoint_calibration" / "dataset" / "REPORT.md"),
            "publication_selection": {
                "publication_endpoint": {
                    "spec": "adaptive_80_det_8",
                    "wght_mode": "freeze-zero",
                    "scale_params": 143,
                    "shelxl_R1all": 0.0996,
                    "shelxl_wR2all": 0.1539,
                    "shelxl_GooF": 1.037,
                    "native_free_wR2_all": 0.2318,
                    "sigma_scale": 0.584,
                }
            },
        },
    }
    args = argparse.Namespace(out_dir=str(out_dir))

    info = cli.write_quick_stats(prep, result, args)
    text = (out_dir / "QUICK_STATS.md").read_text()

    assert info["status"] == "ok"
    assert "requested_resolution_full_scales" in text
    assert "10.00%" in text
    assert "selected_scale_replay" in text
    assert "15.39%" in text
    assert "Resume Warning" in text
    assert "mixed-code resume" in text
    assert "Stage Timings" in text
    assert "Recorded workflow wall time: `0m01s`" in text
    assert "Cumulative completed-stage runtime: `0m03s`" in text
    assert "Total completed stage wall time" not in text
    assert "not workflow wall time" in text
    assert "Input Contract, Hand, And Paired T-Style Diagnostics" in text
    assert "SFAC / endpoint wavelength check | status `ok`; endpoint wavelength `0.025079` A" in text
    assert "selected HKL transform | `identity`" in text
    assert "final hand | `alternate`; post-feedback status `promoted`" in text
    assert (
        "final hand paired t-style, all/work/free | `t 5.000 (n 4, df 3)` / "
        "`t 4.000 (n 2, df 1)` / `t 3.000 (n 2, df 1)`" in text
    )
    assert "not a calibrated normal Z score" in text
    assert "free-row I residual Z | n `2`; RMSZ `1.581`" in text
    assert "thickness | `110.00 nm mean (1100.0 A); uniform-gradient support 0.00-220.00 nm`" in text
    assert "Legacy Native-Objective Audit Matrix (Not Table 1)" in text
    assert "raw signed rows (I), Bloch dynamical" in text
    assert "merged SHELXL endpoint (F/F^2)" in text
    assert "Legacy Native-Objective Delta Matrix (Not Table 1)" in text
    assert "final dynamical rows minus final matched Born rows" in text
    assert "Row Intensity R-Factor Definitions" in text
    assert "Raw signed-row intensity `wR(I)all` is `sqrt(sum_r w_r (Iobs_r - Icalc_r)^2 / sum_r w_r Iobs_r^2)`" in text
    assert "JANA/PETS-Style Row And Post-Refinement Merged Statistics By Resolution/Scale Contract" in text
    assert "Canonical Final Statistics" in text
    assert (out_dir / "TABLE1_FINAL_STATISTICS.md").exists()
    table1 = json.loads((out_dir / "TABLE1_FINAL_STATISTICS.json").read_text())
    assert table1["schema"] == "grond.table1_final_statistics.v1"
    assert table1["reporting_contract"]["row_instability_u"] == pytest.approx(0.02)
    assert table1["jana_pets_rows"][0]["status"] == "not_available"
    assert table1["jana_pets_rows"][1]["model"].startswith("GROND Born")
    assert table1["jana_pets_rows"][1]["Ninput"] == 10
    assert table1["jana_pets_rows"][1]["Nmerge_used"] == 9
    assert table1["jana_pets_rows"][1]["Nmerge_excluded"] == 1
    assert table1["shelxl_endpoints"][1]["status"] == "not_available"
    assert table1["shelxl_endpoints"][1]["WGHT"] is None
    assert table1["shelxl_endpoints"][1]["EXTI"] is None
    table1_text = (out_dir / "TABLE1_FINAL_STATISTICS.md").read_text()
    assert "instability factor `u=0.02`" in table1_text
    assert "GROND Born corrected-HKL endpoint | not available | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a" in table1_text
    endpoint_header = (out_dir / "TABLE1_SHELXL_ENDPOINTS.csv").read_text().splitlines()[0]
    row_header = (out_dir / "TABLE1_JANA_ROWS.csv").read_text().splitlines()[0]
    assert "R1gt" in endpoint_header and "wR2all" in endpoint_header
    assert "Robs" not in endpoint_header and "MRall" not in endpoint_header
    assert "Robs" in row_header and "MRall" in row_header and "MwRall" in row_header
    assert "R1gt" not in row_header and "wR2all" not in row_header and "GooF" not in row_header
    final_text = (out_dir / "FINAL_SUMMARY.md").read_text()
    assert "### A. SHELXL Endpoint Statistics" in final_text
    assert "### B. JANA/PETS-Style Row And Post-Refinement Merged Statistics" in final_text
    assert "Nmerge used/excluded" in final_text
    assert "free/held-out weighted R" not in final_text
    assert "Bloch matrix and row-contract diagnostics" in text
    bloch_diagnostics = out_dir / "BLOCH_DIAGNOSTICS.md"
    assert bloch_diagnostics.exists()
    bloch_text = bloch_diagnostics.read_text()
    assert "GROND Bloch Diagnostics" in bloch_text
    assert "Bethe Approximation" in bloch_text
    assert "Sg, Rocking, And Quadrature" in bloch_text
    assert "center excitation error Sg" in bloch_text
    assert "q = Idyn/Iborn" in bloch_text
    assert "Bloch matrix and row-contract diagnostics" in (out_dir / "FINAL_SUMMARY.md").read_text()
    assert info["console"]["files"]["bloch_diagnostics"] == str(bloch_diagnostics)
    assert info["console"]["final_hand_t_style_all"] == pytest.approx(5.0)
    assert info["console"]["final_hand_t_style_all_n"] == 4
    assert info["console"]["final_hand_t_style_all_df"] == 3
    assert info["console"]["final_hand_t_style_free"] == pytest.approx(3.0)
    assert info["console"]["final_hand_t_style_free_n"] == 2
    assert info["console"]["final_hand_t_style_free_df"] == 1
    assert "legacy field names only" in info["console"]["final_hand_z_compatibility_note"]
