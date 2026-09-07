"""CLI profiles and workers regression tests."""





from pathlib import Path

from grond.cli import (
    apply_speed_profile,
    build_parser,
    estimate_model_nonh_atoms,
    transform_scout_job_count,
)


def test_full_fast_profile_pins_cheap_automatic_work():
    args = build_parser().parse_args(["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", "model.res", "--full-fast"])
    profile = apply_speed_profile(args)
    assert profile["profile"] == "full_fast"
    assert args.screen_n_basis == 24
    assert args.screen_n_points == 10
    assert args.transform_rerank_n_basis == 24
    assert args.transform_rerank_n_points == 10
    assert args.n_points == 10
    assert args.structure_objective_rows == 250
    assert args.structure_max_nfev == 2
    assert args.shelxl_iterations == 1
    assert args.endpoint_feedback == "off"
    assert args.observation_polish == "off"
    assert args.forensic_depth == "quick"


def test_full_fast_profile_can_lower_transform_rerank_cost():
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--full-fast",
            "--full-fast-n-basis",
            "10",
            "--full-fast-n-points",
            "5",
        ]
    )
    apply_speed_profile(args)
    assert args.transform_rerank_n_basis == 10
    assert args.transform_rerank_n_points == 5


def test_transform_scout_outer_jobs_account_for_workers_per_job():
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--workers",
            "12",
            "--transform-scout-workers-per-job",
            "3",
        ]
    )
    assert transform_scout_job_count(args, task_count=20) == 4


def test_transform_scout_outer_jobs_can_be_capped_explicitly():
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--workers",
            "64",
            "--transform-scout-jobs",
            "8",
        ]
    )
    assert transform_scout_job_count(args, task_count=20) == 8
    assert transform_scout_job_count(args, task_count=3) == 3


def test_full_fast_profile_uses_lighter_structure_budget_for_larger_models(tmp_path: Path):
    model = tmp_path / "model.res"
    atoms = "\n".join(f"C{i:03d} 1 0.{i % 9} 0.{(i + 1) % 9} 0.{(i + 2) % 9} 11.0 0.05" for i in range(52))
    model.write_text(f"TITL test\nCELL 0.025 10 10 10 90 90 90\nSFAC C H\n{atoms}\nEND\n")
    assert estimate_model_nonh_atoms(str(model)) == 52
    args = build_parser().parse_args(["auto", "--out-dir", "out", "--xds-dir", "xds", "--structure", str(model), "--full-fast"])
    profile = apply_speed_profile(args, str(model))
    assert profile["structure_budget"]["non_h_atoms"] == 52
    assert args.structure_objective_rows == 160
    assert args.structure_max_nfev == 1
