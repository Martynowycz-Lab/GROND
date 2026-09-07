"""CLI parser backends regression tests."""







import grond.cli as cli
from grond.cli import (
    build_parser,
)


def test_experimental_row_plan_cache_flags_parse():
    default_auto = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
        ]
    )
    assert default_auto.row_plan_cache == "auto"
    assert cli.resolved_row_plan_cache(default_auto, fixed_model=True) == "auto"
    assert cli.resolved_row_plan_cache(default_auto, fixed_model=False) == "off"

    auto = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--row-plan-cache",
            "readwrite",
        ]
    )
    assert auto.row_plan_cache == "readwrite"
    assert cli.resolved_row_plan_cache(auto, fixed_model=True) == "readwrite"
    assert cli.resolved_row_plan_cache(auto, fixed_model=False) == "readwrite"

    exact = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--row-plan-cache",
            "exact",
        ]
    )
    assert exact.row_plan_cache == "exact"

    raw = build_parser().parse_args(
        [
            "raw-polish",
            "--out-dir",
            "out",
            "--hkl",
            "INTEGRATE.HKL",
            "--structure",
            "model.res",
            "--row-plan-cache",
            "rebuild",
        ]
    )
    assert raw.row_plan_cache == "rebuild"

    ls1 = build_parser().parse_args(
        [
            "validate-ls1",
            "--source-summary",
            "summary.json",
            "--start-res",
            "model.res",
            "--out-dir",
            "validate_ls1",
            "--row-plan-cache",
            "readwrite",
        ]
    )
    assert ls1.row_plan_cache == "readwrite"


def test_auto_parser_accepts_cuda_jax_backend_explicitly():
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--compute-backend",
            "jax_cuda_padded",
            "--jax-batch-size",
            "192",
            "--jax-matrix-bucket-size",
            "8",
        ]
    )
    assert args.compute_backend == "jax_cuda_padded"
    assert args.jax_batch_size == 192
    assert args.jax_matrix_bucket_size == 8


def test_auto_parser_accepts_mps_real_expm_backend_explicitly():
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--compute-backend",
            "jax_mps_real_expm_padded",
            "--jax-batch-size",
            "32",
            "--jax-matrix-bucket-size",
            "8",
        ]
    )
    assert args.compute_backend == "jax_mps_real_expm_padded"
    assert args.jax_batch_size == 32
    assert args.jax_matrix_bucket_size == 8


def test_auto_parser_accepts_bethe_backend_controls():
    args = build_parser().parse_args(
        [
            "auto",
            "--out-dir",
            "out",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--compute-backend",
            "scipy_bethe",
            "--bethe-strong-count",
            "64",
            "--bethe-variant",
            "schur_target",
            "--bethe-promote-denom",
            "0.004",
            "--bethe-denom-eps",
            "1e-6",
        ]
    )
    assert args.compute_backend == "scipy_bethe"
    assert args.bethe_strong_count == 64
    assert args.bethe_variant == "schur_target"
    assert args.bethe_promote_denom == 0.004
    assert args.bethe_denom_eps == 1e-6


def test_raw_polish_parser_accepts_bethe_backend_controls():
    args = build_parser().parse_args(
        [
            "raw-polish",
            "--hkl",
            "INTEGRATE.HKL",
            "--structure",
            "model.res",
            "--out-dir",
            "raw",
            "--compute-backend",
            "scipy_bethe",
            "--bethe-strong-count",
            "64",
            "--bethe-variant",
            "schur_target",
        ]
    )
    assert args.command == "raw-polish"
    assert args.compute_backend == "scipy_bethe"
    assert args.bethe_strong_count == 64
    assert args.bethe_variant == "schur_target"


def test_raw_polish_parser_exposes_forward_model_knobs():
    args = build_parser().parse_args(
        [
            "raw-polish",
            "--out-dir",
            "out",
            "--hkl",
            "INTEGRATE.HKL",
            "--structure",
            "model.res",
            "--cbf-dir",
            "cbf",
            "--no-maps",
        ]
    )
    assert args.raw_integration_modes == "sum,bloch_profile_fit"
    assert args.matrix_conventions == "j_minus_i,i_minus_j"
    assert args.prediction_cache_compression == "none"
    assert args.make_maps is False
