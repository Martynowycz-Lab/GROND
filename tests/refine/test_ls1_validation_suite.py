
from types import SimpleNamespace

import grond.commands.validation as owner_commands_validation
from grond.legacy.engine.ls1_validation_suite import as_feedback_args


def test_validate_ls1_cli_forwards_internal_endpoint_controls(monkeypatch, tmp_path):
    from grond import cli

    captured = {}

    def fake_run(cmd, *, check):
        captured["cmd"] = cmd
        captured["check"] = check

    monkeypatch.setattr(owner_commands_validation, "choose_python", lambda _requested: "python")
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    rc = cli.main(
        [
            "validate-ls1",
            "--source-summary",
            str(tmp_path / "summary.json"),
            "--start-res",
            str(tmp_path / "start.res"),
            "--out-dir",
            str(tmp_path / "out"),
            "--endpoint-ls-cycles",
            "7",
            "--endpoint-wght-mode",
            "inherit",
        ]
    )

    assert rc == 0
    assert captured["check"] is True
    cmd = captured["cmd"]
    assert cmd[cmd.index("--endpoint-ls-cycles") + 1] == "7"
    assert cmd[cmd.index("--endpoint-wght-mode") + 1] == "inherit"


def test_feedback_namespace_propagates_inner_ls_and_weight_policy():
    args = SimpleNamespace(
        python="python",
        hkl="rows.hkl",
        baseline_hkl="base.hkl",
        hkl_transform="identity",
        min_d_spacing=0.8,
        min_peak=90.0,
        geometry_percentile=99.0,
        shelxl="shelxl",
        post_bloch_sigma_policy="uniform_z_rms",
        post_bloch_sigma_holdout_fraction=0.2,
        post_bloch_sigma_min_scale=0.25,
        post_bloch_sigma_max_scale=4.0,
        rescale="median-iobs",
        cache_compression="none",
        compute_backend="scipy",
        row_plan_cache="off",
        jax_batch_size=0,
        jax_matrix_bucket_size=0,
        bethe_strong_count=48,
        bethe_variant="schur_target",
        bethe_promote_denom=0.005,
        bethe_denom_eps=1e-5,
        topup_d_min=0.8,
        max_rows=0,
        workers=2,
        chunksize=4,
        endpoint_ls_cycles=7,
        endpoint_wght_mode="freeze-zero",
    )

    feedback = as_feedback_args(args, lambda_grid="1.0", shelxl_iterations=3)

    assert feedback.shelxl_iterations == 3
    assert feedback.shelxl == "shelxl"
    assert feedback.endpoint_ls_cycles == 7
    assert feedback.endpoint_wght_mode == "freeze-zero"
