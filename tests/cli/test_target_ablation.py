"""CLI target ablation regression tests."""







import grond.cli as cli
from grond.cli import (
    build_parser,
)


def test_target_ablation_passes_negative_signed_transform_as_one_argument(monkeypatch, tmp_path):
    monkeypatch.setenv("GROND_ENABLE_INTERNAL_CLI", "1")
    parser = build_parser()
    args = parser.parse_args(
        [
            "target-ablation",
            "--xds-dir",
            "xds",
            "--structure",
            "model.res",
            "--baseline-hkl",
            "model.hkl",
            "--out-dir",
            "ablation",
            "--hkl-transform=-h,k,-l",
        ]
    )
    command = cli._target_ablation_auto_command(args, "recorded_proxy", tmp_path)
    assert "--hkl-transform=-h,k,-l" in command
    inner = parser.parse_args(command[3:])
    assert inner.command == "auto"
    assert inner.hkl_transform == "-h,k,-l"
