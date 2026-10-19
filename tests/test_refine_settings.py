"""grond refine: by default the full grond auto calculation on the new model with the run's options."""

import json

import grond.cli_support.entrypoint as entrypoint


def _run(tmp_path, monkeypatch, *refine_args):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({"argv": [
        "auto", "--prep", "p.json", "--out-dir", "o", "--free-set", "on", "--min-d-spacing", "0.80",
        "--hand-control", "--preset", "fast", "--prediction-cache-dir", "c"]}))
    model = tmp_path / "edited.res"
    model.write_text("TITL edited\n")
    checkpoint = tmp_path / "grond_continue_checkpoint.json"
    checkpoint.write_text(json.dumps({
        "kind": "grond_continue_checkpoint", "status": "ready", "run_dir": str(run), "dataset_count": 2,
        "recommended_start_res": str(model), "hkl_transform": "identity",
        "datasets": [{"name": name, "hkl": f"/data/{name}/INTEGRATE.HKL", "xds_dir": f"/data/{name}",
                      "baseline_hkl": "/data/model.hkl", "hkl_transform": "-h,-k,l", "min_d_spacing": 0.8,
                      "min_peak": 90.0, "split_contract": f"/runs/{name}/split.json"} for name in ("c1", "c2")]}))
    calls = []
    monkeypatch.setattr(entrypoint, "main", lambda argv: calls.append(argv) or 0, raising=True)
    from grond.cli_options.parser import build_parser
    from grond.commands.push import cmd_push_endpoint

    argv = ["refine", "--checkpoint", str(checkpoint), "--out-dir", str(tmp_path / "out"), *refine_args]
    args, extra = build_parser().parse_known_args(argv)
    args.argv = argv
    assert cmd_push_endpoint(args) == 0
    return calls[0], json.loads((tmp_path / "out" / "refine_prep.json").read_text())


def test_refine_runs_auto_with_the_runs_options_and_keeps_indexing_and_split(tmp_path, monkeypatch):
    auto_argv, prep = _run(tmp_path, monkeypatch, "--cycles", "2", "--nested-n-basis", "48")
    assert auto_argv[0] == "auto"
    # the run's own options, not its input and output locations or its hand control
    assert auto_argv[1:7] == ["--free-set", "on", "--min-d-spacing", "0.80", "--preset", "fast"]
    assert "--hand-control" not in auto_argv and "c" not in auto_argv
    # options given to refine go to auto and come after the run's
    assert auto_argv[7:11] == ["--shelxl-feedback-cycles", "2", "--nested-n-basis", "48"]
    assert auto_argv[-3:] == [str(tmp_path / "out"), "--trust-hkl-transform", "--no-hand-control"]
    # no baseline HKL (the new model is not refined again against the kinematic data); indexing and split kept
    assert prep["baseline_hkl"] == "" and all(ds["baseline_hkl"] == "" for ds in prep["datasets"])
    assert [ds["hkl_transform"] for ds in prep["datasets"]] == ["-h,-k,l", "-h,-k,l"]
    assert [ds["split_contract"] for ds in prep["datasets"]] == ["/runs/c1/split.json", "/runs/c2/split.json"]
    assert prep["structure"].endswith("edited.res")


def test_refine_rejects_reuse_only_options_without_reuse_settings(tmp_path, monkeypatch):
    import pytest

    with pytest.raises(SystemExit, match="--reuse-settings"):
        _run(tmp_path, monkeypatch, "--dataset-n-basis", "c1=100")
