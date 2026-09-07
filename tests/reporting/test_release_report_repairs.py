"""Boundary validation and truthful report status without changing calculations."""

import json
from types import SimpleNamespace

import pytest

import grond.commands.finish as owner_commands_finish
import grond.reports.quick_stats as owner_reports_quick_stats
import grond.workflows.finish_maps as owner_workflows_finish_maps
from grond import cli, manual_workflow


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("entrypoint", ["multi", "scout", "manual"])
def test_invalid_hand_threshold_stops_before_computation(tmp_path, monkeypatch, threshold, entrypoint):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid threshold reached geometry, computation, or model materialization")

    for name in ("absolute_structure_policy", "native_stage_cmd", "run_logged", "make_alternate_hand_model", "_combine_adjusted_hand_z"):
        monkeypatch.setattr(cli, name, forbidden)
    class UnreadEvidence(dict):
        def items(self):
            raise AssertionError("invalid threshold reached evidence processing")

    output = tmp_path / "hand_output"
    with pytest.raises(ValueError, match="threshold must be finite and positive"):
        if entrypoint == "multi":
            cli.run_multi_final_elastic_hand_selection(
                SimpleNamespace(datasets=[]), SimpleNamespace(hand_flip_zkp_all=threshold),
                "python", tmp_path / "logs", output,
                final_model="unused.res", selected_hand="original", final_runs={}, shared_hand_decision={},
            )
        elif entrypoint == "scout":
            cli.aggregate_shared_hand_decisions(UnreadEvidence(), SimpleNamespace(hand_min_adjusted_z=threshold, hand_accept_delta=.01))
        else:
            manual_workflow._hand_evidence(UnreadEvidence(), min_z_all=threshold)
    assert not output.exists()


def test_table1_reports_actual_immutable_source_sigma_rule(tmp_path):
    cli._write_final_statistics_table1(
        tmp_path, row_metrics={}, row_metrics_path=tmp_path / "rows.json",
        input_shelxl={}, publication_shelxl={}, publication_endpoint={}, file_block={},
        input_controls={}, final_row_table="rows.csv",
    )
    payload = json.loads((tmp_path / "TABLE1_FINAL_STATISTICS.json").read_text())
    rule = payload["reporting_contract"]["row_sigma_rule"]
    assert "immutable source_sigma in target_I units" in rule
    assert "sigma_eff only as a legacy fallback" in rule
    assert "not otherwise folded" in rule
    text = (tmp_path / "TABLE1_FINAL_STATISTICS.md").read_text()
    assert "Weights use immutable `source_sigma`" in text
    assert "`sigma_eff` only as a legacy fallback" in text
    assert "Weights use `sigma_eff` from the frozen row table" not in text


def _missing_endpoint_result(tmp_path, monkeypatch, kind):
    model = tmp_path / "endpoint.res"
    fcf = model.with_suffix(".fcf")
    model.write_text("endpoint retained\n")
    fcf.write_text("data_test\n_shelx_refln_list_code 6\n")
    endpoint = {} if kind == "missing_endpoint" else {"endpoint_res": str(model)}
    monkeypatch.setattr(owner_commands_finish, "_publication_endpoint_from_result", lambda _result: endpoint)
    monkeypatch.setattr(owner_workflows_finish_maps, "_publication_endpoint_from_result", lambda _result: endpoint)
    monkeypatch.setattr(owner_reports_quick_stats, "_publication_endpoint_from_result", lambda _result: endpoint)
    monkeypatch.setattr(owner_workflows_finish_maps, "_shelx2map_executable", lambda _args: str(tmp_path / "missing_shelx2map"))
    return model


@pytest.mark.parametrize("kind", ["missing_endpoint", "missing_tool"])
def test_maps_report_partial_and_specific_cause(tmp_path, monkeypatch, kind):
    model = _missing_endpoint_result(tmp_path, monkeypatch, kind)
    monkeypatch.setattr(owner_workflows_finish_maps, "run_finish_native_dynamic_maps", lambda *_args: {"status": "ok", "outputs": {}})
    result = cli.run_finish_maps(tmp_path, {}, SimpleNamespace(map_mode="both", make_maps=True))
    assert result["status"] == "partial"
    cause = "missing_endpoint_res" if kind == "missing_endpoint" else "missing_shelx2map"
    assert cause in result["warnings"][0]
    assert model.read_text() == "endpoint retained\n"


def test_all_maps_success_remains_ok_and_physical_error_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(owner_workflows_finish_maps, "run_finish_native_dynamic_maps", lambda *_args: {"status": "ok"})
    monkeypatch.setattr(owner_workflows_finish_maps, "run_finish_endpoint_maps", lambda *_args: {"status": "ok"})
    args = SimpleNamespace(map_mode="both", make_maps=True)
    assert cli.run_finish_maps(tmp_path, {}, args) == {"mode": "both", "native_dynamic": {"status": "ok"}, "shelx2map": {"status": "ok"}, "status": "ok", "warnings": []}
    def failed(*_args):
        raise RuntimeError("native-map row model hash changed")
    monkeypatch.setattr(owner_workflows_finish_maps, "run_finish_native_dynamic_maps", failed)
    result = cli.run_finish_maps(tmp_path, {}, args)
    assert result["status"] == "partial"
    assert result["native_dynamic"]["status"] == "failed"
    assert "row model hash changed" in result["warnings"][0]


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("kind", ["missing_endpoint", "missing_tool"])
def test_finish_retains_endpoint_and_exposes_missing_map_status(tmp_path, monkeypatch, capsys, kind, json_output):
    model = _missing_endpoint_result(tmp_path, monkeypatch, kind)
    (tmp_path / "grond_summary.json").write_text(json.dumps({"model": str(model)}))
    (tmp_path / "run_prep.json").write_text("{}")
    monkeypatch.setattr(owner_commands_finish, "_finish_run_dir", lambda _args: tmp_path)
    monkeypatch.setattr(owner_commands_finish, "load_prep", lambda _path: SimpleNamespace())
    monkeypatch.setattr(owner_commands_finish, "write_quick_stats", lambda *_args: {"console": {"files": {"publication_res": str(model)}}})
    monkeypatch.setattr(owner_commands_finish, "_quick_progress_finish_info", lambda _path: {})
    monkeypatch.setattr(owner_reports_quick_stats, "_quick_progress_finish_info", lambda _path: {})
    monkeypatch.setattr(owner_commands_finish, "_format_terminal_finish_summary", lambda payload: "Endpoint delivered")
    args = SimpleNamespace(map_mode="shelx2map", make_maps=True, json=json_output)
    assert cli.cmd_finish(args) == 0
    output = capsys.readouterr().out
    cause = "missing_endpoint_res" if kind == "missing_endpoint" else "missing_shelx2map"
    manifest = json.loads((tmp_path / "finish_manifest.json").read_text())
    assert manifest["status"] == "warning"
    assert cause in manifest["warnings"][0]
    assert model.read_text() == "endpoint retained\n"
    if json_output:
        payload = json.loads(output)
        assert payload["status"] == "warning"
        assert cause in payload["warnings"][0]
    else:
        assert "Endpoint delivered" in output
        assert "Map warning:" in output and cause in output
