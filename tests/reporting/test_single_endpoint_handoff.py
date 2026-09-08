"""Single-dataset finish must use bound rows and report a worse endpoint."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from grond.reports.terminal import _format_terminal_finish_summary
from grond.workflows import finish_maps


@pytest.mark.parametrize("recomputed_model_matches", [True, False])
def test_native_maps_choose_recomputed_rows_only_for_the_selected_model(tmp_path, monkeypatch, recomputed_model_matches):
    selected = tmp_path / "selected.res"
    selected.write_text("selected endpoint model\n")
    expected = hashlib.sha256(selected.read_bytes()).hexdigest()
    final = tmp_path / "final_row_recompute"
    final.mkdir()
    (final / "native_dyn_calc.csv").write_text("recomputed observations\n")
    (final / "summary.json").write_text(json.dumps({"inputs": {
        "res": str(selected), "res_sha256": expected if recomputed_model_matches else "different-model"}}))
    origin = tmp_path / "cycle_05"
    origin.mkdir()
    original_model = tmp_path / "before_ls1.res"
    original_model.write_text("model used for the earlier Bloch calculation\n")
    (origin / "summary.json").write_text(json.dumps({"inputs": {
        "res": str(original_model), "res_sha256": hashlib.sha256(original_model.read_bytes()).hexdigest()}}))
    (origin / "native_dyn_calc.csv").write_text("earlier observations\n")
    endpoint = {"source_kind": "native_corrected_hkl", "label": "final_ls1",
                "run_dir": str(origin / "endpoint_lam_1p0"), "res_sha256": expected,
                "endpoint_origin_summary": str(origin / "summary.json")}
    monkeypatch.setattr(finish_maps, "_publication_endpoint_from_result", lambda _result: endpoint)
    result = {"primary_run": str(final), "model": str(selected)}
    tables = finish_maps._finish_native_row_tables(result)
    assert tables == [("final_ls1", (final if recomputed_model_matches else origin) / "native_dyn_calc.csv")]
    model, contract = finish_maps._finish_native_model_contract(result, SimpleNamespace(), tables)
    assert model == (selected if recomputed_model_matches else original_model)
    assert contract["status"] == "hash_bound_row_model"


def test_single_dataset_fresh_baseline_warning_keeps_endpoint_available(monkeypatch):
    endpoint = {"artifact_provenance_ok": True, "pathology_status": "complete",
                "metric_consistency_ok": True, "promotion_fields_ok": True, "R1_all": .2008}
    monkeypatch.setattr(finish_maps, "_endpoint_calibration_publication_endpoint", lambda _result: {})
    monkeypatch.setattr(finish_maps, "_chooser_publication_endpoint", lambda _result: endpoint)
    monkeypatch.setattr(finish_maps, "_validated_fresh_conventional_baseline", lambda _result:
                        {"R1_all": .1693, "baseline_source": "fresh_supplied_input_conventional_control"})
    selected = finish_maps._publication_endpoint_from_result({"primary_run": "missing-final-summary"})
    gate = selected["baseline_no_harm_gate"]
    assert gate["passed"] and not gate["no_harm"]
    assert gate["status"] == "warning_worse_than_baseline"
    assert "0.2008" in gate["warnings"][0] and "0.1693" in gate["warnings"][0]
    terminal = _format_terminal_finish_summary({"publication_endpoint": {
        "R1all": .2008, "GooF": .899, "baseline_comparison": gate}})
    assert "Warning: " + gate["warnings"][0] in terminal
