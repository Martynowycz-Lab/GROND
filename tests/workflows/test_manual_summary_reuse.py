"""Historical summary reuse must distinguish a saved hash from present-day equality."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from grond.io.config import DatasetSpec
from grond.manual.scouting import _validate_reused_summary_binding, manual_thickness


def _inputs(tmp_path, *, hash_kind="flat"):
    hkl = tmp_path / "INTEGRATE.HKL"
    hkl.write_text("original measured observations\n")
    digest = hashlib.sha256(hkl.read_bytes()).hexdigest()
    inputs = {"hkl": str(hkl)}
    if hash_kind == "flat":
        inputs["hkl_sha256"] = digest
    elif hash_kind == "structured":
        inputs["hkl"] = {"path": str(hkl), "sha256": digest}
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "inputs": inputs,
        "config": {"hkl_transform": "identity", "min_d_spacing": 0.7, "min_peak": 90, "geometry_percentile": 99},
        "best_native": {"nuisance": {"thickness_A": 1000.0}},
    }))
    dataset = DatasetSpec(name="one", hkl=str(hkl), hkl_transform="identity")
    return dataset, hkl, summary, digest


@pytest.mark.parametrize("hash_kind", ["flat", "structured"])
def test_unchanged_manual_summary_reuse_verifies_recorded_hash(tmp_path, hash_kind):
    dataset, hkl, summary, digest = _inputs(tmp_path, hash_kind=hash_kind)
    binding = _validate_reused_summary_binding(dataset, summary)

    assert binding["historical_source_verified"] is True
    assert binding["verification_status"] == "verified_recorded_source_hash"
    assert set(binding["recorded_source_hkl_hashes"].values()) == {digest}
    assert binding["source_hkl"] == str(hkl)
    assert binding["source_hkl_sha256"] == digest
    assert binding["warnings"] == []


def test_same_path_hkl_mutation_rejects_old_manual_summary(tmp_path):
    dataset, hkl, summary, original_hash = _inputs(tmp_path)
    hkl.write_text("modified measured observations\n")
    current_hash = hashlib.sha256(hkl.read_bytes()).hexdigest()

    with pytest.raises(SystemExit, match="source HKL changed") as caught:
        _validate_reused_summary_binding(dataset, summary)

    message = str(caught.value)
    assert str(hkl) in message and str(summary) in message
    assert original_hash in message and current_hash in message
    assert "original_path=" in message and "current_dataset_path=" in message


def test_all_available_recorded_hkl_hashes_must_agree(tmp_path):
    dataset, _hkl, summary, _digest = _inputs(tmp_path, hash_kind="structured")
    payload = json.loads(summary.read_text())
    payload["inputs"]["hkl_sha256"] = "0" * 64
    summary.write_text(json.dumps(payload))

    with pytest.raises(SystemExit, match="source HKL changed"):
        _validate_reused_summary_binding(dataset, summary)


def test_malformed_recorded_hash_is_not_treated_as_legacy_missing_hash(tmp_path):
    dataset, _hkl, summary, _digest = _inputs(tmp_path)
    payload = json.loads(summary.read_text())
    payload["inputs"]["hkl_sha256"] = "not-a-source-hash"
    summary.write_text(json.dumps(payload))

    with pytest.raises(SystemExit, match="invalid recorded source-HKL SHA256"):
        _validate_reused_summary_binding(dataset, summary)


def test_legacy_missing_hash_remains_explicitly_unverified_in_handoff(tmp_path, capsys):
    dataset, _hkl, summary, digest = _inputs(tmp_path, hash_kind="missing")
    binding = _validate_reused_summary_binding(dataset, summary)
    assert binding["historical_source_verified"] is False
    assert binding["verification_status"] == "unverified_legacy_missing_source_hash"
    assert binding["source_hkl_sha256"] == digest
    assert binding["recorded_source_hkl_hashes"] == {}
    assert "historical calculation is unverified" in binding["warnings"][0]

    model = tmp_path / "model.res"
    model.write_text("TITL source model\n")
    prep = tmp_path / "prep.json"
    prep.write_text(json.dumps({"structure": str(model), "datasets": [{"name": "one", "hkl": dataset.hkl}]}))
    workflow_path = tmp_path / "manual_workflow.json"
    workflow_path.write_text(json.dumps({
        "kind": "grond_manual_workflow", "root": str(tmp_path), "prep": str(prep),
        "stages": {"scout": {"status": "ready"}},
    }))
    args = SimpleNamespace(workflow=str(workflow_path), skip=True, summary=[f"one={summary}"])

    assert manual_thickness(args) == 0
    handoff = json.loads((tmp_path / "02_thickness" / "handoff.json").read_text())
    assert handoff["command"]["status"] == "skipped_with_supplied_summaries_historical_hash_unavailable"
    assert handoff["command"]["bindings"]["one"]["historical_source_verified"] is False
    assert any("historical calculation is unverified" in warning for warning in handoff["warnings"])
    assert "historical calculation is unverified" in capsys.readouterr().out
