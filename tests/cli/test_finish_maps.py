"""CLI finish maps regression tests."""

import argparse
import json
from pathlib import Path

import pytest

import grond.cli as cli
import grond.workflows.finish_maps as owner_workflows_finish_maps


def test_finish_native_maps_use_publication_runs_for_combined_endpoint(tmp_path: Path):
    run_a = tmp_path / "weighted_runs" / "a"
    run_b = tmp_path / "weighted_runs" / "b"
    run_c = tmp_path / "weighted_runs" / "c_endpoint_only"
    result = {
        "combined": str(tmp_path / "runs" / "combined"),
        "publication_runs": {"b": str(run_b), "a": str(run_a)},
        "combined_input_runs": {"b": str(run_b), "a": str(run_a), "c": str(run_c)},
        "publication_endpoint_chooser": {
            "status": "ok",
            "best_physical_endpoint": {
                "label": "combined",
                "source_kind": "combined_corrected_hkl",
                "GooF": 1.0,
                "R1_all": 0.10,
                "wR2_all": 0.20,
                "lst": str(tmp_path / "runs" / "combined" / "combined_endpoint_lam_1p0" / "multi_dyn_corrected.lst"),
            },
        },
    }

    row_tables = cli._finish_native_row_tables(result)

    assert row_tables == [
        ("a", run_a / "native_dyn_calc.csv"),
        ("b", run_b / "native_dyn_calc.csv"),
    ]


def test_finish_native_model_is_hash_bound_to_frozen_rows_not_endpoint(tmp_path: Path):
    import hashlib

    row_model = tmp_path / "row_model.res"
    endpoint_model = tmp_path / "endpoint_model.res"
    row_model.write_text("TITL row model\n")
    endpoint_model.write_text("TITL endpoint model\n")
    run_a = tmp_path / "weighted" / "a"
    run_b = tmp_path / "weighted" / "b"
    for run in (run_a, run_b):
        run.mkdir(parents=True)
        (run / "native_dyn_calc.csv").write_text("h,k,l\n")
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "inputs": {
                        "res": str(row_model),
                        "res_sha256": hashlib.sha256(row_model.read_bytes()).hexdigest(),
                    }
                }
            )
        )
    result = {
        "combined": str(tmp_path / "combined"),
        "model": str(row_model),
        "publication_runs": {"a": str(run_a), "b": str(run_b)},
        "publication_endpoint_chooser": {
            "status": "ok",
            "best_physical_endpoint": {
                "source_kind": "combined_corrected_hkl",
                "res": str(endpoint_model),
            },
        },
    }
    row_tables = cli._finish_native_row_tables(result)

    selected, contract = cli._finish_native_model_contract(
        result,
        argparse.Namespace(model=""),
        row_tables,
    )

    assert selected == row_model.resolve()
    assert selected != endpoint_model
    assert contract["status"] == "hash_bound_row_model"
    assert len(contract["row_models"]) == 2


def test_finish_native_model_rejects_mutated_row_model(tmp_path: Path):
    row_model = tmp_path / "row_model.res"
    row_model.write_text("TITL changed row model\n")
    run = tmp_path / "weighted" / "a"
    run.mkdir(parents=True)
    row_table = run / "native_dyn_calc.csv"
    row_table.write_text("h,k,l\n")
    (run / "summary.json").write_text(
        json.dumps({"inputs": {"res": str(row_model), "res_sha256": "0" * 64}})
    )

    with pytest.raises(RuntimeError, match="row model hash changed"):
        cli._finish_native_model_contract(
            {"model": str(row_model)},
            argparse.Namespace(model=""),
            [("a", row_table)],
        )


def test_finish_native_model_rejects_mixed_provenance(tmp_path: Path):
    import hashlib

    row_model = tmp_path / "row_model.res"
    row_model.write_text("TITL row model\n")
    digest = hashlib.sha256(row_model.read_bytes()).hexdigest()
    row_tables = []
    for label in ("verified", "legacy"):
        run = tmp_path / label
        run.mkdir()
        table = run / "native_dyn_calc.csv"
        table.write_text("h,k,l\n")
        payload = {"inputs": {"res": str(row_model), "res_sha256": digest}} if label == "verified" else {}
        (run / "summary.json").write_text(json.dumps(payload))
        row_tables.append((label, table))

    with pytest.raises(RuntimeError, match="mixed verified/unverified"):
        cli._finish_native_model_contract(
            {"model": str(row_model)},
            argparse.Namespace(model=""),
            row_tables,
        )


def test_finish_endpoint_maps_replays_list3_as_zero_cycle_list6(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    endpoint = tmp_path / "endpoint"
    endpoint.mkdir()
    res = endpoint / "model.res"
    hkl = endpoint / "model.hkl"
    fcf = endpoint / "model.fcf"
    lst = endpoint / "model.lst"
    res.write_text(
        "TITL endpoint\n"
        "CELL 0.025079 5 6 7 90 90 90\n"
        "L.S. 1\n"
        "LIST 3\n"
        "WGHT 0 0\n"
        "HKLF 4\n"
        "END\n"
    )
    hkl.write_text("   0   0   0    0.00    0.00\n")
    fcf.write_text("data_model\n_shelx_refln_list_code 3\n")
    metrics = (
        "wR2 = 0.2923, GooF = S = 1.074 for all data\n"
        "R1 = 0.1281 for 441 Fo > 4sig(Fo) and 0.2004 for all 1894 data\n"
        "0 atoms may be split and 0 atoms NPD\n"
    )
    lst.write_text(metrics)
    fake_shelxl = tmp_path / "shelxl"
    fake_shelx2map = tmp_path / "shelx2map"
    for executable in (fake_shelxl, fake_shelx2map):
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o755)

    calls: list[list[str]] = []

    def fake_run(command, *, cwd, text, stdout, stderr, timeout=None):
        calls.append(list(command))
        work = Path(cwd)
        if Path(command[0]) == fake_shelxl:
            stem = command[1]
            assert "L.S. 0" in (work / f"{stem}.ins").read_text()
            assert "LIST 6" in (work / f"{stem}.ins").read_text()
            (work / f"{stem}.fcf").write_text("data_model\n_shelx_refln_list_code 6\n")
            (work / f"{stem}.lst").write_text(metrics)
            return SimpleNamespace(returncode=0, stdout="SHELXL replay ok\n")
        assert timeout == 120
        out_map = Path(command[command.index("-o") + 1])
        out_map.write_text("map\n")
        return SimpleNamespace(returncode=0, stdout=f'Map data written to "{out_map}"\n')

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(owner_workflows_finish_maps, "_publication_endpoint_from_result", lambda _result: {"endpoint_res": str(res), "endpoint_hkl": str(hkl)})
    args = argparse.Namespace(
        shelxl=str(fake_shelxl),
        shelxl_requested=str(fake_shelxl),
        tool_config={"tools": {"shelx2map": str(fake_shelx2map)}},
    )

    result = cli.run_finish_endpoint_maps({}, args)

    assert result["status"] == "ok"
    assert result["list6_replay"]["status"] == "ok"
    assert result["list6_replay"]["source_list_code"] == 3
    assert result["list6_replay"]["replay_list_code"] == 6
    assert result["list6_replay"]["metrics_unchanged"] is True
    assert len(calls) == 5
    map_calls = calls[1:]
    assert all(call[1:3] == ["-w", "1"] for call in map_calls[:3])
    assert map_calls[3][1:3] == ["-w", "0"]
    assert result["maps"]["fofc"]["shelx2map_weight"] == 1.0
    assert result["maps"]["fofc"]["display_role"] == "primary_weighted"
    assert result["maps"]["fofc_unweighted_diagnostic"]["display_role"] == "diagnostic_unweighted"
    assert all(record["exists"] for record in result["maps"].values())
