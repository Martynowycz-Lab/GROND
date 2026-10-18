"""Crystal-independent orbit-hash free split, orbit exclusion and their engine contract."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grond.bloch.splits import (
    build_transform_split_contract,
    candidate_invariant_split_membership,
    orbit_hash_split_masks,
    resolve_split_masks,
    split_masks,
    validate_transform_split_contract,
)
from grond.cli_options.parser import build_parser
from grond.io.config import DatasetSpec
from grond.validation import heldout as H
from grond.workflows.native_commands import native_stage_cmd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests" / "reference"))
import reference_lib as G  # noqa: E402

MMM = [{"rot": np.diag(d)} for d in ([1, -1, -1], [-1, 1, -1], [-1, -1, 1])]
KLEIN_FOUR = ["identity", "h,-k,-l", "-h,k,-l", "-h,-k,l"]
P212121 = "\n".join(
    [
        "TITL orbit-hash split control",
        "CELL 0.025079 8 9 12 90 90 90",
        "LATT -1",
        "SYMM 1/2-X, -Y, 1/2+Z",
        "SYMM -X, 1/2+Y, 1/2-Z",
        "SYMM 1/2+X, 1/2-Y, -Z",
        "SFAC C 1 1 1 1 1 1 1 1 0",
        "UNIT 1",
        "FVAR 1.0",
        "C1 1 0.13 0.27 0.39 1.0 0.04",
        "HKLF 4",
        "END",
        "",
    ]
)


def _write_xds(path: Path, hkls, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    lines = [
        "!X-RAY_WAVELENGTH= 0.025079",
        "!STARTING_FRAME= 1",
        "!STARTING_ANGLE= 0",
        "!OSCILLATION_RANGE= 0.5",
        "!ROTATION_AXIS= 1 0 0",
        "!INCIDENT_BEAM_DIRECTION= 0 0 1",
        "!SPACE_GROUP_NUMBER= 19",
        "!UNIT_CELL_CONSTANTS= 8 9 12 90 90 90",
        "!UNIT_CELL_A-AXIS= 8 0 0",
        "!UNIT_CELL_B-AXIS= 0 9 0",
        "!UNIT_CELL_C-AXIS= 0 0 12",
    ]
    for i, (h, k, ell) in enumerate(hkls):
        z = 1.0 + (i % 20)
        values = [h, k, ell, 100.0 + 50.0 * rng.random(), 5.0, 100.0, 100.0, z, 1.0, 100.0, 100.0, 0.0, 100.0, 100.0, z]
        lines.append(" ".join(str(v) for v in [*values, 0, 0, 0, 0, 0, 0]))
    path.write_text("\n".join(lines) + "\n")


def _grid_hkls():
    """Index grid without the P212121 axial absences, whose predictions are exactly zero and never written."""

    def absent(h, k, m):
        return (k == m == 0 and h % 2) or (h == m == 0 and k % 2) or (h == k == 0 and m % 2)

    return [h for h in itertools.product(range(-2, 3), range(-2, 3), range(-3, 4)) if h != (0, 0, 0) and not absent(*h)]


def _write_lockbox(path: Path, orbits: list[str]) -> str:
    sha = hashlib.sha256("\n".join(orbits).encode()).hexdigest()
    path.write_text(json.dumps({"lockbox_orbits": orbits, "lockbox_sha256": sha, "shells": [{"d_min": 1.0}]}))
    return sha


def test_orbit_hash_free_set_is_the_reference_free_fold_in_every_crystal():
    rng = np.random.default_rng(4)
    crystal_a = rng.integers(-9, 10, size=(600, 3))
    crystal_b = np.concatenate([crystal_a[::3] * np.array([1, -1, 1]), rng.integers(-9, 10, size=(400, 3))])
    rng.shuffle(crystal_b)
    masks_a, excluded_a = orbit_hash_split_masks(crystal_a, 0.10, H.DEFAULT_FREE_SALT, MMM)
    masks_b, _ = orbit_hash_split_masks(crystal_b, 0.10, H.DEFAULT_FREE_SALT, MMM)
    assert not excluded_a.any()
    reference_free = [G.hash_mod(H.DEFAULT_FREE_SALT, G.orbit_key(*h), 10) == 0 for h in crystal_a]
    assert masks_a["free"].tolist() == reference_free
    assert np.array_equal(masks_a["work"], ~masks_a["free"])
    status_a = {G.orbit_key(*h): bool(f) for h, f in zip(crystal_a, masks_a["free"], strict=True)}
    status_b = {G.orbit_key(*h): bool(f) for h, f in zip(crystal_b, masks_b["free"], strict=True)}
    common = set(status_a) & set(status_b)
    assert len(common) > 50
    assert all(status_a[key] == status_b[key] for key in common)
    # The legacy per-dataset shuffle puts some shared orbits on opposite sides (the leak).
    legacy_a = split_masks(crystal_a, 0.10, 42, symmetry_ops=MMM)
    legacy_b = split_masks(crystal_b, 0.10, 42, symmetry_ops=MMM)
    legacy_status_a = {G.orbit_key(*h): bool(f) for h, f in zip(crystal_a, legacy_a["free"], strict=True)}
    legacy_status_b = {G.orbit_key(*h): bool(f) for h, f in zip(crystal_b, legacy_b["free"], strict=True)}
    assert any(legacy_status_a[key] != legacy_status_b[key] for key in common)


def test_orbit_exclusion_rows_are_neither_work_nor_free():
    hkls = np.asarray(_grid_hkls())
    keys = H.orbit_keys(hkls, MMM)
    excluded_keys = {keys[0], keys[5], keys[40]}
    masks, excluded = orbit_hash_split_masks(hkls, 0.10, "salt", MMM, excluded_orbits=excluded_keys)
    assert excluded.tolist() == [key in excluded_keys for key in keys]
    assert not np.any((masks["work"] | masks["free"]) & excluded)
    assert np.array_equal(masks["work"] | masks["free"], ~excluded)


def test_candidate_union_orbit_hash_membership_for_laue_equivalent_candidates():
    raw = np.asarray(_grid_hkls())
    keys = H.orbit_keys(raw, MMM)
    lockbox = {keys[3], keys[17]}
    result = candidate_invariant_split_membership(
        raw, MMM, KLEIN_FOUR, free_fraction=0.10, scheme="orbit-hash", salt="S", excluded_orbits=lockbox
    )
    free = H.free_mask(keys, "S", 0.10)
    expected = np.where(np.isin(keys.astype(str), sorted(lockbox)), 2, np.where(free, 1, 0))
    assert result["membership"].tolist() == expected.tolist()
    assert result["multi_orbit_group_count"] == 0
    assert result["mixed_status_group_count"] == 0
    assert result["excluded_group_count"] == 2
    legacy = candidate_invariant_split_membership(raw, MMM, KLEIN_FOUR, free_fraction=0.10, seed=42)
    assert set(np.unique(legacy["membership"])) <= {0, 1}
    with pytest.raises(ValueError, match="orbit exclusion requires"):
        candidate_invariant_split_membership(raw, MMM, KLEIN_FOUR, excluded_orbits=lockbox)


def test_candidate_union_orbit_hash_frees_a_component_when_any_orbit_is_free():
    raw = np.asarray(_grid_hkls())
    result = candidate_invariant_split_membership(raw, [], ["identity", "k,h,-l"], free_fraction=0.25, scheme="orbit-hash", salt="S")
    groups = result["orbit_group_by_row"]
    for group_id in np.unique(groups):
        rows = np.where(groups == group_id)[0]
        keys = {H.orbit_key(raw[r]) for r in rows} | {H.orbit_key((raw[r][1], raw[r][0], -raw[r][2])) for r in rows}
        any_free = bool(np.any(H.free_mask(sorted(keys), "S", 0.25)))
        assert set(result["membership"][rows].tolist()) == {1 if any_free else 0}
    assert result["multi_orbit_group_count"] > 0


def test_orbit_hash_contract_round_trip_records_scheme_salt_and_exclusion(tmp_path: Path):
    hkl = tmp_path / "x.HKL"
    model = tmp_path / "m.res"
    _write_xds(hkl, _grid_hkls())
    model.write_text(P212121)
    keys = sorted({G.orbit_key(*h) for h in _grid_hkls()})
    lockbox = [keys[7], keys[2]]
    lock_path = tmp_path / "lockbox.json"
    sha = _write_lockbox(lock_path, lockbox)

    legacy = build_transform_split_contract(hkl, model, KLEIN_FOUR, dataset="d")
    assert legacy["algorithm"] == "candidate_union_model_symmetry_friedel_orbits_v1"
    assert "split_scheme" not in legacy
    payload = build_transform_split_contract(
        hkl, model, KLEIN_FOUR, dataset="d", scheme="orbit-hash", salt="S", excluded_orbits=lockbox
    )
    assert payload["algorithm"] == "candidate_union_model_symmetry_friedel_orbit_hash_v1"
    assert payload["split_salt"] == "S" and payload["excluded_orbits_sha256"] == sha
    validation = validate_transform_split_contract(payload, source_hkl=hkl, current_transform="identity")
    # The calibration rebuild (transform_binding) reproduces the contract from its recorded fields.
    rebuilt = build_transform_split_contract(
        hkl, model, payload["candidate_transforms"], dataset="d", free_fraction=payload["free_fraction"],
        seed=payload["split_seed"], include_friedel=payload["include_friedel"], scheme=payload["split_scheme"],
        salt=payload["split_salt"], excluded_orbits=payload["excluded_orbits"],
    )
    assert rebuilt["contract_payload_sha256"] == payload["contract_payload_sha256"]
    masks = validation["masks"]
    row_keys = [G.orbit_key(*h) for h in _grid_hkls()]
    assert masks["excluded"].tolist() == [key in lockbox for key in row_keys]
    assert masks["free"].tolist() == [
        key not in lockbox and G.hash_mod("S", key, 10) == 0 for key in row_keys
    ]
    assert validation["split_scheme"] == "orbit-hash" and validation["excluded_row_count"] == int(masks["excluded"].sum())

    tampered = dict(payload, excluded_orbits=lockbox[:1])
    tampered.pop("contract_payload_sha256")
    tampered["contract_payload_sha256"] = __import__("grond.bloch.splits", fromlist=["x"])._canonical_json_sha256(tampered)
    with pytest.raises(ValueError, match="excluded-orbit listing hash mismatch"):
        validate_transform_split_contract(tampered)

    contract = tmp_path / "split.json"
    contract.write_text(json.dumps(payload))
    from grond.bloch.model import parse_model
    from grond.io.xds import XDSData

    data = XDSData(hkl, hkl_transform="identity")
    parsed = parse_model(model)
    common = {"split_contract": str(contract), "free_fraction": 0.10, "split_scheme": "orbit-hash", "split_salt": "S"}
    got, excluded, summary, scheme = resolve_split_masks(data, parsed, exclude_orbits_json=str(lock_path), **common)
    assert np.array_equal(excluded, masks["excluded"]) and np.array_equal(got["free"], masks["free"])
    assert "excluded" not in got and scheme["excluded_orbits_sha256"] == sha
    with pytest.raises(ValueError, match="--split-salt does not match"):
        resolve_split_masks(data, parsed, exclude_orbits_json=str(lock_path), **{**common, "split_salt": "T"})
    with pytest.raises(ValueError, match="--exclude-orbits-json does not match"):
        resolve_split_masks(data, parsed, **common)
    with pytest.raises(ValueError, match="--split-scheme legacy does not match"):
        resolve_split_masks(data, parsed, **{**common, "split_scheme": "legacy"})
    with pytest.raises(ValueError, match="requires --split-scheme orbit-hash"):
        resolve_split_masks(data, parsed, split_scheme="legacy", exclude_orbits_json=str(lock_path))


def _engine(tmp_path: Path, name: str, *extra: str) -> tuple[list[dict], dict]:
    hkl = tmp_path / "x.HKL"
    model = tmp_path / "m.res"
    if not hkl.exists():
        _write_xds(hkl, _grid_hkls())
        model.write_text(P212121)
    out = tmp_path / name
    subprocess.run(
        [
            sys.executable,
            str(REPO / "src" / "grond" / "legacy" / "engine" / "native_row_regularize.py"),
            "--hkl", str(hkl), "--res", str(model), "--out-dir", str(out),
            "--prediction-cache-dir", str(tmp_path / "cache" / "predictions"),
            "--ug-cache-dir", str(tmp_path / "cache" / "ug"),
            "--n-basis", "4", "--n-points", "1", "--thickness-grid", "100", "--thickness-grad-grid", "0",
            "--thickness-points", "1", "--frame-offset-grid", "0", "--group-width-grid", "2frame",
            "--skip-endpoint-refinement", "--workers", "1", "--min-peak", "0", "--min-d-spacing", "0.5",
            "--topup-d-min", "0.5", "--max-support-sg", "1.0", "--geometry-percentile", "100", *extra,
        ],
        check=True,
        capture_output=True,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
    )
    with (out / "native_dyn_calc.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows, json.loads((out / "summary.json").read_text())


def test_engine_drops_excluded_orbits_before_fitting_and_keeps_legacy_default(tmp_path: Path):
    keys = sorted({G.orbit_key(*h) for h in _grid_hkls()})
    lockbox = keys[::7]
    lock_path = tmp_path / "lockbox.json"
    sha = _write_lockbox(lock_path, lockbox)
    rows, summary = _engine(tmp_path, "hash", "--split-scheme", "orbit-hash", "--exclude-orbits-json", str(lock_path))
    row_keys = [G.orbit_key(int(r["h"]), int(r["k"]), int(r["l"])) for r in rows]
    assert rows and not set(row_keys) & set(lockbox)
    assert len(rows) == sum(G.orbit_key(*h) not in lockbox for h in _grid_hkls())
    assert [r["split"] == "free" for r in rows] == [G.hash_mod(H.DEFAULT_FREE_SALT, key, 10) == 0 for key in row_keys]
    selection = summary["row_selection"]
    assert selection["selected_rows"] == len(rows) == selection["work_rows"] + selection["free_rows"]
    assert selection["split_scheme"]["excluded_orbits_sha256"] == sha
    assert summary["config"]["split_scheme"] == "orbit-hash"

    legacy_rows, legacy_summary = _engine(tmp_path, "legacy")
    assert len(legacy_rows) == len(_grid_hkls())
    assert legacy_summary["row_selection"]["split_scheme"]["split_scheme"] == "legacy"
    legacy = split_masks(np.asarray(_grid_hkls()), 0.10, 42, symmetry_ops=MMM)
    assert [r["split"] == "free" for r in legacy_rows] == legacy["free"].tolist()


def test_auto_defaults_forward_orbit_hash_split_and_legacy_flags_restore_it(tmp_path: Path):
    lock_path = tmp_path / "lockbox.json"
    _write_lockbox(lock_path, ["2|0|0"])
    base = ["auto", "--out-dir", str(tmp_path / "o"), "--hkl", "x.HKL", "--structure", "m.res"]
    args = build_parser().parse_args(base)
    assert (args.split_scheme, args.split_salt, args.exclude_orbits_json) == ("orbit-hash", H.DEFAULT_FREE_SALT, "")
    ds = DatasetSpec(name="d", hkl="x.HKL", split_contract=str(tmp_path / "split.json"))
    cmd = native_stage_cmd("python", ds, "m.res", "", tmp_path / "run", args, n_basis=8, n_points=3, stage="screen")
    assert cmd[cmd.index("--split-scheme") + 1] == "orbit-hash"
    assert cmd[cmd.index("--split-salt") + 1] == H.DEFAULT_FREE_SALT
    assert "--exclude-orbits-json" not in cmd

    lock_args = build_parser().parse_args([*base, "--exclude-orbits-json", str(lock_path)])
    cmd = native_stage_cmd("python", ds, "m.res", "", tmp_path / "run", lock_args, n_basis=8, n_points=3, stage="screen")
    assert cmd[cmd.index("--exclude-orbits-json") + 1] == str(lock_path.resolve())

    legacy = build_parser().parse_args([*base, "--split-scheme", "legacy", "--split-seed", "42"])
    cmd = native_stage_cmd("python", ds, "m.res", "", tmp_path / "run", legacy, n_basis=8, n_points=3, stage="screen")
    assert cmd[cmd.index("--split-scheme") + 1] == "legacy"
    # Parsers without the new options keep the legacy split.
    old = SimpleNamespace(**{k: v for k, v in vars(legacy).items() if k not in {"split_scheme", "split_salt"}})
    cmd = native_stage_cmd("python", ds, "m.res", "", tmp_path / "run", old, n_basis=8, n_points=3, stage="screen")
    assert cmd[cmd.index("--split-scheme") + 1] == "legacy"


def test_transform_scout_contract_follows_the_split_flags(tmp_path: Path):
    from grond.io.config import PrepConfig
    from grond.workflows.transform_selection import _materialize_transform_hand_split_contract

    hkl = tmp_path / "x.HKL"
    model = tmp_path / "m.res"
    _write_xds(hkl, _grid_hkls())
    model.write_text(P212121)
    lock_path = tmp_path / "lockbox.json"
    sha = _write_lockbox(lock_path, ["2|0|0", "0|1|1"])
    prep = PrepConfig(structure=str(model), datasets=[DatasetSpec(name="d", hkl=str(hkl))])
    args = SimpleNamespace(free_fraction=0.10, split_seed=42, split_scheme="orbit-hash", split_salt="S", exclude_orbits_json=str(lock_path))
    ds, binding = _materialize_transform_hand_split_contract(prep, prep.datasets[0], KLEIN_FOUR, args, tmp_path)
    payload = json.loads(Path(ds.split_contract).read_text())
    assert payload["split_scheme"] == "orbit-hash" and payload["excluded_orbits_sha256"] == sha
    assert binding["split_salt"] == "S" and binding["excluded_row_count"] > 0
    legacy_args = SimpleNamespace(free_fraction=0.10, split_seed=42, split_scheme="legacy", split_salt="S", exclude_orbits_json="")
    ds, _ = _materialize_transform_hand_split_contract(prep, prep.datasets[0], KLEIN_FOUR, legacy_args, tmp_path / "legacy")
    assert json.loads(Path(ds.split_contract).read_text())["algorithm"] == "candidate_union_model_symmetry_friedel_orbits_v1"
    with pytest.raises(ValueError, match="requires --split-scheme orbit-hash"):
        _materialize_transform_hand_split_contract(
            prep, prep.datasets[0], KLEIN_FOUR, SimpleNamespace(**{**vars(legacy_args), "exclude_orbits_json": str(lock_path)}), tmp_path
        )


def _native_parser():
    """Capture the native engine's own argument parser (no calculation is run)."""

    import argparse
    from unittest.mock import patch

    from grond.legacy.engine import native_row_regularize as nr

    captured = {}

    class Captured(Exception):
        pass

    def capture(self, *_args, **_kwargs):
        captured["parser"] = self
        raise Captured()

    with patch.object(argparse.ArgumentParser, "parse_args", capture), pytest.raises(Captured):
        nr.main()
    return captured["parser"]


def test_single_dataset_shelxl_feedback_cycles_replay_the_orbit_hash_split(tmp_path: Path):
    """Each feedback cycle re-runs the native engine; it must carry the source scheme with the contract."""

    import argparse

    from grond.bloch.model import parse_model
    from grond.io.xds import XDSData
    from grond.legacy.engine import shelxl_feedback_loop as feedback

    hkl = tmp_path / "x.HKL"
    model = tmp_path / "m.res"
    _write_xds(hkl, _grid_hkls())
    model.write_text(P212121)
    lock_path = tmp_path / "lockbox.json"
    _write_lockbox(lock_path, ["2|0|0"])
    payload = build_transform_split_contract(
        hkl, model, KLEIN_FOUR, dataset="d", scheme="orbit-hash", salt="S", excluded_orbits=["2|0|0"]
    )
    contract = tmp_path / "split.json"
    contract.write_text(json.dumps(payload))
    args = argparse.Namespace(
        python="python", hkl=str(hkl), baseline_hkl="", hkl_transform="identity",
        min_d_spacing=0.7, min_peak=0.0, geometry_percentile=100.0, topup_d_min=0.7,
        lambda_grid="1", shelxl="shelxl", shelxl_iterations=1, endpoint_ls_cycles=1,
        post_bloch_sigma_policy="none", post_bloch_sigma_holdout_fraction=0.2,
        post_bloch_sigma_min_scale=0.25, post_bloch_sigma_max_scale=4.0,
        rescale="none", workers=1, chunksize=1, cache_compression="none", max_rows=0,
    )
    # The source summary records the engine config (vars(args)) of an AUTO orbit-hash stage.
    config = {
        "split_contract": str(contract), "split_scheme": "orbit-hash", "split_salt": "S",
        "exclude_orbits_json": str(lock_path), "free_fraction": 0.10, "split_seed": 42,
    }
    summary = {"config": config, "best_native": {"nuisance": {"thickness_A": 700.0}}}
    command = feedback.native_command(args, summary, model, tmp_path / "cycle", n_basis=8, n_points=3)
    parsed = _native_parser().parse_args(command[2:])
    assert (parsed.split_scheme, parsed.split_salt, parsed.exclude_orbits_json) == ("orbit-hash", "S", str(lock_path))
    data = XDSData(hkl, hkl_transform="identity")
    masks, excluded, _contract, scheme = resolve_split_masks(
        data, parse_model(model), split_contract=parsed.split_contract, free_fraction=parsed.free_fraction,
        split_seed=parsed.split_seed, split_scheme=parsed.split_scheme, split_salt=parsed.split_salt,
        exclude_orbits_json=parsed.exclude_orbits_json,
    )
    assert scheme["split_scheme"] == "orbit-hash" and excluded.any()
    assert np.array_equal(masks["free"], validate_transform_split_contract(payload)["masks"]["free"])
    # Without the scheme the cycle's engine default (legacy) would refuse the orbit-hash contract.
    with pytest.raises(ValueError, match="--split-scheme legacy does not match"):
        resolve_split_masks(data, parse_model(model), split_contract=str(contract), split_scheme="legacy")

    # Historical summaries that never recorded a scheme keep the engine's legacy split.
    old = feedback.native_command(
        args, {"config": {"free_fraction": 0.10, "split_seed": 42}, "best_native": {"nuisance": {"thickness_A": 700.0}}},
        model, tmp_path / "old", n_basis=8, n_points=3,
    )
    assert "--split-scheme" not in old and "--split-salt" not in old and "--exclude-orbits-json" not in old
    assert _native_parser().parse_args(old[2:]).split_scheme == "legacy"


def test_engine_max_rows_subset_is_ranked_without_excluded_orbits(tmp_path: Path):
    """Excluded orbits never take (or displace) a --max-rows selection slot."""

    from grond.io.xds import XDSData
    from grond.regularize.observations import row_selection, target_arrays

    keys = sorted({G.orbit_key(*h) for h in _grid_hkls()})
    lockbox = keys[::4]
    lock_path = tmp_path / "lockbox.json"
    _write_lockbox(lock_path, lockbox)
    rows, summary = _engine(
        tmp_path, "subset", "--split-scheme", "orbit-hash", "--exclude-orbits-json", str(lock_path), "--max-rows", "40"
    )
    row_keys = [G.orbit_key(int(r["h"]), int(r["k"]), int(r["l"])) for r in rows]
    assert not set(row_keys) & set(lockbox)
    assert len(rows) == summary["row_selection"]["selected_rows"] == 40
    # The subset is the stratified strongest-row selection over the non-excluded rows alone.
    data = XDSData(tmp_path / "x.HKL", hkl_transform="identity")
    target, _sig = target_arrays(data, "iobs")
    allowed = np.asarray([G.orbit_key(*h) not in lockbox for h in np.asarray(data.hkl, dtype=int)], dtype=bool)
    expected = row_selection(data, np.where(allowed, target, np.nan), True, 0.5, 0.0, 40, 100.0)
    assert sorted(int(r["row_index"]) for r in rows) == sorted(int(i) for i in expected)
