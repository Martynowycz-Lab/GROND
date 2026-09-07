
import json
from pathlib import Path

import numpy as np
import pytest

import grond.cli as cli
import grond.io.structure_contract as owner_io_structure_contract
from grond.legacy.engine.bloch_engine import fcalc_res, parse_model
from grond.legacy.engine.make_res_variants import write_alternate_hand_variant
from grond.progress import ProgressState


def _p41_res(*, x: float = 0.13) -> str:
    return "\n".join(
        [
            "TITL P41 alternate-hand materialization",
            "CELL 0.025079 8 8 12 90 90 90",
            "LATT -1",
            "SYMM -Y, X, Z+1/4",
            "SYMM -X, -Y, Z+1/2",
            "SYMM Y, -X, Z+3/4",
            "EQIV $1 -Y, X, Z+1/4",
            "SFAC C 1 1 1 1 1 1 1 1 0",
            "UNIT 2",
            "FVAR 1.0",
            f"C1 1 {x:.6f} 0.27 0.39 1.0 0.04",
            "C2 1 0.41 0.18 0.72 1.0 0.05",
            "HKLF 4",
            "END",
            "",
        ]
    )


def _symm_lines(path: Path) -> list[str]:
    return [line.strip().lower().replace(" ", "") for line in path.read_text().splitlines() if line.upper().startswith("SYMM")]


def test_p41_materializes_as_p43_and_round_trips_back_to_p41(tmp_path: Path):
    p41 = tmp_path / "p41.res"
    p43 = tmp_path / "p43.res"
    p41_roundtrip = tmp_path / "p41_roundtrip.res"
    p41.write_text(_p41_res())

    write_alternate_hand_variant(p41, p43)
    write_alternate_hand_variant(p43, p41_roundtrip)

    assert _symm_lines(p43) == ["symm-y,x,z-1/4", "symm-x,-y,z-1/2", "symmy,-x,z-3/4"]
    assert _symm_lines(p41_roundtrip) == ["symm-y,x,z+1/4", "symm-x,-y,z+1/2", "symmy,-x,z+3/4"]
    assert "EQIV $1 -y,x,z-1/4" in p43.read_text()
    assert "EQIV $1 -y,x,z+1/4" in p41_roundtrip.read_text()

    original_model = parse_model(p41)
    alternate_model = parse_model(p43)
    roundtrip_model = parse_model(p41_roundtrip)
    for hkl in ((1, 2, 3), (2, -1, 4), (3, 1, -2)):
        assert fcalc_res(alternate_model, hkl) == pytest.approx(np.conj(fcalc_res(original_model, hkl)))
        assert fcalc_res(roundtrip_model, hkl) == pytest.approx(fcalc_res(original_model, hkl))


def test_alternate_model_cache_regenerates_for_same_stem_new_or_changed_source(tmp_path: Path, monkeypatch):
    source_a = tmp_path / "source_a" / "model.res"
    source_b = tmp_path / "source_b" / "model.res"
    source_a.parent.mkdir()
    source_b.parent.mkdir()
    source_a.write_text(_p41_res(x=0.13))
    source_b.write_text(_p41_res(x=0.21))
    out_root = tmp_path / "run"
    calls: list[str] = []

    def fake_run_logged(cmd, _logs, _label, **_kwargs):
        src = Path(cmd[cmd.index("--input") + 1])
        dst = Path(cmd[cmd.index("--output") + 1])
        dst.parent.mkdir(parents=True, exist_ok=True)
        write_alternate_hand_variant(src, dst)
        calls.append(str(src))

    monkeypatch.setattr(owner_io_structure_contract, "run_logged", fake_run_logged)
    first = cli.make_alternate_hand_model("python", str(source_a), out_root, tmp_path / "logs", False)
    reused = cli.make_alternate_hand_model("python", str(source_a), out_root, tmp_path / "logs", False)
    replaced = cli.make_alternate_hand_model("python", str(source_b), out_root, tmp_path / "logs", False)
    source_b.write_text(_p41_res(x=0.29))
    refreshed = cli.make_alternate_hand_model("python", str(source_b), out_root, tmp_path / "logs", False)

    assert first == reused == replaced == refreshed
    assert calls == [str(source_a.resolve()), str(source_b.resolve()), str(source_b.resolve())]
    written_c1 = next(atom for atom in parse_model(refreshed)["atoms"] if atom["name"] == "C1")
    assert written_c1["x"] == pytest.approx(0.71)
    manifest_path = refreshed.with_suffix(refreshed.suffix + ".source.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_path"] == str(source_b.resolve())
    assert manifest["variant"] == "alternate_hand"
    assert manifest["materialization_contract"] == "invert_atoms_and_shelx_symmetry_translations_v1"
    assert manifest["output_sha256"]


def test_resume_fingerprint_tracks_make_variant_output_file(tmp_path: Path):
    source = tmp_path / "model.res"
    output = tmp_path / "model_alternate.res"
    log = tmp_path / "variant.log"
    source.write_text(_p41_res())
    output.write_text("first output\n")
    log.write_text("completed\n")
    command = ["python", "make_res_variants.py", "--input", str(source), "--output", str(output), "--alternate-hand"]
    state = ProgressState(root=tmp_path / "progress", resume=True, total=1)
    state.mark_completed("make_alternate_hand_model", command, 0.1, 0, log)

    assert state.should_skip("make_alternate_hand_model", command)
    output.write_text("corrupted output\n")
    assert not state.should_skip("make_alternate_hand_model", command)


def test_chiv_model_fail_closes_automatic_hand_materialization(tmp_path: Path):
    source = tmp_path / "chiv_p41.res"
    output = tmp_path / "must_not_be_written.res"
    source.write_text(_p41_res().replace("HKLF 4", "CHIV 0.1 1.0 C1 C2 C1 C2\nHKLF 4"))

    policy = cli.absolute_structure_policy(str(source))

    assert policy["run_hand_control"] is False
    assert "CHIV" in policy["reason"]
    with pytest.raises(ValueError, match="CHIV restraints"):
        write_alternate_hand_variant(source, output)
    assert not output.exists()
