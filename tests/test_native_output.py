from pathlib import Path

import pytest

from grond.bloch.model import parse_res
from grond.refine.native_output import update_res_coordinates


def source_model(path: Path, hydrogen_uiso: str):
    path.write_text(
        "TITL output round trip\nCELL 0.025 10 11 12 90 90 90\nLATT -1\n"
        "SFAC C 1 1 1 1 1 1 1 1 0\nSFAC H 1 1 1 1 1 1 1 1 0\n"
        "FVAR 1\nC1 1 0.1 0.2 0.3 11 0.020\n"
        f"H1 2 0.2 0.2 0.3 11 {hydrogen_uiso}\nHKLF 4\nEND\n"
    )
    return parse_res(path)


def test_updated_free_hydrogen_uiso_survives_res_round_trip(tmp_path):
    source, output = tmp_path / "start.res", tmp_path / "updated.res"
    model = source_model(source, "0.025")
    model["atoms"][1].update(x=0.21, uiso=0.085)

    update_res_coordinates(source, output, model)

    saved = parse_res(output)["atoms"][1]
    assert saved["x"] == pytest.approx(0.21)
    assert saved["uiso"] == pytest.approx(0.085)
    assert saved["occ"] == 1.0


def test_riding_hydrogen_keeps_its_parent_uiso_multiplier(tmp_path):
    source, output = tmp_path / "start.res", tmp_path / "updated.res"
    model = source_model(source, "-1.2")
    model["atoms"][0]["uiso"] = 0.04

    update_res_coordinates(source, output, model)

    assert "-1.2" in next(line for line in output.read_text().splitlines() if line.startswith("H1"))
    assert parse_res(output)["atoms"][1]["uiso"] == pytest.approx(1.2 * 0.04)


def test_free_variable_and_fixed_uiso_codes_are_kept(tmp_path):
    source, output = tmp_path / "start.res", tmp_path / "updated.res"
    for code in ("31.00000", "10.03000"):
        source_model(source, "0.025")
        source.write_text(source.read_text().replace("FVAR 1\n", "FVAR 1 0.5 0.06\n").replace(" 11 0.025", f" 11 {code}"))
        model = parse_res(source)
        model["atoms"][1]["uiso"] = 0.07
        update_res_coordinates(source, output, model)
        assert code in next(line for line in output.read_text().splitlines() if line.startswith("H1"))
