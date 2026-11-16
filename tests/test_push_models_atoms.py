"""SHELX atom-record detection used by the refine start-model preparations (free-H, ADP-stabilized)."""

from __future__ import annotations

import argparse
import json

from grond.workflows.push_models import is_shelx_atom_line, prepare_push_endpoint_model

# Z = 4 and H is SFAC 4, so ZERR has the shape of an atom record of element H
MODEL = """TITL test in P2(1)2(1)2(1)
CELL 0.02508 5.8300 21.0100 6.9700 90.000 90.000 90.000
ZERR 4 0.0103 0.0192 0.0323 0.000 0.000 0.000
LATT -1
SYMM 0.5-X,-Y,0.5+Z
SFAC C N O H
UNIT 36 4 12 44
L.S. 10
FVAR 1.00000
C1    1    0.596081    0.573080    0.498878    11.00000    0.01888    0.03684 =
         0.02000    0.00100    0.00200    0.00300
H1    4    0.756979    0.618360    0.519518    11.00000   -1.20000
HKLF 4
END
"""


def test_atom_records_are_atoms():
    assert is_shelx_atom_line("C001  1    0.596081    0.573080    0.498878    11.00000    0.01888    0.03684 =")
    assert is_shelx_atom_line("H001  4    0.756979    0.618360    0.519518    11.00000   -1.20000")


def test_instruction_records_shaped_like_atoms_are_not_atoms():
    assert not is_shelx_atom_line("ZERR 4 0.0103 0.0192 0.0323 0.000 0.000 0.000")
    assert not is_shelx_atom_line("HKLF 4 1 0 0 0 1 0 0 0 1")
    assert not is_shelx_atom_line("TWIN 1 0 0 0 -1 0 0 0 -1 2")


def test_free_h_preparation_keeps_zerr(tmp_path):
    source = tmp_path / "model.res"
    source.write_text(MODEL)
    args = argparse.Namespace(start_res=str(source), model_prep="free-h-anis-no-restraints", set_ls_cycles=1)
    prepared, audit = prepare_push_endpoint_model(args, tmp_path / "out")
    lines = prepared.read_text().splitlines()
    assert "ZERR 4 0.0103 0.0192 0.0323 0.000 0.000 0.000" in lines
    assert [h["label"] for h in json.loads(audit.read_text())["converted_hydrogens"]] == ["H1"]


def test_adp_stabilized_preparation_puts_restraints_after_the_header(tmp_path):
    source = tmp_path / "model.res"
    source.write_text(MODEL)
    args = argparse.Namespace(start_res=str(source), model_prep="adp-stabilized", set_ls_cycles=1)
    prepared, _audit = prepare_push_endpoint_model(args, tmp_path / "out")
    lines = prepared.read_text().splitlines()
    # the RIGU/SIMU block goes before the first atom, not before ZERR (SHELXL reads CELL, ZERR, LATT, ... first)
    assert lines.index("RIGU") > lines.index("UNIT 36 4 12 44")
    assert lines.index("RIGU") < next(i for i, line in enumerate(lines) if line.startswith("C1 "))
