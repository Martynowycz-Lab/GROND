"""GROND reads DIALS files without DIALS and gets the values that DIALS reads.

tests/reference/dials_tiny holds a synthetic table and experiment list written by DIALS 3.30.0, and the values
that DIALS read from them (expected.json): a hexagonal R -3 c cell with scan-varying A matrices, a panel inside an
offset detector group, a goniometer setting rotation, a shoebox column and every row flag GROND uses.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from grond.io.dials_files import read_dials

TINY = Path(__file__).parent / "reference" / "dials_tiny"


def test_values_match_dials():
    expected = json.loads((TINY / "expected.json").read_text())
    arrays, meta = read_dials(TINY / "tiny.refl", TINY / "tiny.expt")
    assert sorted(arrays) == sorted(expected["arrays"])
    for name, values in expected["arrays"].items():
        want = np.asarray(values, dtype=arrays[name].dtype)
        np.testing.assert_array_equal(arrays[name], want, err_msg=name)
    # dxtbx returns the static A matrix through its unit cell (U B), which can change the last bit.
    np.testing.assert_allclose(meta.pop("A_static"), expected["meta"].pop("A_static"), rtol=1e-14, atol=0)
    meta.pop("source_refl")
    meta.pop("source_expt")
    assert meta == expected["meta"]


def test_one_experiment_only(tmp_path):
    experiments = json.loads((TINY / "tiny.expt").read_text())
    experiments["experiment"] *= 2
    expt = tmp_path / "two.expt"
    expt.write_text(json.dumps(experiments))
    with pytest.raises(ValueError, match="one experiment per dataset"):
        read_dials(TINY / "tiny.refl", expt)


def test_not_a_reflection_file(tmp_path):
    refl = tmp_path / "integrated.refl"
    refl.write_bytes(b"\x80\x04 not msgpack")
    with pytest.raises(ValueError, match="not a DIALS reflection file"):
        read_dials(refl, TINY / "tiny.expt")
