import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grond.legacy.engine.native_row_regularize import row_contract_selection


class _Cell:
    def calculate_d(self, hkl):
        return 1.0 / (1.0 + sum(abs(int(value)) for value in hkl))


def _data() -> SimpleNamespace:
    return SimpleNamespace(
        hkl=np.asarray([[1, 0, 0], [0, 2, 0], [-1, 1, 2]], dtype=int),
        sigma=np.asarray([2.0, 3.0, 4.0], dtype=float),
        cell=_Cell(),
    )


def _write_contract(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ("row_index", "h", "k", "l", "target_I", "xds_sigma", "d_A")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _row(data: SimpleNamespace, target: np.ndarray, index: int) -> dict[str, object]:
    h, k, ell = data.hkl[index]
    return {
        "row_index": index,
        "h": int(h),
        "k": int(k),
        "l": int(ell),
        "target_I": float(target[index]),
        "xds_sigma": float(data.sigma[index]),
        "d_A": data.cell.calculate_d((h, k, ell)),
    }


def test_row_contract_selects_exact_membership_and_records_provenance(tmp_path: Path):
    data = _data()
    target = np.asarray([10.0, 20.0, 30.0])
    contract = tmp_path / "policy_selected.csv"
    _write_contract(contract, [_row(data, target, 2), _row(data, target, 0)])

    indices, audit = row_contract_selection(data, target, contract)

    assert indices.tolist() == [0, 2]
    assert audit["status"] == "verified_exact_row_contract"
    assert audit["requested_rows"] == audit["selected_rows"] == 2
    assert len(audit["sha256"]) == 64
    assert audit["target_rows_verified"] == 2
    assert audit["sigma_rows_verified"] == 2
    assert audit["d_spacing_rows_verified"] == 2


def test_row_contract_fails_closed_on_hkl_or_observation_mismatch(tmp_path: Path):
    data = _data()
    target = np.asarray([10.0, 20.0, 30.0])
    contract = tmp_path / "stale.csv"
    stale = _row(data, target, 1)
    stale["h"] = 9
    _write_contract(contract, [stale])

    with pytest.raises(ValueError, match="HKL mismatch"):
        row_contract_selection(data, target, contract)

    stale = _row(data, target, 1)
    stale["target_I"] = 999.0
    _write_contract(contract, [stale])
    with pytest.raises(ValueError, match="target_I mismatch"):
        row_contract_selection(data, target, contract)
