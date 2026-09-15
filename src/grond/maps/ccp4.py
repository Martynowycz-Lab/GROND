"""Preserve explicit CCP4 symmetry settings when expanding partial maps."""
from pathlib import Path

import gemmi
import numpy as np


def read_ccp4_map(path: Path) -> gemmi.Ccp4Map:
    """Read without expansion, resolving the setting from stored operators."""
    path = Path(path)
    result = gemmi.read_ccp4_map(str(path))
    count = result.header_i32(24)
    if count:
        if count < 0 or count % 80:
            raise ValueError("CCP4 symmetry records must occupy complete 80-byte cards")
        with path.open("rb") as handle:
            handle.seek(1024)
            records = handle.read(count)
        if len(records) != count:
            raise ValueError("CCP4 symmetry records are truncated")
        operations = [gemmi.Op(records[i : i + 80].decode("ascii").strip()) for i in range(0, count, 80)]
        group = gemmi.find_spacegroup_by_ops(gemmi.GroupOps(operations))
        if group is None:
            raise ValueError("CCP4 operators do not identify a space-group setting")
        if result.header_i32(23) not in {0, group.number, group.ccp4}:
            raise ValueError("CCP4 symmetry operators conflict with the space-group number")
        # Group 14 alone selects P21/c in Gemmi, even when the file stores P21/n.
        result.grid.spacegroup = group
    return result


def normalize_ccp4_map(path: Path) -> dict[str, object]:
    """Write a full-cell P1 map so viewers cannot choose a different setting."""
    path = Path(path)
    loaded = read_ccp4_map(path)
    if any(abs(loaded.header_float(word)) > 1e-6 for word in (50, 51, 52)):
        raise ValueError("Resolve the nonzero Cartesian map origin before full-cell export")
    source_group = loaded.grid.spacegroup.xhm() if loaded.grid.spacegroup else "unknown"
    source_shape = list(loaded.grid.shape)
    loaded.setup(float("nan"))
    values = np.array(loaded.grid, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("CCP4 map does not cover the full cell after symmetry expansion")
    grid = gemmi.FloatGrid(*values.shape)
    grid.unit_cell = loaded.grid.unit_cell
    grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    grid.array[:] = values
    result = gemmi.Ccp4Map()
    result.grid = grid
    result.update_ccp4_header()
    temporary = path.with_name(path.name + ".tmp")
    result.write_ccp4_map(str(temporary))
    temporary.replace(path)
    return {"source_spacegroup": source_group, "source_shape": source_shape,
            "output_spacegroup": "P 1", "output_shape": list(values.shape),
            "definition": "full cell expanded using the stored CCP4 symmetry operators"}
