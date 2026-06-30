"""Small SHELX output diagnostics shared by GROND endpoint runners."""

from __future__ import annotations

import re
from pathlib import Path


def parse_shelx_lst_pathologies(path: Path) -> dict[str, object]:
    """Parse ADP/split pathologies from a SHELXL LST file.

    SHELXL can write a syntactically usable RES file while warning that one or
    more anisotropic displacement tensors are non-positive definite.  GROND
    keeps those outputs for diagnostics, but feedback loops should not promote
    them when a clean previous model exists.
    """

    out: dict[str, object] = {
        "lst": str(path),
        "pathology_status": "missing",
        "npd_count": 0,
        "split_count": 0,
        "npd_atoms": [],
        "has_npd": False,
        "has_split_warnings": False,
    }
    if not Path(path).exists():
        return out
    text = Path(path).read_text(errors="replace")
    out["pathology_status"] = "ok"
    summaries = re.findall(
        r"(?:\*\*\s*Warning:\s*)?(\d+)\s+atoms may be split and\s+(\d+)\s+atoms NPD",
        text,
        flags=re.IGNORECASE,
    )
    if summaries:
        split_text, npd_text = summaries[-1]
        out["split_count"] = int(split_text)
        out["npd_count"] = int(npd_text)

    atoms: list[str] = []
    for line in text.splitlines():
        if "NON POSITIVE DEFINITE" not in line.upper():
            continue
        match = re.search(
            r"^\s*[-+0-9.Ee]+\s+[-+0-9.Ee]+\s+[-+0-9.Ee]+\s+(\S+)\s+\*\*\s+NON POSITIVE DEFINITE\s+\*\*",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            atoms.append(match.group(1))
    if atoms:
        out["npd_atoms"] = atoms
        out["npd_count"] = max(int(out["npd_count"]), len(atoms))

    out["has_npd"] = int(out["npd_count"]) > 0
    out["has_split_warnings"] = int(out["split_count"]) > 0
    return out


def shelx_pathology_safe_for_feedback(pathology: dict[str, object]) -> bool:
    """Return true when SHELXL pathology diagnostics are clean or unavailable."""

    if str(pathology.get("pathology_status", "missing")) == "missing":
        return True
    return not bool(pathology.get("has_npd")) and not bool(pathology.get("has_split_warnings"))
