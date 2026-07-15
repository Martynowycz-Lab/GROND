"""Small SHELX output diagnostics shared by GROND endpoint runners."""

from __future__ import annotations

import re
from pathlib import Path


def parse_shelx_lst_metrics(path: Path) -> dict[str, object]:
    """Parse the final printed SHELXL endpoint statistics from an LST file."""

    out: dict[str, object] = {
        "lst": str(path),
        "metric_status": "missing",
        "printed_decimals": {},
    }

    def decimals(value: str) -> int:
        return len(value.partition(".")[2]) if "." in value else 0
    if not path.exists():
        return out
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        return {**out, "metric_status": "read_error", "error": str(exc)}
    out["metric_status"] = "incomplete"
    wr2_matches = re.findall(r"wR2 =\s*([0-9.]+), GooF = S =\s*([0-9.]+).* for all data", text)
    if wr2_matches:
        wr2, goof = wr2_matches[-1]
        out.update({"wR2_all": float(wr2), "GooF": float(goof)})
        out["printed_decimals"].update({"wR2_all": decimals(wr2), "GooF": decimals(goof)})
    plus_wr2_matches = re.findall(r"\+\s+wR2:\s*([0-9.]+),\s*GooF:\s*([0-9.]+)", text)
    if plus_wr2_matches:
        wr2, goof = plus_wr2_matches[-1]
        out.update({"wR2_all": float(wr2), "GooF": float(goof)})
        out["printed_decimals"].update({"wR2_all": decimals(wr2), "GooF": decimals(goof)})
    r1_matches = re.findall(
        r"R1 =\s*([0-9.]+) for\s+(\d+) Fo > 4sig\(Fo\) and\s+([0-9.]+) for all\s+(\d+) data",
        text,
    )
    if r1_matches:
        r1_gt, n_gt, r1_all, n_all = r1_matches[-1]
        out.update(
            {
                "R1_gt": float(r1_gt),
                "N_gt": int(n_gt),
                "R1_all": float(r1_all),
                "N_all": int(n_all),
            }
        )
        out["printed_decimals"].update({"R1_gt": decimals(r1_gt), "R1_all": decimals(r1_all)})
    plus_gt = re.findall(r"\+\s+R1:\s*([0-9.]+) for\s+(\d+) reflections", text)
    plus_all = re.findall(r"\+\s+R1 \(all\):\s*([0-9.]+) for\s+(\d+) reflections", text)
    if plus_gt and plus_all:
        r1_gt, n_gt = plus_gt[-1]
        r1_all, n_all = plus_all[-1]
        out.update(
            {
                "R1_gt": float(r1_gt),
                "N_gt": int(n_gt),
                "R1_all": float(r1_all),
                "N_all": int(n_all),
            }
        )
        out["printed_decimals"].update({"R1_gt": decimals(r1_gt), "R1_all": decimals(r1_all)})
    required = ("R1_gt", "R1_all", "wR2_all", "GooF", "N_gt", "N_all")
    if all(key in out for key in required):
        out["metric_status"] = "complete"
    return out


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
    # Merely existing is not proof that a SHELXL run reached its final
    # pathology summary.  Interrupted runs can leave an empty or metrics-only
    # LST beside a syntactically valid RES.  Keep that distinction explicit so
    # endpoint feedback can fail closed instead of interpreting absence of a
    # warning as evidence of zero warnings.
    out["pathology_status"] = "incomplete"
    summaries = re.findall(
        r"(?:\*\*\s*Warning:\s*)?(\d+)\s+atoms may be split and\s+(\d+)\s+atoms NPD",
        text,
        flags=re.IGNORECASE,
    )
    if summaries:
        split_text, npd_text = summaries[-1]
        out["split_count"] = int(split_text)
        out["npd_count"] = int(npd_text)
        out["pathology_status"] = "complete"

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
    """Return true when SHELXL diagnostics are usable as feedback.

    NPD atoms indicate an unstable ADP update and are not fed into later
    Bloch cycles automatically. Split warnings are reported but remain a
    modeler-facing decision rather than a GROND feedback blocker.
    """

    # A feedback endpoint needs affirmative evidence that SHELXL completed the
    # final pathology audit.  Missing or truncated diagnostics are not a clean
    # result.  Callers that intentionally validate a user-supplied input model
    # without an LST handle that structural-only case separately.
    if str(pathology.get("pathology_status", "missing")) != "complete":
        return False
    return not bool(pathology.get("has_npd"))
