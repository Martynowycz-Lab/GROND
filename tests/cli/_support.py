"""Synthetic models and CLI test helpers shared by functional groups."""





from pathlib import Path


def valid_shelx_res_text(title: str = "ok") -> str:
    """Return a minimal model that is structurally usable by the Bloch parser."""

    return (
        f"TITL {title}\n"
        "CELL 0.02508 5.0 6.0 7.0 90 90 90\n"
        "SFAC C 1 0 1 0 1 0 1 0\n"
        "FVAR 1\n"
        "C1 1 0.10 0.20 0.30 11.0 0.05\n"
        "HKLF 4\n"
        "END\n"
    )

def bound_shelx_health(path: Path, **updates) -> dict:
    """Return a path/hash-bound health record for feedback-resume tests."""

    from grond.legacy.engine.native_row_regularize import shelx_res_health

    lst = path.with_suffix(".lst")
    if not lst.exists():
        lst.write_text("0 atoms may be split and 0 atoms NPD\n")
    health = shelx_res_health(path, lst)
    health.update(updates)
    return health

def combined_endpoint_summary(path: Path, *, status: str = "ok", **health_updates) -> dict:
    """Return a current-format combined endpoint summary bound to ``path``."""

    health = bound_shelx_health(path, **health_updates)
    return {
        "combined_endpoint": {
            "lambda_q": "1.0",
            "mod_p95": 1.0,
            "modulation_metric": "scale_invariant_symmetric_fold_change_p95",
            "artifacts": {
                "res": str(path),
                "res_sha256": health.get("sha256", ""),
            },
            "shelxl": {
                "status": status,
                "res_health": health,
            },
        }
    }

def cli_command_names(parser):
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if choices:
            return set(choices)
    raise AssertionError("No CLI subcommands found")

def _write_hydrogen_audit_model(
    path: Path,
    *,
    cell_a: float = 10.0,
    include_h: bool,
    symm: str = "",
    coordinate_shift: float = 0.0,
) -> None:
    atoms = [
        f"C1 1 {0.10 + coordinate_shift:.4f} 0.20 0.30 11.0 0.05",
        f"C2 1 {0.25 + coordinate_shift:.4f} 0.35 0.40 11.0 0.05",
        f"C3 1 {0.45 + coordinate_shift:.4f} 0.55 0.60 11.0 0.05",
    ]
    if include_h:
        atoms.append("H1 2 0.15 0.20 0.30 11.0 0.05")
    path.write_text(
        "\n".join(
            [
                "TITL H audit",
                f"CELL 0.02508 {cell_a:g} 11 12 90 90 90",
                "LATT -1",
                *([f"SYMM {symm}"] if symm else []),
                "SFAC C 1 0 1 0 1 0 1 0 0",
                "SFAC H 1 0 1 0 1 0 1 0 0",
                "UNIT 1 1",
                *atoms,
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )

def _write_hydrogen_audit_map(path: Path, *, cell_a: float = 10.0) -> None:
    import gemmi
    import numpy as np

    grid = gemmi.FloatGrid(8, 8, 8)
    grid.unit_cell = gemmi.UnitCell(cell_a, 11.0, 12.0, 90.0, 90.0, 90.0)
    grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    np.asarray(grid)[:] = 0.0
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header(2, True)
    ccp4.write_ccp4_map(str(path))
