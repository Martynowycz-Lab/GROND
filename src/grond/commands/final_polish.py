"""The optional dynamical (DiffBloch-style) final polish: ``push-endpoint --final-polish diffbloch`` and AUTO's last stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grond.cli_support.tools import choose_python, shelxl_is_available

POLISH_DIR = "final_polish_diffbloch"
# The push-endpoint options the polish uses.  Every other push-endpoint option belongs to a checkpoint
# continuation; the polish takes those settings from the run and refuses any that differ from their defaults, so
# no requested setting is silently ignored.
POLISH_PUSH_OPTIONS = {
    "command", "func", "out_dir", "run_dir", "start_res", "python", "shelxl", "workers", "dry_run", "config",
    "allow_missing_tools", "exclude_orbits_json",
}


def ignored_push_options(args: argparse.Namespace) -> list[str]:
    """push-endpoint options set away from their defaults that the polish would not use."""

    from grond.cli_options.parser import build_parser

    defaults = vars(build_parser().parse_args(["push-endpoint", "--out-dir", "unused"]))
    return sorted(
        f"--{key.replace('_', '-')}"
        for key, default in defaults.items()
        if key not in POLISH_PUSH_OPTIONS and not key.startswith("final_polish") and getattr(args, key, default) != default
    )


def unused_polish_options(args: argparse.Namespace) -> list[str]:
    """--final-polish-* options (and push-endpoint's --run-dir) given while --final-polish is off."""

    from grond.cli_options.run_endpoints import add_final_polish

    probe = argparse.ArgumentParser(add_help=False)
    add_final_polish(probe)
    defaults = {key: value for key, value in vars(probe.parse_args([])).items() if key != "final_polish"}
    defaults["run_dir"] = ""
    return sorted(
        f"--{key.replace('_', '-')}" for key, default in defaults.items() if getattr(args, key, default) != default
    )


def refuse_unused_polish_options(args: argparse.Namespace) -> None:
    """Exit when --final-polish-* options (or --run-dir) are given with --final-polish off."""

    if str(getattr(args, "final_polish", "off") or "off") == "off":
        unused = unused_polish_options(args)
        if unused:
            raise SystemExit(f"{', '.join(unused)} only apply with --final-polish diffbloch, which is off")


def polish_options(args: argparse.Namespace):
    """Return the polish settings (workers, iteration, task and step limits, thickness mode) from ``args``."""

    from grond.polish.refine import PolishOptions

    return PolishOptions(
        workers=max(1, int(getattr(args, "workers", 1) or 1)),
        max_iterations=int(args.final_polish_max_iterations),
        rows_per_task=int(args.final_polish_rows_per_task),
        max_xyz_step_A=float(args.final_polish_max_xyz_step),
        max_u_step_A2=float(args.final_polish_max_u_step),
        refine_thickness=str(args.final_polish_thickness) == "refine",
        jacobian_memory_gib=float(args.final_polish_jacobian_memory_gib),
        hydrogen_mode=str(getattr(args, "final_polish_hydrogens", "translate")),
    )


def validate_final_polish(args: argparse.Namespace, *, structure: str | Path | None = None) -> None:
    """Refuse invalid polish options, a missing SHELXL and (with ``structure``) a model the polish cannot represent.

    AUTO calls this before its first stage, so a polish that could not run fails the run at once.
    """

    for name in ("final_polish_max_iterations", "final_polish_rows_per_task", "final_polish_shelxl_iterations",
                 "final_polish_shelxl_ls_cycles"):
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be at least 1")
    for name in ("final_polish_max_xyz_step", "final_polish_max_u_step", "final_polish_jacobian_memory_gib"):
        if not float(getattr(args, name)) > 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if str(args.final_polish_shelxl) == "run" and not shelxl_is_available(args.shelxl):
        raise SystemExit(
            f"--final-polish diffbloch writes SHELXL outputs, but SHELXL {args.shelxl!r} is not available; install it or "
            "pass --final-polish-shelxl skip for a polish without SHELXL files"
        )
    if structure is not None and Path(structure).suffix.lower() in {".res", ".ins"}:
        from grond.polish.structure import StructureModel

        # The publication model keeps the input's instructions and atom list, so its refusals are known now.
        try:
            model = StructureModel(structure, hydrogen_mode=str(getattr(args, "final_polish_hydrogens", "translate")))
        except ValueError as exc:
            raise SystemExit(f"--final-polish diffbloch cannot polish {structure}: {exc}") from exc
        if model.n_params == 0:
            raise SystemExit(f"--final-polish diffbloch: {structure} has no refinable coordinates or displacement parameters")


def cmd_final_polish(args: argparse.Namespace) -> int:
    """Polish a finished run (push-endpoint --final-polish diffbloch --run-dir RUN --out-dir OUT)."""

    if str(args.final_polish) != "diffbloch":
        raise SystemExit(f"unknown --final-polish {args.final_polish!r}")
    run_dir = str(getattr(args, "run_dir", "") or "").strip()
    if not run_dir:
        raise SystemExit("--final-polish diffbloch needs --run-dir, the finished GROND run to polish")
    if str(getattr(args, "checkpoint", "") or "").strip() or str(getattr(args, "source_summary", "") or "").strip():
        raise SystemExit(
            "--final-polish diffbloch polishes a finished run from --run-dir; it does not continue a --checkpoint or "
            "--source-summary. Run the continuation first, or polish the run directly"
        )
    ignored = ignored_push_options(args)
    if ignored:
        raise SystemExit(
            "--final-polish diffbloch takes the physics, split, scales and SHELXL handoff from the finished run and "
            f"would ignore {', '.join(ignored)}; drop them (the polish options are the --final-polish-* flags)"
        )
    validate_final_polish(args)
    exclusion = str(getattr(args, "exclude_orbits_json", "") or "").strip()
    python = choose_python(getattr(args, "python", ""))
    options = polish_options(args)
    out_dir = Path(args.out_dir).expanduser().resolve()
    if bool(getattr(args, "dry_run", False)):
        plan = {
            "mode": "final_polish_diffbloch",
            "run_dir": str(Path(run_dir).expanduser().resolve()),
            "start_res": str(getattr(args, "start_res", "") or "publication model of the run"),
            "out_dir": str(out_dir),
            "options": options.__dict__,
            "shelxl": str(args.shelxl) if str(args.final_polish_shelxl) == "run" else "skip",
            "exclude_orbits_json": str(Path(exclusion).expanduser().resolve()) if exclusion else "",
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final_polish_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(plan, indent=2))
        return 0
    from grond.polish.run import run_diffbloch_polish

    summary = run_diffbloch_polish(
        run_dir,
        out_dir,
        start_res=str(getattr(args, "start_res", "") or "") or None,
        options=options,
        python=python,
        shelxl=str(args.shelxl),
        run_shelxl=str(args.final_polish_shelxl) == "run",
        shelxl_iterations=int(args.final_polish_shelxl_iterations),
        shelxl_ls_cycles=int(args.final_polish_shelxl_ls_cycles),
        exclude_orbits_json=exclusion or None,
    )
    free = ((summary.get("heldout") or {}).get("free") or {})
    delta = ((free.get("polished_minus_start") or {}).get("r_obs") or {})
    if delta:
        lo, hi = delta.get("ci95", [None, None])
        print(f"final polish: free-row R_obs polished - start {delta.get('delta'):+.3f} pp [{lo:+.3f}, {hi:+.3f}]")
    print(f"final polish report: {out_dir / 'FINAL_POLISH.md'}")
    return 0


def final_polish_command(args: argparse.Namespace, python: str) -> list[str]:
    """The logged push-endpoint command AUTO runs as its last stage."""

    from grond.paths import engine_script

    root = Path(args.out_dir).expanduser().resolve()
    cmd = [
        python,
        str(engine_script("diffbloch_polish.py")),
        "--final-polish",
        str(args.final_polish),
        "--run-dir",
        str(root),
        "--out-dir",
        str(root / POLISH_DIR),
        "--python",
        python,
        "--shelxl",
        str(args.shelxl),
        "--workers",
        str(max(1, int(getattr(args, "workers", 1) or 1))),
        "--final-polish-max-iterations",
        str(int(args.final_polish_max_iterations)),
        "--final-polish-hydrogens",
        str(getattr(args, "final_polish_hydrogens", "translate")),
        "--final-polish-thickness",
        str(args.final_polish_thickness),
        "--final-polish-max-xyz-step",
        f"{float(args.final_polish_max_xyz_step):g}",
        "--final-polish-max-u-step",
        f"{float(args.final_polish_max_u_step):g}",
        "--final-polish-rows-per-task",
        str(int(args.final_polish_rows_per_task)),
        "--final-polish-jacobian-memory-gib",
        f"{float(args.final_polish_jacobian_memory_gib):g}",
        "--final-polish-shelxl",
        str(args.final_polish_shelxl),
        "--final-polish-shelxl-iterations",
        str(int(args.final_polish_shelxl_iterations)),
        "--final-polish-shelxl-ls-cycles",
        str(int(args.final_polish_shelxl_ls_cycles)),
    ]
    exclusion = str(getattr(args, "exclude_orbits_json", "") or "").strip()
    if exclusion:
        # The polish verifies that every stage dropped the same lockbox orbits.
        cmd += ["--exclude-orbits-json", str(Path(exclusion).expanduser().resolve())]
    return cmd


def run_final_polish_stage(args: argparse.Namespace, python: str, logs: Path, result: dict | None = None) -> dict:
    """AUTO's optional last stage; runs after grond_summary.json exists and fails closed on any error.

    A run that stopped before its final endpoint (``stopped_bad_preflight``, ``stopped_bad_fit``) has no
    publication model to polish; the stage is then not run, and the reason is recorded and printed.
    """

    from grond.cli_support.execution import run_logged

    stopped = str((result or {}).get("status") or "")
    if stopped.startswith("stopped_"):
        print(f"final_polish_diffbloch not run: the run stopped before its final endpoint ({stopped})", flush=True)
        return {"status": "not_run", "reason": f"run status {stopped}: no final endpoint to polish", "mode": str(args.final_polish)}
    validate_final_polish(args)
    root = Path(args.out_dir).expanduser().resolve()
    cmd = final_polish_command(args, python)
    run_logged(cmd, logs, "final_polish_diffbloch", dry_run=bool(getattr(args, "dry_run", False)))
    if bool(getattr(args, "dry_run", False)):
        return {"status": "dry_run", "command": cmd}
    path = root / POLISH_DIR / "final_polish_summary.json"
    summary = json.loads(path.read_text())
    free = ((summary.get("heldout") or {}).get("free") or {})
    shelxl = summary.get("shelxl") or {}
    return {
        "status": summary.get("status"),
        "mode": str(args.final_polish),
        "summary": str(path),
        "report": str(root / POLISH_DIR / "FINAL_POLISH.md"),
        "polished_model": summary.get("polished_model"),
        "optimizer_status": (summary.get("optimizer") or {}).get("status"),
        "free_r_obs": {
            "start": ((free.get("start") or {}).get("r_obs") or {}),
            "polished": ((free.get("polished") or {}).get("r_obs") or {}),
            "polished_minus_start": ((free.get("polished_minus_start") or {}).get("r_obs") or {}),
        },
        "shelxl_refined_res": (shelxl.get("refined") or {}).get("res"),
        "shelxl_as_polished_res": (shelxl.get("as_polished") or {}).get("res"),
        "note": "additional output; the AUTO publication endpoint is unchanged",
    }
