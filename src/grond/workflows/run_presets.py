"""Named AUTO presets (``--preset``): fill the run settings the user did not give.

A preset is a list of ordinary option values.  It changes nothing the user set on the command line, it adds no
calculation of its own, and it is recorded in ``run_config.json`` (``preset``) and in the held-out report, so a
preset run is reproduced by its explicit flags alone.  ``standard`` (the default) changes nothing.

``fast`` lowers fidelity and search breadth only.  It keeps every piece of the honest validation of the standard
AUTO: the orbit-hash free set and the lockbox exclusion, the leak-free start, frame scales fitted on work rows,
nuisance selection on work rows (the nested inner orbit CV), the early held-out hand decision and its final
confirmation, the SHELXL handoff without free rows, and the held-out report with CIs and leak flags.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

PRESET_SCHEMA = "grond.run-preset.v1"
DEFAULT_PRESET = "standard"


@dataclass(frozen=True)
class PresetSetting:
    """One option value a preset fills when the user did not give the option."""

    dest: str
    flag: str
    value: object
    why: str
    # Options that only act when this one keeps its standard value: if the user gave any of them, the preset
    # leaves this option alone, so the user's option is not silently made moot.
    unless_given: tuple[str, ...] = ()


# Settings of --preset fast: lower fidelity and narrower searches; every validation step is kept.
FAST_PRESET: tuple[PresetSetting, ...] = (
    PresetSetting(
        "sg_window_mode", "--sg-window-mode", "fixed",
        "fixed 0.010 1/A rocking window with linear Sg sampling instead of the per-row DIALS bbox window "
        "(up to 0.03 1/A)",
    ),
    PresetSetting("n_points", "--n-points", 20, "20 final profile points (standard: 60 DIALS bbox, 50 fixed 0.010)"),
    PresetSetting(
        "screen_n_points", "--screen-n-points", 10, "10 screen, gate and nested-spectra points (standard: 30 or 25)"
    ),
    PresetSetting(
        "nested_n_points", "--nested-n-points", "screen",
        "nested cone x dose x thickness selection on the screen points (standard: final points)",
    ),
    PresetSetting(
        "mosaicity_sigma_grid", "--mosaicity-sigma-grid", "0,0.1,0.2",
        "three cone widths, none, 0.1 and 0.2 deg (standard: 0, 0.1, 0.2, 0.3 deg); 0.1 is kept so that a weakly "
        "mosaic crystal is not forced to 0.2",
    ),
    PresetSetting(
        "dose_b_total_grid", "--dose-b-total-grid", "0,2",
        "two dose values, 0 and 2 A^2 over the scan (standard: 0, 0.5, 1, 2, 3)",
    ),
    PresetSetting(
        "n_basis", "--n-basis", 120,
        "beam cap 120 without the per-stage support scan (standard: support scan, max support + 10)",
        # The support-scan options act only with --n-basis 0.
        unless_given=("final_basis_scan_limit", "final_basis_margin", "final_basis_min", "final_basis_max"),
    ),
    PresetSetting(
        "shelxl_feedback_cycles", "--shelxl-feedback-cycles", 1,
        "one SHELXL macro-cycle (standard: up to 5, with the 0.002 A shift stop)",
    ),
    PresetSetting(
        "feedback_thickness_update", "--feedback-thickness-update", "off",
        "the nested thickness pick is kept through the macro-cycles (standard: re-selected for every structure)",
    ),
    PresetSetting(
        "thickness_grid", "--thickness-grid", "1200",
        "one screen thickness seed (standard: 1000, 1200, 1400); the nested stage selects the thickness",
    ),
    PresetSetting(
        "thickness_grad_grid", "--thickness-grad-grid", "0.4",
        "one screen thickness-gradient seed (standard: 0.4, 1.0)",
    ),
    PresetSetting(
        "final_thickness_points", "--final-thickness-points", 5,
        "5 thickness quadrature points in the final stages (standard: 9); unused by the nested gamma/wedge table "
        "evaluators, and it lets the final stage reuse the final row recompute's cached prediction",
    ),
    PresetSetting(
        "hand_early_escalation", "--hand-early-escalation", "off",
        "an inconclusive early hand test is not recomputed at final fidelity (the final confirmation still runs)",
    ),
    PresetSetting(
        "unmerged_endpoint", "--unmerged-endpoint", "off", "no unmerged HKLF 4 batch endpoint (standard: on)"
    ),
    PresetSetting("forensic_depth", "--forensic-depth", "quick", "quick forensic reports (standard: standard)"),
)

PRESETS: dict[str, tuple[PresetSetting, ...]] = {DEFAULT_PRESET: (), "fast": FAST_PRESET}
PRESET_DESCRIPTIONS = {
    DEFAULT_PRESET: "the AUTO defaults",
    "fast": (
        "lower fidelity and search breadth for a quick answer; every validation step of the standard AUTO is kept "
        "and the held-out numbers are reported as usual"
    ),
}


def _flag_value(value: object) -> str:
    """Format a preset value as a command-line token (floats with %g)."""

    return f"{value:g}" if isinstance(value, float) else str(value)


def _suppress_defaults(parser: argparse.ArgumentParser) -> None:
    """Give every option of the parser and its subparsers the SUPPRESS default, so a parse returns only given options."""

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                _suppress_defaults(sub)
        elif action.dest != "help":
            action.default = argparse.SUPPRESS
    parser._defaults.clear()


def explicit_option_dests(argv: Sequence[str]) -> frozenset[str]:
    """The option destinations given on this command line (abbreviations and ``--no-`` forms included)."""

    from grond.cli_options.parser import build_parser

    parser = build_parser()
    _suppress_defaults(parser)
    namespace = parser.parse_args(list(argv))
    return frozenset(name for name in vars(namespace) if name not in {"command", "func"})


def _non_default_dests(args: argparse.Namespace) -> frozenset[str]:
    """Fallback for namespaces built without a command line: every option that differs from its parser default."""

    from grond.cli_options.parser import build_parser

    defaults = vars(build_parser().parse_args(["auto", "--out-dir", "_"]))
    return frozenset(name for name, value in vars(args).items() if name in defaults and defaults[name] != value)


def apply_run_preset(args: argparse.Namespace, explicit: Iterable[str] | None = None) -> dict[str, object]:
    """Fill the preset's settings the user did not give, and return the provenance record.

    ``explicit`` is the set of option destinations given on the command line; without it the namespace's
    ``explicit_options`` (recorded by the CLI entry point) is used, and failing that every option whose value
    differs from its parser default counts as given.
    """

    name = str(getattr(args, "preset", DEFAULT_PRESET) or DEFAULT_PRESET)
    if name not in PRESETS:
        raise SystemExit(f"unknown --preset {name!r}; choices are {', '.join(PRESETS)}")
    record: dict[str, object] = {
        "schema": PRESET_SCHEMA,
        "name": name,
        "description": PRESET_DESCRIPTIONS[name],
        "filled": {},
        "kept_explicit": {},
        "left_for_explicit": {},
    }
    if name == DEFAULT_PRESET:
        args.preset_record = record
        return record
    mode = str(getattr(args, "mode", "auto") or "auto")
    if mode != "auto":
        # The preset's record states that every AUTO validation step is kept; that is not true of other modes.
        raise SystemExit(f"--preset {name} applies to the AUTO workflow (--mode auto), not --mode {mode}")
    for legacy in ("fast", "full_fast"):
        if bool(getattr(args, legacy, False)):
            raise SystemExit(
                f"--preset {name} cannot be combined with --{legacy.replace('_', '-')}: --fast is the quick-look "
                "frozen-model run and --full-fast the cheap-N/Q profile; the preset keeps the standard AUTO "
                "stages at lower fidelity"
            )
    if explicit is None:
        recorded = getattr(args, "explicit_options", None)
        given = frozenset(recorded) if recorded is not None else _non_default_dests(args)
        record["explicit_source"] = "command_line" if recorded is not None else "differs_from_parser_default"
    else:
        given = frozenset(explicit)
        record["explicit_source"] = "caller"
    filled: dict[str, object] = {}
    kept: dict[str, object] = {}
    left: dict[str, object] = {}
    for setting in PRESETS[name]:
        if setting.dest in given:
            kept[setting.dest] = {"flag": setting.flag, "value": getattr(args, setting.dest, None),
                                  "preset_value": setting.value}
            continue
        because = [f"--{dest.replace('_', '-')}" for dest in setting.unless_given if dest in given]
        if because:
            left[setting.dest] = {"flag": setting.flag, "value": getattr(args, setting.dest, None),
                                  "preset_value": setting.value, "because": because}
            continue
        filled[setting.dest] = {"flag": setting.flag, "value": setting.value,
                                "standard": getattr(args, setting.dest, None), "why": setting.why}
        setattr(args, setting.dest, setting.value)
    record["filled"] = filled
    record["kept_explicit"] = kept
    record["left_for_explicit"] = left
    record["equivalent_flags"] = [token for dest, entry in filled.items() for token in (entry["flag"], _flag_value(entry["value"]))]
    args.preset_record = record
    return record


def preset_source(args: argparse.Namespace, dest: str) -> str | None:
    """``preset:<name>`` when a named preset filled ``dest`` and the value is still the preset's, else None.

    Provenance records that label a set option ``explicit`` use this so a preset value is not reported as given.
    """

    record = getattr(args, "preset_record", None)
    filled = (record.get("filled") or {}).get(dest) if isinstance(record, dict) else None
    if isinstance(filled, dict) and filled.get("value") == getattr(args, dest, None):
        return f"preset:{record.get('name')}"
    return None


def preset_console_lines(record: dict[str, object] | None) -> list[str]:
    """Startup lines naming the preset, what it filled and what the user kept."""

    if not record or record.get("name") in (None, DEFAULT_PRESET):
        return []
    filled = record.get("filled") or {}
    kept = record.get("kept_explicit") or {}
    lines = [f"preset {record['name']}: {record.get('description')}"]
    if filled:
        lines.append("preset " + str(record["name"]) + " sets " + " ".join(record.get("equivalent_flags") or []))
    if kept:
        lines.append(
            "preset " + str(record["name"]) + " keeps your "
            + ", ".join(f"{entry['flag']} {_flag_value(entry['value'])}" for entry in kept.values())
        )
    for entry in (record.get("left_for_explicit") or {}).values():
        lines.append(
            f"preset {record['name']} leaves {entry['flag']} {_flag_value(entry['value'])} (preset value "
            f"{_flag_value(entry['preset_value'])}) because you gave {', '.join(entry['because'])}"
        )
    return lines


def preset_markdown_lines(record: dict[str, object] | None) -> list[str]:
    """Report lines (held-out report, QUICK_STATS, FINAL_SUMMARY) for a non-standard preset."""

    if not record or record.get("name") in (None, DEFAULT_PRESET):
        return []
    filled = record.get("filled") or {}
    kept = record.get("kept_explicit") or {}
    lines = [
        f"**Preset `{record['name']}`**: {record.get('description')}. Settings filled by the preset: "
        + (", ".join(f"`{entry['flag']} {_flag_value(entry['value'])}`" for entry in filled.values()) or "none")
        + "."
    ]
    if kept:
        lines[0] += " Given explicitly and kept: " + ", ".join(
            f"`{entry['flag']} {_flag_value(entry['value'])}`" for entry in kept.values()
        ) + "."
    left = record.get("left_for_explicit") or {}
    if left:
        lines[0] += " Left at the standard value because a dependent option was given: " + ", ".join(
            f"`{entry['flag']} {_flag_value(entry['value'])}` ({', '.join(entry['because'])})" for entry in left.values()
        ) + "."
    return [*lines, ""]
