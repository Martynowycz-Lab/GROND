"""Held-out validation of the SHELXL endpoints: the refined models scored on the orbits AUTO held out.

AUTO holds out about 10% of the Laue orbits (orbit-hash free set, the same orbits in every crystal).  Free rows never
enter the frame scales, the nuisance selection, the dataset policy, the leak-free start or any refinement HKL, so a
final model has never seen them.  Here each final model is scored on them with SHELXL itself, zero cycles, in one run
of its own work HKL plus the free reflections under SHELXL's free-R flag (grond.refine.zero_cycle): R1(Fo > 4sig(Fo)),
R1(all), wR2 and GooF on the free reflections next to the same statistics of the same model on its own work
reflections, and the gap work minus free.

* GROND (combine engine, --endpoint-free-validation on): the corrected free rows of the final merged and unmerged
  endpoints, built exactly like the work rows (same final-model q and kappa scale, same q-exclusion reference, the
  per-crystal error model fitted on work rows, same dataset weights, inter-dataset scales and merge rule).
* Conventional (leak-free start): the free reflections of the conventional baseline HKL (same orbit-hash masks),
  scored with the conventional model refined on the work orbits only.

Reporting only: nothing in GROND is chosen on these numbers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from grond.refine.zero_cycle import run_zero_cycle

ROLE = (
    "reporting only: nothing is chosen on these numbers; free rows never enter scales, nuisance selection, the "
    "dataset policy, the leak-free start or any refinement HKL"
)
DEFINITION = (
    "one SHELXL L.S. 0 run (no parameter moves; final WGHT, BASF and EXTI as refined) of the final refined model on "
    "its own work HKL plus the held-out (free) reflections flagged -1 (L.S. 0 -1, SHELXL's free-R flag). Work: SHELXL's "
    "statistics as printed (GooF with N - P). Free: R1(Free) as printed (R1gt: Fo > 4sig(Fo)); wR2 and GooF from the "
    ".fcf on the model's scale with the final weights, GooF with N (nothing was fitted to the free reflections). "
    "Gap = work minus free"
)
STATISTICS = ("R1_gt", "R1_all", "wR2_all", "GooF")
FREE_FLAG = -1
# The work-statistics reproduction rule, defined once for every L.S. 0 score of a refined model (the GROND merged and
# unmerged endpoints and the conventional leak-free start).  The model and the work reflections are the endpoint's own
# bytes: the score fails unless L.S. 0 left the model unchanged (score_model_on_work_and_free), and grond
# validate-endpoint compares the HKL and model bytes.  SHELXL prints R1 and wR2 with four decimals and S (GooF) with
# three.  The work half of the score must print the same N_gt and N_all as the endpoint's own listing; R1gt, R1all,
# wR2 and GooF may each differ by one unit of their last printed digit (0.0001 for R1 and wR2, 0.001 for GooF).  The
# endpoint's L.S. N listing prints the statistics of its last cycle, computed before that cycle's final shifts are
# applied, while L.S. 0 scores the shifted parameters in the .res, so at a rounding edge the two round differently
# (for example GooF 1.088 at L.S. 10 and 1.089 at L.S. 0 when S(N - P) from the .fcf is 1.0885).  Nothing else is
# allowed to differ.
WORK_STATISTICS_RULE: dict[str, dict[str, int]] = {
    "R1_gt": {"printed_decimals": 4, "allowed_units_of_last_digit": 1},
    "R1_all": {"printed_decimals": 4, "allowed_units_of_last_digit": 1},
    "wR2_all": {"printed_decimals": 4, "allowed_units_of_last_digit": 1},
    "GooF": {"printed_decimals": 3, "allowed_units_of_last_digit": 1},
    "N_gt": {"printed_decimals": 0, "allowed_units_of_last_digit": 0},
    "N_all": {"printed_decimals": 0, "allowed_units_of_last_digit": 0},
}
WORK_STATISTICS_RULE_TEXT = (
    "model and work HKL: the endpoint's own bytes, and L.S. 0 leaves the model unchanged; N_gt, N_all: equal as "
    "printed; R1gt, R1all, wR2 (printed with 4 decimals) and GooF (printed with 3 decimals): each within one unit of "
    "the last printed digit (0.0001, 0.001), since the endpoint's listing prints the statistics of its last L.S. "
    "cycle before that cycle's shifts and L.S. 0 scores the shifted parameters, which can round differently"
)
# The unrounded value the .fcf recomputation of the same L.S. 0 run gives for each printed statistic.
_UNROUNDED_KEY = {"R1_gt": "R1_gt", "R1_all": "R1_all", "wR2_all": "wR2_all", "GooF": "GooF_n_minus_p",
                  "N_gt": "N_gt", "N_all": "N_all"}
HKLF4_TERMINATOR = "   0   0   0    0.00    0.00"


def gap_work_minus_free(work: Mapping[str, object] | None, free: Mapping[str, object] | None) -> dict[str, float | None]:
    """work minus free of every statistic (fractions; GooF unitless)."""

    out: dict[str, float | None] = {}
    for key in STATISTICS:
        try:
            value = float(work[key]) - float(free[key])  # type: ignore[arg-type,index]
        except (KeyError, TypeError, ValueError):
            value = None
        out[key] = value if value is not None and math.isfinite(value) else None
    return out


def hklf4_reflection_lines(path: Path) -> list[str]:
    """Reflection lines of a fixed-format HKLF 4 file before the 0 0 0 terminator, verbatim."""

    out = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if len(line) < 12 or not line[:12].strip():
            continue
        if (int(line[0:4]), int(line[4:8]), int(line[8:12])) == (0, 0, 0):
            break
        out.append(line.rstrip())
    return out


def _rescaled(line: str, factor: float) -> str:
    """Return one HKLF 4 line with I and sigma multiplied by factor, rewritten in their F8.2 columns."""

    intensity, sigma = float(line[12:20]) * factor, float(line[20:28]) * factor
    return f"{line[:12]}{intensity:8.2f}{sigma:8.2f}{line[28:]}"


def flagged_validation_lines(
    work_lines: list[str], free_lines: list[str], *, work_rescale: float | None = None
) -> list[str]:
    """Work reflections verbatim, then the free reflections with SHELXL's free-R flag -1 in columns 29-32.

    ``work_rescale`` (below 1 when the free rows needed an extra factor to fit F8.2; they arrive with it applied)
    multiplies I and sigma of the work rows too, and the run's HKLF scale undoes it for every row.
    """

    work = [_rescaled(line, work_rescale) if work_rescale else line for line in work_lines]
    free = [line[:28].ljust(28) + f"{FREE_FLAG:4d}" for line in free_lines]
    return [*work, *free]


def write_hklf4_lines(path: Path, lines: list[str], *, batch_column: bool = False) -> None:
    """HKLF 4 file of fixed-format reflection lines with the 0 0 0 terminator."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    terminator = HKLF4_TERMINATOR + ("   0" if batch_column else "")
    Path(path).write_text("\n".join([*lines, terminator]) + "\n")


def score_model_on_work_and_free(
    directory: Path,
    *,
    res: Path,
    work_hkl: Path,
    free_lines: list[str],
    shelxl: str,
    wght_line: str | None = None,
    work_rescale: float | None = None,
    stem: str = "validation",
) -> dict[str, object]:
    """One zero-cycle SHELXL run of ``res`` on its work HKL plus ``free_lines`` flagged -1 (``directory/stem.*``)."""

    directory = Path(directory)
    work_lines = hklf4_reflection_lines(work_hkl)
    if any(line[28:32].strip() == str(FREE_FLAG) for line in work_lines):
        raise ValueError(f"{work_hkl} already carries free-R flags ({FREE_FLAG} in columns 29-32)")
    lines = flagged_validation_lines(work_lines, free_lines, work_rescale=work_rescale)
    write_hklf4_lines(directory / f"{stem}.hkl", lines, batch_column=True)
    run = run_zero_cycle(
        directory, stem, res=Path(res), hkl=directory / f"{stem}.hkl", shelxl=shelxl, wght_line=wght_line,
        hklf_scale=(1.0 / work_rescale) if work_rescale else None, free_flag=FREE_FLAG,
    )
    work, free = run.get("work"), run.get("free")
    status = run.get("status")
    unchanged = (run.get("model_changes") or {}).get("unchanged")
    if status == "ok" and unchanged is not True:
        # a score whose model moved (or could not be compared) is not a score of the refined model
        status = "model_changed_by_zero_cycle" if unchanged is False else "model_unchanged_not_verified"
    return {
        "status": status,
        "role": ROLE,
        "definition": DEFINITION,
        "model": str(Path(res).resolve()),
        "wght": run.get("wght"),
        "free": free,
        "work": work,
        "gap_work_minus_free": gap_work_minus_free(work, free),
        "model_unchanged_by_zero_cycle": bool((run.get("model_changes") or {}).get("unchanged")),
        "zero_cycle": {key: value for key, value in run.items() if key not in {"work", "free"}},
        "work_reflections_in_file": len(work_lines),
        "free_reflections_in_file": len(free_lines),
    }


def units_of_last_printed_digit(recorded: object, scored: object, printed_decimals: int) -> int:
    """|recorded - scored| in units of the last printed digit, each taken at its printed precision."""

    scale = 10 ** int(printed_decimals)
    return abs(round(float(recorded) * scale) - round(float(scored) * scale))  # type: ignore[arg-type]


def endpoint_reproduction(zero_cycle_work: Mapping[str, object], endpoint: Mapping[str, object]) -> dict[str, object]:
    """Does the zero-cycle work score reproduce the endpoint's own recorded SHELXL statistics (WORK_STATISTICS_RULE)?

    Each compared statistic records the endpoint's printed value, the zero-cycle printed value, the zero-cycle value
    recomputed from its .fcf without rounding (when the score recorded it), the difference in units of the last
    printed digit and the allowance.
    """

    unrounded = zero_cycle_work.get("recomputed_from_fcf")
    unrounded = unrounded if isinstance(unrounded, Mapping) else {}
    checks: dict[str, object] = {"rule": WORK_STATISTICS_RULE_TEXT}
    missing, invalid = [], []
    for key, rule in WORK_STATISTICS_RULE.items():
        recorded = endpoint.get(key)
        scored = zero_cycle_work.get(key)
        if recorded is None or scored is None:
            checks[key] = None
            missing.append(key)
            continue
        try:
            values = (float(recorded), float(scored))
            if not all(math.isfinite(value) for value in values):
                raise ValueError("non-finite statistic")
            if rule["printed_decimals"] == 0 and not all(value >= 0 and value.is_integer() for value in values):
                raise ValueError("reflection counts must be nonnegative integers")
            units = units_of_last_printed_digit(recorded, scored, rule["printed_decimals"])
        except (TypeError, ValueError, OverflowError):
            checks[key] = {"recorded": recorded, "zero_cycle": scored, "same": False,
                           "reason": "invalid_statistic"}
            invalid.append(key)
            continue
        checks[key] = {
            "recorded": recorded,
            "zero_cycle": scored,
            "zero_cycle_unrounded": unrounded.get(_UNROUNDED_KEY[key]),
            "printed_decimals": rule["printed_decimals"],
            "difference_units_of_last_digit": units,
            "allowed_units_of_last_digit": rule["allowed_units_of_last_digit"],
            "same": units <= rule["allowed_units_of_last_digit"],
        }
    compared = [value for value in checks.values() if isinstance(value, dict)]
    checks["missing_statistics"] = missing
    checks["invalid_statistics"] = invalid
    checks["complete"] = not missing and not invalid
    checks["all_same"] = checks["complete"] and all(bool(value["same"]) for value in compared)
    return checks


def record_endpoint_reproduction(scored: dict[str, object], endpoint: Mapping[str, object]) -> dict[str, object]:
    """Add ``work_reproduces_endpoint`` to a score; a work half that does not reproduce the endpoint is not ``ok``.

    The model half of the rule is the score's own status (a zero cycle that moved the model is not ``ok``); it is
    repeated in the record next to the statistics.
    """

    reproduction = endpoint_reproduction(scored.get("work") or {}, endpoint)  # type: ignore[arg-type]
    if "model_unchanged_by_zero_cycle" in scored:
        reproduction["model_unchanged_by_zero_cycle"] = scored.get("model_unchanged_by_zero_cycle")
    scored["work_reproduces_endpoint"] = reproduction
    if scored.get("status") == "ok" and not reproduction["all_same"]:
        scored["status"] = ("work_does_not_reproduce_endpoint" if reproduction["complete"]
                            else "work_reproduction_incomplete")
    return scored


def conventional_free_validation(
    out_dir: Path,
    rows: list[tuple[tuple[int, int, int], str]],
    masks: Mapping[str, object],
    *,
    res: Path,
    wght_line: str | None,
    shelxl: str,
    wght_source: str,
) -> dict[str, object]:
    """Score the leak-free conventional model on the baseline HKL's free reflections (same orbit-hash masks).

    ``rows`` are the (hkl, original line) pairs of the conventional baseline HKL and ``masks`` the masks the leak-free
    start used; ``wght_line`` is the WGHT the model was refined with in its final call.  Files go to
    ``out_dir/free_validation``: free.hkl (the free reflections), work.hkl (the refinement's own work reflections) and
    the scoring run validation.* (work plus free flagged -1).
    """

    free_lines = [line.rstrip() for (_hkl, line), keep in zip(rows, masks["free"], strict=True) if keep]
    work_lines = [line.rstrip() for (_hkl, line), keep in zip(rows, masks["work"], strict=True) if keep]
    directory = Path(out_dir) / "free_validation"
    directory.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "directory": str(directory.resolve()),
        "free_reflections_read": len(free_lines),
        "work_reflections_read": len(work_lines),
        "wght_source": wght_source,
    }
    if not free_lines:
        return {**record, "status": "skipped_no_free_reflections", "role": ROLE}
    batched = any(line[28:32].strip() for line in work_lines + free_lines)
    if batched and any(line.split()[:1] == ["BASF"] for line in Path(res).read_text(errors="replace").splitlines()):
        # a free row flagged -1 would lose its batch scale
        return {**record, "status": "not_supported_batch_scaled_conventional_model", "role": ROLE}
    if any(line[28:32].strip() == str(FREE_FLAG) for line in work_lines + free_lines):
        return {**record, "status": "not_supported_baseline_hkl_has_free_r_flags", "role": ROLE}
    write_hklf4_lines(directory / "free.hkl", free_lines)
    write_hklf4_lines(directory / "work.hkl", work_lines)
    scored = score_model_on_work_and_free(
        directory, res=Path(res), work_hkl=directory / "work.hkl", free_lines=free_lines, shelxl=shelxl,
        wght_line=wght_line,
    )
    return {**record, **scored}
