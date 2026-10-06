"""Final all-data refinement, the last step of ``--free-set off`` (the AUTO default).

AUTO always chooses the hand, the nuisance settings and the publication endpoint with the orbit-hash free set held
out, and scores that endpoint on it (HELDOUT_VALIDATION).  With ``--free-set off`` the published model is then refined
once more against every reflection: the endpoint's own work HKL plus its free reflections corrected by the same recipe
(the endpoint free validation's ``validation.hkl`` with SHELXL's free flag removed), with the endpoint's SHELXL contract
(weighting mode, L.S. cycles, convergence tolerance, NPD recovery, native outputs).  That all-data model is the run's
result; the held-out statistics describe the published endpoint before this step.  ``--free-set on`` keeps the free
reflections out of every fit, the final model included.

The free reflections exist for the merged combined endpoint of a multi-crystal run (written by the combine engine's
free validation) and are rebuilt for the per-crystal corrected-HKL endpoint of a single-crystal run
(grond.validation.native_endpoint_free).  Any other published endpoint is reported as ``not_available`` and stays the
result.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

from grond.io.config import PrepConfig
from grond.reports.values import try_read_json
from grond.validation.endpoint_free import FREE_FLAG, hklf4_reflection_lines, write_hklf4_lines

ALL_DATA_DIR = "final_all_data"
STEM = "all_data"
NATIVE_SOURCE_KIND = "native_corrected_hkl"
SUMMARY_NAME = "all_data_refinement.json"
STATISTICS = ("R1_gt", "R1_all", "wR2_all", "GooF", "N_gt", "N_all")
DEFINITION = (
    "the published endpoint refined again with its SHELXL contract against its work reflections plus its free "
    "reflections, both corrected by the endpoint's recipe; the held-out statistics describe the endpoint before this step"
)


def free_set_held_out(args: argparse.Namespace) -> bool:
    """True when the free reflections stay out of every fit, the final model included (``--free-set on``)."""

    return str(getattr(args, "free_set", "on") or "on") == "on"


def all_data_lines(validation_hkl: Path) -> tuple[list[str], int, int]:
    """HKLF 4 lines of a work-plus-free validation file with the free flag removed: (lines, work count, free count)."""

    lines = hklf4_reflection_lines(validation_hkl)
    free = sum(line[28:32].strip() == str(FREE_FLAG) for line in lines)
    return [line[:28] for line in lines], len(lines) - free, free


def scale_hklf_record(ins: Path, factor: float) -> None:
    """Multiply the intensity scale (second HKLF argument) of an .ins by ``factor``."""

    out = []
    for line in Path(ins).read_text().splitlines():
        parts = line.split()
        if parts and parts[0].upper() == "HKLF":
            scale = float(parts[2]) if len(parts) > 2 else 1.0
            line = " ".join([parts[0], parts[1] if len(parts) > 1 else "4", f"{scale * factor:.10g}", *parts[3:]])
        out.append(line)
    Path(ins).write_text("\n".join(out) + "\n")


def endpoint_free_reflections(endpoint: Mapping[str, object], result: Mapping[str, object], out_dir: Path,
                              shelxl: str) -> dict[str, object]:
    """The published endpoint's work-plus-free validation HKL and its extra HKL factor, or why there is none."""

    res = Path(str(endpoint.get("endpoint_res") or endpoint.get("res") or ""))
    if result.get("combined"):
        summary = try_read_json(res.parent.parent / "combined_summary.json")
        merged = summary.get("combined_endpoint") if isinstance(summary.get("combined_endpoint"), Mapping) else {}
        merged_res = str((merged.get("artifacts") or {}).get("res") or "")
        if not merged_res or Path(merged_res).resolve() != res.resolve():
            return {"status": "not_available",
                    "reason": f"the published endpoint {res} is not the merged combined endpoint of {res.parent.parent}"}
        validation = merged.get("free_validation") if isinstance(merged.get("free_validation"), Mapping) else {}
    elif str(endpoint.get("source_kind") or "") == NATIVE_SOURCE_KIND:
        from grond.validation.native_endpoint_free import ReplayRefused, replay_native_endpoint

        try:
            replay = replay_native_endpoint(endpoint, out_dir / "free_reflections", shelxl=shelxl)
        except ReplayRefused as exc:
            return {"status": exc.status, "reason": exc.reason}
        validation = replay.get("free_validation") if isinstance(replay.get("free_validation"), Mapping) else {}
    else:
        return {"status": "not_available",
                "reason": f"published endpoint kind {endpoint.get('source_kind')!r} has no free-reflection recipe"}
    hkl = Path(str((validation.get("artifacts") or {}).get("validation_hkl") or ""))
    free_rows = validation.get("free_rows") if isinstance(validation.get("free_rows"), Mapping) else {}
    overlap = free_rows.get("merge_key_classes_also_in_work_hkl", free_rows.get("symmetry_classes_also_in_work_hkl"))
    if not hkl.is_file():
        return {"status": "not_available",
                "reason": f"no free-validation HKL for the published endpoint (free validation {validation.get('status')})"}
    if overlap:
        return {"status": "free_reflections_overlap_work", "reason": f"{overlap} free symmetry classes are also work"}
    extra = free_rows.get("extra_hkl_factor")
    return {"status": "ok", "validation_hkl": str(hkl), "extra_hkl_factor": float(extra) if extra else None,
            "free_validation_status": validation.get("status")}


def run_all_data_refinement(prep: PrepConfig, result: Mapping[str, object], args: argparse.Namespace) -> dict:
    """Refine the published endpoint against all reflections when the free set is off; return the record."""

    if free_set_held_out(args):
        return {"status": "skipped_free_set_on"}
    if getattr(args, "dry_run", False):
        return {"status": "dry_run"}
    if str(getattr(args, "endpoint_free_validation", "on")) != "on":
        return {"status": "not_available", "reason": "--endpoint-free-validation off writes no free reflections"}
    from grond.legacy.engine.combine_native_nuisance_runs import endpoint_native_outputs, run_shelxl
    from grond.refine.shelx_input import write_endpoint_ins
    from grond.refine.shelxl_iterations import endpoint_initial_wght
    from grond.workflows.finish_maps import _publication_endpoint_from_result

    out_dir = Path(getattr(args, "out_dir", ".")) / ALL_DATA_DIR
    endpoint = _publication_endpoint_from_result(result)
    if not endpoint:
        return {"status": "not_available", "reason": "the run published no endpoint"}
    shelxl = str(getattr(args, "shelxl", "") or "")
    record: dict[str, object] = {
        "free_set": "off",
        "definition": DEFINITION,
        "source_endpoint": {key: endpoint.get(key) for key in ("selection_source", "source_kind", "id", "label",
                                                               "endpoint_res", "endpoint_hkl", "res_sha256")},
    }
    free = endpoint_free_reflections(endpoint, result, out_dir, shelxl)
    record["free_reflections"] = free
    if free.get("status") != "ok":
        return _written(out_dir, {**record, "status": str(free.get("status")), "reason": free.get("reason")})
    lines, n_work, n_free = all_data_lines(Path(str(free["validation_hkl"])))
    write_hklf4_lines(out_dir / f"{STEM}.hkl", lines)
    record["reflections"] = {"work": n_work, "free": n_free}
    mode = str(getattr(args, "endpoint_wght_mode", "freeze-zero"))
    cleanup = str(getattr(args, "shelx_ins_cleanup", "off") or "off") == "on"
    native = SimpleNamespace(zerr_esds=str(getattr(args, "resolved_zerr_esds", "") or ""), strip_q_peaks=cleanup,
                             endpoint_list_code=6 if cleanup else 0, acta=bool(getattr(args, "shelx_acta", False)))
    write_endpoint_ins(
        Path(str(endpoint["endpoint_res"])), out_dir / f"{STEM}.ins", hkl_name=f"{STEM}.hkl",
        keep_exti=bool(endpoint.get("keep_exti", False)), wght=endpoint_initial_wght(mode),
        ls_cycles=int(getattr(args, "endpoint_ls_cycles", 1)), keep_damp=bool(getattr(args, "shelxl_carry_damp", False)),
        **endpoint_native_outputs(native),
    )
    if free.get("extra_hkl_factor"):
        # the validation file carries every reflection times the extra fixed-width factor; the HKLF scale undoes it
        scale_hklf_record(out_dir / f"{STEM}.ins", 1.0 / float(free["extra_hkl_factor"]))
    refined = run_shelxl(
        out_dir, STEM, shelxl, iterations=int(getattr(args, "shelxl_iterations", 5)), wght_mode=mode,
        endpoint_ls_cycles=int(getattr(args, "endpoint_ls_cycles", 1)),
        convergence_shift_esd=float(getattr(args, "shelxl_convergence_shift_esd", 0.0) or 0.0),
        npd_recovery=str(getattr(args, "shelxl_npd_recovery", "damp") or "damp"),
        keep_damp=bool(getattr(args, "shelxl_carry_damp", False)),
        wght_converge_max_calls=int(getattr(args, "wght_converge_max_calls", 20) or 20),
    )
    record["shelxl"] = {key: value for key, value in refined.items() if key != "iteration_history"}
    record["statistics"] = {key: refined.get(key) for key in STATISTICS}
    record["artifacts"] = {suffix: str((out_dir / f"{STEM}.{suffix}").resolve())
                           for suffix in ("res", "hkl", "lst", "fcf", "cif") if (out_dir / f"{STEM}.{suffix}").is_file()}
    return _written(out_dir, {**record, "status": str(refined.get("status") or "unknown")})


def all_data_console_line(record: Mapping[str, object]) -> str:
    """One console line on the final all-data refinement."""

    status = str(record.get("status"))
    if status == "skipped_free_set_on":
        return "free set on: the free reflections stay out of the final model"
    if status == "dry_run":
        return "free set off: final all-data refinement (dry run)"
    if status != "ok":
        return (f"final all-data refinement not done ({status}): {record.get('reason') or ''}; "
                "the published endpoint stays the result")
    stats, counts = record.get("statistics") or {}, record.get("reflections") or {}

    def pct(key: str) -> str:
        value = stats.get(key)
        return f"{100.0 * float(value):.2f}%" if value is not None else "n/a"

    return (f"final all-data refinement (free set off): R1gt {pct('R1_gt')}, R1all {pct('R1_all')}, "
            f"wR2 {pct('wR2_all')}, GooF {stats.get('GooF')} on {counts.get('work')} work + {counts.get('free')} free "
            f"reflections -> {(record.get('artifacts') or {}).get('res', '')}")


def _written(out_dir: Path, record: dict) -> dict:
    """Write the record to ``out_dir/all_data_refinement.json`` and return it."""

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SUMMARY_NAME).write_text(json.dumps(record, indent=2, default=str) + "\n")
    return record
