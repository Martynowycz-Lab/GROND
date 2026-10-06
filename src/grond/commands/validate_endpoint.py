"""``grond validate-endpoint``: held-out SHELXL validation of a finished AUTO run, written to a new directory.

The run directory is only read.  Two parts:

* GROND, ``--endpoint published`` (default): the endpoint the run published, the endpoint chooser's best physical
  endpoint recorded in grond_summary.json (the kept feedback cycle's combine or the final one).  The engine command that
  wrote it (logs/commands.jsonl, matched by its output directory) is re-run with this checkout's engine in score-only
  mode (``--score-res``): its rows are rebuilt with the recorded recipe, including the error-model pass the endpoint
  kept, the rebuilt work HKL must equal the published HKL byte for byte, and the published .res itself is scored at
  L.S. 0 on that HKL plus the corrected free rows, with the WGHT its own last refinement applied.  Nothing is
  re-refined, so a later build's refinement differences (the NPD recovery's tight ISOR tier) cannot break the
  reproduction.  The published endpoint is merged; the unmerged endpoint is validated by ``--endpoint final``.
  A single-crystal run publishes its per-crystal native endpoint: grond.validation.native_endpoint_free rebuilds its
  work HKL from the endpoint command's recorded row table (byte-identical to the published HKL) and scores the
  published .res the same way.
* GROND, ``--endpoint final`` (the behaviour before the option): the run's last ``combine_final`` command is re-run
  with this checkout's engine, its output directory moved to OUT/runs/combined and ``--endpoint-free-validation on``
  added (``--unmerged on`` also adds the unmerged batch endpoint when the run had none).  The re-run's merged endpoint
  must reproduce the recorded one (same HKL bytes, same refined model, same R1, wR2, GooF and counts), otherwise the
  command reports the mismatch and exits 1.  runs/combined is the published endpoint only when the chooser published
  the final combine.
* Conventional: the leak-free start's masks are rebuilt from its recorded split (salt, fraction, lockbox, shared
  candidate group) and must give back the run's own start.hkl byte for byte; start_final.res is then scored with
  SHELXL L.S. 0 on the free reflections of the baseline HKL, next to its work reflections.

Every L.S. 0 work score must reproduce its endpoint's own listing by grond.validation.endpoint_free's
WORK_STATISTICS_RULE (N_gt and N_all as printed; R1gt, R1all, wR2 and GooF each within one unit of the last printed digit).
Reporting only: nothing is chosen on these numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import try_read_json
from grond.paths import engine_script
from grond.refine.zero_cycle import MODEL_UNCHANGED_RULE_TEXT, SPECIAL_POSITION_ALLOWED_UNITS
from grond.reports.endpoint_free_report import (
    endpoint_free_block_from_rows,
    endpoint_free_markdown_lines,
    endpoint_free_row,
    endpoint_free_rows,
    endpoint_free_terminal_line,
)
from grond.validation.endpoint_free import WORK_STATISTICS_RULE, WORK_STATISTICS_RULE_TEXT

SCHEMA = "grond.endpoint_free_validation.v1"
REPRODUCED_FIELDS = ("R1_gt", "R1_all", "wR2_all", "GooF", "N_gt", "N_all")
ENDPOINT_MODES = ("published", "final")
PUBLISHED_SOURCE = "grond_summary.json publication_endpoint_chooser.best_physical_endpoint"
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """Relative path -> (size, mtime_ns) of every file under ``root``: evidence that a command left it unchanged."""

    out = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() or path.is_symlink():
            stat = path.lstat()
            out[str(path.relative_to(root))] = (int(stat.st_size), int(stat.st_mtime_ns))
    return out


def recorded_command(run_dir: Path, label: str = "combine_final") -> tuple[list[str] | None, int | None]:
    """The last recorded command with ``label`` in logs/commands.jsonl and its return code."""

    path = Path(run_dir) / "logs" / "commands.jsonl"
    command, returncode = None, None
    if not path.is_file():
        return None, None
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("label") != label:
            continue
        if "command" in entry:
            command, returncode = list(entry["command"]), None
        elif "returncode" in entry:
            returncode = entry["returncode"]
    return command, returncode


def _option_value(command: list[str], option: str) -> str | None:
    """Return the value after ``option`` in a command line, or None when the option is absent."""

    return command[command.index(option) + 1] if option in command and command.index(option) + 1 < len(command) else None


def _without_option(command: list[str], option: str, *, takes_value: bool) -> list[str]:
    """Return a command line without ``option`` (and its value when ``takes_value``)."""

    out, skip = [], False
    for token in command:
        if skip:
            skip = False
            continue
        if token == option:
            skip = takes_value
            continue
        out.append(token)
    return out


def free_validation_command(
    recorded: list[str], out_dir: Path, *, python: str, shelxl: str = "", unmerged: str = "recorded"
) -> list[str]:
    """The recorded combine command with this checkout's engine, a new --out-dir and --endpoint-free-validation on."""

    if len(recorded) < 3 or not recorded[1].endswith("combine_native_nuisance_runs.py"):
        raise ValueError(f"not a combine engine command: {recorded[:2]}")
    command = [python, str(engine_script("combine_native_nuisance_runs.py")), *recorded[2:]]
    if "--out-dir" not in command:
        raise ValueError("the recorded combine command has no --out-dir")
    command[command.index("--out-dir") + 1] = str(out_dir)
    command = _without_option(command, "--endpoint-free-validation", takes_value=True)
    if shelxl:
        if "--shelxl" in command:
            command[command.index("--shelxl") + 1] = shelxl
        else:
            command += ["--shelxl", shelxl]
    if unmerged == "on" and "--unmerged-endpoint" not in command:
        command.append("--unmerged-endpoint")
    return [*command, "--endpoint-free-validation", "on"]


def _res_model_digest(path: Path) -> str:
    """SHA-256 of a .res without SHELXL's 'created by ... at <time>' title line (the run time, not the model)."""

    lines = [line for line in Path(path).read_text(errors="replace").splitlines() if "created by SHELXL" not in line]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def endpoint_reproduction(recorded: Mapping[str, object], rerun: Mapping[str, object]) -> dict[str, object]:
    """Same statistics, same HKL bytes and same refined model (ignoring SHELXL's time stamp)?"""

    record_shelxl = recorded.get("shelxl") if isinstance(recorded.get("shelxl"), Mapping) else {}
    rerun_shelxl = rerun.get("shelxl") if isinstance(rerun.get("shelxl"), Mapping) else {}
    checks: dict[str, object] = {}
    for key in REPRODUCED_FIELDS:
        checks[key] = {"recorded": record_shelxl.get(key), "rerun": rerun_shelxl.get(key),
                       "same": record_shelxl.get(key) == rerun_shelxl.get(key)}
    record_artifacts = recorded.get("artifacts") if isinstance(recorded.get("artifacts"), Mapping) else {}
    rerun_artifacts = rerun.get("artifacts") if isinstance(rerun.get("artifacts"), Mapping) else {}
    checks["hkl_sha256"] = {"recorded": record_artifacts.get("hkl_sha256"), "rerun": rerun_artifacts.get("hkl_sha256"),
                            "same": bool(record_artifacts.get("hkl_sha256"))
                            and record_artifacts.get("hkl_sha256") == rerun_artifacts.get("hkl_sha256")}
    digests = {}
    for side, artifacts in (("recorded", record_artifacts), ("rerun", rerun_artifacts)):
        res = Path(str(artifacts.get("res") or ""))
        digests[side] = _res_model_digest(res) if res.is_file() else None
    checks["res_model"] = {**digests, "same": digests["recorded"] is not None and digests["recorded"] == digests["rerun"]}
    checks["exact"] = all(bool(value["same"]) for value in checks.values() if isinstance(value, dict))
    return checks


def run_engine(command: list[str], out_dir: Path, label: str) -> dict[str, object] | None:
    """Run one engine command in ``out_dir`` (thread variables 1), logged under ``label``; a failure record or None."""

    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **{name: "1" for name in THREAD_VARIABLES}}
    with (logs / "commands.jsonl").open("a") as handle:
        handle.write(json.dumps({"label": label, "cwd": str(out_dir), "command": command}) + "\n")
    proc = subprocess.run(command, cwd=out_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    (logs / f"{label}.log").write_text(proc.stdout + f"\n[returncode] {proc.returncode}\n")
    if proc.returncode != 0:
        return {"status": "failed", "reason": f"combine engine returned {proc.returncode}", "command": command,
                "log": str(logs / f"{label}.log")}
    return None


def grond_part(run_dir: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    """--endpoint final: re-run the last recorded combine_final and require it to reproduce runs/combined."""

    recorded, returncode = recorded_command(run_dir)
    if recorded is None:
        return {"status": "not_available", "reason": "no combine_final command in logs/commands.jsonl (single-crystal run?)"}
    if returncode not in (0, None):
        return {"status": "not_available", "reason": f"the run's combine_final returned {returncode}"}
    recorded_out = Path(str(_option_value(recorded, "--out-dir") or ""))
    recorded_summary_path = recorded_out / "combined_summary.json"
    recorded_summary = try_read_json(recorded_summary_path)
    if not recorded_summary:
        return {"status": "not_available", "reason": f"no recorded {recorded_summary_path}"}
    rerun_dir = out_dir / "runs" / "combined"
    command = free_validation_command(
        recorded, rerun_dir, python=str(args.python or sys.executable), shelxl=str(args.shelxl or ""),
        unmerged=str(args.unmerged),
    )
    failure = run_engine(command, out_dir, "combine_final_free_validation")
    if failure:
        return failure
    rerun_summary = try_read_json(rerun_dir / "combined_summary.json")
    reproduction = {
        "merged": endpoint_reproduction(recorded_summary.get("combined_endpoint") or {}, rerun_summary.get("combined_endpoint") or {}),
    }
    recorded_unmerged = recorded_summary.get("combined_unmerged_endpoint") or {}
    if str(recorded_unmerged.get("status")) == "ok":
        reproduction["unmerged"] = endpoint_reproduction(recorded_unmerged, rerun_summary.get("combined_unmerged_endpoint") or {})
    elif str((rerun_summary.get("combined_unmerged_endpoint") or {}).get("status")) == "ok":
        reproduction["unmerged"] = {"exact": None, "note": "the run had no unmerged endpoint; added by --unmerged on"}
    exact = all(block.get("exact") is not False for block in reproduction.values())
    return {
        "status": "ok" if exact else "endpoint_not_reproduced",
        "recorded_command": recorded,
        "command": command,
        "recorded_summary": str(recorded_summary_path),
        "summary": str(rerun_dir / "combined_summary.json"),
        "reproduction": reproduction,
        "combined_summary": rerun_summary,
    }


PUBLISHED_FIELDS = (
    "id", "label", "source_kind", "res", "hkl", "lst", "res_sha256", "hkl_sha256", "lst_sha256", "endpoint_origin_summary",
    "endpoint_row_table", *REPRODUCED_FIELDS,
)
# The source kind of a single-crystal run's per-crystal corrected-HKL endpoint (native_row_regularize).
NATIVE_SOURCE_KIND = "native_corrected_hkl"


def published_endpoint(run_dir: Path) -> dict[str, object]:
    """The endpoint the run published: the chooser's best physical endpoint as grond_summary.json records it.

    ``chooser_file_res_sha256`` is the same endpoint's hash in endpoint_chooser/publication_endpoint_choice.json (None
    when that file is absent or records no best physical endpoint); the two records must name the same file, and a
    missing chooser record is not a confirmation.
    """

    summary = try_read_json(Path(run_dir) / "grond_summary.json")
    chooser = summary.get("publication_endpoint_chooser") if isinstance(summary.get("publication_endpoint_chooser"), Mapping) else {}
    endpoint = chooser.get("best_physical_endpoint") if isinstance(chooser.get("best_physical_endpoint"), Mapping) else {}
    if not endpoint or not str(endpoint.get("res") or "").strip():
        return {}
    record: dict[str, object] = {key: endpoint.get(key) for key in PUBLISHED_FIELDS}
    record["source"] = PUBLISHED_SOURCE
    record["run_is_combined"] = bool(summary.get("combined"))
    choice = try_read_json(Path(run_dir) / "endpoint_chooser" / "publication_endpoint_choice.json")
    other = choice.get("best_physical_endpoint") if isinstance(choice.get("best_physical_endpoint"), Mapping) else {}
    record["chooser_file_res_sha256"] = other.get("res_sha256") if other else None
    return record


def recorded_command_writing(run_dir: Path, combine_out: Path) -> tuple[str | None, list[str] | None, int | None]:
    """The last recorded combine command whose --out-dir is ``combine_out``: (label, command, return code).

    The return code is the one recorded after that command: a later command with the same label that wrote another
    directory ends the match, so its return code is never taken for this one.
    """

    path = Path(run_dir) / "logs" / "commands.jsonl"
    label, command, returncode = None, None, None
    current = False
    if not path.is_file():
        return None, None, None
    target = Path(combine_out).resolve()
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "command" in entry:
            recorded = [str(token) for token in entry["command"]]
            out = _option_value(recorded, "--out-dir")
            if len(recorded) > 1 and recorded[1].endswith("combine_native_nuisance_runs.py") and out and Path(out).resolve() == target:
                label, command, returncode, current = str(entry.get("label") or ""), recorded, None, True
            elif label is not None and str(entry.get("label") or "") == label:
                current = False
        elif current and entry.get("label") == label and "returncode" in entry:
            returncode = entry["returncode"]
    return label, command, returncode


def published_validation_command(
    recorded: list[str], out_dir: Path, *, python: str, shelxl: str, score_res: Path, kept_pass: int, sigma_model: str
) -> list[str]:
    """The recorded combine as a score-only run: no refinement, the merged endpoint only, --score-res the published .res."""

    command = free_validation_command(recorded, out_dir, python=python, shelxl=shelxl, unmerged="recorded")
    command = _without_option(command, "--unmerged-endpoint", takes_value=False)
    for option in ("--score-res", "--score-error-model-pass", "--score-sigma-model"):
        command = _without_option(command, option, takes_value=True)
    command += ["--score-res", str(score_res), "--score-error-model-pass", str(int(kept_pass))]
    if sigma_model:
        command += ["--score-sigma-model", str(sigma_model)]
    return command


def published_reproduction(published: Mapping[str, object], rerun_endpoint: Mapping[str, object]) -> dict[str, object]:
    """Work HKL bytes, model bytes, the L.S. 0 work statistics and the listing the published record came from."""

    score_only = rerun_endpoint.get("score_only") if isinstance(rerun_endpoint.get("score_only"), Mapping) else {}
    block = rerun_endpoint.get("free_validation") if isinstance(rerun_endpoint.get("free_validation"), Mapping) else {}
    statistics = block.get("work_reproduces_endpoint") if isinstance(block.get("work_reproduces_endpoint"), Mapping) else {}
    listing = score_only.get("listing_statistics") if isinstance(score_only.get("listing_statistics"), Mapping) else {}
    record_checks = {
        key: {"published_record": published.get(key), "listing": listing.get(key), "same": published.get(key) == listing.get(key)}
        for key in REPRODUCED_FIELDS
        if published.get(key) is not None
    }
    checks: dict[str, object] = {
        "work_hkl_bytes": {
            "published": published.get("hkl_sha256"),
            "published_file": score_only.get("scored_hkl_sha256"),
            "rebuilt": score_only.get("rebuilt_hkl_sha256"),
            "same": bool(score_only.get("rebuilt_hkl_sha256"))
            and score_only.get("rebuilt_hkl_sha256") == score_only.get("scored_hkl_sha256")
            and (not published.get("hkl_sha256") or published.get("hkl_sha256") == score_only.get("rebuilt_hkl_sha256")),
        },
        "model_bytes": {
            "published": published.get("res_sha256"),
            "scored": score_only.get("scored_res_sha256"),
            "unchanged_by_zero_cycle": block.get("model_unchanged_by_zero_cycle"),
            "same": bool(score_only.get("scored_res_sha256"))
            and score_only.get("scored_res_sha256") == published.get("res_sha256")
            and block.get("model_unchanged_by_zero_cycle") is True,
        },
        "work_statistics": {**dict(statistics), "same": statistics.get("all_same") is True},
        "listing_matches_published_record": {
            **record_checks,
            "same": bool(record_checks) and all(item["same"] for item in record_checks.values()),
        },
        "wght": {"applied_in_listing": score_only.get("wght_applied_in_listing"), "scored_with": score_only.get("wght"),
                 "same": score_only.get("wght_same") is True},
    }
    checks["exact"] = all(bool(value.get("same")) for value in checks.values() if isinstance(value, dict))
    return checks


def published_part(run_dir: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    """--endpoint published: score the published endpoint itself at L.S. 0 on its rebuilt HKL (no re-refinement)."""

    published = published_endpoint(run_dir)
    if not published:
        return {"status": "not_available", "reason": f"the run records no published endpoint ({PUBLISHED_SOURCE})"}
    res = Path(str(published["res"]))
    hkl = Path(str(published.get("hkl") or res.with_suffix(".hkl")))
    base = {"endpoint": published}
    if not published["run_is_combined"] and published.get("source_kind") != NATIVE_SOURCE_KIND:
        return {**base, "status": "not_available",
                "reason": f"single-crystal run: the published endpoint ({published.get('source_kind')}) is neither a "
                          f"combined nor a {NATIVE_SOURCE_KIND} endpoint"}
    if published.get("chooser_file_res_sha256") is None:
        return {**base, "status": "published_endpoint_ambiguous",
                "reason": "endpoint_chooser/publication_endpoint_choice.json is missing or records no best physical "
                          "endpoint, so grond_summary.json's published endpoint is not confirmed"}
    if published.get("chooser_file_res_sha256") != published.get("res_sha256"):
        return {**base, "status": "published_endpoint_ambiguous",
                "reason": "grond_summary.json and endpoint_chooser/publication_endpoint_choice.json name different endpoints"}
    if not res.is_file() or not hkl.is_file() or hkl.resolve() != res.with_suffix(".hkl").resolve():
        return {**base, "status": "published_endpoint_missing",
                "reason": f"the published endpoint's .res and .hkl are not beside each other at {res.parent}"}
    changed = [name for name, path, key in (("res", res, "res_sha256"), ("hkl", hkl, "hkl_sha256"))
               if published.get(key) and _sha256(path) != published.get(key)]
    if changed:
        return {**base, "status": "published_endpoint_changed", "reason": f"SHA-256 of the published {', '.join(changed)} differs from the record"}
    if not published["run_is_combined"]:
        return native_published_part(run_dir, out_dir, args, published)
    combine_out = res.parent.parent
    label, recorded, returncode = recorded_command_writing(run_dir, combine_out)
    if recorded is None:
        return {**base, "status": "published_combine_not_found", "reason": f"no recorded combine command wrote {combine_out}"}
    if returncode not in (0, None):
        return {**base, "status": "published_combine_not_found", "reason": f"the run's {label} returned {returncode}"}
    recorded_summary_path = combine_out / "combined_summary.json"
    recorded_summary = try_read_json(recorded_summary_path)
    recorded_endpoint = recorded_summary.get("combined_endpoint") or {}
    artifacts = recorded_endpoint.get("artifacts") if isinstance(recorded_endpoint.get("artifacts"), Mapping) else {}
    unmerged_artifacts = (recorded_summary.get("combined_unmerged_endpoint") or {}).get("artifacts") or {}
    if unmerged_artifacts and Path(str(unmerged_artifacts.get("res") or "")).resolve() == res.resolve():
        return {**base, "status": "not_supported_unmerged_published_endpoint",
                "reason": "the run published its unmerged batch endpoint; --endpoint published scores a merged endpoint "
                          "(--endpoint final validates the unmerged endpoint of combine_final)"}
    if not artifacts or Path(str(artifacts.get("res") or "")).resolve() != res.resolve() or (
        published.get("res_sha256") and artifacts.get("res_sha256") != published.get("res_sha256")
    ):
        return {**base, "status": "published_combine_not_found",
                "reason": f"{recorded_summary_path} does not record the published endpoint"}
    passes = [item for item in recorded_endpoint.get("error_model_passes") or [] if isinstance(item, Mapping)]
    kept = [item for item in passes if item.get("kept") is True]
    if passes and len(kept) != 1:
        return {**base, "status": "published_combine_not_found",
                "reason": f"{recorded_summary_path} records {len(kept)} kept error-model passes"}
    kept_pass = int(kept[0]["pass"]) if kept else 1
    sigma_model = str(kept[0].get("sigma_model") or "") if kept and kept_pass > 1 else ""
    try:
        relative = combine_out.resolve().relative_to(Path(run_dir).resolve())
    except ValueError:
        relative = Path("runs") / "published_combine"
    rerun_dir = out_dir / relative
    command = published_validation_command(
        recorded, rerun_dir, python=str(args.python or sys.executable), shelxl=str(args.shelxl or ""), score_res=res,
        kept_pass=kept_pass, sigma_model=sigma_model,
    )
    failure = run_engine(command, out_dir, "published_endpoint_free_validation")
    if failure:
        return {**base, **failure, "recorded_label": label}
    rerun_summary = try_read_json(rerun_dir / "combined_summary.json")
    reproduction = {"merged": published_reproduction(published, rerun_summary.get("combined_endpoint") or {})}
    return {
        **base,
        "status": "ok" if reproduction["merged"]["exact"] else "endpoint_not_reproduced",
        "recorded_label": label,
        "recorded_command": recorded,
        "kept_error_model_pass": kept_pass,
        "sigma_model": sigma_model or None,
        "command": command,
        "recorded_summary": str(recorded_summary_path),
        "summary": str(rerun_dir / "combined_summary.json"),
        "reproduction": reproduction,
        "combined_summary": rerun_summary,
    }


def native_published_part(run_dir: Path, out_dir: Path, args: argparse.Namespace, published: Mapping[str, object]) -> dict[str, object]:
    """--endpoint published of a single-crystal run: the per-crystal endpoint scored without re-refinement.

    grond.validation.native_endpoint_free replays the endpoint command's recipe from its recorded row table into
    ``out_dir/<the endpoint's relative path>``: the rebuilt work HKL must equal the published HKL byte for byte, and the
    published .res is scored at L.S. 0 on it plus the corrected free rows, with its applied WGHT.
    """

    from grond.validation.native_endpoint_free import ReplayRefused, replay_native_endpoint, write_summary

    base = {"endpoint": dict(published)}
    res = Path(str(published["res"]))
    try:
        relative = res.parent.resolve().relative_to(Path(run_dir).resolve())
    except ValueError:
        relative = Path("runs") / "published_endpoint"
    rerun_dir = out_dir / relative
    try:
        endpoint = replay_native_endpoint(published, rerun_dir, shelxl=str(args.shelxl or ""))
    except ReplayRefused as exc:
        return {**base, "status": exc.status, "reason": exc.reason}
    summary_path = write_summary(rerun_dir, endpoint)
    reproduction = {"single_crystal": published_reproduction(published, endpoint)}
    replay = endpoint.get("replay") if isinstance(endpoint.get("replay"), Mapping) else {}
    return {
        **base,
        "status": "ok" if reproduction["single_crystal"]["exact"] else "endpoint_not_reproduced",
        "recorded_label": "native_row_regularize.py export_corrected_endpoint",
        "recorded_command": replay.get("command"),
        "recorded_command_record": replay.get("command_record"),
        "kept_error_model_pass": replay.get("kept_error_model_pass"),
        "sigma_model": replay.get("sigma_model"),
        "summary": str(summary_path),
        "reproduction": reproduction,
        "single_crystal_endpoint": endpoint,
    }


def _shared_candidate_union(run_dir: Path, recorded: Mapping[str, object]) -> dict[str, object] | None:
    """The shared candidate group of the run's split contracts, checked against the leak-free start's hash."""

    from grond.bloch.splits import SHARED_CANDIDATE_POLICY, shared_candidate_group_sha256

    if str(recorded.get("policy") or "") != SHARED_CANDIDATE_POLICY:
        return None
    for contract_path in sorted((Path(run_dir) / "decisions").glob("*/transform_hand_split_contract.json")):
        candidate_set = try_read_json(contract_path).get("candidate_set") or {}
        shared = [str(value) for value in candidate_set.get("shared_group") or []]
        laue = [str(value) for value in candidate_set.get("laue_proper_rotations_hkl") or []]
        if shared and shared_candidate_group_sha256(shared, laue) == str(recorded.get("shared_group_sha256")):
            return {"policy": SHARED_CANDIDATE_POLICY, "shared_group": shared,
                    "shared_group_sha256": str(recorded["shared_group_sha256"]),
                    "laue_coset_count": int(recorded.get("laue_coset_count", 1))}
    raise ValueError("no split contract under decisions/ carries the leak-free start's shared candidate group")


def conventional_part(run_dir: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    """Score the leak-free start model on the free reflections of the baseline HKL (L.S. 0); return the record.

    The work/free masks are rebuilt from the recorded split and must reproduce the run's start.hkl byte for byte; files
    are copied to ``out_dir``, the run is only read.
    """

    from grond.bloch.model import parse_model
    from grond.bloch.splits import load_split_exclusion
    from grond.workflows.leak_free_start import (
        leak_free_split_masks,
        read_hklf4_rows,
        score_leak_free_start,
        work_hkl_text,
    )

    source = Path(run_dir) / "controls" / "leak_free_start"
    summary = try_read_json(source / "summary.json")
    if str(summary.get("status")) != "ok":
        return {"status": "not_available", "reason": f"leak-free start status {summary.get('status', 'not recorded')!r}"}
    split = try_read_json(Path(run_dir) / "decisions" / "split_summary.json")
    exclusion = None
    if str(split.get("excluded_orbits_json") or "").strip():
        exclusion = load_split_exclusion(split["excluded_orbits_json"])
        if exclusion["sha256"] != split.get("excluded_orbits_sha256"):
            raise ValueError("the lockbox file changed since the run (SHA-256 differs from decisions/split_summary.json)")
    rows = read_hklf4_rows(Path(str(summary["input_hkl"])))
    masks, excluded, _candidate = leak_free_split_masks(
        rows,
        parse_model(Path(str(summary["input_res"]))),
        free_fraction=float(summary["free_fraction"]),
        salt=str(summary["split_salt"]),
        excluded_orbits=exclusion["orbits"] if exclusion else None,
        union=_shared_candidate_union(run_dir, summary.get("split_candidate_set") or {}),
    )
    counts = {"n_work": int(masks["work"].sum()), "n_free_removed": int(masks["free"].sum()),
              "n_excluded_removed": int(excluded.sum())}
    rebuilt = hashlib.sha256(work_hkl_text(rows, masks).encode()).hexdigest()
    recorded = _sha256(source / "start.hkl")
    reproduction = {
        "start_hkl_sha256": {"recorded": recorded, "rebuilt": rebuilt, "same": recorded == rebuilt},
        **{key: {"recorded": summary.get(key), "rebuilt": value, "same": summary.get(key) == value} for key, value in counts.items()},
    }
    if not all(item["same"] for item in reproduction.values()):
        return {"status": "masks_not_reproduced", "reproduction": reproduction}
    target = out_dir / "controls" / "leak_free_start"
    target.mkdir(parents=True, exist_ok=True)
    # score_leak_free_start reads start.ins and start_final.res; point it at the run's files through copies in OUT
    for name in ("start.ins", "start_final.res", "start.lst"):
        (target / name).write_bytes((source / name).read_bytes())
    record = score_leak_free_start(target, rows, masks, shelxl=str(args.shelxl or summary_shelxl(run_dir)))
    record["source_model"] = str((source / "start_final.res").resolve())
    record["source_model_sha256"] = _sha256(source / "start_final.res")
    record["reproduction"] = reproduction
    (target / "free_validation.json").write_text(json.dumps(record, indent=2, allow_nan=True) + "\n")
    return {"status": record.get("status"), "leak_free_start": {**summary, "free_validation": record},
            "reproduction": reproduction}


def summary_shelxl(run_dir: Path) -> str:
    """Return the SHELXL executable the run's combine_final command used (``shelxl`` if not recorded)."""

    command, _returncode = recorded_command(run_dir)
    return str(_option_value(command or [], "--shelxl") or "shelxl")


def _yes(value: object) -> str:
    """Format True as ``yes``, None as ``n/a`` and anything else as ``NO``."""

    return "yes" if value is True else ("n/a" if value is None else "NO")


def _published_markdown_lines(grond: Mapping[str, object]) -> list[str]:
    """The published endpoint's reproduction: HKL and model bytes, L.S. 0 work statistics, listing record, WGHT."""

    reproduction = grond.get("reproduction") or {}
    kind = "single_crystal" if "single_crystal" in reproduction else "merged"
    block = reproduction.get(kind) or {}
    if not block:
        return []
    statistics = block.get("work_statistics") or {}
    details = []
    for key in WORK_STATISTICS_RULE:
        value = statistics.get(key)
        if isinstance(value, Mapping):
            units = int(value.get("difference_units_of_last_digit") or 0)
            details.append(f"{key} {value.get('recorded')} / {value.get('zero_cycle')}"
                           + (f" ({units} unit of the last digit, allowed {value.get('allowed_units_of_last_digit')})" if units else ""))
    wght = block.get("wght") or {}
    label = "GROND single-crystal" if kind == "single_crystal" else "GROND merged"
    lines = [
        f"- {label}, published endpoint: exact {'yes' if block.get('exact') else 'NO'}. Work HKL bytes (published / "
        f"rebuilt) same {_yes((block.get('work_hkl_bytes') or {}).get('same'))}; model bytes same and unchanged by L.S. 0 "
        f"{_yes((block.get('model_bytes') or {}).get('same'))}; L.S. 0 work statistics reproduce the endpoint's listing "
        f"{_yes((block.get('work_statistics') or {}).get('same'))} (listing / L.S. 0: {', '.join(details) or 'n/a'}); "
        f"listing equals the published record {_yes((block.get('listing_matches_published_record') or {}).get('same'))}; "
        f"scored with the applied WGHT `{wght.get('scored_with')}` {_yes(wght.get('same'))}.",
    ]
    if kind == "merged":
        lines.append("- GROND unmerged: not scored with `--endpoint published` (the published endpoint is the merged one; "
                     "`--endpoint final` validates the unmerged endpoint of combine_final).")
    return lines


def _markdown(record: Mapping[str, object]) -> str:
    """Render the validation record as ENDPOINT_FREE_VALIDATION.md text."""

    grond = record.get("grond") or {}
    endpoint = record.get("endpoint") or {}
    lines = [
        "# GROND Endpoint Free Validation",
        "",
        f"Run: `{record['run_dir']}` (read only; unchanged: {'yes' if record.get('run_directory_unchanged') else 'NO'}).",
        "",
    ]
    if endpoint.get("mode") == "published":
        published = grond.get("endpoint") or {}
        lines += [
            f"Endpoint scored: the published endpoint (`--endpoint published`; {PUBLISHED_SOURCE}): "
            f"`{published.get('res', 'not recorded')}` (SHA-256 `{str(published.get('res_sha256') or '')[:16]}`), "
            f"written by the recorded `{grond.get('recorded_label', 'n/a')}`. Its rows were rebuilt with that command's "
            f"recipe (kept error-model pass {grond.get('kept_error_model_pass', 'n/a')}); the rebuilt work HKL must equal "
            "the published HKL byte for byte, and the published .res itself is scored at L.S. 0 with the WGHT its last "
            "refinement applied. Nothing is re-refined.",
            "",
            f"Work-statistics reproduction rule: {WORK_STATISTICS_RULE_TEXT}.",
            "",
        ]
    lines += [
        *endpoint_free_markdown_lines(record.get("block")),
        "### Reproduction of the run's own endpoints",
        "",
    ]
    if endpoint.get("mode") == "published":
        lines += _published_markdown_lines(grond)
    else:
        for label, block in (grond.get("reproduction") or {}).items():
            if block.get("exact") is None and block.get("note"):
                lines.append(f"- GROND {label}: {block['note']}.")
                continue
            details = ", ".join(
                f"{key} {value.get('recorded')} / {value.get('rerun')}" for key, value in block.items()
                if isinstance(value, dict) and key in REPRODUCED_FIELDS
            )
            lines.append(f"- GROND {label}: exact {'yes' if block.get('exact') else 'NO'} (recorded / re-run: {details}; "
                         f"HKL bytes same {block.get('hkl_sha256', {}).get('same')}, refined model same "
                         f"{block.get('res_model', {}).get('same')}).")
    if grond.get("status") not in ("ok", None) and not grond.get("reproduction"):
        lines.append(f"- GROND: {grond.get('status')} ({grond.get('reason', '')}).")
    conventional = record.get("conventional") or {}
    reproduction = conventional.get("reproduction") or {}
    if reproduction:
        lines.append(f"- Conventional: leak-free masks rebuilt, start.hkl identical: "
                     f"{'yes' if (reproduction.get('start_hkl_sha256') or {}).get('same') else 'NO'}.")
    elif conventional:
        lines.append(f"- Conventional: {conventional.get('status')} ({conventional.get('reason', '')}).")
    return "\n".join([*lines, ""])


def cmd_validate_endpoint(args: argparse.Namespace) -> int:
    """Run ``grond validate-endpoint``: score the GROND and conventional endpoints on held-out rows.

    Writes ENDPOINT_FREE_VALIDATION.json and .md to a new ``--out-dir``. Returns 1 unless the run directory is
    unchanged, each part is ok or not available, and every scored row block is ok.
    """

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    mode = str(getattr(args, "endpoint", "published") or "published")
    if mode not in ENDPOINT_MODES:
        raise SystemExit(f"validate-endpoint: --endpoint must be one of {', '.join(ENDPOINT_MODES)}")
    if mode == "published" and str(getattr(args, "unmerged", "recorded")) != "recorded":
        raise SystemExit("validate-endpoint: --unmerged on applies to --endpoint final (the published endpoint is merged)")
    if not run_dir.is_dir():
        raise SystemExit(f"validate-endpoint: no run directory {run_dir}")
    if out_dir == run_dir or run_dir in out_dir.parents:
        raise SystemExit("validate-endpoint: --out-dir must be outside the run directory (the run is never modified)")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"validate-endpoint: {out_dir} exists and is not empty; give a new directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    before = tree_snapshot(run_dir)
    record: dict[str, object] = {"schema": SCHEMA, "run_dir": str(run_dir), "out_dir": str(out_dir),
                                 "role": "reporting only: nothing is chosen on these numbers"}
    record["endpoint"] = (
        {"mode": "published", "source": PUBLISHED_SOURCE,
         "definition": "the published .res scored at L.S. 0 on its combine's rebuilt HKL (single-crystal run: its native "
                       "endpoint command's HKL rebuilt from the recorded row table; work part byte-identical to the "
                       "published HKL) plus the corrected free rows, with its applied WGHT; no re-refinement"}
        if mode == "published"
        else {"mode": "final", "definition": "the last recorded combine_final re-run (runs/combined), which must "
                                             "reproduce the recorded endpoint exactly"}
    )
    record["reproduction_rule"] = {
        "bytes": "the scored endpoint's work HKL and model are the run's own bytes (published: the rebuilt HKL equals "
                 "the published HKL and the published .res is scored unchanged; final: the re-run's HKL and refined "
                 "model equal the recorded ones)",
        "work_statistics": WORK_STATISTICS_RULE_TEXT,
        "work_statistics_rule": WORK_STATISTICS_RULE,
        "model_unchanged": MODEL_UNCHANGED_RULE_TEXT,
        "model_unchanged_special_position_allowed_units_of_last_digit": SPECIAL_POSITION_ALLOWED_UNITS,
    }
    try:
        record["grond"] = published_part(run_dir, out_dir, args) if mode == "published" else grond_part(run_dir, out_dir, args)
    except Exception as exc:
        record["grond"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    try:
        record["conventional"] = conventional_part(run_dir, out_dir, args)
    except Exception as exc:
        record["conventional"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    grond = record["grond"] if isinstance(record["grond"], Mapping) else {}
    conventional = record["conventional"] if isinstance(record["conventional"], Mapping) else {}
    rows = endpoint_free_rows(
        grond.get("combined_summary"), conventional.get("leak_free_start"),
        combined_source=str(grond.get("summary") or ""),
        leak_free_source=str(out_dir / "controls" / "leak_free_start" / "free_validation.json"),
    )
    single = grond.get("single_crystal_endpoint") if isinstance(grond.get("single_crystal_endpoint"), Mapping) else {}
    if isinstance(single.get("free_validation"), Mapping):
        rows.insert(0, endpoint_free_row("grond_single_crystal", single["free_validation"], source=str(grond.get("summary") or "")))
    record["block"] = endpoint_free_block_from_rows(rows)
    record["run_directory_unchanged"] = tree_snapshot(run_dir) == before
    # the engine summaries are files of their own (grond.summary)
    stored = {**record, "grond": {key: value for key, value in grond.items()
                                  if key not in {"combined_summary", "single_crystal_endpoint"}}}
    (out_dir / "ENDPOINT_FREE_VALIDATION.json").write_text(json.dumps(stored, indent=2, allow_nan=True) + "\n")
    (out_dir / "ENDPOINT_FREE_VALIDATION.md").write_text(_markdown(record))
    line = endpoint_free_terminal_line(record["block"]).strip()
    print(f"{line}\n  Details: {out_dir / 'ENDPOINT_FREE_VALIDATION.md'}", flush=True)
    ok = (
        record["run_directory_unchanged"]
        and grond.get("status") in ("ok", "not_available")
        and conventional.get("status") in ("ok", "not_available")
        and all(row.get("status") == "ok" for row in rows)
    )
    return 0 if ok else 1
