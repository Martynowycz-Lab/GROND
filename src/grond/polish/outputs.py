"""SHELXL-compatible outputs of the dynamical polish.

* ``polished_model.res``: the source SHELXL model with the polished coordinates and displacement parameters.
  Only plain numeric tokens of refined and riding atoms change; fixed or free-variable encoded tokens, SOF,
  AFIX/HFIX, restraints, SFAC, WGHT and every other instruction are kept.
* ``tables/<crystal>/native_dyn_calc.csv``: the run's row table with the polished Bloch rows and scales, so the
  unchanged GROND combine writes the matching corrected HKL (kappa*|F|^2 * Iobs/Icalc, work rows
  only, pathological correction ratios excluded).
* ``shelxl/refined``: a converged SHELXL re-refinement of the polished model against that HKL (.res/.hkl/.fcf/.lst).
* ``shelxl/as_polished``: SHELXL with ``L.S. 0`` on the polished model itself (Fc, maps and statistics of the
  dynamical optimum, no least-squares shift).
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np

from grond.bloch.model import SHELX_NON_ATOM_RECORDS, is_shelx_q_peak
from grond.paths import engine_script
from grond.polish.structure import U6_KEYS, _encoded

SUMMARY_NAME = "final_polish_summary.json"
# SHELXL statuses of a finished re-refinement (adp_pathology: finished, NPD atoms remain; flagged in the summary).
COMPLETED_SHELXL = {"ok", "adp_pathology"}


def _fmt(value: float) -> str:
    """Format a coordinate or displacement value with six decimals for the polished .res."""

    return f"{float(value):.6f}"


def write_polished_res(src: Path, dst: Path, model: dict, moved: set[str], rem: list[str]) -> None:
    """Rewrite the numeric tokens of moved atoms in place; every other byte of the instruction model is kept."""

    by_name = {str(a["name"]): a for a in model["atoms"]}
    lines = Path(src).read_text(errors="replace").splitlines()
    out: list[str] = []
    i = 0
    wrote_rem = False
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        tokens = stripped.split()
        if not wrote_rem and tokens and tokens[0].upper() == "TITL":
            out.append(line)
            # SHELXL refuses instruction lines longer than 80 characters.
            for text in rem:
                out.extend(f"REM {part}" for part in textwrap.wrap(text, 74))
            wrote_rem = True
            i += 1
            continue
        is_atom = (
            len(tokens) >= 6
            and tokens[0] in by_name
            and tokens[1].lstrip("-").isdigit()
            and tokens[0].upper() not in SHELX_NON_ATOM_RECORDS
            and not is_shelx_q_peak(tokens[0])
        )
        if not is_atom or tokens[0] not in moved:
            out.append(line)
            i += 1
            continue
        # Gather the logical record (continuation lines end with "=").
        record = [line]
        while record[-1].rstrip().endswith("=") and i + 1 < len(lines):
            i += 1
            record.append(lines[i])
        values = " ".join(part.rstrip().rstrip("=") for part in record).split()
        atom = by_name[tokens[0]]
        new = list(values)
        for position, key in ((2, "x"), (3, "y"), (4, "z")):
            if not _encoded(float(values[position])):
                new[position] = _fmt(atom[key])
        adp = values[6:]
        if atom.get("adp_type") == "anis" and len(adp) >= 6:
            for offset, key in enumerate(U6_KEYS):
                if not _encoded(float(adp[offset])):
                    new[6 + offset] = f"{float(atom[key]):.5f}"
        elif (
            adp and not _encoded(float(adp[0])) and float(adp[0]) > 0.0
        ):
            new[6] = f"{float(atom['uiso']):.5f}"
        prefix = line[: len(line) - len(line.lstrip())]
        if len(new) > 8:
            out.append(prefix + " ".join(new[:8]) + " =")
            out.append("    " + " ".join(new[8:]))
        else:
            out.append(prefix + " ".join(new))
        i += 1
    if not any(line.strip().upper().startswith("LIST") for line in out):
        # Olex2/SHELXLE maps need the .fcf: request it before FVAR (instructions precede the atom list).
        for j, line in enumerate(out):
            if line.strip().upper().split()[:1] in (["FVAR"], ["HKLF"]):
                out.insert(j, "LIST 6")
                break
    Path(dst).write_text("\n".join(out) + "\n")


def write_polished_tables(problem, solution: dict, out_dir: Path) -> dict[str, Path]:
    """Save polished rows with their own model, frozen-plan state and recomputed summary."""

    from grond.polish.artifacts import save_state, seal_rows
    from grond.regularize.multidataset import native_metrics_by_split, selected_native_wght
    from grond.regularize.scales import fit_group_scales
    from grond.validation import heldout as H

    final_ev = solution["fit"]["evaluation"]
    final = solution["fit"]["profile"]
    runs: dict[str, Path] = {}
    for crystal in problem.crystals:
        name = crystal.name
        block = final_ev[name]
        part = final["parts"][name]
        intensity = block["intensity"]
        dyn_raw = intensity * crystal.row_factor
        born = block["born"]
        fit = problem.scales[name].fit
        if crystal.scale["kind"] == "fixed_design":
            from grond.polish.refine import ScaleBlock

            born_block = ScaleBlock(crystal)
            born_block.freeze_penalty(born)
            coefficients, _active = born_block.solve(born)
            born_scaled = born_block.scales(coefficients, born) * born
        elif crystal.scale["kind"] == "pspline":
            born_scaled = H.pspline_fit_predict(
                crystal.y,
                crystal.sigma_source,
                born,
                crystal.angle,
                crystal.group,
                fit,
                {"n_basis": crystal.scale["born_n_basis"], "lambda": crystal.scale["born_lambda"], "objective": crystal.scale.get("objective", "row_mr")},
                bool(crystal.scale.get("clamp", True)),
            )["prediction"]
        else:
            # The engine's matched Born control: its own per-frame fit with the stage's SHELX weights.
            born_scaled, _scales = fit_group_scales(crystal.y, crystal.sigma_eff, born, crystal.group, fit,
                                                    crystal.scale["wght_a"], crystal.scale["wght_b"])
        pred = part["pred"]
        target = out_dir / name
        target.mkdir(parents=True, exist_ok=True)
        manifest, plan_fields = save_state(problem, solution, crystal, target)
        fieldnames = list(dict.fromkeys([*crystal.fieldnames, "born_scaled", *plan_fields]))
        output_rows = []
        with (target / "native_dyn_calc.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for j, row in enumerate(crystal.table_rows):
                out = dict(row)
                sigma = float(crystal.sigma_eff[j])
                out["dyn_raw_unmodulated"] = f"{intensity[j]:.17g}"
                out["dyn_raw"] = f"{dyn_raw[j]:.17g}"
                out["born_raw"] = f"{born[j]:.17g}"
                out["q_dyn_over_born"] = f"{dyn_raw[j] / max(born[j], 1e-300):.17g}"
                out["dyn_scaled"] = f"{pred[j]:.17g}"
                out["born_scaled"] = f"{born_scaled[j]:.17g}"
                for key, values in plan_fields.items():
                    out[key] = str(values[j])
                # The engine's sign and sigma floor (regularize.row_output): (prediction - observation) / sigma_eff.
                if "dyn_residual_over_sigma" in out:
                    out["dyn_residual_over_sigma"] = f"{(pred[j] - crystal.y[j]) / max(sigma, 1e-12):.17g}"
                if "born_residual_over_sigma" in out:
                    out["born_residual_over_sigma"] = f"{(born_scaled[j] - crystal.y[j]) / max(sigma, 1e-12):.17g}"
                writer.writerow(out)
                output_rows.append(out)
        seal_rows(target)
        wght_a, wght_b = selected_native_wght(crystal.summary)
        # Preserve the full old summary only in frozen/source_summary.json. Trial scores, selected thickness,
        # model identities and endpoint statistics in it do not describe the polished predictions.
        nuisance = dict(crystal.summary["best_native"]["nuisance"])
        nuisance.update(thickness_A=manifest["config"]["thickness_A"],
                        thickness_grad_frac=manifest["config"]["thickness_grad_frac"])
        sizes = plan_fields["basis_size"]
        summary = {
            "schema": "grond.polish.rows.v1", "prediction_kind": "frozen_dynamical_polish",
            "frozen_state": "frozen/manifest.json", "source_summary": "frozen/source_summary.json",
            "inputs": {"model_json": "frozen/model.json", "hkl": "frozen/input.hkl"},
            "config": manifest["config"],
            "row_selection": crystal.summary.get("row_selection", {}),
            "shelx_weight": {"selected_native_a": wght_a, "selected_native_b": wght_b},
            "best_native": {
                "nuisance": nuisance, "native_weight": {"a": wght_a, "b": wght_b},
                "scale_model": {**crystal.scale, "edf": float(part["edf"]),
                                "coefficient_count": int(len(part["coef"])),
                                "penalty_multiplier": problem.scales[name].kappa,
                                "weights": "frozen at the starting prediction"},
                "scale_count": int(math.ceil(part["edf"])), "scale_fit_split": "work",
                "metrics": native_metrics_by_split(output_rows, wght_a, wght_b),
                "metrics_contract": "current exported rows; sigma_eff; observed threshold I > 2 sigma_eff",
                "basis_size_stats": {"min": int(np.min(sizes)), "median": float(np.median(sizes)),
                                     "max": int(np.max(sizes))},
            },
            "note": "Replay with grond.polish.artifacts.replay_rows; the rounded RES and a rebuilt stock plan are not the calculation saved here.",
        }
        (target / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
        runs[name] = target
    return runs


def combine_command(python: str, runs: dict[str, Path], res: Path, out_dir: Path, *, shelxl: str, iterations: int, ls_cycles: int,
                    contract: dict) -> list[str]:
    """Build the combine command that writes the polished rows' corrected HKL and refines the polished model on it.

    The handoff is fixed: work rows only, the absolute kappa*|F|^2 scale, pathological
    correction ratios excluded, no interdataset scale, WGHT promote-gold, SHELXL to |shift/esd| < 0.01 and ISOR-only
    NPD recovery.  ``contract`` (run.endpoint_contract) supplies the run's own merge key, sigma calibrations and
    resolution cutoff.  It does not replay the current AUTO endpoint: AUTO's converged WGHT (--endpoint-wght-mode
    converge and its call cap), error-model refits, ZERR cell e.s.d.s, the Olex2/SHELXLE input cleanup, the unmerged
    endpoint, ACTA and the endpoint-free validation are not passed.
    """

    cmd = [
        python,
        str(engine_script("combine_native_nuisance_runs.py")),
        "--out-dir", str(out_dir),
        "--res", str(res),
        "--endpoint-lambda", "1.0",
        "--endpoint-source", "native",
        "--interdataset-scale", "none",
        "--endpoint-merge-key", str(contract["merge_key"]),
        "--dataset-sigma-policy", str(contract["dataset_sigma_policy"]),
        "--dataset-sigma-holdout-fraction", f"{float(contract['dataset_sigma_holdout_fraction']):g}",
        "--dataset-sigma-min-scale", f"{float(contract['dataset_sigma_min_scale']):g}",
        "--dataset-sigma-max-scale", f"{float(contract['dataset_sigma_max_scale']):g}",
        "--merge-sigma-policy", str(contract["merge_sigma_policy"]),
        "--post-bloch-sigma-policy", str(contract["post_bloch_sigma_policy"]),
        "--post-bloch-sigma-holdout-fraction", f"{float(contract['post_bloch_sigma_holdout_fraction']):g}",
        "--post-bloch-sigma-min-scale", f"{float(contract['post_bloch_sigma_min_scale']):g}",
        "--post-bloch-sigma-max-scale", f"{float(contract['post_bloch_sigma_max_scale']):g}",
        "--shelxl", str(shelxl),
        "--shelxl-iterations", str(int(iterations)),
        "--endpoint-ls-cycles", str(int(ls_cycles)),
        "--endpoint-wght-mode", "promote-gold",
        "--shelxl-convergence-shift-esd", "0.01",
        "--shelxl-npd-recovery", "isor",
        "--endpoint-q-policy", "exclude",
        "--rescale", "absolute",
        "--no-endpoint-include-free-rows",
    ]
    if float(contract["min_d_spacing"]) > 0.0:
        cmd += ["--min-d-spacing", f"{float(contract['min_d_spacing']):g}"]
    if bool(contract["dataset_sigma_holdout_floor"]):
        cmd.append("--dataset-sigma-holdout-floor")
    for name, run in runs.items():
        cmd += ["--run", f"{name}={run}"]
    return cmd


def _active_wght(res_path: Path) -> tuple[float, float] | None:
    """Return (a, b) of the first WGHT instruction of a .res file (b = 0 when absent), or None without one."""

    from grond.refine.shelx_input import parse_shelx_wght_terms

    for line in Path(res_path).read_text(errors="replace").splitlines():
        if line.strip().upper().startswith("WGHT"):
            terms = parse_shelx_wght_terms(line)
            if terms:
                return float(terms[0]), float(terms[1]) if len(terms) > 1 else 0.0
    return None


def run_shelxl_outputs(polished_res: Path, runs: dict[str, Path], out_dir: Path, *, python: str, shelxl: str,
                       iterations: int, ls_cycles: int, contract: dict, log=print) -> dict:
    """Converged SHELXL re-refinement against the polished corrected HKL, and the L.S. 0 as-polished model."""

    from grond.refine.shelx_input import write_endpoint_ins
    from grond.refine.shelxl_diagnostics import parse_lst

    out_dir.mkdir(parents=True, exist_ok=True)
    refined = out_dir / "refined"
    cmd = combine_command(python, runs, polished_res, refined, shelxl=shelxl, iterations=iterations,
                          ls_cycles=ls_cycles, contract=contract)
    log_path = out_dir / "combine_polished.log"
    with log_path.open("w") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"SHELXL re-refinement of the polished model failed (returncode {proc.returncode}); see {log_path}")
    combined = json.loads((refined / "combined_summary.json").read_text())
    endpoint = combined.get("combined_endpoint") or {}
    artifacts = endpoint.get("artifacts") or {}
    stats = endpoint.get("shelxl") or {}
    ep_dir = Path(str(artifacts.get("directory") or ""))
    # adp_pathology is a finished refinement whose model has NPD atoms: kept, flagged and reported (as AUTO does).
    if str(stats.get("status", "")) not in COMPLETED_SHELXL or not (ep_dir / "multi_dyn_corrected.res").is_file():
        raise RuntimeError(f"SHELXL re-refinement of the polished model did not finish: status {stats.get('status')!r}; see {log_path}")
    if bool((endpoint.get("free_rows") or {}).get("free_rows_in_hkl", True)):
        raise RuntimeError(f"the corrected HKL of the polished model does not record that it excludes free rows; see {log_path}")
    kappa = float((endpoint.get("intensity_scale") or {}).get("kappa_effective") or 0.0)
    # As-polished: the polished model itself against the same HKL, no least-squares shift.
    as_dir = out_dir / "as_polished"
    as_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ep_dir / "multi_dyn_corrected.hkl", as_dir / "polished_model.hkl")
    refined_res = ep_dir / "multi_dyn_corrected.res"
    wght = _active_wght(refined_res) or (0.1, 0.0)
    write_endpoint_ins(
        polished_res,
        as_dir / "polished_model.ins",
        hkl_name="polished_model.hkl",
        keep_exti=False,
        wght=wght,
        ls_cycles=1,
        keep_damp=False,
        osf=math.sqrt(kappa) if kappa > 0.0 else None,
    )
    ins = (as_dir / "polished_model.ins").read_text()
    ins = re.sub(r"^(\s*)(L\.S\.|CGLS)\s+\d+", r"\1L.S. 0", ins, count=1, flags=re.MULTILINE | re.IGNORECASE)
    (as_dir / "polished_model.ins").write_text(ins)
    with (as_dir / "shelxl_stdout.log").open("w") as handle:
        run = subprocess.run([str(shelxl), "polished_model"], cwd=str(as_dir), stdout=handle, stderr=subprocess.STDOUT, text=True)
    as_stats = parse_lst(as_dir / "polished_model.lst")
    if run.returncode != 0 or as_stats.get("status") != "ok" or not (as_dir / "polished_model.res").is_file():
        raise RuntimeError(f"SHELXL L.S. 0 on the polished model failed: {as_stats.get('status')}; see {as_dir / 'shelxl_stdout.log'}")
    return {
        "refined": {
            "directory": str(ep_dir),
            "res": str(refined_res),
            "hkl": str(ep_dir / "multi_dyn_corrected.hkl"),
            "fcf": str(ep_dir / "multi_dyn_corrected.fcf") if (ep_dir / "multi_dyn_corrected.fcf").is_file() else "",
            "lst": str(ep_dir / "multi_dyn_corrected.lst"),
            "shelxl": {key: stats.get(key) for key in ("status", "R1_gt", "R1_all", "wR2_all", "GooF", "N_gt", "N_all", "npd_count", "npd_atoms", "max_shift_esd", "iterations_completed", "converged", "isor_atoms") if key in stats},
            "pathology": str(stats.get("status")) != "ok",
            "combined_summary": str(refined / "combined_summary.json"),
            "free_rows_in_hkl": bool((endpoint.get("free_rows") or {}).get("free_rows_in_hkl", True)),
            "n_excluded_q_rows": endpoint.get("n_excluded_q_rows"),
            "kappa_effective": kappa,
        },
        "as_polished": {
            "directory": str(as_dir),
            "res": str(as_dir / "polished_model.res"),
            "hkl": str(as_dir / "polished_model.hkl"),
            "fcf": str(as_dir / "polished_model.fcf") if (as_dir / "polished_model.fcf").is_file() else "",
            "lst": str(as_dir / "polished_model.lst"),
            "shelxl": {key: as_stats.get(key) for key in ("status", "R1_gt", "R1_all", "wR2_all", "GooF", "N_gt", "N_all", "npd_count", "npd_atoms") if key in as_stats},
            "wght": list(wght),
            "osf": math.sqrt(kappa) if kappa > 0.0 else None,
        },
        "log": str(log_path),
    }


def _pct(value) -> str:
    """Format a percentage with two decimals, or ``n/a`` when it is not numeric."""

    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _ci(block: dict, key: str) -> str:
    """Format ``block[key]`` as value [95% CI], or a paired delta as signed delta [CI]; ``n/a`` when missing."""

    entry = (block or {}).get(key) or {}
    if "value" in entry:
        lo, hi = entry.get("ci95", [float("nan"), float("nan")])
        return f"{_pct(entry['value'])} [{_pct(lo)}, {_pct(hi)}]"
    if "delta" in entry:
        lo, hi = entry.get("ci95", [float("nan"), float("nan")])
        return f"{float(entry['delta']):+.2f} [{float(lo):+.2f}, {float(hi):+.2f}]"
    return "n/a"


def write_report(summary: dict, path: Path) -> None:
    """Write the Markdown report of a finished polish: held-out and work R1, model, thickness, SHELXL and restraints."""

    lines = [
        "# GROND dynamical final polish (DiffBloch-style)",
        "",
        f"Status: `{summary.get('status')}`; optimizer: `{(summary.get('optimizer') or {}).get('status')}` after "
        f"{(summary.get('optimizer') or {}).get('iterations')} iteration(s), {summary.get('evaluations')} Bloch evaluations.",
        "",
        "Free rows were predicted and scored only; no free row entered the objective, the frame scales, the "
        "thickness, the stopping rule or the SHELXL HKL.",
        "",
        "## Held-out rows (free), amplitude R1 %",
        "",
        "| model | R_all [95% CI] | R_obs [95% CI] |",
        "|---|---|---|",
    ]
    free = (summary.get("heldout") or {}).get("free") or {}
    for label, key in (("start (SHELXL publication model)", "start"), ("polished", "polished")):
        lines.append(f"| {label} | {_ci(free.get(key), 'r_all')} | {_ci(free.get(key), 'r_obs')} |")
    lines.append(f"| polished - start (pp) | {_ci(free.get('polished_minus_start'), 'r_all')} | {_ci(free.get('polished_minus_start'), 'r_obs')} |")
    work = (summary.get("heldout") or {}).get("work") or {}
    lines += [
        "",
        "## Work rows (fitted), amplitude R1 %",
        "",
        f"start R_obs {_ci(work.get('start'), 'r_obs')}; polished R_obs {_ci(work.get('polished'), 'r_obs')}.",
        "",
        "## Model",
        "",
    ]
    model = summary.get("model") or {}
    lines.append(
        f"Refined parameters: {model.get('n_structure_parameters')} structural, "
        f"{model.get('n_thickness_parameters')} thickness; max atom shift {_pct(model.get('max_shift_A'))} A "
        f"(rms {_pct(model.get('rms_shift_A'))} A). NPD atoms at start: {model.get('npd_atoms_start') or 'none'}; "
        f"after polish: {model.get('npd_atoms') or 'none'} (a step may not make a positive-definite atom NPD)."
    )
    for name, entry in (summary.get("thickness") or {}).items():
        lines.append(f"- {name}: thickness {entry['start_t0_A']:.1f} -> {entry['final_t0_A']:.1f} A; gradient {entry['start_grad']:.3f} -> {entry['final_grad']:.3f}")
    shelxl = summary.get("shelxl") or {}
    if shelxl:
        lines += ["", "## SHELXL (work rows only in the HKL)", ""]
        for label, key in (("as polished (L.S. 0)", "as_polished"), ("converged re-refinement", "refined")):
            block = shelxl.get(key) or {}
            stats = block.get("shelxl") or {}
            lines.append(
                f"- {label}: status {stats.get('status')}, R1gt {stats.get('R1_gt')}, R1all {stats.get('R1_all')}, "
                f"wR2 {stats.get('wR2_all')}, GooF {stats.get('GooF')}, NPD {stats.get('npd_count', 0)}; `{block.get('res')}`"
            )
    restraints = (summary.get("model") or {}).get("restraints_not_applied") or []
    if restraints:
        lines += ["", "Restraints present in the RES are not terms of the dynamical objective (they stay in the SHELXL input): "
                  + "; ".join(restraints)]
    Path(path).write_text("\n".join(lines) + "\n")
