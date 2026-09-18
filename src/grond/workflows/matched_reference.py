# ruff: noqa: E741
"""Load the exact observations and fitted values used by a full-H endpoint."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from grond.bloch.config import Config
from grond.bloch.model import parse_model
from grond.legacy.engine.bloch_engine import XDSData
from grond.legacy.engine.combine_native_nuisance_runs import read_hkl, read_rows as read_native_rows
from grond.macromolecule import sha256_file
from grond.regularize.multidataset import (
    endpoint_key_function,
    endpoint_rows_and_corrections_from_native,
    merge_hkl_rows,
)
from grond.regularize.prediction_cache import external_geometry_fingerprints


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")
    temporary.replace(path)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ReferencePaths:
    def __init__(self, replacements):
        self.replacements = []
        for item in replacements:
            old, separator, new = item.partition("=")
            if not separator or not old or not new:
                raise ValueError("--path-map must be OLD_PREFIX=NEW_PREFIX")
            self.replacements.append((old.rstrip("/"), new.rstrip("/")))

    def __call__(self, value) -> Path:
        text = str(value)
        for old, new in sorted(self.replacements, key=lambda pair: -len(pair[0])):
            if text == old or text.startswith(old + "/"):
                text = new + text[len(old):]
                break
        path = Path(text).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Reference file missing: {path}. Use --path-map if the run moved.")
        return path


def frozen_config(summary: dict, paths: ReferencePaths) -> Config:
    config, nuisance = summary["config"], summary["best_native"]["nuisance"]
    missing = {"n_basis", "n_points", "thickness_points", "topup_d_min", "max_support_sg"} - config.keys()
    missing |= {"thickness_A", "thickness_model", "frame_offset", "absorption_mode", "sg_jitter_mode"} - nuisance.keys()
    if missing:
        raise ValueError(f"Incomplete reference physics: {sorted(missing)}")
    values = {field.name: nuisance.get(field.name, config.get(field.name, field.default)) for field in fields(Config)}
    values["topup_d_min"] = min(float(config["topup_d_min"]), float(config["min_d_spacing"]))
    geometry = nuisance.get("geometry_path_file") or config.get("geometry_path")
    values["geometry_path_file"] = str(paths(geometry)) if geometry else ""
    # Rebuild each basis from the fixed reference pool; no cross-model row cache.
    values["row_plan_cache"] = "off"
    values["paired_inverse_hand"] = False
    return Config(**values)


def load_reference(root: Path, paths: ReferencePaths) -> dict:
    result = json.loads((root / "grond_summary.json").read_text())
    if not result.get("combined"):
        raise ValueError("never-h currently requires a completed multi-crystal combined endpoint.")
    selected = (result.get("publication_endpoint_chooser") or {}).get("best_physical_endpoint") or {}
    combined_path = (paths(selected["endpoint_origin_summary"]) if selected.get("endpoint_origin_summary")
                     else paths(result["combined"]) / "combined_summary.json")
    combined = json.loads(combined_path.read_text())
    if combined["provenance"]["endpoint_source"] != "native":
        raise ValueError("The reference endpoint must have been built from saved native observation tables.")
    model_record = combined["provenance"]["structure"]
    model_path = paths(model_record["path"])
    if sha256_file(model_path) != model_record["sha256"]:
        raise ValueError("The full-H reference model changed after its calculation.")
    if not any(atom["element"].upper() in {"H", "D"} for atom in parse_model(model_path)["atoms"]):
        raise ValueError("The full-H reference model contains no hydrogen atoms.")
    endpoint = combined["combined_endpoint"]
    if float(endpoint["lambda_q"]) != 1 or endpoint.get("keep_exti"):
        raise ValueError("A matched reference needs the full q correction and an EXTI-off endpoint.")
    datasets = []
    for item in combined["runs"]:
        if float(item.get("endpoint_weight", 1)) <= 0 and float(item.get("raw_row_weight", 1)) <= 0:
            continue
        artifacts = item["source_artifacts"]
        summary_path = paths(artifacts["summary"]["path"])
        table = paths(artifacts["row_table"]["path"])
        for field, path in (("summary", summary_path), ("row_table", table)):
            expected = artifacts[field].get("sha256")
            if not expected or sha256_file(path) != expected:
                raise ValueError(f"Changed or unbound reference {field}: {path}")
        summary = json.loads(summary_path.read_text())
        if summary["config"].get("q_control", "real") != "real":
            raise ValueError("A shuffled-q diagnostic cannot be a never-H reference.")
        bound = summary.get("inputs", {}).get("res_sha256")
        if bound != model_record["sha256"]:
            raise ValueError(f"{item['name']}: native predictions and reference model are not bound to the same coordinates.")
        if summary["config"].get("alternate_hand"):
            raise ValueError("Canonicalize the alternate-hand reference before a matched never-H run.")
        hkl = paths(summary["config"]["hkl"])
        rows = read_native_rows(table.parent, item["name"])
        if not rows:
            raise ValueError(f"Empty reference table: {table}")
        cfg = frozen_config(summary, paths)
        data = XDSData(hkl, hkl_transform=summary["config"]["hkl_transform"])
        datasets.append({"name": item["name"], "summary": summary, "rows": rows, "row_table": str(table),
                         "hkl": str(hkl), "config": asdict(cfg),
                         "endpoint_weight": float(item.get("endpoint_weight", 1)),
                         "raw_row_weight": float(item.get("raw_row_weight", 1)),
                         "provenance": {"summary": str(summary_path), "summary_sha256": sha256_file(summary_path),
                                        "row_table_sha256": sha256_file(table), "hkl_sha256": sha256_file(hkl),
                                        "geometry": external_geometry_fingerprints(data, cfg)}})
    reference = {"root": str(root), "model": str(model_path), "model_sha256": model_record["sha256"],
                 "combined": combined, "combined_path": str(combined_path), "datasets": datasets,
                 "selected_endpoint": selected.get("id", "final_combined"),
                 "endpoint_hkl": str(paths(endpoint["artifacts"]["hkl"]))}
    return reference


def reference_endpoint_rows(reference: dict) -> dict[str, list]:
    """Recover the fitted per-observation endpoint correction without fitting anything."""
    endpoint = reference["combined"]["combined_endpoint"]
    inter = reference["combined"]["interdataset_scale"]["scales"]
    minimum = float(endpoint.get("min_d_spacing_A", 0))
    model = parse_model(Path(reference["model"]))
    import gemmi
    cell = gemmi.UnitCell(*model["cell"])
    result = {}
    for ds in reference["datasets"]:
        if ds["endpoint_weight"] <= 0:
            result[ds["name"]] = []
            continue
        selected = [row for row in ds["rows"] if cell.calculate_d([int(row[k]) for k in ("h", "k", "l")]) >= minimum - 1e-8]
        endpoint_rows, _ = endpoint_rows_and_corrections_from_native(selected, 1.0)
        if len(endpoint_rows) != len(selected):
            raise ValueError(f"{ds['name']}: invalid reference endpoint observations")
        sigma = endpoint["dataset_sigma"][ds["name"]]
        sigma_factor = float(sigma.get("sigma_scale", 1)) if str(sigma.get("status", "")).startswith("ok") else 1.0
        factor = float(inter[ds["name"]])
        weight = ds["endpoint_weight"]
        result[ds["name"]] = [(int(row["row_index"]), (*hkl[:3], hkl[3] * factor, hkl[4] * factor * sigma_factor / math.sqrt(weight)))
                              for row, hkl in zip(selected, endpoint_rows, strict=True)]
    return result


def frozen_merge_contract(reference: dict) -> dict:
    model = parse_model(Path(reference["model"]))
    key = endpoint_key_function(reference["combined"]["combined_endpoint"]["merge_key"], model["symmetry_ops"])
    contributions = reference_endpoint_rows(reference)
    formal, _ = merge_hkl_rows([hkl for group in contributions.values() for _, hkl in group], "formal", key)
    written = {key(*hkl[:3]): hkl for hkl in read_hkl(Path(reference["endpoint_hkl"]))}
    safe = float(reference["combined"]["combined_endpoint"]["hkl_safe_factor"])
    multipliers = {}
    for h, k, l, intensity, sigma in formal:
        observed = written.get((h, k, l))
        if observed is None or not np.isclose(observed[3], intensity * safe, rtol=2e-6, atol=0.006):
            raise ValueError(f"Reference endpoint cannot be reproduced at {(h,k,l)}; its row tables or scales differ.")
        if sigma <= 0 or observed[4] <= 0:
            raise ValueError("Reference endpoint has a nonpositive uncertainty")
        multipliers[(h, k, l)] = observed[4] / (sigma * safe)
    if len(written) != len(formal):
        raise ValueError("Reference merged reflection membership differs")
    return {"contributions": contributions, "sigma_multipliers": multipliers, "safe_factor": safe}


def merge_changed_predictions(reference: dict, frozen: dict, rows_by_name: dict) -> list:
    """Change only q; preserve empirical sigma factors, intercrystal scales and membership."""
    all_rows = []
    for ds in reference["datasets"]:
        current = {int(row["row_index"]): row for row in rows_by_name[ds["name"]]}
        old = {int(row["row_index"]): row for row in ds["rows"]}
        for index, hkl in frozen["contributions"][ds["name"]]:
            q = float(current[index]["q_dyn_over_born"])
            if not math.isfinite(q) or q <= 0:
                raise ValueError(f"Undefined dynamical correction in {ds['name']} observation {index}")
            ratio = np.clip(float(old[index]["q_dyn_over_born"]), 0.05, 20) / np.clip(q, 0.05, 20)
            all_rows.append((*hkl[:3], hkl[3] * ratio, hkl[4] * ratio))
    model = parse_model(Path(reference["model"]))
    key = endpoint_key_function(reference["combined"]["combined_endpoint"]["merge_key"], model["symmetry_ops"])
    merged, _ = merge_hkl_rows(all_rows, "formal", key)
    return [(h, k, l, intensity * frozen["safe_factor"], sigma * frozen["safe_factor"] * frozen["sigma_multipliers"][(h, k, l)])
            for h, k, l, intensity, sigma in merged]
