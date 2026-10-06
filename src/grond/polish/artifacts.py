"""Portable, hash-checked state for replaying a polish's frozen Bloch calculation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from grond.polish import crystals as C

SCHEMA = "grond.polish.frozen.v1"


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_state(problem, solution: dict, crystal, target: Path) -> tuple[dict, dict]:
    """Save the accepted model, potential, scales and plans, independently of mutable worker state.

    Every file needed for prediction replay lives below target. The original stage summary and inputs are
    retained separately; they describe the source calculation, not these final predictions.
    """

    directory = target / "frozen"
    directory.mkdir(parents=True, exist_ok=True)
    name = crystal.name
    x = np.asarray(solution["fit"]["x"], dtype=float)
    params = x[:problem.n_struct]
    model = problem.structure.model_at(params)
    (directory / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    for source, filename in (
        (problem.structure.res_path, "start_model.res"), (crystal.stage_res, "stage_model.res"),
        (crystal.hkl, "input.hkl"), (crystal.summary_path, "source_summary.json"),
        (crystal.row_table, "source_rows.csv"),
    ):
        shutil.copy2(source, directory / filename)
    # evaluate() may last have written a rejected trial. Recompute U from the accepted parameter vector.
    potential = problem.potentials[name]
    u, _ = potential.values_and_jacobian(params, jacobian=False)
    np.save(directory / "potential.npy", u)
    np.save(directory / "reciprocal_indices.npy", potential.g)
    geo = problem.geometry[name]
    shutil.copy2(geo["lookup"], directory / "lookup.npy")
    t0, gradient = problem.thickness(x, name)
    nodes, _, _ = C.thickness_nodes(problem.quad[name], t0, gradient)
    weights, _, _ = C.thickness_weights(problem.quad[name], t0, gradient)
    part = solution["fit"]["profile"]["parts"][name]
    scales = problem.scales[name]
    np.savez(directory / "parameters.npz", x=x, scales=part["scales"], coefficients=part["coef"],
             row_factor=crystal.row_factor, rows=crystal.rows, fit=scales.fit, scale_weights=scales.w,
             scale_basis=scales.basis if scales.basis is not None else np.zeros((0, 0)),
             scale_penalty=scales.penalty,
             thickness_nodes=nodes, thickness_weights=weights)
    plans = []
    counts = np.zeros(len(crystal.rows), dtype=int)
    profiles = np.zeros(len(crystal.rows), dtype=int)
    profile_sums = np.zeros(len(crystal.rows))
    hashes = [""] * len(crystal.rows)
    seen = np.zeros(len(crystal.rows), dtype=bool)
    for number, source in enumerate(problem.chunks[name]):
        path = directory / f"plan_{number:05d}.npz"
        shutil.copy2(source, path)
        plans.append(path.name)
        with np.load(path) as chunk:
            for i, basis, _sg, profile_weights in C._chunk_views(chunk):
                position = int(chunk["positions"][i])
                if seen[position] or int(chunk["rows"][i]) != int(crystal.rows[position]):
                    raise ValueError(f"{name}: frozen plan row identities do not match the output table")
                seen[position] = True
                counts[position] = len(basis)
                profiles[position] = len(profile_weights)
                profile_sums[position] = np.sum(profile_weights)
                hashes[position] = hashlib.sha256(np.asarray(basis, dtype=np.int32).tobytes()).hexdigest()[:16]
    if not np.all(seen):
        raise ValueError(f"{name}: frozen plans do not cover every output row")
    cfg = C.config_dict(crystal.cfg)
    cfg.update(thickness_A=t0, thickness_grad_frac=gradient, double_precision=True)
    manifest = {
        "schema": SCHEMA, "crystal": name, "config": cfg,
        "geometry": {key: value for key, value in geo.items() if key != "lookup"},
        "plans": plans, "scale_contract": crystal.scale, "scale_penalty_multiplier": scales.kappa,
        "dataset_fit_weight": scales.objective_weight, "hydrogen_mode": problem.options.hydrogen_mode,
        "optimizer_status": solution["fit"]["status"],
        "thickness_bounds_A": [float(value) if np.isfinite(value) else None
                               for value in problem.thickness_bounds[name]],
        "model_precision": "model.json and potential.npy hold the accepted full-precision state; RES is rounded",
        "basis_policy": "membership and rocking geometry frozen at start_model.res",
        "source_paths": {"summary": str(crystal.summary_path), "rows": str(crystal.row_table),
                         "hkl": str(crystal.hkl), "stage_model": str(crystal.stage_res),
                         "start_model": str(problem.structure.res_path)},
        "files": {path.name: _digest(path) for path in sorted(directory.iterdir()) if path.name != "manifest.json"},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, {"basis_size": counts, "basis_support_hash64": hashes,
                      "sg_profile_points_total": profiles, "sg_profile_weight_sum": profile_sums}


def seal_rows(target: Path) -> None:
    """Bind the completed prediction table to its frozen state."""

    path = target / "frozen" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["row_table_sha256"] = _digest(target / "native_dyn_calc.csv")
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def replay_rows(directory: Path, *, workers: int = 1) -> dict[str, np.ndarray]:
    """Verify a frozen directory and replay its raw and scaled predictions without the original run."""

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported frozen polish state")
    table = directory.parent / "native_dyn_calc.csv"
    if not table.is_file() or _digest(table) != manifest.get("row_table_sha256"):
        raise ValueError("frozen polish prediction table missing or changed")
    for filename, digest in manifest["files"].items():
        path = directory / filename
        if not path.is_file() or _digest(path) != digest:
            raise ValueError(f"frozen polish artifact missing or changed: {filename}")
    with np.load(directory / "parameters.npz") as state:
        rows, scales, factor = state["rows"], state["scales"], state["row_factor"]
        nodes, weights = state["thickness_nodes"], state["thickness_weights"]
    geo = manifest["geometry"]
    tasks = [{"crystal": manifest["crystal"], "chunk": str(directory / filename),
              "u": str(directory / "potential.npy"), "lookup": str(directory / "lookup.npy"),
              "lookup_lim": geo["lookup_lim"], "a_matrix": geo["a_matrix"], "k": geo["k"],
              "matrix_convention": manifest["config"]["matrix_convention"],
              "t_nodes": nodes.tolist(), "t_weights": weights.tolist(),
              "dt_dt0": np.zeros_like(nodes).tolist(), "dt_dgrad": np.zeros_like(nodes).tolist(),
              "grad_positions": []} for filename in manifest["plans"]]
    raw, born = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    for result in C.run_tasks(C.evaluate_chunk, tasks, workers):
        raw[result["positions"]] = result["intensity"]
        born[result["positions"]] = result["born"]
    if not (np.all(np.isfinite(raw)) and np.all(np.isfinite(born))):
        raise ValueError("incomplete frozen polish prediction replay")
    return {"rows": rows, "dyn_raw_unmodulated": raw, "dyn_raw": factor * raw,
            "dyn_scaled": scales * factor * raw, "born_raw": born}
