"""Restore endpoint-push arguments and dataset contracts from a saved checkpoint."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import read_json, try_read_json
from grond.io.config import DatasetSpec, PrepConfig
from grond.macromolecule import sha256_file
from grond.workflows.checkpoints import _checkpoint_path


def apply_push_checkpoint(args: argparse.Namespace) -> dict:
    """Fill push-endpoint inputs from a continue checkpoint when requested."""

    checkpoint_path = str(getattr(args, "checkpoint", "") or "")
    if not checkpoint_path:
        if not getattr(args, "source_summary", ""):
            raise SystemExit("Provide --source-summary, or use --checkpoint from a finished GROND auto run.")
        if not getattr(args, "start_res", ""):
            raise SystemExit("Provide --start-res, or use --checkpoint with a recorded start model.")
        return {}

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Continue checkpoint not found: {path}")
    checkpoint = read_json(path)
    if checkpoint.get("kind") != "grond_continue_checkpoint":
        raise SystemExit(f"{path} is not a GROND continue checkpoint.")
    if checkpoint.get("status") != "ready":
        reason = checkpoint.get("status_reason") or "checkpoint is not marked ready"
        raise SystemExit(f"{path} cannot be used for endpoint continuation: {reason}")

    base = path.parent
    dataset_count = int(checkpoint.get("dataset_count") or len(checkpoint.get("datasets") or []) or 1)
    if not getattr(args, "source_summary", ""):
        args.source_summary = _checkpoint_path(checkpoint.get("source_summary"), base=base)
    if not getattr(args, "start_res", ""):
        args.start_res = _checkpoint_path(checkpoint.get("recommended_start_res"), base=base)
    if dataset_count == 1 and not getattr(args, "hkl", "") and checkpoint.get("hkl"):
        args.hkl = _checkpoint_path(checkpoint.get("hkl"), base=base)
    if dataset_count == 1 and not getattr(args, "baseline_hkl", "") and checkpoint.get("baseline_hkl"):
        args.baseline_hkl = _checkpoint_path(checkpoint.get("baseline_hkl"), base=base)
    if dataset_count == 1 and not getattr(args, "hkl_transform", "") and checkpoint.get("hkl_transform"):
        args.hkl_transform = str(checkpoint.get("hkl_transform"))
    if (
        dataset_count > 1
        and not getattr(args, "multi_endpoint_merge_key", "")
        and checkpoint.get("multi_endpoint_merge_key")
    ):
        args.multi_endpoint_merge_key = str(checkpoint.get("multi_endpoint_merge_key"))

    if dataset_count == 1 and not getattr(args, "source_summary", ""):
        raise SystemExit(f"{path} does not record source_summary; provide --source-summary.")
    if not getattr(args, "start_res", ""):
        raise SystemExit(f"{path} does not record recommended_start_res; provide --start-res.")
    return {
        "checkpoint": str(path),
        "checkpoint_status": checkpoint.get("status"),
        "dataset_count": dataset_count,
        "checkpoint_data": checkpoint,
    }


def hydrate_push_args_for_multi(args: argparse.Namespace) -> argparse.Namespace:
    """Fill missing multi-dataset continuation attributes from normal run defaults."""

    from grond.cli_options.parser import build_parser

    defaults = build_parser().parse_args(
        ["run", "--out-dir", ".", "--hkl", "INTEGRATE.HKL", "--structure", "model.res"]
    )
    for key, value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args.fast = False
    args.native_only = False
    args.dry_run = bool(getattr(args, "dry_run", False))
    args.shelxl_feedback_cycles = int(getattr(args, "cycles", 3) or 0)
    args.shelxl_feedback_shelxl_iterations = int(getattr(args, "shelxl_iterations", 1) or 1)
    args.shelxl_feedback_endpoint_ls_cycles = int(getattr(args, "endpoint_ls_cycles", 1) or 1)
    args.shelxl_iterations = int(getattr(args, "shelxl_iterations", 1) or 1)
    args.n_points = int(getattr(args, "n_points", 20) or 20)
    args.n_basis = int(getattr(args, "n_basis", 108) or 108)
    args.final_n_basis = int(getattr(args, "final_n_basis", 0) or 0)
    args.final_n_points = int(getattr(args, "final_n_points", args.n_points) or args.n_points)
    args.prediction_cache_compression = getattr(args, "prediction_cache_compression", "none")
    args.make_maps = bool(getattr(args, "make_maps_on_final", False))
    args.multi_endpoint_merge_key = getattr(args, "multi_endpoint_merge_key", "model_symmetry") or "model_symmetry"
    return args


def resolve_dataset_row_contracts(
    specifications: list[str] | tuple[str, ...] | None,
    datasets: list[DatasetSpec],
) -> dict[str, str]:
    """Parse strict ``dataset=policy-selected-native_dyn_calc.csv`` bindings."""

    raw = list(specifications or [])
    if not raw:
        return {}
    names = {dataset.name for dataset in datasets}
    resolved: dict[str, str] = {}
    for spec in raw:
        text = str(spec).strip()
        if "=" not in text:
            raise SystemExit("--dataset-row-contract must be DATASET=/path/to/native_dyn_calc.csv")
        name, path_text = text.split("=", 1)
        name = name.strip()
        path = Path(path_text).expanduser()
        if name not in names:
            raise SystemExit(
                f"--dataset-row-contract names unknown dataset {name!r}; expected: {', '.join(sorted(names))}"
            )
        if not str(path_text).strip():
            raise SystemExit(f"--dataset-row-contract for {name!r} has an empty path")
        if name in resolved:
            raise SystemExit(f"--dataset-row-contract was supplied more than once for {name!r}")
        resolved[name] = str(path.resolve())
    missing = sorted(names - set(resolved))
    if missing:
        raise SystemExit(
            "immutable multi-dataset continuation requires one --dataset-row-contract for every dataset; "
            f"missing: {', '.join(missing)}"
        )
    return resolved


def resolve_dataset_scale_specs(
    specifications: list[str] | tuple[str, ...] | None,
    datasets: list[DatasetSpec],
) -> dict[str, str]:
    """Parse all-or-none ``DATASET=compact-scale-spec`` bindings."""

    raw = list(specifications or [])
    if not raw:
        return {}
    names = {dataset.name for dataset in datasets}
    resolved: dict[str, str] = {}
    for value in raw:
        text = str(value).strip()
        if "=" not in text:
            raise SystemExit("--dataset-scale-spec must be DATASET=SPEC, for example timepix4_01=adaptive:28:det8")
        name, spec = text.split("=", 1)
        name = name.strip()
        spec = spec.strip()
        if name not in names:
            raise SystemExit(
                f"--dataset-scale-spec names unknown dataset {name!r}; expected: {', '.join(sorted(names))}"
            )
        if not spec:
            raise SystemExit(f"--dataset-scale-spec for {name!r} has an empty compact scale specification")
        if name in resolved:
            raise SystemExit(f"--dataset-scale-spec was supplied more than once for {name!r}")
        resolved[name] = spec
    missing = sorted(names - set(resolved))
    if missing:
        raise SystemExit(
            "compact multi-dataset continuation requires one --dataset-scale-spec for every dataset; "
            f"missing: {', '.join(missing)}"
        )
    return resolved


def dataset_policy_from_row_contracts(
    contracts: Mapping[str, str],
    datasets: list[DatasetSpec],
) -> dict:
    """Recover immutable multi-dataset weights from policy-selected row tables."""

    required = ("dataset", "dataset_role", "raw_row_weight", "endpoint_weight")
    decisions = []
    for dataset in datasets:
        path = Path(contracts[dataset.name]).expanduser()
        if not path.is_file():
            raise SystemExit(f"row contract for {dataset.name!r} is missing: {path}")
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing = [field for field in required if field not in fields]
            if missing:
                raise SystemExit(f"row contract {path} lacks policy annotation column(s): {', '.join(missing)}")
            rows = list(reader)
        if not rows:
            raise SystemExit(f"row contract {path} contains no rows")
        names = {str(row.get("dataset", "")).strip() for row in rows}
        roles = {str(row.get("dataset_role", "")).strip() for row in rows}
        try:
            raw_weights = {float(str(row["raw_row_weight"]).strip()) for row in rows}
            endpoint_weights = {float(str(row["endpoint_weight"]).strip()) for row in rows}
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"row contract {path} has invalid policy weight: {exc}") from exc
        if names != {dataset.name}:
            raise SystemExit(
                f"row contract {path} dataset annotation does not bind to {dataset.name!r}: {sorted(names)}"
            )
        if len(roles) != 1 or roles.pop() not in {"primary", "downweight", "completeness_only", "exclude"}:
            raise SystemExit(f"row contract {path} has inconsistent or invalid dataset_role annotations")
        role = str(rows[0]["dataset_role"]).strip()
        if len(raw_weights) != 1 or len(endpoint_weights) != 1:
            raise SystemExit(f"row contract {path} has inconsistent policy weights across its rows")
        raw_weight = raw_weights.pop()
        endpoint_weight = endpoint_weights.pop()
        if not all(math.isfinite(value) and value >= 0.0 for value in (raw_weight, endpoint_weight)):
            raise SystemExit(f"row contract {path} has non-finite or negative policy weights")
        decisions.append(
            {
                "name": dataset.name,
                "run_dir": str(path.parent.resolve()),
                "action": role,
                "raw_row_weight": raw_weight,
                "endpoint_weight": endpoint_weight,
                "include_in_row_regularization": role in {"primary", "downweight"} and raw_weight > 0.0,
                "include_in_endpoint": role != "exclude" and endpoint_weight > 0.0,
                "reasons": ["immutable policy annotation from --dataset-row-contract"],
                "row_contract": str(path.resolve()),
                "row_contract_sha256": sha256_file(path),
            }
        )
    return {
        "status": "immutable_row_contract",
        "definition": "multi-dataset membership and weights fixed by policy-selected row contracts",
        "datasets": decisions,
    }


def prep_from_multi_checkpoint(
    checkpoint: Mapping[str, object],
    start_model: Path | str,
    *,
    min_d_spacing: float = 0.0,
) -> tuple[PrepConfig, dict[str, dict]]:
    """Build a multi-dataset PrepConfig and nuisance map from a continue checkpoint."""

    rows = checkpoint.get("datasets")
    if not isinstance(rows, list) or len(rows) < 2:
        raise SystemExit("Multi-dataset endpoint continuation needs at least two datasets in the checkpoint.")
    datasets: list[DatasetSpec] = []
    nuisance: dict[str, dict] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise SystemExit("Malformed dataset entry in continue checkpoint.")
        values = {
            field: raw[field] for field in DatasetSpec.__dataclass_fields__ if field in raw and raw[field] is not None
        }
        if float(min_d_spacing) > 0.0:
            values["min_d_spacing"] = float(min_d_spacing)
        ds = DatasetSpec(**values)
        datasets.append(ds)
        nuisance[ds.name] = dict(raw.get("nuisance") or {})
        if not nuisance[ds.name]:
            summary = _checkpoint_path(raw.get("source_summary"))
            if summary:
                nuisance[ds.name] = (try_read_json(Path(summary)).get("best_native") or {}).get("nuisance") or {}
        if not nuisance[ds.name]:
            raise SystemExit(f"Checkpoint does not record a pinned nuisance block for dataset {ds.name!r}.")
    prep = PrepConfig(
        structure=str(Path(start_model).expanduser().resolve()),
        datasets=datasets,
        baseline_hkl=str(checkpoint.get("baseline_hkl") or ""),
        default_hkl_transform=str(checkpoint.get("hkl_transform") or "identity"),
        geometry_percentile=float(checkpoint.get("geometry_percentile") or 99.0),
        parity_setup_id=str(checkpoint.get("parity_setup_id") or ""),
        parity_setup_manifest_sha256=str(checkpoint.get("parity_setup_manifest_sha256") or ""),
        metadata={"continued_from": checkpoint.get("run_dir", "")},
    )
    return prep, nuisance


def override_push_native_weights(
    nuisance: dict[str, dict],
    args: argparse.Namespace,
) -> dict[str, dict]:
    """Apply an explicit native row-weight contract to a continuation replay.

    This is a diagnostic/recovery control for checkpoints written before the
    native feedback loop pinned its starting WGHT contract.  The override is
    deliberately global so every dataset in the shared endpoint loop is
    evaluated under the same declared row objective.
    """

    native_wght_a = getattr(args, "native_wght_a", None)
    native_wght_b = getattr(args, "native_wght_b", None)
    if native_wght_a is None and native_wght_b is None:
        return nuisance
    for block in nuisance.values():
        if native_wght_a is not None:
            block["native_wght_a"] = float(native_wght_a)
        if native_wght_b is not None:
            block["native_wght_b"] = float(native_wght_b)
    return nuisance
