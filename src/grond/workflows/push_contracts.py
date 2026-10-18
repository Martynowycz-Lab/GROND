"""Restore endpoint-push arguments and dataset contracts from a saved checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import read_json, try_read_json
from grond.hashing import sha256_file
from grond.io.config import DatasetSpec, PrepConfig
from grond.validation.heldout import DEFAULT_FREE_SALT
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


# The continuation contract a push must replay from the run it continues: the free split, the P-spline
# frame-scale grids and the dynamical SHELXL handoff.  Each entry is (push attribute = source-summary config
# key, value of the historical behaviour for summaries that predate the key).  The current AUTO defaults are
# never used.
CONTINUATION_SPLIT_SCALE_KEYS = (
    ("split_scheme", "legacy"),
    ("split_salt", DEFAULT_FREE_SALT),
    ("exclude_orbits_json", ""),
    ("free_fraction", 0.10),
    ("split_seed", 42),
    ("scale_spline_n_basis_grid", "32"),
    ("scale_spline_lambda_grid", "0.3,1,3,10,30,100"),
)
CONTINUATION_HANDOFF_KEYS = (
    ("endpoint_include_free_rows", True),
    ("endpoint_q_policy", "clip"),
    ("rescale", "median-iobs"),
    ("shelxl_npd_recovery", "damp"),
    ("shelxl_convergence_shift_esd", 0.0),
    ("endpoint_wght_mode", "freeze-zero"),
    # Before --keep-damp existed the endpoint writer always carried an active DAMP forward.
    ("keep_damp", True),
)
CONTINUATION_CHOICES = {
    "split_scheme": ("orbit-hash", "legacy"),
    "endpoint_q_policy": ("clip", "exclude"),
    "rescale": ("median-iobs", "none", "absolute"),
    "shelxl_npd_recovery": ("damp", "isor"),
    "endpoint_wght_mode": ("freeze-zero", "inherit", "promote", "promote-gold", "converge"),
}
_CONTINUATION_BOOL_KEYS = {"endpoint_include_free_rows", "keep_damp"}
_CONTINUATION_FLOAT_KEYS = {"free_fraction", "shelxl_convergence_shift_esd"}
_CONTINUATION_INT_KEYS = {"split_seed"}


def _continuation_value(key: str, value: object) -> object:
    """Convert a recorded continuation value to the type of its contract key (bool, float, int or str)."""

    if key in _CONTINUATION_BOOL_KEYS:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return bool(value)
    if key in _CONTINUATION_FLOAT_KEYS:
        return float(value)
    if key in _CONTINUATION_INT_KEYS:
        return int(float(value))
    return str(value)


def _recorded_continuation_value(summary: Mapping[str, object], key: str) -> object | None:
    """The value a source native summary recorded for one contract key, or None when it does not record it."""

    config = summary.get("config") if isinstance(summary.get("config"), Mapping) else {}
    if key in config and config[key] is not None:
        return config[key]
    best = summary.get("best_native") if isinstance(summary.get("best_native"), Mapping) else {}
    scale_model = best.get("scale_model") if isinstance(best.get("scale_model"), Mapping) else {}
    if key == "scale_spline_n_basis_grid" and scale_model.get("n_basis_grid"):
        return ",".join(str(int(value)) for value in scale_model["n_basis_grid"])
    if key == "scale_spline_lambda_grid" and scale_model.get("lambda_grid"):
        return ",".join(f"{float(value):g}" for value in scale_model["lambda_grid"])
    return None


def resolve_push_continuation_contract(
    args: argparse.Namespace,
    summaries: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Resolve the split, scale and handoff contract of a continuation, with the provenance of every value.

    An explicit push-endpoint option wins.  Otherwise every source summary must record the same value (the
    datasets of one run share one free split, scale grids and handoff); a summary that does not record a key
    gives that key's legacy value, so an older checkpoint continues as it ran.  The P-spline grids have no legacy
    value: a run that recorded none (per-frame scales) needs them passed explicitly.
    """

    contract: dict[str, dict[str, object]] = {}
    for key, legacy in (*CONTINUATION_SPLIT_SCALE_KEYS, *CONTINUATION_HANDOFF_KEYS):
        recorded_raw = {name: _recorded_continuation_value(summary, key) for name, summary in summaries.items()}
        recorded = {
            name: _continuation_value(key, legacy if value is None else value) for name, value in recorded_raw.items()
        }
        override = getattr(args, key, None)
        if override is not None:
            value, source = _continuation_value(key, override), "override"
        else:
            distinct = {json.dumps(item, sort_keys=True) for item in recorded.values()}
            if len(distinct) > 1:
                raise SystemExit(
                    f"the continuation datasets record different {key} values {recorded}; one push-endpoint "
                    f"continues one run contract. Pass --{key.replace('_', '-')} explicitly only if that is intended."
                )
            value = next(iter(recorded.values())) if recorded else _continuation_value(key, legacy)
            if any(item is not None for item in recorded_raw.values()):
                source = "source_summary"
            else:
                source = "historical_default_not_recorded"
        choices = CONTINUATION_CHOICES.get(key)
        if choices and value not in choices:
            raise SystemExit(f"continuation {key} {value!r} is not one of {', '.join(choices)}")
        contract[key] = {"value": value, "source": source, "recorded": recorded}
    for key in ("scale_spline_n_basis_grid", "scale_spline_lambda_grid"):
        if contract[key]["source"] == "historical_default_not_recorded":
            raise SystemExit(
                f"no source summary records the P-spline frame-scale grid {key}; "
                f"pass --{key.replace('_', '-')} explicitly"
            )
    return contract


def continuation_source_summaries(checkpoint: Mapping[str, object], base: Path) -> dict[str, dict]:
    """Per-dataset source native summaries of a multi-dataset checkpoint (empty for a dataset without one)."""

    summaries: dict[str, dict] = {}
    for raw in checkpoint.get("datasets") or []:
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name") or "")
        path = _checkpoint_path(raw.get("source_summary"), base=base)
        if path and not Path(path).is_file():
            raise SystemExit(f"checkpoint dataset {name!r} records source_summary {path}, which is missing")
        summaries[name] = read_json(Path(path)) if path else {}
    return summaries


def apply_push_continuation_contract_multi(args: argparse.Namespace, contract: Mapping[str, Mapping]) -> None:
    """Set the resolved contract on a multi-dataset push namespace under the AUTO attribute names."""

    for key, _legacy in CONTINUATION_SPLIT_SCALE_KEYS:
        setattr(args, key, contract[key]["value"])
    rescale = str(contract["rescale"]["value"])
    if rescale not in {"absolute", "median-iobs"}:
        raise SystemExit(
            f"multi-dataset continuation supports the absolute or median-iobs endpoint intensity scale, not {rescale!r}"
        )
    args.endpoint_intensity_scale = rescale
    args.rescale = rescale
    args.endpoint_include_free_rows = bool(contract["endpoint_include_free_rows"]["value"])
    args.endpoint_q_policy = str(contract["endpoint_q_policy"]["value"])
    args.shelxl_npd_recovery = str(contract["shelxl_npd_recovery"]["value"])
    args.shelxl_convergence_shift_esd = float(contract["shelxl_convergence_shift_esd"]["value"])
    args.endpoint_wght_mode = str(contract["endpoint_wght_mode"]["value"])
    args.keep_damp = bool(contract["keep_damp"]["value"])
    args.shelxl_carry_damp = bool(contract["keep_damp"]["value"])


def continuation_contract_record(contract: Mapping[str, Mapping]) -> dict[str, dict[str, object]]:
    """JSON record of the resolved continuation contract (value and source per key)."""

    return {key: {"value": item["value"], "source": item["source"]} for key, item in contract.items()}


def refuse_single_push_split_scale_overrides(contract: Mapping[str, Mapping]) -> None:
    """A single-dataset push replays the split and scale recorded in its source summary; it cannot change them.

    shelxl_feedback_loop.py takes the free split and the frame-scale family from --source-summary, so an
    override that differs from the recorded value would be ignored; refuse it instead.
    """

    for key, _legacy in CONTINUATION_SPLIT_SCALE_KEYS:
        item = contract[key]
        recorded = next(iter(item["recorded"].values()), None)
        if item["source"] == "override" and item["value"] != recorded:
            raise SystemExit(
                f"single-dataset push-endpoint replays {key}={recorded!r} from --source-summary; "
                f"--{key.replace('_', '-')} {item['value']!r} cannot change it. Continue from a source summary "
                "made with that setting instead."
            )


def single_push_handoff_args(contract: Mapping[str, Mapping]) -> list[str]:
    """shelxl_feedback_loop.py handoff options of a single-dataset push (the loop defaults are never relied on)."""

    return [
        "--endpoint-wght-mode",
        str(contract["endpoint_wght_mode"]["value"]),
        "--shelxl-convergence-shift-esd",
        f"{float(contract['shelxl_convergence_shift_esd']['value']):g}",
        "--shelxl-npd-recovery",
        str(contract["shelxl_npd_recovery"]["value"]),
        "--endpoint-q-policy",
        str(contract["endpoint_q_policy"]["value"]),
        "--endpoint-include-free-rows"
        if bool(contract["endpoint_include_free_rows"]["value"])
        else "--no-endpoint-include-free-rows",
        *(["--keep-damp"] if bool(contract["keep_damp"]["value"]) else []),
    ]


def warn_historical_continuation_defaults(contract: Mapping[str, Mapping]) -> None:
    """State which contract values no source summary recorded (the historical value is used for them)."""

    unrecorded = [key for key, item in contract.items() if item["source"] == "historical_default_not_recorded"]
    if unrecorded:
        print(
            "WARN push-endpoint: the source summaries do not record "
            + ", ".join(f"{key}={contract[key]['value']!r}" for key in unrecorded)
            + "; the continuation uses these historical values",
            flush=True,
        )


def hydrate_push_args_for_multi(
    args: argparse.Namespace,
    source_summaries: Mapping[str, Mapping[str, object]] | None = None,
) -> argparse.Namespace:
    """Fill missing multi-dataset continuation attributes; the continuation contract comes from the checkpoint.

    The split, scale and handoff contract is resolved from ``source_summaries`` (the checkpoint datasets'
    source native summaries) and explicit push-endpoint overrides, never from the current AUTO defaults;
    every other missing attribute takes the normal run default.
    """

    from grond.cli_options.parser import build_parser

    contract = resolve_push_continuation_contract(args, dict(source_summaries or {}))
    apply_push_continuation_contract_multi(args, contract)
    args.continuation_contract = continuation_contract_record(contract)
    warn_historical_continuation_defaults(contract)
    defaults = build_parser().parse_args(
        ["auto", "--out-dir", ".", "--hkl", "INTEGRATE.HKL", "--structure", "model.res"]
    )
    for key, value in vars(defaults).items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    args.fast = False
    args.native_only = False
    # A continuation replays the checkpoint's pinned nuisance; the per-macro-cycle thickness update of the AUTO
    # nested nuisance stage needs that run's nested contracts and does not apply.
    args.feedback_thickness_update = "off"
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
