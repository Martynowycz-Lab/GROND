"""Scout proper HKL transforms with geometry checks and a fixed observation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from grond.cli_support.execution import (
    run_logged,
    transform_scout_job_count,
    transform_scout_semaphore,
    transform_scout_worker_count,
)
from grond.cli_support.values import read_json
from grond.hkl_transform import (
    absolute_configuration_claim_contract,
    parse_hkl_transform_info,
    source_orientation_contract,
)
from grond.io.config import DatasetSpec, PrepConfig
from grond.macromolecule import sha256_file
from grond.workflows.decision import (
    ALL_PROPER_SIGNED_PERMUTATIONS,
    expand_transform_candidates,
    safe_float,
    safe_label,
    score_transform_summary,
    write_json,
)
from grond.workflows.hand_scout import _dataset_decision_identity
from grond.workflows.model_selection import dataset_baseline_hkl
from grond.workflows.native_commands import native_stage_cmd
from grond.workflows.nuisance_policy import transform_scout_nuisance
from grond.workflows.parity_evidence import replay_transform_work_born
from grond.workflows.transform_metadata import metadata_prefilter_transforms


def dataset_with_transform(ds: DatasetSpec, transform: str) -> DatasetSpec:
    """Return a dataset copy using a different HKL transform."""

    return DatasetSpec(
        name=ds.name,
        hkl=ds.hkl,
        xds_dir=ds.xds_dir,
        baseline_hkl=ds.baseline_hkl,
        hkl_transform=transform,
        min_d_spacing=ds.min_d_spacing,
        min_peak=ds.min_peak,
        target_mode=ds.target_mode,
        include_nonpositive=ds.include_nonpositive,
        geometry_percentile=ds.geometry_percentile,
        split_contract=ds.split_contract,
        physical_crystal_id=ds.physical_crystal_id,
        acquisition_fingerprint=ds.acquisition_fingerprint,
    )


def dataset_with_split_contract(ds: DatasetSpec, split_contract: str) -> DatasetSpec:
    """Return a dataset copy bound to one immutable raw-row work/free split."""

    return DatasetSpec(
        name=ds.name,
        hkl=ds.hkl,
        xds_dir=ds.xds_dir,
        baseline_hkl=ds.baseline_hkl,
        hkl_transform=ds.hkl_transform,
        min_d_spacing=ds.min_d_spacing,
        min_peak=ds.min_peak,
        target_mode=ds.target_mode,
        include_nonpositive=ds.include_nonpositive,
        geometry_percentile=ds.geometry_percentile,
        split_contract=str(split_contract),
        physical_crystal_id=ds.physical_crystal_id,
        acquisition_fingerprint=ds.acquisition_fingerprint,
    )


def _transform_candidate_generation_contract(
    requested_spec: str,
    generated_candidates: list[str],
    retained_candidates: list[str],
    metric_filter: Mapping[str, object],
    *,
    parity_policy: str,
) -> dict[str, object]:
    """Describe whether candidate generation is calibration-grade automatic."""

    requested = str(requested_spec or "auto").strip()
    tokens = [token.strip().lower() for token in requested.split(";") if token.strip()]
    automatic_only = bool(tokens) and all(token in {"auto", "common"} for token in tokens)
    canonical_candidates = [parse_hkl_transform_info(candidate).canonical for candidate in generated_candidates]
    required = [parse_hkl_transform_info(candidate).canonical for candidate in ALL_PROPER_SIGNED_PERMUTATIONS]
    exact_full24_proper_set = bool(
        len(canonical_candidates) == len(required) and set(canonical_candidates) == set(required)
    )
    canonical_retained = [parse_hkl_transform_info(candidate).canonical for candidate in retained_candidates]
    metric_filter_valid = bool(
        metric_filter.get("policy") == "source_model_reciprocal_cell_metric_only_v1"
        and metric_filter.get("uses_intensities") is False
        and metric_filter.get("uses_atomic_coordinates_or_hand") is False
        and metric_filter.get("uses_atom_types_or_known_chemistry") is False
        and metric_filter.get("uses_known_configuration") is False
        and list(metric_filter.get("retained_transforms") or []) == canonical_retained
    )
    calibration_grade = bool(
        parity_policy == "proper-only"
        and automatic_only
        and exact_full24_proper_set
        and metric_filter_valid
        and canonical_retained
        and all(parse_hkl_transform_info(candidate).proper for candidate in canonical_retained)
    )
    return {
        "policy": (
            "automatic_full24_proper_metric_filtered_v1" if calibration_grade else "custom_or_restricted_candidates_v1"
        ),
        "requested_spec": requested,
        "automatic_tokens_only": automatic_only,
        "full_proper_signed_permutation_universe": required,
        "exact_full24_proper_set": exact_full24_proper_set,
        "generated_candidates_before_metadata": canonical_candidates,
        "candidate_count_before_metadata": len(canonical_candidates),
        "metric_filter": dict(metric_filter),
        "retained_candidates_after_metric_filter": canonical_retained,
        "metric_filter_contract_valid": metric_filter_valid,
        "calibration_grade_candidate_generation": calibration_grade,
    }


def _materialize_transform_hand_split_contract(
    prep: PrepConfig,
    ds: DatasetSpec,
    candidates: list[str],
    args: argparse.Namespace,
    root: Path,
) -> tuple[DatasetSpec, dict[str, object]]:
    """Write and bind the candidate-union raw-row split used by all scouts."""

    from grond.legacy.engine.bloch_engine import (
        build_transform_split_contract,
        validate_transform_split_contract,
    )

    path = (root / "decisions" / ds.name / "transform_hand_split_contract.json").resolve()
    payload = build_transform_split_contract(
        Path(ds.hkl),
        Path(prep.structure),
        candidates,
        dataset=ds.name,
        free_fraction=float(args.free_fraction),
        seed=int(args.split_seed),
        include_friedel=True,
    )
    validation = validate_transform_split_contract(
        payload,
        source_hkl=Path(ds.hkl),
        current_transform=candidates[0],
    )
    validation.pop("masks", None)
    write_json(path, payload)
    binding = {
        **validation,
        "path": str(path),
        "sha256": sha256_file(path),
    }
    return dataset_with_split_contract(ds, str(path)), binding


def _transform_record_metadata(
    transform: str,
    *,
    source_hkl: str = "",
) -> dict[str, object]:
    """Return flat determinant metadata suitable for scout tables."""

    info = parse_hkl_transform_info(transform)
    record = {
        "transform": info.canonical,
        "transform_determinant": info.determinant,
        "transform_parity": info.parity,
        "transform_is_proper": info.proper,
        "sign_paired_transform": info.sign_paired_transform,
        "proper_canonical_transform": info.proper_canonical_transform,
        "coordinate_inversion_required_for_proper_canonicalization": (
            info.coordinate_inversion_required_for_proper_canonicalization
        ),
    }
    if source_hkl:
        record["source_orientation_contract"] = source_orientation_contract(source_hkl, info.canonical)
    return record


def _selected_transform_claim_contract(
    transform: str,
    *,
    selection_source: str,
    parity_policy: str,
    source_hkl: str = "",
) -> dict[str, object]:
    """Bind one selected transform to GROND's fail-closed claim policy."""

    contract = absolute_configuration_claim_contract(transform)
    contract.update(
        {
            "selection_source": selection_source,
            "transform_parity_policy": parity_policy,
        }
    )
    if source_hkl:
        orientation = source_orientation_contract(source_hkl, transform)
        contract["source_orientation_contract"] = orientation
        if not bool(orientation.get("absolute_claim_geometry_ready")):
            contract["configuration_status"] = "parity_unresolved"
            contract["absolute_configuration_claim_allowed"] = False
            contract["reason"] = (
                f"source orientation contract is {orientation.get('status', 'missing')}; "
                "a positive right-handed raw real-space axis matrix and positive transformed "
                "determinant are required before named absolute configuration can be claimed"
            )
    return contract


def transform_scout_records(
    prep: PrepConfig,
    ds: DatasetSpec,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    scout_root: Path,
    transforms: list[str],
    *,
    scout_stage: str,
    n_basis: int,
    n_points: int,
    max_rows: int,
) -> list[dict]:
    """Run one transform-scout stage and return scored decision records."""

    from grond.workflows.crystal_parameters import crystal_arguments

    args = crystal_arguments(args, ds.name)
    n_basis = getattr(args, "fixed_n_basis", None) or n_basis
    n_points = getattr(args, "fixed_n_points", None) or n_points
    frozen_nuisance = {**transform_scout_nuisance(args), **getattr(args, "fixed_thickness", {})}
    frozen_nuisance_sha256 = hashlib.sha256(
        json.dumps(
            frozen_nuisance,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    def run_one(rank: int, transform: str) -> dict:
        cand = dataset_with_transform(ds, transform)
        out_dir = scout_root / safe_label(transform)
        base_record = {
            **_transform_record_metadata(transform, source_hkl=ds.hkl),
            "out_dir": str(out_dir),
            "scout_stage": scout_stage,
            "n_basis": int(n_basis),
            "n_points": int(n_points),
            "max_rows": int(max_rows),
            "rank_in": rank,
            "frozen_nuisance_contract": frozen_nuisance,
            "frozen_nuisance_contract_sha256": frozen_nuisance_sha256,
            "engine_trial_selection_metric_requested": "work_born_wR2",
        }
        try:
            cmd = native_stage_cmd(
                python,
                cand,
                prep.structure,
                dataset_baseline_hkl(prep, cand),
                out_dir,
                args,
                n_basis=n_basis,
                n_points=n_points,
                stage=f"scout_transform_{scout_stage}_{safe_label(transform)}",
                pinned=frozen_nuisance,
                make_maps=False,
                endpoint=False,
                max_rows_override=max_rows,
                workers_override=transform_scout_worker_count(args),
                scale_fit_split_override="work",
                selection_metric_override="work_born_wR2",
            )
            if args.dry_run:
                run_logged(cmd, logs, f"{ds.name}_scout_transform_{scout_stage}_{safe_label(transform)}", dry_run=True)
            else:
                semaphore = transform_scout_semaphore(args)
                with semaphore:
                    run_logged(
                        cmd, logs, f"{ds.name}_scout_transform_{scout_stage}_{safe_label(transform)}", dry_run=False
                    )
            if args.dry_run:
                return {**base_record, "status": "dry_run", "score": float("inf")}
            summary_path = out_dir / "summary.json"
            summary = read_json(summary_path)
            summary_score = score_transform_summary(summary)
            replay = replay_transform_work_born(summary_path)
            replay_score = float(replay["score"])
            if not math.isclose(
                replay_score,
                float(summary_score["score"]),
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "transform work-Born score does not replay from the bound "
                    f"row artifact: summary={summary_score['score']}, "
                    f"replay={replay_score}"
                )
            return {
                **base_record,
                "status": "ok",
                **summary_score,
                "score": replay_score,
                "artifact_score_replay": replay,
                "engine_trial_selection_metric": str(summary.get("selection_metric") or ""),
                "nuisance_trial_count": int(safe_float(summary.get("nuisance_trial_count"), 0.0)),
                "prediction_geometry_trial_count": int(safe_float(summary.get("prediction_geometry_trial_count"), 0.0)),
            }
        except Exception as exc:
            return {**base_record, "status": "failed", "error": str(exc), "score": float("inf")}

    if len(transforms) <= 1:
        return [run_one(rank, transform) for rank, transform in enumerate(transforms, start=1)]
    records: list[dict] = []
    max_workers = transform_scout_job_count(args, len(transforms))
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix=f"grond-{ds.name}-{scout_stage}-transform"
    ) as pool:
        futures = [pool.submit(run_one, rank, transform) for rank, transform in enumerate(transforms, start=1)]
        for future in as_completed(futures):
            records.append(future.result())
    return sorted(records, key=lambda row: int(row.get("rank_in", 0)))


def ranked_transform_records(records: list[dict], *, dry_run: bool) -> list[dict]:
    """Return transform scout records in promotion order."""

    if dry_run:
        return [r for r in records if r.get("status") == "dry_run"]
    ok = [r for r in records if r.get("status") == "ok" and float(r.get("score", float("inf"))) < float("inf")]
    # Metadata overlap is allowed to affect scheduling only.  Exact scout ties
    # use a canonical transform spelling, never the metadata-derived input
    # order, so all-row overlap cannot silently make the selected decision.
    priority = {
        parse_hkl_transform_info(candidate).canonical: rank
        for rank, candidate in enumerate(ALL_PROPER_SIGNED_PERMUTATIONS)
    }
    return sorted(
        ok,
        key=lambda r: (
            float(r["score"]),
            priority.get(
                parse_hkl_transform_info(str(r.get("transform") or "identity")).canonical,
                len(priority),
            ),
            parse_hkl_transform_info(str(r.get("transform") or "identity")).canonical,
        ),
    )


def choose_transform_by_scout(
    prep: PrepConfig,
    ds: DatasetSpec,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    *,
    scout_max_rows: int | None = None,
) -> tuple[DatasetSpec, dict]:
    """Choose the HKL alignment using cheap all-candidate scouts, then rerank top hits."""

    parity_policy = str(getattr(args, "transform_parity_policy", "proper-only"))
    if getattr(args, "trust_hkl_transform", False):
        selected_info = parse_hkl_transform_info(ds.hkl_transform or "identity")
        if not selected_info.proper and parity_policy != "all-diagnostic":
            raise ValueError(
                f"trusted transform {selected_info.canonical!r} is improper (determinant -1) "
                "and cannot be accepted under --transform-parity-policy=proper-only; use "
                "GROND canonicalize-hand to invert the endpoint model/HKLF4 pair and adopt "
                f"{selected_info.proper_canonical_transform!r}, or explicitly use "
                "all-diagnostic mode without a named absolute-configuration claim"
            )
        selected_transform = selected_info.canonical
        selected = {
            **_transform_record_metadata(selected_transform, source_hkl=ds.hkl),
            "status": "trusted_input",
            "score": None,
            "reason": "user supplied --trust-hkl-transform; transform scouting was skipped",
        }
        claim_contract = _selected_transform_claim_contract(
            selected_transform,
            selection_source="trusted_input",
            parity_policy=parity_policy,
            source_hkl=ds.hkl,
        )
        payload = {
            "records": [],
            "coarse_records": [],
            "rerank_records": [],
            "coarse_selected": selected,
            "selected": selected,
            "rerank_top": 0,
            "metadata_transform_decision": {},
            "dataset": ds.name,
            "input_identity": _dataset_decision_identity(prep, ds),
            "transform_parity_policy": parity_policy,
            "selected_transform_claim_contract": claim_contract,
            "scout_parallelism": {
                "max_concurrent_scout_jobs": 0,
                "workers_per_scout_job": 0,
                "coarse_candidate_count": 0,
                "coarse_parallel_jobs": 0,
                "reranked_candidate_count": 0,
                "reranked_parallel_jobs": 0,
            },
            "rule": (
                "trusted user-supplied HKL transform; no Bloch transform scout was run; "
                "an improper trusted transform is diagnostic-only and forbids a named "
                "absolute-configuration claim"
            ),
            "selection_target": "trusted_input_no_ranking",
            "known_configuration_used_for_candidate_generation_or_ranking": False,
            "metadata_candidate_elimination_used": False,
        }
        write_json(root / "decisions" / ds.name / "transform_decision.json", payload)
        return dataset_with_transform(ds, selected_transform), payload

    candidates = expand_transform_candidates(
        ds.hkl_transform,
        args.transform_candidates,
        parity_policy=parity_policy,
    )
    generated_candidates = list(candidates)
    from grond.legacy.engine.bloch_engine import metric_filter_signed_axis_transforms

    candidates, metric_filter_contract = metric_filter_signed_axis_transforms(
        Path(ds.hkl),
        Path(prep.structure),
        generated_candidates,
        relative_tolerance=float(getattr(args, "transform_metric_tolerance", 0.02)),
        dataset_label=ds.name,
    )
    candidate_generation = _transform_candidate_generation_contract(
        str(args.transform_candidates),
        generated_candidates,
        candidates,
        metric_filter_contract,
        parity_policy=parity_policy,
    )
    split_contract: dict[str, object] = {}
    if not args.dry_run:
        ds, split_contract = _materialize_transform_hand_split_contract(
            prep,
            ds,
            candidates,
            args,
            root,
        )
    metadata_transform_decision: dict = {}
    metadata_candidate_elimination_used = False
    baseline_hkl = dataset_baseline_hkl(prep, ds)
    if getattr(args, "transform_metadata_prefilter", "auto") != "off" and baseline_hkl:
        metadata_candidates, metadata_transform_decision = metadata_prefilter_transforms(
            ds.hkl,
            baseline_hkl,
            candidates,
            mode=getattr(args, "transform_metadata_prefilter", "auto"),
            max_candidates=int(getattr(args, "transform_metadata_max_candidates", 3)),
            min_pairs=int(getattr(args, "transform_metadata_min_pairs", 100)),
            min_correlation=float(getattr(args, "transform_metadata_min_correlation", 0.70)),
            min_correlation_gap=float(getattr(args, "transform_metadata_min_correlation_gap", 0.05)),
            max_rows=int(getattr(args, "transform_metadata_max_rows", 0)),
        )
        metadata_mode = str(getattr(args, "transform_metadata_prefilter", "auto"))
        if metadata_mode == "on":
            candidates = metadata_candidates
            metadata_candidate_elimination_used = True
        else:
            # Default auto mode may order the expensive proper candidates but
            # may not eliminate any of them: its all-row overlap is not part
            # of the held-out-safe transform selection target.
            ranked_names = [
                str(record.get("transform"))
                for record in (metadata_transform_decision.get("records") or [])
                if isinstance(record, Mapping)
            ]
            candidates = [
                *[name for name in ranked_names if name in candidates],
                *[name for name in candidates if name not in ranked_names],
            ]
            metadata_transform_decision["prefilter_status"] = metadata_transform_decision.get("status")
            metadata_transform_decision["status"] = "diagnostic_order_only"
            metadata_transform_decision["selected_transforms"] = list(candidates)
        metadata_transform_decision["candidate_elimination_used"] = metadata_candidate_elimination_used
    scout_root = root / "decisions" / ds.name / "transform_scout"
    coarse_max_rows = args.scout_max_rows if scout_max_rows is None else scout_max_rows
    coarse_records = transform_scout_records(
        prep,
        ds,
        args,
        python,
        logs,
        scout_root,
        candidates,
        scout_stage="coarse",
        n_basis=args.scout_n_basis,
        n_points=args.scout_n_points,
        max_rows=coarse_max_rows,
    )
    ranked_coarse = ranked_transform_records(coarse_records, dry_run=args.dry_run)
    if not ranked_coarse and not args.dry_run:
        write_json(
            root / "decisions" / ds.name / "transform_decision.json",
            {"records": coarse_records, "coarse_records": coarse_records, "selected": None},
        )
        raise RuntimeError(f"all transform scouts failed for dataset {ds.name}; see {scout_root}")

    coarse_selected = ranked_coarse[0] if ranked_coarse else {"transform": ds.hkl_transform}
    rerank_top = max(0, int(getattr(args, "transform_rerank_top", 0)))
    rerank_records: list[dict] = []
    selected = coarse_selected
    if len(candidates) > 1 and rerank_top > 0 and ranked_coarse:
        top_records = ranked_coarse[: min(rerank_top, len(ranked_coarse))]
        top_transforms = [str(r["transform"]) for r in top_records]
        rerank_max_rows = int(getattr(args, "transform_rerank_max_rows", 0) or 0)
        if rerank_max_rows <= 0:
            rerank_max_rows = int(coarse_max_rows)
        rerank_records = transform_scout_records(
            prep,
            ds,
            args,
            python,
            logs,
            scout_root / "updated",
            top_transforms,
            scout_stage="updated",
            n_basis=int(args.transform_rerank_n_basis),
            n_points=int(args.transform_rerank_n_points),
            max_rows=rerank_max_rows,
        )
        ranked_rerank = ranked_transform_records(rerank_records, dry_run=args.dry_run)
        if ranked_rerank:
            selected = ranked_rerank[0]

    records = [*coarse_records, *rerank_records]
    selected_transform = str(selected["transform"])
    claim_contract = _selected_transform_claim_contract(
        selected_transform,
        selection_source="automatic_transform_scout",
        parity_policy=parity_policy,
        source_hkl=ds.hkl,
    )
    payload = {
        "records": records,
        "coarse_records": coarse_records,
        "rerank_records": rerank_records,
        "coarse_selected": coarse_selected,
        "selected": selected,
        "rerank_top": rerank_top,
        "metadata_transform_decision": metadata_transform_decision,
        "metadata_candidate_elimination_used": metadata_candidate_elimination_used,
        "transform_parity_policy": parity_policy,
        "candidate_parity": [_transform_record_metadata(candidate, source_hkl=ds.hkl) for candidate in candidates],
        "candidate_generation": candidate_generation,
        "ranking_tie_break": {
            "policy": "fixed_full24_proper_priority_then_canonical_v1",
            "full24_proper_priority": [
                parse_hkl_transform_info(candidate).canonical for candidate in ALL_PROPER_SIGNED_PERMUTATIONS
            ],
            "metadata_order_used_for_ties_or_rerank_admission": False,
        },
        "transform_hand_split_contract": split_contract,
        "dataset": ds.name,
        "input_identity": _dataset_decision_identity(prep, ds),
        "selected_transform_claim_contract": claim_contract,
        "scout_parallelism": {
            "max_concurrent_scout_jobs": transform_scout_job_count(args),
            "workers_per_scout_job": transform_scout_worker_count(args),
            "coarse_candidate_count": len(candidates),
            "coarse_parallel_jobs": transform_scout_job_count(args, len(candidates)),
            "reranked_candidate_count": len(rerank_records),
            "reranked_parallel_jobs": transform_scout_job_count(args, len(rerank_records)) if rerank_records else 0,
        },
        "rule": (
            "proper-only metadata/Bloch transform scouting by default, followed by higher N/Q "
            "rerank of the top candidates; transform rankings use independently scaled matched-Born "
            "work rows only, reserve free rows, and never use coordinate-hand-sensitive Bloch terms; "
            "default metadata overlap may order but never eliminate candidates; all-diagnostic is an "
            "explicit mixed-parity mode that cannot by itself establish named absolute configuration"
        ),
        "selection_target": "matched_born_work_only_hand_invariant_v1",
        "selection_nuisance_contract": transform_scout_nuisance(args),
        "known_configuration_used_for_candidate_generation_or_ranking": False,
    }
    write_json(root / "decisions" / ds.name / "transform_decision.json", payload)
    return dataset_with_transform(ds, selected["transform"]), payload
