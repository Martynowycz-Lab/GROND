"""Replay the bindings that make a saved transform decision valid for this dataset."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import read_json
from grond.hkl_transform import parse_hkl_transform_info
from grond.macromolecule import sha256_file
from grond.workflows.decision import ALL_PROPER_SIGNED_PERMUTATIONS, safe_float
from grond.workflows.decision_bindings import _validated_decision_identity
from grond.workflows.parity_evidence import replay_transform_work_born


def _fixed_transform_decision_binding(
    path_text: str,
    expected_transform: str = "",
    *,
    require_proper_geometry: bool = False,
    require_calibration_grade: bool = False,
) -> dict[str, object]:
    """Validate and hash a transform decision made without known chemistry."""

    if not path_text:
        return {}
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"transform decision does not exist: {path}")
    payload = read_json(path)
    selected = payload.get("selected")
    selected = selected if isinstance(selected, Mapping) else {}
    recorded_transform = str(selected.get("transform") or payload.get("selected_transform") or "").strip()
    if not recorded_transform:
        raise ValueError("transform decision does not record a selected transform")
    recorded = parse_hkl_transform_info(recorded_transform).canonical
    expected = parse_hkl_transform_info(expected_transform).canonical if expected_transform else recorded
    if expected_transform and recorded != expected:
        raise ValueError(
            f"transform decision does not match the transform being canonicalized: {recorded!r} != {expected!r}"
        )
    input_identity = _validated_decision_identity(
        payload.get("input_identity"),
        label="transform decision",
        require_complete=require_calibration_grade,
    )
    if (
        require_calibration_grade
        and payload.get("known_configuration_used_for_candidate_generation_or_ranking") is not False
    ):
        raise ValueError(
            "transform decision does not explicitly prove that known configuration was excluded "
            "from candidate generation and ranking"
        )
    candidate_generation = payload.get("candidate_generation")
    candidate_generation = candidate_generation if isinstance(candidate_generation, Mapping) else {}
    ranking_tie_break = payload.get("ranking_tie_break")
    ranking_tie_break = ranking_tie_break if isinstance(ranking_tie_break, Mapping) else {}
    full_proper_universe = [
        parse_hkl_transform_info(candidate).canonical for candidate in ALL_PROPER_SIGNED_PERMUTATIONS
    ]
    retained_candidates: list[str] = []
    metric_filter_contract: Mapping[str, object] = {}
    if require_calibration_grade:
        generated = [
            parse_hkl_transform_info(candidate).canonical
            for candidate in (candidate_generation.get("generated_candidates_before_metadata") or [])
        ]
        if (
            candidate_generation.get("policy") != "automatic_full24_proper_metric_filtered_v1"
            or candidate_generation.get("calibration_grade_candidate_generation") is not True
            or candidate_generation.get("exact_full24_proper_set") is not True
            or len(generated) != len(full_proper_universe)
            or set(generated) != set(full_proper_universe)
        ):
            raise ValueError(
                "transform decision candidate generation is not the exact automatic "
                "24 determinant-positive signed-axis permutation universe required "
                "for parity calibration"
            )
        metric_filter_contract = candidate_generation.get("metric_filter")
        metric_filter_contract = metric_filter_contract if isinstance(metric_filter_contract, Mapping) else {}
        retained_candidates = [
            parse_hkl_transform_info(candidate).canonical
            for candidate in (candidate_generation.get("retained_candidates_after_metric_filter") or [])
        ]
        if (
            candidate_generation.get("metric_filter_contract_valid") is not True
            or metric_filter_contract.get("policy") != "source_model_reciprocal_cell_metric_only_v1"
            or metric_filter_contract.get("uses_intensities") is not False
            or metric_filter_contract.get("uses_atomic_coordinates_or_hand") is not False
            or metric_filter_contract.get("uses_atom_types_or_known_chemistry") is not False
            or metric_filter_contract.get("uses_known_configuration") is not False
            or list(metric_filter_contract.get("retained_transforms") or []) != retained_candidates
            or not retained_candidates
        ):
            raise ValueError(
                "transform decision does not contain a valid hand/chemistry/intensity-"
                "independent source/model unit-cell metric filter contract"
            )
        if (
            ranking_tie_break.get("policy") != "fixed_full24_proper_priority_then_canonical_v1"
            or ranking_tie_break.get("metadata_order_used_for_ties_or_rerank_admission") is not False
            or list(ranking_tie_break.get("full24_proper_priority") or []) != full_proper_universe
        ):
            raise ValueError(
                "transform decision does not prove a metadata-independent fixed tie-break/rerank-admission policy"
            )
        coarse = payload.get("coarse_records")
        coarse = coarse if isinstance(coarse, list) else []
        coarse_by_transform = {
            parse_hkl_transform_info(str(record.get("transform") or "")).canonical: record
            for record in coarse
            if isinstance(record, Mapping) and record.get("transform")
        }
        missing_or_invalid = []
        nuisance_contract_hashes = set()
        replayed_coarse_scores: dict[str, float] = {}
        coarse_fidelity_contract: dict[str, object] | None = None
        for candidate in retained_candidates:
            record = coarse_by_transform.get(candidate)
            if (
                not isinstance(record, Mapping)
                or record.get("status") != "ok"
                or record.get("selection_target") != "matched_born_work_only_hand_invariant_v1"
                or record.get("coordinate_hand_sensitive_terms_used") is not False
                or record.get("free_rows_used_for_transform_selection") is not False
                or record.get("engine_trial_selection_metric") != "work_born_wR2"
                or record.get("engine_trial_selection_metric_requested") != "work_born_wR2"
                or int(record.get("nuisance_trial_count") or 0) != 1
                or int(record.get("prediction_geometry_trial_count") or 0) != 1
            ):
                missing_or_invalid.append(candidate)
            elif str(record.get("frozen_nuisance_contract_sha256") or ""):
                nuisance_contract_hashes.add(str(record["frozen_nuisance_contract_sha256"]))
                recorded_replay = record.get("artifact_score_replay")
                recorded_replay = recorded_replay if isinstance(recorded_replay, Mapping) else {}
                binding = recorded_replay.get("artifact_binding")
                binding = binding if isinstance(binding, Mapping) else {}
                try:
                    recomputed_replay = replay_transform_work_born(Path(str(binding.get("summary_path") or "")))
                except Exception:
                    missing_or_invalid.append(candidate)
                    continue
                recomputed_binding = recomputed_replay.get("artifact_binding")
                recomputed_binding = recomputed_binding if isinstance(recomputed_binding, Mapping) else {}
                if (
                    recorded_replay.get("status") != "replayed"
                    or dict(binding) != dict(recomputed_binding)
                    or not math.isclose(
                        safe_float(recorded_replay.get("score")),
                        safe_float(recomputed_replay.get("score")),
                        rel_tol=1e-12,
                        abs_tol=1e-14,
                    )
                    or not math.isclose(
                        safe_float(record.get("score")),
                        safe_float(recomputed_replay.get("score")),
                        rel_tol=1e-12,
                        abs_tol=1e-14,
                    )
                ):
                    missing_or_invalid.append(candidate)
                    continue
                job = recomputed_replay.get("job_contract")
                job = job if isinstance(job, Mapping) else {}
                candidate_fidelity = {
                    field: job.get(field)
                    for field in (
                        "n_basis",
                        "n_points",
                        "max_rows",
                        "selected_rows",
                        "work_rows",
                        "free_rows",
                        "scale_fit_split",
                        "selection_metric",
                        "split_contract_sha256",
                        "split_membership_sha256",
                        "declared_nuisance_contract_sha256",
                        "grid_physics_contract_sha256",
                        "selected_nuisance_sha256",
                        "bloch_kernel_contract_sha256",
                        "row_support_sha256",
                    )
                }
                if (
                    str(job.get("source_hkl_sha256") or "") != str(input_identity.get("source_hkl_sha256") or "")
                    or str(job.get("source_model_sha256") or "") != str(input_identity.get("source_model_sha256") or "")
                    or parse_hkl_transform_info(str(job.get("hkl_transform") or "identity")).canonical != candidate
                    or job.get("alternate_hand") is not False
                    or str(job.get("model_tag") or "").split(";", 1)[0] != "original"
                    or str(job.get("split_contract_sha256") or "")
                    != str(input_identity.get("split_contract_sha256") or "")
                    or str(job.get("split_membership_sha256") or "")
                    != str(input_identity.get("split_membership_sha256") or "")
                    or str(job.get("scale_fit_split") or "") != "work"
                    or str(job.get("selection_metric") or "") != "work_born_wR2"
                    or dict(job.get("declared_nuisance_contract") or {})
                    != dict(record.get("frozen_nuisance_contract") or {})
                    or str(job.get("declared_nuisance_contract_sha256") or "")
                    != str(record.get("frozen_nuisance_contract_sha256") or "")
                    or int(job.get("n_basis") or 0) != int(record.get("n_basis") or 0)
                    or int(job.get("n_points") or 0) != int(record.get("n_points") or 0)
                    or int(job.get("max_rows") or 0) != int(record.get("max_rows") or 0)
                    or not str(job.get("bloch_kernel_contract_sha256") or "")
                    or not str(job.get("row_support_sha256") or "")
                ):
                    missing_or_invalid.append(candidate)
                    continue
                if coarse_fidelity_contract is None:
                    coarse_fidelity_contract = candidate_fidelity
                elif candidate_fidelity != coarse_fidelity_contract:
                    missing_or_invalid.append(candidate)
                    continue
                replayed_coarse_scores[candidate] = float(recomputed_replay["score"])
        if missing_or_invalid:
            raise ValueError(
                "calibration-grade transform decision did not successfully score "
                "every metric-compatible proper candidate with the work-only matched-Born target: "
                + ", ".join(missing_or_invalid)
            )
        if len(nuisance_contract_hashes) != 1:
            raise ValueError(
                "calibration-grade transform candidates do not share one exact single-trial nuisance contract"
            )
        priority = {candidate: rank for rank, candidate in enumerate(full_proper_universe)}
        ranked_coarse_candidates = sorted(
            retained_candidates,
            key=lambda candidate: (
                replayed_coarse_scores[candidate],
                priority[candidate],
                candidate,
            ),
        )
        rerank = payload.get("rerank_records")
        rerank = rerank if isinstance(rerank, list) else []
        rerank_top = max(0, int(payload.get("rerank_top") or 0))
        expected_selected = ranked_coarse_candidates[0]
        if rerank:
            expected_admitted = ranked_coarse_candidates[: min(rerank_top, len(ranked_coarse_candidates))]
            rerank_by_transform = {
                parse_hkl_transform_info(str(record.get("transform") or "")).canonical: record
                for record in rerank
                if isinstance(record, Mapping) and record.get("transform")
            }
            if set(rerank_by_transform) != set(expected_admitted):
                raise ValueError(
                    "transform rerank admission does not reproduce from the fixed-priority coarse work-Born ranking"
                )
            replayed_rerank_scores: dict[str, float] = {}
            rerank_fidelity_contract: dict[str, object] | None = None
            for candidate in expected_admitted:
                record = rerank_by_transform[candidate]
                recorded_replay = record.get("artifact_score_replay")
                recorded_replay = recorded_replay if isinstance(recorded_replay, Mapping) else {}
                binding = recorded_replay.get("artifact_binding")
                binding = binding if isinstance(binding, Mapping) else {}
                recomputed_replay = replay_transform_work_born(Path(str(binding.get("summary_path") or "")))
                job = recomputed_replay.get("job_contract")
                job = job if isinstance(job, Mapping) else {}
                candidate_fidelity = {
                    field: job.get(field)
                    for field in (
                        "n_basis",
                        "n_points",
                        "max_rows",
                        "selected_rows",
                        "work_rows",
                        "free_rows",
                        "scale_fit_split",
                        "selection_metric",
                        "split_contract_sha256",
                        "split_membership_sha256",
                        "declared_nuisance_contract_sha256",
                        "grid_physics_contract_sha256",
                        "selected_nuisance_sha256",
                        "bloch_kernel_contract_sha256",
                        "row_support_sha256",
                    )
                }
                if (
                    record.get("status") != "ok"
                    or record.get("selection_target") != "matched_born_work_only_hand_invariant_v1"
                    or record.get("engine_trial_selection_metric") != "work_born_wR2"
                    or int(record.get("nuisance_trial_count") or 0) != 1
                    or int(record.get("prediction_geometry_trial_count") or 0) != 1
                    or dict(recorded_replay.get("artifact_binding") or {})
                    != dict(recomputed_replay.get("artifact_binding") or {})
                    or not math.isclose(
                        safe_float(record.get("score")),
                        safe_float(recomputed_replay.get("score")),
                        rel_tol=1e-12,
                        abs_tol=1e-14,
                    )
                    or str(job.get("source_hkl_sha256") or "") != str(input_identity.get("source_hkl_sha256") or "")
                    or str(job.get("source_model_sha256") or "") != str(input_identity.get("source_model_sha256") or "")
                    or parse_hkl_transform_info(str(job.get("hkl_transform") or "identity")).canonical != candidate
                    or job.get("alternate_hand") is not False
                    or str(job.get("model_tag") or "").split(";", 1)[0] != "original"
                    or str(job.get("split_contract_sha256") or "")
                    != str(input_identity.get("split_contract_sha256") or "")
                    or str(job.get("split_membership_sha256") or "")
                    != str(input_identity.get("split_membership_sha256") or "")
                    or dict(job.get("declared_nuisance_contract") or {})
                    != dict(record.get("frozen_nuisance_contract") or {})
                    or int(job.get("n_basis") or 0) != int(record.get("n_basis") or 0)
                    or int(job.get("n_points") or 0) != int(record.get("n_points") or 0)
                    or int(job.get("max_rows") or 0) != int(record.get("max_rows") or 0)
                    or not str(job.get("bloch_kernel_contract_sha256") or "")
                    or not str(job.get("row_support_sha256") or "")
                ):
                    raise ValueError(f"transform rerank artifact does not replay for {candidate}")
                if rerank_fidelity_contract is None:
                    rerank_fidelity_contract = candidate_fidelity
                elif candidate_fidelity != rerank_fidelity_contract:
                    raise ValueError(
                        "transform rerank candidates do not share one exact fidelity and row-support contract"
                    )
                replayed_rerank_scores[candidate] = float(recomputed_replay["score"])
            expected_selected = min(
                expected_admitted,
                key=lambda candidate: (
                    replayed_rerank_scores[candidate],
                    priority[candidate],
                    candidate,
                ),
            )
        if recorded != expected_selected:
            raise ValueError(
                "selected transform does not reproduce from hash-bound row "
                f"artifacts: recorded={recorded!r}, expected={expected_selected!r}"
            )
        if (
            selected.get("status") != "ok"
            or selected.get("selection_target") != "matched_born_work_only_hand_invariant_v1"
            or selected.get("engine_trial_selection_metric") != "work_born_wR2"
            or int(selected.get("nuisance_trial_count") or 0) != 1
            or int(selected.get("prediction_geometry_trial_count") or 0) != 1
            or str(selected.get("frozen_nuisance_contract_sha256") or "") not in nuisance_contract_hashes
        ):
            raise ValueError("selected transform is not bound to the validated single-trial work-Born scout contract")
    if require_calibration_grade and payload.get("metadata_candidate_elimination_used") is not False:
        raise ValueError(
            "transform decision used metadata candidate elimination; a calibration-grade "
            "decision must run every proper candidate through the work-only matched-Born target"
        )
    selection_target = str(payload.get("selection_target") or "")
    if require_calibration_grade and selection_target != "matched_born_work_only_hand_invariant_v1":
        raise ValueError(
            "transform decision selection_target must be "
            "'matched_born_work_only_hand_invariant_v1' before external calibration"
        )
    selected_info = parse_hkl_transform_info(recorded)
    orientation = selected.get("source_orientation_contract")
    if not isinstance(orientation, Mapping):
        claim = payload.get("selected_transform_claim_contract")
        claim = claim if isinstance(claim, Mapping) else {}
        orientation = claim.get("source_orientation_contract")
    orientation = orientation if isinstance(orientation, Mapping) else {}
    if require_proper_geometry and (
        not selected_info.proper
        or orientation.get("absolute_claim_geometry_ready") is not True
        or safe_float(orientation.get("source_basis_determinant"), -1.0) <= 0.0
        or safe_float(orientation.get("transformed_basis_determinant"), -1.0) <= 0.0
    ):
        raise ValueError(
            "transform decision is not bound to a proper operator and positive right-handed "
            "source/transformed orientation contract"
        )
    split_binding = payload.get("transform_hand_split_contract")
    split_binding = split_binding if isinstance(split_binding, Mapping) else {}
    if require_calibration_grade:
        from grond.legacy.engine.bloch_engine import (
            build_transform_split_contract,
            validate_transform_split_contract,
        )

        split_path = Path(str(split_binding.get("path") or "")).expanduser().resolve()
        if not split_path.is_file():
            raise ValueError(f"transform decision split contract is missing: {split_path}")
        if sha256_file(split_path) != str(split_binding.get("sha256") or ""):
            raise ValueError("transform decision split-contract file hash mismatch")
        split_payload = read_json(split_path)
        split_validation = validate_transform_split_contract(
            split_payload,
            source_hkl=Path(str(input_identity["source_hkl"])),
            current_transform=recorded,
        )
        split_validation.pop("masks", None)
        if (
            str(split_validation["membership_sha256"]) != str(input_identity["split_membership_sha256"])
            or str(split_binding.get("membership_sha256") or "") != str(input_identity["split_membership_sha256"])
            or list(split_validation["candidate_transforms"]) != retained_candidates
        ):
            raise ValueError(
                "transform decision split contract does not match its input identity and exact proper candidate set"
            )
        if str(input_identity["split_contract_sha256"]) != sha256_file(split_path):
            raise ValueError("transform decision input identity has a different split-contract hash")
        if str(input_identity["dataset"]) != str(payload.get("dataset") or ""):
            raise ValueError("transform decision dataset does not match its bound input identity")
        from grond.legacy.engine.bloch_engine import metric_filter_signed_axis_transforms

        recomputed_retained, recomputed_metric = metric_filter_signed_axis_transforms(
            Path(str(input_identity["source_hkl"])),
            Path(str(input_identity["source_model"])),
            generated,
            relative_tolerance=float(metric_filter_contract.get("relative_frobenius_tolerance")),
            dataset_label=str(input_identity["dataset"]),
        )
        if recomputed_retained != retained_candidates or str(
            recomputed_metric.get("contract_payload_sha256") or ""
        ) != str(metric_filter_contract.get("contract_payload_sha256") or ""):
            raise ValueError(
                "transform decision unit-cell metric filter cannot be reproduced from its bound source HKL/model"
            )
        recomputed_split = build_transform_split_contract(
            Path(str(input_identity["source_hkl"])),
            Path(str(input_identity["source_model"])),
            retained_candidates,
            dataset=str(input_identity["dataset"]),
            free_fraction=float(split_payload["free_fraction"]),
            seed=int(split_payload["split_seed"]),
            include_friedel=bool(split_payload["include_friedel"]),
        )
        split_reproducibility_fields = (
            "contract_payload_sha256",
            "membership_sha256",
            "orbit_group_by_row_sha256",
            "source_raw_hkl_sha256",
            "model_symmetry_rotations_sha256",
            "candidate_transforms_sha256",
        )
        split_mismatches = {
            field: {
                "recorded": split_payload.get(field),
                "recomputed": recomputed_split.get(field),
            }
            for field in split_reproducibility_fields
            if str(split_payload.get(field) or "") != str(recomputed_split.get(field) or "")
        }
        if split_mismatches:
            raise ValueError(
                "transform decision split contract cannot be reproduced from "
                f"the bound HKL/model/candidates/seed: {split_mismatches}"
            )
    else:
        split_validation = {}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "selected_transform": recorded,
        "selection_target": selection_target,
        "transform_determinant": selected_info.determinant,
        "source_orientation_contract": dict(orientation),
        "calibration_grade_validated": require_calibration_grade,
        "candidate_generation": dict(candidate_generation),
        "ranking_tie_break": dict(ranking_tie_break),
        "dataset_identity": input_identity,
        "transform_hand_split_contract": {
            **dict(split_validation),
            "path": str(split_binding.get("path") or ""),
            "sha256": str(split_binding.get("sha256") or ""),
        }
        if split_binding
        else {},
    }
