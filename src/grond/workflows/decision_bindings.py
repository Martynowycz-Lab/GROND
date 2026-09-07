"""Verify saved hand decisions against immutable acquisition, model and row identities."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import _json_contract_equal, read_json
from grond.hkl_transform import parse_hkl_transform_info
from grond.macromolecule import sha256_file
from grond.workflows.decision import safe_float
from grond.workflows.hand_row_contracts import _pinned_nuisance_mismatches
from grond.workflows.hand_statistics import _adjusted_hand_blocks_match
from grond.workflows.parity_evidence import build_common_sigma_hand_evidence, select_branch_symmetric_nuisance


def _validated_decision_identity(
    raw: object,
    *,
    label: str,
    require_complete: bool,
) -> dict[str, object]:
    """Validate one decision's exact dataset/model/raw-row identity."""

    identity = raw if isinstance(raw, Mapping) else {}
    fields = (
        "dataset",
        "setup_id",
        "setup_manifest_sha256",
        "physical_crystal_id",
        "acquisition_fingerprint",
        "source_hkl",
        "source_hkl_sha256",
        "source_model",
        "source_model_sha256",
        "split_contract_path",
        "split_contract_sha256",
        "split_membership_sha256",
    )
    normalized = {field: str(identity.get(field) or "").strip() for field in fields}
    if require_complete:
        missing = [field for field in fields if not normalized[field]]
        if missing:
            raise ValueError(f"{label} is missing exact dataset identity field(s): " + ", ".join(missing))
        normalized["setup_manifest_sha256"] = _validated_sha256_text(
            normalized["setup_manifest_sha256"],
            label=f"{label} setup-manifest SHA-256",
        )
        normalized["acquisition_fingerprint"] = _validated_sha256_text(
            normalized["acquisition_fingerprint"],
            label=f"{label} acquisition fingerprint",
        )
        hkl_path = Path(normalized["source_hkl"]).expanduser().resolve()
        model_path = Path(normalized["source_model"]).expanduser().resolve()
        split_path = Path(normalized["split_contract_path"]).expanduser().resolve()
        for artifact, path, expected_hash in (
            ("source HKL", hkl_path, normalized["source_hkl_sha256"]),
            ("source model", model_path, normalized["source_model_sha256"]),
            ("split contract", split_path, normalized["split_contract_sha256"]),
        ):
            if not path.is_file():
                raise ValueError(f"{label} {artifact} is missing: {path}")
            if sha256_file(path) != expected_hash:
                raise ValueError(f"{label} {artifact} hash mismatch: {path}")
        split_payload = read_json(split_path)
        if str(split_payload.get("membership_sha256") or "") != normalized["split_membership_sha256"]:
            raise ValueError(f"{label} split-membership hash mismatch")
        normalized["source_hkl"] = str(hkl_path)
        normalized["source_model"] = str(model_path)
        normalized["split_contract_path"] = str(split_path)
    return normalized


def _fixed_hand_decision_binding(
    path_text: str,
    expected_transform: str,
    *,
    require_decisive: bool = False,
) -> dict[str, object]:
    """Validate and hash a frozen matched-hand decision artifact."""

    if not path_text:
        return {}
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"matched hand decision does not exist: {path}")
    payload = read_json(path)
    selected_hand = str(payload.get("selected_hand") or payload.get("selected") or "").strip()
    if selected_hand not in {"original", "alternate"}:
        raise ValueError("matched hand decision must explicitly freeze selected_hand as original or alternate")
    comparison = payload.get("comparison_contract")
    comparison = comparison if isinstance(comparison, Mapping) else {}
    identity = _validated_decision_identity(
        comparison.get("dataset_identity"),
        label="matched hand decision",
        require_complete=require_decisive,
    )
    recorded_transform = str(comparison.get("hkl_transform") or payload.get("selected_transform") or "").strip()
    if recorded_transform:
        recorded = parse_hkl_transform_info(recorded_transform).canonical
        expected = parse_hkl_transform_info(expected_transform).canonical
        if recorded != expected:
            raise ValueError(
                "matched hand decision transform does not match the transform being canonicalized: "
                f"{recorded!r} != {expected!r}"
            )
    delta = safe_float(payload.get("score_delta_original_minus_alternate"), float("nan"))
    accept_delta = safe_float(payload.get("accept_delta"), float("nan"))
    comparison = payload.get("comparison_contract")
    comparison = comparison if isinstance(comparison, Mapping) else {}
    common_sigma_evidence: dict[str, object] = {}
    if require_decisive:
        if str(payload.get("method") or "") == "nakane_negative_thickness_elastic_pair_v1":
            raise ValueError(
                "current final-elastic negative-thickness hand artifacts are conditional hand results and are not "
                "yet accepted by the legacy parity-calibration replay. Preserve the setup and known-chemistry "
                "assignment externally; do not claim a named configuration from this residual sign alone."
            )
        reasons: list[str] = []
        if str(payload.get("quality_gate_status") or "") != "passed":
            reasons.append("quality gate did not pass")
        if payload.get("warning"):
            reasons.append(f"decision warning: {payload.get('warning')}")
        if comparison.get("row_selection_match") is not True:
            reasons.append("original/alternate row selections do not match")
        # Free observations are diagnostic only; named sign calibration still
        # requires exact physical contracts and decisive all-row evidence below.
        recorded_nuisance_selection = comparison.get("branch_symmetric_nuisance_selection")
        recorded_nuisance_selection = (
            recorded_nuisance_selection if isinstance(recorded_nuisance_selection, Mapping) else {}
        )
        original_grid_binding = recorded_nuisance_selection.get("original_artifact_binding")
        original_grid_binding = original_grid_binding if isinstance(original_grid_binding, Mapping) else {}
        alternate_grid_binding = recorded_nuisance_selection.get("alternate_artifact_binding")
        alternate_grid_binding = alternate_grid_binding if isinstance(alternate_grid_binding, Mapping) else {}
        try:
            recomputed_nuisance_selection = select_branch_symmetric_nuisance(
                Path(str(original_grid_binding.get("summary_path") or "")),
                Path(str(alternate_grid_binding.get("summary_path") or "")),
            )
        except Exception as exc:
            reasons.append(f"branch-symmetric nuisance selection does not replay: {exc}")
            recomputed_nuisance_selection = {}
        if recomputed_nuisance_selection:
            for field in (
                "status",
                "selection_rule",
                "selection_objective",
                "hand_label_invariant",
                "free_rows_used_for_selection",
                "known_configuration_used",
                "candidate_count",
                "candidate_grid_sha256",
                "selected_candidate_sha256",
                "selected_paired_mean_work_dyn_wR2",
                "paired_candidate_records",
                "original_artifact_binding",
                "alternate_artifact_binding",
                "original_job_contract",
                "alternate_job_contract",
            ):
                if not _json_contract_equal(
                    recorded_nuisance_selection.get(field),
                    recomputed_nuisance_selection.get(field),
                ):
                    reasons.append(f"branch-symmetric nuisance selection mismatch for {field}")
            selected_nuisance_mismatches = _pinned_nuisance_mismatches(
                dict(recomputed_nuisance_selection.get("selected_nuisance") or {}),
                dict(recorded_nuisance_selection.get("selected_nuisance") or {}),
            )
            if selected_nuisance_mismatches:
                reasons.append(
                    "branch-symmetric selected nuisance mismatch: " + "; ".join(selected_nuisance_mismatches)
                )
        recorded_evidence = comparison.get("common_sigma_hand_evidence")
        recorded_evidence = recorded_evidence if isinstance(recorded_evidence, Mapping) else {}
        original_binding = recorded_evidence.get("original_artifact_binding")
        original_binding = original_binding if isinstance(original_binding, Mapping) else {}
        alternate_binding = recorded_evidence.get("alternate_artifact_binding")
        alternate_binding = alternate_binding if isinstance(alternate_binding, Mapping) else {}
        minimum_abs_z = safe_float(
            recorded_evidence.get("minimum_abs_adjusted_z"),
            float("nan"),
        )
        try:
            recomputed_evidence = build_common_sigma_hand_evidence(
                Path(str(original_binding.get("summary_path") or "")),
                Path(str(alternate_binding.get("summary_path") or "")),
                minimum_abs_adjusted_z=minimum_abs_z,
            )
        except Exception as exc:
            reasons.append(f"common-sigma hand evidence does not replay: {exc}")
            recomputed_evidence = {}
        if recomputed_evidence:
            common_sigma_evidence = dict(recomputed_evidence)
            for field in (
                "status",
                "comparison_rule",
                "common_sigma_rule",
                "common_sigma_sha256",
                "common_sigma_row_count",
                "scale_design_rule",
                "scale_design_sha256",
                "original_artifact_binding",
                "alternate_artifact_binding",
                "original_job_contract",
                "alternate_job_contract",
                "residual_preferred_branch",
                "branch_aligned_adjusted_z",
                "minimum_abs_adjusted_z",
                "branch_z_sign_agreement",
                "adjusted_z_threshold_passed",
                "known_configuration_used",
            ):
                if not _json_contract_equal(
                    recorded_evidence.get(field),
                    recomputed_evidence.get(field),
                ):
                    reasons.append(f"common-sigma hand evidence mismatch for {field}")
            recomputed_delta = safe_float(
                recomputed_evidence.get("score_delta_original_minus_alternate"),
                float("nan"),
            )
            if not math.isclose(
                delta,
                recomputed_delta,
                rel_tol=1e-12,
                abs_tol=1e-14,
            ):
                reasons.append("recorded hand score delta does not match common-sigma replay")
            if selected_hand != str(recomputed_evidence.get("residual_preferred_branch") or ""):
                reasons.append("selected hand does not match the common-sigma residual preference")
            original_job = recomputed_evidence.get("original_job_contract")
            original_job = original_job if isinstance(original_job, Mapping) else {}
            expected_canonical = parse_hkl_transform_info(expected_transform).canonical
            try:
                job_transform = parse_hkl_transform_info(str(original_job.get("hkl_transform") or "")).canonical
            except ValueError:
                job_transform = ""
            if (
                str(original_job.get("source_hkl_sha256") or "") != str(identity.get("source_hkl_sha256") or "")
                or str(original_job.get("source_model_sha256") or "") != str(identity.get("source_model_sha256") or "")
                or str(original_job.get("split_contract_sha256") or "")
                != str(identity.get("split_contract_sha256") or "")
                or str(original_job.get("split_membership_sha256") or "")
                != str(identity.get("split_membership_sha256") or "")
                or job_transform != expected_canonical
            ):
                reasons.append(
                    "common-sigma hand job contract is not bound to the decision HKL/model/transform/split identity"
                )
            if recomputed_nuisance_selection:
                pinned_mismatches = _pinned_nuisance_mismatches(
                    dict(recomputed_nuisance_selection.get("selected_nuisance") or {}),
                    dict(original_job.get("selected_nuisance") or {}),
                )
                if pinned_mismatches:
                    reasons.append(
                        "final matched-hand evidence does not use the "
                        "branch-symmetric selected nuisance: " + "; ".join(pinned_mismatches)
                    )
            recorded_z = recorded_evidence.get("adjusted_hand_z_original_favored")
            recorded_z = recorded_z if isinstance(recorded_z, Mapping) else {}
            recomputed_z = recomputed_evidence.get("adjusted_hand_z_original_favored")
            recomputed_z = recomputed_z if isinstance(recomputed_z, Mapping) else {}
            for split in ("all", "free"):
                recorded_block = recorded_z.get(split)
                recorded_block = recorded_block if isinstance(recorded_block, Mapping) else {}
                recomputed_block = recomputed_z.get(split)
                recomputed_block = recomputed_block if isinstance(recomputed_block, Mapping) else {}
                if not _adjusted_hand_blocks_match(
                    recorded_block,
                    recomputed_block,
                    allow_unavailable=split == "free",
                ):
                    reasons.append(f"common-sigma adjusted hand Z mismatch for {split} rows")
            all_z = recomputed_z.get("all")
            all_z = all_z if isinstance(all_z, Mapping) else {}
            if (
                all_z.get("status") != "ok"
                or int(all_z.get("n") or 0) <= 0
                or recomputed_evidence.get("branch_z_sign_agreement") is not True
                or recomputed_evidence.get("adjusted_z_threshold_passed") is not True
            ):
                reasons.append(
                    "common-sigma Klar-Palatinus adjusted hand Z is not "
                    "finite, sign-consistent, and above its frozen threshold"
                )
        if not math.isfinite(delta) or not math.isfinite(accept_delta):
            reasons.append("hand score delta or acceptance threshold is nonfinite")
        elif selected_hand == "alternate" and not delta > accept_delta:
            reasons.append(f"alternate branch delta {delta:.6g} does not exceed +{accept_delta:.6g}")
        elif selected_hand == "original" and not delta < -accept_delta:
            reasons.append(f"original branch delta {delta:.6g} does not exceed -{accept_delta:.6g}")
        if reasons:
            raise ValueError(
                "matched hand decision is not decisive enough for parity calibration: " + "; ".join(reasons)
            )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "selected_hand": selected_hand,
        "recorded_transform": recorded_transform,
        "score_delta_original_minus_alternate": (delta if math.isfinite(delta) else None),
        "accept_delta": accept_delta if math.isfinite(accept_delta) else None,
        "decisive_for_calibration": require_decisive,
        "dataset_identity": identity,
        "common_sigma_hand_evidence": common_sigma_evidence,
    }


def _external_parity_calibration_binding(path_text: str) -> dict[str, object]:
    """Bind an external parity-calibration artifact without interpreting chemistry."""

    if not path_text:
        return {}
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"external parity-calibration reference does not exist: {path}")
    payload = read_json(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "payload": payload,
    }


def _require_matching_decision_identities(
    transform: Mapping[str, object],
    hand: Mapping[str, object],
    *,
    expected_dataset_id: str,
    label: str,
) -> dict[str, object]:
    """Require transform and hand evidence from the same exact crystal bytes."""

    transform_identity = transform.get("dataset_identity")
    transform_identity = transform_identity if isinstance(transform_identity, Mapping) else {}
    hand_identity = hand.get("dataset_identity")
    hand_identity = hand_identity if isinstance(hand_identity, Mapping) else {}
    compared_fields = (
        "dataset",
        "setup_id",
        "setup_manifest_sha256",
        "physical_crystal_id",
        "acquisition_fingerprint",
        "source_hkl_sha256",
        "source_model_sha256",
        "split_contract_sha256",
        "split_membership_sha256",
    )
    mismatches = {
        field: {
            "transform": transform_identity.get(field),
            "hand": hand_identity.get(field),
        }
        for field in compared_fields
        if str(transform_identity.get(field) or "") != str(hand_identity.get(field) or "")
    }
    if mismatches:
        raise ValueError(
            f"{label} transform and matched-hand decisions do not bind the same "
            f"dataset/model/split identity: {mismatches}"
        )
    dataset = str(transform_identity.get("dataset") or "")
    if dataset != str(expected_dataset_id):
        raise ValueError(
            f"{label} dataset ID {dataset!r} does not match the requested dataset ID {str(expected_dataset_id)!r}"
        )
    return dict(transform_identity)


def _validated_sha256_text(value: object, *, label: str) -> str:
    """Validate a caller-supplied immutable external provenance fingerprint."""

    text = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{label} must be exactly 64 hexadecimal SHA-256 characters")
    return text
