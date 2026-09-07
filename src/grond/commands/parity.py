"""Canonicalize hand conventions and create or apply external parity-calibration records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

from grond.hkl_transform import (
    absolute_configuration_claim_contract,
    parse_hkl_transform_info,
    validate_external_parity_calibration,
)
from grond.macromolecule import sha256_file
from grond.workflows.decision import write_json
from grond.workflows.decision_bindings import (
    _external_parity_calibration_binding,
    _fixed_hand_decision_binding,
    _require_matching_decision_identities,
    _validated_sha256_text,
)
from grond.workflows.transform_binding import _fixed_transform_decision_binding


def cmd_canonicalize_hand(args: argparse.Namespace) -> int:
    """Canonicalize ``(model, T)`` to an explicit model plus proper transform."""

    from grond.legacy.engine.make_res_variants import write_alternate_hand_variant
    from grond.legacy.engine.shelx_hkl import (
        audit_fixed_width_hklf4,
        write_sign_paired_hklf4,
    )

    source = Path(args.input_res).expanduser().resolve()
    output = Path(args.output_res).expanduser().resolve()
    source_hkl = Path(args.input_hkl).expanduser().resolve()
    output_hkl = Path(args.output_hkl).expanduser().resolve()
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else output.with_suffix(output.suffix + ".hand_canonicalization.json")
    )
    if not source.is_file():
        raise ValueError(f"input RES does not exist: {source}")
    if not source_hkl.is_file():
        raise ValueError(f"input HKLF4 does not exist: {source_hkl}")
    if source == output or source_hkl == output_hkl:
        raise ValueError("input and output RES/HKL paths must differ")
    for path in (output, output_hkl, manifest_path):
        if path.exists() and not args.overwrite:
            raise ValueError(f"refusing to overwrite existing artifact without --overwrite: {path}")

    input_info = parse_hkl_transform_info(args.hkl_transform)
    if input_info.proper and not args.allow_proper_noop:
        raise ValueError(
            f"transform {input_info.canonical!r} is already proper; use --allow-proper-noop "
            "only when an audited copy/calibration manifest is intentionally required"
        )
    hand_decision = _fixed_hand_decision_binding(
        args.matched_hand_decision,
        input_info.canonical,
    )
    transform_decision = _fixed_transform_decision_binding(
        args.transform_decision,
        input_info.canonical,
    )
    output_transform = input_info.proper_canonical_transform
    output_info = parse_hkl_transform_info(output_transform)
    if not output_info.proper:
        raise RuntimeError("internal error: canonicalized output transform is not proper")

    output.parent.mkdir(parents=True, exist_ok=True)
    output_hkl.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".grond-hand-", dir=output.parent) as temp_dir:
        temp_res = Path(temp_dir) / output.name
        # Keep the HKL temporary on its destination filesystem.
        with tempfile.TemporaryDirectory(prefix=".grond-hand-hkl-", dir=output_hkl.parent) as temp_hkl_dir:
            temp_hkl = Path(temp_hkl_dir) / output_hkl.name
            if input_info.proper:
                shutil.copy2(source, temp_res)
                shutil.copy2(source_hkl, temp_hkl)
                source_hkl_audit = audit_fixed_width_hklf4(source_hkl)
                output_hkl_audit = audit_fixed_width_hklf4(temp_hkl)
                if source_hkl_audit != output_hkl_audit:
                    raise ValueError("proper no-op copy changed the HKLF4 row/value contract")
                hkl_audit = {
                    "source": source_hkl_audit,
                    "output": output_hkl_audit,
                    "reflection_count_preserved": True,
                    "observation_payload_preserved": True,
                    "index_operation": "identity",
                }
                operation = "audited_proper_noop_copy"
                coordinate_inversion_applied = False
            else:
                write_alternate_hand_variant(source, temp_res)
                hkl_audit = write_sign_paired_hklf4(source_hkl, temp_hkl)
                operation = "invert_coordinates_and_negate_hkl_transform"
                coordinate_inversion_applied = True
            os.replace(temp_res, output)
            os.replace(temp_hkl, output_hkl)
    claim_contract = absolute_configuration_claim_contract(output_transform)
    manifest = {
        "schema_version": 1,
        "kind": "grond_hand_transform_canonicalization",
        "operation": operation,
        "gauge_identity": "(model,T) == (inverted_model,-T)",
        "known_configuration_used_for_candidate_generation_or_ranking": False,
        "input": {
            "res": str(source),
            "res_sha256": sha256_file(source),
            "hkl": str(source_hkl),
            "hkl_sha256": sha256_file(source_hkl),
            "transform": input_info.to_dict(),
        },
        "output": {
            "res": str(output),
            "res_sha256": sha256_file(output),
            "hkl": str(output_hkl),
            "hkl_sha256": sha256_file(output_hkl),
            "transform": output_info.to_dict(),
        },
        "hklf4_gauge_pair_audit": hkl_audit,
        "coordinate_inversion_applied": coordinate_inversion_applied,
        "symmetry_translation_inversion": (
            "materialized_and_validated" if coordinate_inversion_applied else "not_required"
        ),
        "matched_hand_decision": hand_decision,
        "transform_decision": transform_decision,
        "external_parity_calibration": {},
        "configuration_claim_contract": claim_contract,
        "cif_status": (
            "not_generated; regenerate CIF by replaying the canonical RES/HKL pair through "
            "the configured endpoint program"
        ),
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2))
    return 0


def cmd_create_parity_calibration(args: argparse.Namespace) -> int:
    """Create a reserved calibrant manifest after blinded decisions are frozen."""

    if not args.reserved_from_validation:
        raise ValueError("a parity calibrant must be explicitly --reserved-from-validation")
    transform = _fixed_transform_decision_binding(
        args.transform_decision,
        require_proper_geometry=True,
        require_calibration_grade=True,
    )
    fixed_transform = str(transform["selected_transform"])
    hand = _fixed_hand_decision_binding(
        args.matched_hand_decision,
        fixed_transform,
        require_decisive=True,
    )
    calibrant_identity = _require_matching_decision_identities(
        transform,
        hand,
        expected_dataset_id=str(args.calibrant_dataset_id),
        label="calibrant",
    )
    calibrant_physical_crystal_id = str(args.calibrant_physical_crystal_id).strip()
    if not calibrant_physical_crystal_id:
        raise ValueError("calibrant physical-crystal ID must not be empty")
    calibrant_acquisition_fingerprint = _validated_sha256_text(
        args.calibrant_acquisition_fingerprint,
        label="calibrant acquisition fingerprint",
    )
    setup_manifest_sha256 = _validated_sha256_text(
        args.setup_manifest_sha256,
        label="setup manifest SHA-256",
    )
    frozen_values = {
        "setup_id": str(calibrant_identity["setup_id"]),
        "setup_manifest_sha256": str(calibrant_identity["setup_manifest_sha256"]),
        "physical_crystal_id": str(calibrant_identity["physical_crystal_id"]),
        "acquisition_fingerprint": str(calibrant_identity["acquisition_fingerprint"]),
    }
    requested_values = {
        "setup_id": str(args.setup_id),
        "setup_manifest_sha256": setup_manifest_sha256,
        "physical_crystal_id": calibrant_physical_crystal_id,
        "acquisition_fingerprint": calibrant_acquisition_fingerprint,
    }
    if requested_values != frozen_values:
        raise ValueError(
            "calibrant CLI identity does not match the identity frozen before "
            f"transform/hand selection: requested={requested_values}, "
            f"frozen={frozen_values}"
        )
    output = Path(args.out).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise ValueError(f"refusing to overwrite parity calibration without --overwrite: {output}")
    orientation = transform["source_orientation_contract"]
    orientation = orientation if isinstance(orientation, Mapping) else {}
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "grond_external_parity_calibration",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "setup_id": frozen_values["setup_id"],
        "calibrant_dataset_id": str(args.calibrant_dataset_id),
        "calibrant_physical_crystal_id": frozen_values["physical_crystal_id"],
        "calibrant_acquisition_fingerprint": frozen_values["acquisition_fingerprint"],
        "setup_manifest_sha256": frozen_values["setup_manifest_sha256"],
        "fixed_transform": fixed_transform,
        "fixed_transform_determinant": 1,
        "transform_decision_path": str(transform["path"]),
        "transform_decision_sha256": str(transform["sha256"]),
        "matched_hand_decision_path": str(hand["path"]),
        "matched_hand_decision_sha256": str(hand["sha256"]),
        "expected_known_configuration": str(args.expected_known_configuration),
        "expected_known_branch": str(args.expected_known_branch),
        "observed_branch": str(hand["selected_hand"]),
        "calibrant_identity": calibrant_identity,
        "observed_sign_mapping": {"original": 1, "alternate": -1},
        "reserved_from_validation": True,
        "source_orientation_determinant": orientation.get("source_basis_determinant"),
        "transformed_orientation_determinant": orientation.get("transformed_basis_determinant"),
        "source_orientation_right_handed": bool(orientation.get("right_handed_source")),
        "selection_independence": {
            "calibration_used_for_transform_candidate_generation": False,
            "known_configuration_used_for_transform_ranking": False,
            "calibration_attached_after_transform_and_hand_frozen": True,
        },
    }
    validation = validate_external_parity_calibration(
        payload,
        setup_id=str(args.setup_id),
        proper_transform=fixed_transform,
        transform_decision_sha256=str(transform["sha256"]),
        matched_hand_decision_sha256=str(hand["sha256"]),
        matched_selected_hand=str(hand["selected_hand"]),
    )
    payload["validation"] = validation
    payload["setup_parity_multiplier"] = validation["setup_parity_multiplier"]
    payload["manifest_payload_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    print(json.dumps({"parity_calibration": str(output), **payload}, indent=2))
    return 0


def cmd_apply_parity_calibration(args: argparse.Namespace) -> int:
    """Bind one reserved setup calibrant to a different validation decision."""

    calibration_binding = _external_parity_calibration_binding(args.calibration)
    payload = calibration_binding.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("parity calibration manifest must contain a JSON object")
    if str(payload.get("setup_id") or "") != str(args.setup_id):
        raise ValueError("validation setup ID does not match the parity calibration")
    setup_manifest_sha256 = _validated_sha256_text(
        args.setup_manifest_sha256,
        label="validation setup manifest SHA-256",
    )
    if setup_manifest_sha256 != str(payload.get("setup_manifest_sha256") or ""):
        raise ValueError("validation setup-manifest hash does not match the reserved calibrant")
    validation_physical_crystal_id = str(args.validation_physical_crystal_id).strip()
    if not validation_physical_crystal_id:
        raise ValueError("validation physical-crystal ID must not be empty")
    if validation_physical_crystal_id == str(payload.get("calibrant_physical_crystal_id") or ""):
        raise ValueError("reserved calibrant physical crystal cannot also be used as validation")
    validation_acquisition_fingerprint = _validated_sha256_text(
        args.validation_acquisition_fingerprint,
        label="validation acquisition fingerprint",
    )
    if validation_acquisition_fingerprint == str(payload.get("calibrant_acquisition_fingerprint") or ""):
        raise ValueError(
            "reserved calibrant acquisition/raw-image fingerprint cannot be reused "
            "as validation after reintegration or renaming"
        )
    manifest_hash = str(payload.get("manifest_payload_sha256") or "")
    manifest_without_hash = dict(payload)
    manifest_without_hash.pop("manifest_payload_sha256", None)
    current_manifest_hash = hashlib.sha256(
        json.dumps(
            manifest_without_hash,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if not manifest_hash or manifest_hash != current_manifest_hash:
        raise ValueError("parity calibration manifest payload hash mismatch")
    if str(payload.get("calibrant_dataset_id") or "") == str(args.validation_dataset_id):
        raise ValueError("reserved calibrant dataset cannot also be used as a validation dataset")
    calibrant_transform = _fixed_transform_decision_binding(
        str(payload.get("transform_decision_path") or ""),
        str(payload.get("fixed_transform") or ""),
        require_proper_geometry=True,
        require_calibration_grade=True,
    )
    calibrant_hand = _fixed_hand_decision_binding(
        str(payload.get("matched_hand_decision_path") or ""),
        str(calibrant_transform["selected_transform"]),
        require_decisive=True,
    )
    if str(calibrant_transform["sha256"]) != str(payload.get("transform_decision_sha256") or ""):
        raise ValueError("reserved calibrant transform-decision hash mismatch")
    if str(calibrant_hand["sha256"]) != str(payload.get("matched_hand_decision_sha256") or ""):
        raise ValueError("reserved calibrant matched-hand-decision hash mismatch")
    calibrant_identity = _require_matching_decision_identities(
        calibrant_transform,
        calibrant_hand,
        expected_dataset_id=str(payload.get("calibrant_dataset_id") or ""),
        label="reserved calibrant",
    )
    recorded_identity = payload.get("calibrant_identity")
    recorded_identity = recorded_identity if isinstance(recorded_identity, Mapping) else {}
    for field in (
        "dataset",
        "setup_id",
        "setup_manifest_sha256",
        "physical_crystal_id",
        "acquisition_fingerprint",
        "source_hkl_sha256",
        "source_model_sha256",
        "split_contract_sha256",
        "split_membership_sha256",
    ):
        if str(recorded_identity.get(field) or "") != str(calibrant_identity.get(field) or ""):
            raise ValueError(f"parity calibration calibrant identity mismatch for {field}")
    if str(payload.get("observed_branch") or "") != str(calibrant_hand["selected_hand"]):
        raise ValueError(
            "parity calibration observed branch does not match the re-parsed frozen calibrant hand decision"
        )
    calibration_validation = validate_external_parity_calibration(
        payload,
        setup_id=str(args.setup_id),
        proper_transform=str(payload.get("fixed_transform") or ""),
        transform_decision_sha256=str(calibrant_transform["sha256"]),
        matched_hand_decision_sha256=str(calibrant_hand["sha256"]),
        matched_selected_hand=str(calibrant_hand["selected_hand"]),
    )
    if int(payload.get("setup_parity_multiplier") or 0) != int(calibration_validation["setup_parity_multiplier"]):
        raise ValueError(
            "parity calibration recorded setup multiplier does not match the recomputed calibrant branch mapping"
        )

    validation_transform = _fixed_transform_decision_binding(
        args.transform_decision,
        require_proper_geometry=True,
        require_calibration_grade=True,
    )
    validation_hand = _fixed_hand_decision_binding(
        args.matched_hand_decision,
        str(validation_transform["selected_transform"]),
        require_decisive=True,
    )
    validation_identity = _require_matching_decision_identities(
        validation_transform,
        validation_hand,
        expected_dataset_id=str(args.validation_dataset_id),
        label="validation",
    )
    validation_frozen_values = {
        "setup_id": str(validation_identity["setup_id"]),
        "setup_manifest_sha256": str(validation_identity["setup_manifest_sha256"]),
        "physical_crystal_id": str(validation_identity["physical_crystal_id"]),
        "acquisition_fingerprint": str(validation_identity["acquisition_fingerprint"]),
    }
    validation_requested_values = {
        "setup_id": str(args.setup_id),
        "setup_manifest_sha256": setup_manifest_sha256,
        "physical_crystal_id": validation_physical_crystal_id,
        "acquisition_fingerprint": validation_acquisition_fingerprint,
    }
    if validation_requested_values != validation_frozen_values:
        raise ValueError(
            "validation CLI identity does not match the identity frozen before "
            f"transform/hand selection: requested={validation_requested_values}, "
            f"frozen={validation_frozen_values}"
        )
    if str(validation_transform["sha256"]) == str(calibrant_transform["sha256"]):
        raise ValueError("reserved calibrant transform decision cannot be reused as validation")
    if str(validation_hand["sha256"]) == str(calibrant_hand["sha256"]):
        raise ValueError("reserved calibrant matched-hand decision cannot be reused as validation")
    if str(validation_identity["source_hkl_sha256"]) == str(calibrant_identity["source_hkl_sha256"]):
        raise ValueError(
            "reserved calibrant source reflection content cannot be reused as validation under a different dataset name"
        )
    setup_multiplier = int(calibration_validation["setup_parity_multiplier"])
    branch_mapping = calibration_validation["observed_sign_mapping"]
    branch_mapping = branch_mapping if isinstance(branch_mapping, Mapping) else {}
    observed_branch = str(validation_hand["selected_hand"])
    branch_sign = int(branch_mapping[observed_branch])
    calibrated_coordinate_hand_factor = setup_multiplier * branch_sign
    calibration_reference = f"{calibration_binding['path']}#sha256={calibration_binding['sha256']}"
    claim = absolute_configuration_claim_contract(
        str(validation_transform["selected_transform"]),
        externally_calibrated_parity=True,
        calibration_reference=calibration_reference,
        calibration_validation_status="validated",
    )
    binding = {
        "schema_version": 1,
        "kind": "grond_applied_parity_calibration",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "setup_id": validation_frozen_values["setup_id"],
        "validation_dataset_id": str(args.validation_dataset_id),
        "validation_physical_crystal_id": validation_frozen_values["physical_crystal_id"],
        "validation_acquisition_fingerprint": validation_frozen_values["acquisition_fingerprint"],
        "setup_manifest_sha256": validation_frozen_values["setup_manifest_sha256"],
        "reserved_calibrant_dataset_id": calibration_validation["calibrant_dataset_id"],
        "calibration_manifest": {
            "path": str(calibration_binding["path"]),
            "sha256": str(calibration_binding["sha256"]),
        },
        "validation_transform_decision": validation_transform,
        "validation_matched_hand_decision": validation_hand,
        "validation_identity": validation_identity,
        "setup_parity_multiplier": setup_multiplier,
        "validation_observed_branch": observed_branch,
        "validation_branch_sign": branch_sign,
        "calibrated_coordinate_hand_factor": calibrated_coordinate_hand_factor,
        "configuration_relative_to_validation_input_coordinates": (
            "as_supplied" if calibrated_coordinate_hand_factor == 1 else "inverted"
        ),
        "configuration_claim_contract": claim,
        "known_configuration_used_for_validation_selection": False,
        "reserved_calibrant_excluded_from_validation": True,
    }
    output = Path(args.out).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise ValueError(f"refusing to overwrite applied parity calibration without --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, binding)
    print(json.dumps({"applied_parity_calibration": str(output), **binding}, indent=2))
    return 0
