"""Shared row-contract dynrow contract for the XDS/DIALS Bloch pipeline.

The scientific point of this module is narrow: observation builders and native
audits must carry the same frozen row identity and Bloch-kernel assumptions
that were accepted for the audited signed-row production path.  It is not a place to add
new Hamiltonian options.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

try:
    from .bloch_engine import DEFAULT_MAX_SUPPORT_SG
except ImportError:  # pragma: no cover - direct script execution
    from bloch_engine import DEFAULT_MAX_SUPPORT_SG

DYNROW_SCHEMA_VERSION = "row_contract_signed_dynrow_v1.0"
FROZEN_ENGINE_CONTRACT_VERSION = "audited_signed_row_simple_cycles_v1.0"


FROZEN_ENGINE_CONTRACT: dict[str, Any] = {
    "version": FROZEN_ENGINE_CONTRACT_VERSION,
    "scope": "production_default_for_xds_bloch_row_contract_dynrow_audits",
    "row_identity": {
        "signed_hkl_required": True,
        "simulation_hkl_source": "exact signed observation row",
        "canonicalization_for_simulation": False,
        "asu_mapping_for_simulation": False,
        "friedel_merge_for_simulation": False,
        "canonicalization_allowed_for": ["reporting", "diagnostics"],
    },
    "hamiltonian": {
        "matrix_convention": "simple_cycles",
        "diagonal": "H_ii = S_i",
        "offdiagonal": "H_ij = U(h_j - h_i) / (2*k)",
        "offdiagonal_sign": "basis[j] - basis[i]",
        "propagation": "psi(t) = exp(2*pi*i*H*t) psi(0)",
        "solver_default": "scipy.linalg.eigh for elastic Hermitian H",
    },
    "profile": {
        "mode": "target_sg_to_angle",
        "quadrature": "uniform_target_sg",
        "accumulation": "summed_detector_subexposures",
        "legacy_accumulation": "normalized_mean",
        "default_sg_window_inv_A": 0.006,
        "default_n_points": 20,
        "basis_sg_recompute": "recompute every basis-beam Sg at every solved target-Sg angle",
    },
    "basis": {
        "mode": "direct_target_topup",
        "first_beams": ["direct_000", "exact_signed_target_hkl"],
        "topup_d_min_A": 0.8,
        "max_support_sg_inv_A": DEFAULT_MAX_SUPPORT_SG,
        "score": "abs(U_g)^2 / (abs(Sg) + 1e-5)",
    },
    "thickness": {
        "model": "uniform_gradient",
        "referenceed_reference_t0_A": 1245.053,
        "referenceed_reference_grad_frac": 0.4,
        "referenceed_reference_points": 9,
    },
    "disabled_by_default": [
        "row contract measured-intensity support ranking",
        "frame-level measured-plus-topup basis",
        "raw pre-scale EXTI",
        "post-scale EXTI",
        "Lorentz correction modes",
        "profile cutoff masks",
        "non-Hermitian absorption modes",
    ],
}


REQUIRED_DYNROW_FIELDS = (
    "dynrow_schema_version",
    "engine_contract_version",
    "obs_id",
    "dataset_id",
    "h",
    "k",
    "l",
    "signed_h",
    "signed_k",
    "signed_l",
    "hkl",
    "signed_hkl_source",
    "simulation_hkl",
    "simulation_hkl_policy",
    "frame_id",
    "observation_group_id",
    "group_id",
    "target_alpha0_deg",
    "target_zero_sg_angle_deg",
    "target_sg_at_zcal_inv_A",
    "xcal_px",
    "ycal_px",
    "zcal_frame",
    "Iobs_raw",
    "sigma_Iobs_raw",
    "primary_intensity_field",
    "primary_sigma_field",
    "quality",
)


def contract_metadata(source: str) -> dict[str, Any]:
    return {
        "dynrow_schema_version": DYNROW_SCHEMA_VERSION,
        "engine_contract_version": FROZEN_ENGINE_CONTRACT_VERSION,
        "source": source,
        "frozen_engine_contract": FROZEN_ENGINE_CONTRACT,
    }


def frame_contract_fields(source: str) -> dict[str, Any]:
    return {
        "dynrow_schema_version": DYNROW_SCHEMA_VERSION,
        "engine_contract_version": FROZEN_ENGINE_CONTRACT_VERSION,
        "frame_contract_source": source,
        "row_identity_policy": FROZEN_ENGINE_CONTRACT["row_identity"],
    }


def observation_contract_fields(source: str, h: int, k: int, l: int) -> dict[str, Any]:
    signed = [int(h), int(k), int(l)]
    return {
        "dynrow_schema_version": DYNROW_SCHEMA_VERSION,
        "engine_contract_version": FROZEN_ENGINE_CONTRACT_VERSION,
        "row_identity_policy": "exact_signed_hkl_only_no_asu_no_friedel_no_p222_canonicalization",
        "simulation_hkl": signed,
        "simulation_hkl_source": source,
        "simulation_hkl_policy": "use exact signed observation HKL for target, basis selection, and U(h_j-h_i) lookup",
        "canonical_hkl_reporting_only": None,
    }


def validate_dynrow_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_DYNROW_FIELDS:
        if key not in record:
            errors.append(f"missing required field {key}")

    try:
        signed = [int(record["signed_h"]), int(record["signed_k"]), int(record["signed_l"])]
        hkl = [int(v) for v in record["hkl"]]
        scalar = [int(record["h"]), int(record["k"]), int(record["l"])]
        sim = [int(v) for v in record.get("simulation_hkl", [])]
        if scalar != signed:
            errors.append(f"h/k/l fields {scalar} do not match signed_h/k/l {signed}")
        if hkl != signed:
            errors.append(f"hkl field {hkl} does not match signed_h/k/l {signed}")
        if sim != signed:
            errors.append(f"simulation_hkl {sim} does not match signed_h/k/l {signed}")
    except Exception as exc:
        errors.append(f"could not validate signed HKL identity: {exc}")

    if record.get("dynrow_schema_version") != DYNROW_SCHEMA_VERSION:
        errors.append(
            f"dynrow_schema_version {record.get('dynrow_schema_version')!r} != {DYNROW_SCHEMA_VERSION!r}"
        )
    if record.get("engine_contract_version") != FROZEN_ENGINE_CONTRACT_VERSION:
        errors.append(
            f"engine_contract_version {record.get('engine_contract_version')!r} != {FROZEN_ENGINE_CONTRACT_VERSION!r}"
        )
    policy = str(record.get("simulation_hkl_policy", ""))
    if "exact signed" not in policy:
        errors.append("simulation_hkl_policy does not state exact signed HKL use")
    return errors


def summarize_contract_validation(records: list[dict[str, Any]], max_examples: int = 20) -> dict[str, Any]:
    versions = Counter(str(r.get("dynrow_schema_version", "missing")) for r in records)
    engine_versions = Counter(str(r.get("engine_contract_version", "missing")) for r in records)
    examples = []
    n_bad = 0
    error_counts: Counter[str] = Counter()
    for i, rec in enumerate(records):
        errors = validate_dynrow_record(rec)
        if not errors:
            continue
        n_bad += 1
        for err in errors:
            error_counts[err] += 1
        if len(examples) < int(max_examples):
            examples.append(
                {
                    "local_row_index": int(i),
                    "obs_id": rec.get("obs_id"),
                    "hkl": rec.get("hkl"),
                    "errors": errors,
                }
            )
    return {
        "dynrow_schema_version_expected": DYNROW_SCHEMA_VERSION,
        "engine_contract_version_expected": FROZEN_ENGINE_CONTRACT_VERSION,
        "row_count": int(len(records)),
        "invalid_row_count": int(n_bad),
        "valid": n_bad == 0,
        "schema_versions": dict(versions),
        "engine_contract_versions": dict(engine_versions),
        "error_counts": dict(error_counts),
        "invalid_examples": examples,
    }
