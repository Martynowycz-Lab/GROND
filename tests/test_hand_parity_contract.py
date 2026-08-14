from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import grond.cli as cli
from grond.hkl_transform import (
    absolute_configuration_claim_contract,
    parse_hkl_transform_info,
    source_orientation_contract,
    validate_external_parity_calibration,
)
from grond.io.config import DatasetSpec, PrepConfig
from grond.legacy.engine.bloch_engine import (
    build_transform_split_contract,
    candidate_invariant_split_membership,
    metric_filter_signed_axis_transforms,
    parse_model,
    validate_transform_split_contract,
)
from grond.legacy.engine.native_row_regularize import rank_trials
from grond.workflows.decision import (
    ALL_PROPER_SIGNED_PERMUTATIONS,
    expand_transform_candidates,
    score_transform_summary,
)
from grond.workflows.parity_evidence import (
    build_common_sigma_hand_evidence,
    replay_transform_work_born,
)


def _p41_res() -> str:
    return "\n".join(
        [
            "TITL parity canonicalization",
            "CELL 0.025079 8 8 12 90 90 90",
            "LATT -1",
            "SYMM -Y, X, Z+1/4",
            "SYMM -X, -Y, Z+1/2",
            "SYMM Y, -X, Z+3/4",
            "EQIV $1 -Y, X, Z+1/4",
            "SFAC C 1 1 1 1 1 1 1 1 0",
            "UNIT 1",
            "FVAR 1.0",
            "C1 1 0.13 0.27 0.39 1.0 0.04",
            "HKLF 4",
            "END",
            "",
        ]
    )


def _orthorhombic_res() -> str:
    return "\n".join(
        [
            "TITL orthorhombic metric control",
            "CELL 0.025079 8 9 12 90 90 90",
            "LATT -1",
            "SFAC C 1 1 1 1 1 1 1 1 0",
            "UNIT 1",
            "FVAR 1.0",
            "C1 1 0.13 0.27 0.39 1.0 0.04",
            "HKLF 4",
            "END",
            "",
        ]
    )


def _p212121_res_with_cell(cell: tuple[float, float, float]) -> str:
    a, b, c = cell
    return "\n".join(
        [
            "TITL XDS P1 integration-cell fallback control",
            f"CELL 0.01970 {a} {b} {c} 90 90 90",
            "LATT -1",
            "SYMM 1/2-X, -Y, 1/2+Z",
            "SYMM -X, 1/2+Y, 1/2-Z",
            "SYMM 1/2+X, 1/2-Y, -Z",
            "SFAC C 1 1 1 1 1 1 1 1 0",
            "UNIT 1",
            "FVAR 1.0",
            "C1 1 0.13 0.27 0.39 1.0 0.04",
            "HKLF 4",
            "END",
            "",
        ]
    )


def _summary(
    *,
    born_wr2: float,
    dyn_wr2: float,
    free_born_wr2: float | None = None,
) -> dict:
    return {
        "best_native": {
            "metrics": {
                "all": {
                    "born": {
                        "wR2_all": born_wr2,
                        "R1_all": born_wr2 + 0.01,
                        "R1_gt": born_wr2 + 0.02,
                        "CC": 0.9,
                    },
                    "dyn": {
                        "wR2_all": dyn_wr2,
                        "R1_all": dyn_wr2 + 0.01,
                        "CC": 0.9,
                    },
                },
                "work": {
                    "born": {
                        "wR2_all": born_wr2,
                        "R1_all": born_wr2 + 0.01,
                        "R1_gt": born_wr2 + 0.02,
                        "CC": 0.9,
                    },
                    "dyn": {
                        "wR2_all": dyn_wr2,
                        "R1_all": dyn_wr2 + 0.01,
                        "CC": 0.9,
                    },
                },
                "free": {
                    "born": {
                        "wR2_all": free_born_wr2
                        if free_born_wr2 is not None
                        else born_wr2 + 0.02,
                    },
                    "dyn": {"wR2_all": dyn_wr2 + 0.02},
                },
            }
        }
    }


def _hklf4_text() -> str:
    rows = [
        (1, -2, 3, 123.5, 4.25, 7),
        (-4, 5, -6, -12.0, 3.5, 9),
        (0, 0, 0, 0.0, 0.0, 0),
    ]
    return "".join(
        f"{h:4d}{k:4d}{ell:4d}{intensity:8.2f}{sigma:8.2f}{batch:4d}\n"
        for h, k, ell, intensity, sigma, batch in rows
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_calibration_xds(
    path: Path,
    *,
    intensity_offset: float,
    cell: tuple[float, float, float] = (8.0, 8.0, 12.0),
) -> None:
    a, b, c = cell
    lines = [
        "!X-RAY_WAVELENGTH= 0.025079",
        "!STARTING_FRAME= 1",
        "!STARTING_ANGLE= 0",
        "!OSCILLATION_RANGE= 0.5",
        "!ROTATION_AXIS= 1 0 0",
        "!INCIDENT_BEAM_DIRECTION= 0 0 1",
        "!SPACE_GROUP_NUMBER= 1",
        f"!UNIT_CELL_CONSTANTS= {a} {b} {c} 90 90 90",
        f"!UNIT_CELL_A-AXIS= {a} 0 0",
        f"!UNIT_CELL_B-AXIS= 0 {b} 0",
        f"!UNIT_CELL_C-AXIS= 0 0 {c}",
    ]
    hkls = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
        (2, 1, 0),
        (2, 0, 1),
    ]
    for row_index, (h, k, ell) in enumerate(hkls):
        values = [
            h,
            k,
            ell,
            100.0 + intensity_offset + row_index,
            5.0,
            100.0,
            100.0,
            1.0 + row_index,
            1.0,
            100.0,
            100.0,
            0.0,
            100.0,
            100.0,
            1.0 + row_index,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        lines.append(" ".join(str(value) for value in values))
    path.write_text("\n".join(lines) + "\n")


def _write_distorted_p1_xds(
    path: Path,
    *,
    cell: tuple[float, float, float, float, float, float],
) -> None:
    _write_calibration_xds(
        path,
        intensity_offset=0.0,
        cell=cell[:3],
    )
    a, b, c, alpha, beta, gamma = cell
    text = path.read_text()
    text = text.replace(
        f"!UNIT_CELL_CONSTANTS= {a} {b} {c} 90 90 90",
        f"!UNIT_CELL_CONSTANTS= {a} {b} {c} {alpha} {beta} {gamma}",
    )
    path.write_text(text)


def _write_transform_decision(
    path: Path,
    transform: str = "identity",
    *,
    dataset: str,
    intensity_offset: float,
    setup_id: str = "setup",
    setup_manifest_sha256: str = "b" * 64,
    physical_crystal_id: str = "",
    acquisition_fingerprint: str = "a" * 64,
) -> dict[str, str]:
    model = path.with_name(f"{dataset}.res")
    hkl = path.with_name(f"{dataset}.INTEGRATE.HKL")
    split = path.with_name(f"{dataset}.split.json")
    model.write_text(_p41_res())
    _write_calibration_xds(hkl, intensity_offset=intensity_offset)
    retained, metric = metric_filter_signed_axis_transforms(
        hkl,
        model,
        ALL_PROPER_SIGNED_PERMUTATIONS,
        relative_tolerance=0.02,
    )
    assert parse_hkl_transform_info(transform).canonical in retained
    split_payload = build_transform_split_contract(
        hkl,
        model,
        retained,
        dataset=dataset,
        free_fraction=0.10,
        seed=42,
    )
    split.write_text(json.dumps(split_payload, indent=2) + "\n")
    split_validation = validate_transform_split_contract(
        split_payload,
        source_hkl=hkl,
        current_transform=transform,
    )
    split_validation.pop("masks")
    info = parse_hkl_transform_info(transform)
    identity = {
        "dataset": dataset,
        "setup_id": setup_id,
        "setup_manifest_sha256": setup_manifest_sha256,
        "physical_crystal_id": physical_crystal_id or f"physical_{dataset}",
        "acquisition_fingerprint": acquisition_fingerprint,
        "source_hkl": str(hkl.resolve()),
        "source_hkl_sha256": _sha256(hkl),
        "source_model": str(model.resolve()),
        "source_model_sha256": _sha256(model),
        "split_contract_path": str(split.resolve()),
        "split_contract_sha256": _sha256(split),
        "split_membership_sha256": split_payload["membership_sha256"],
    }
    candidate_generation = cli._transform_candidate_generation_contract(
        "auto",
        list(ALL_PROPER_SIGNED_PERMUTATIONS),
        retained,
        metric,
        parity_policy="proper-only",
    )
    frozen_nuisance = cli.transform_scout_nuisance(
        SimpleNamespace(
            fast=False,
            thickness_grid="1200",
            thickness_grad_grid="0.0",
            frame_offset_grid="0.0",
            group_width_grid="frame",
            detector_scale_bins="0",
        )
    )
    frozen_nuisance.update(
        {
            "group_width_deg": 1.0,
            "sg_jitter_points": 1,
            "sg_jitter_student_df": 4.0,
            "sg_jitter_width_factor": 3.0,
            "dose_b_ref_frame": float("nan"),
            "matrix_convention": "j_minus_i",
            "native_wght_a": 0.0,
            "native_wght_b": 0.0,
            "scale_smoothness": 0.0,
        }
    )

    def production_config(*, alternate_hand: bool) -> dict[str, object]:
        nuisance_to_grid = {
            "thickness_A": "thickness_grid",
            "thickness_model": "thickness_model_grid",
            "thickness_grad_frac": "thickness_grad_grid",
            "thickness_sigma_frac": "thickness_sigma_frac_grid",
            "frame_offset": "frame_offset_grid",
            "group_width_deg": "group_width_grid",
            "detector_scale_bins": "detector_scale_bins",
            "sigma_a": "sigma_a_grid",
            "sigma_b_corr": "sigma_b_corr_grid",
            "sigma_c_peak": "sigma_c_peak_grid",
            "sg_jitter_mode": "sg_jitter_mode_grid",
            "sg_jitter_sigma0": "sg_jitter_sigma0_grid",
            "sg_jitter_sigma_d_inv": "sg_jitter_sigma_d_inv_grid",
            "sg_jitter_sigma_frame": "sg_jitter_sigma_frame_grid",
            "absorption_mode": "absorption_mode_grid",
            "absorption_mu": "absorption_mu_grid",
            "absorption_frac": "absorption_frac_grid",
            "absorption_mu_g2": "absorption_mu_g2_grid",
            "absorption_mu_theta2": "absorption_mu_theta2_grid",
            "absorption_tds_b": "absorption_tds_b_grid",
            "dose_b_mode": "dose_b_mode_grid",
            "dose_b_per_frame": "dose_b_per_frame_grid",
            "row_resolution_b": "row_resolution_b_grid",
        }
        return {
            **{
                option: frozen_nuisance[field]
                for field, option in nuisance_to_grid.items()
            },
            "split_contract": str(split.resolve()),
            "alternate_hand": alternate_hand,
            "n_basis": 20,
            "n_points": 10,
            "max_rows": 0,
            "split_seed": 42,
            "free_fraction": 0.10,
        }

    split_binding = {
        "sha256": _sha256(split),
        "membership_sha256": split_payload["membership_sha256"],
    }
    records = []
    for rank, candidate in enumerate(retained):
        candidate_dir = path.with_name(
            f"{dataset}.transform_{rank:02d}"
        )
        candidate_dir.mkdir()
        row_table = candidate_dir / "native_dyn_calc.csv"
        fields = [
            "row_index",
            "h",
            "k",
            "l",
            "target_I",
            "sigma_eff",
            "xds_sigma",
            "group",
            "detector_bin",
            "split",
            "dyn_raw",
            "born_raw",
            "dyn_scaled",
            "born_scaled",
        ]
        selected_candidate = (
            parse_hkl_transform_info(candidate).canonical
            == parse_hkl_transform_info(transform).canonical
        )
        with row_table.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row_index, raw_hkl in enumerate(
                [
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                    (1, 1, 0),
                    (1, 0, 1),
                    (0, 1, 1),
                    (1, 1, 1),
                    (2, 1, 0),
                    (2, 0, 1),
                ]
            ):
                target = 100.0 + intensity_offset + row_index
                relative_error = 0.01 if selected_candidate else 0.15 + 0.01 * rank
                prediction = target * (1.0 + relative_error)
                writer.writerow(
                    {
                        "row_index": row_index,
                        "h": raw_hkl[0],
                        "k": raw_hkl[1],
                        "l": raw_hkl[2],
                        "target_I": target,
                        "sigma_eff": 5.0,
                        "xds_sigma": 5.0,
                        "group": 0,
                        "detector_bin": 0,
                        "split": (
                            "free"
                            if split_payload["membership"][row_index]
                            else "work"
                        ),
                        "dyn_raw": prediction,
                        "born_raw": prediction,
                        "dyn_scaled": prediction,
                        "born_scaled": prediction,
                    }
                )
        summary_path = candidate_dir / "summary.json"
        work_rows = int(
            sum(value == 0 for value in split_payload["membership"])
        )
        free_rows = int(
            sum(value == 1 for value in split_payload["membership"])
        )
        summary_path.write_text(
            json.dumps(
                {
                    "artifacts": {
                        "native_row_table": str(row_table.resolve()),
                        "native_row_table_sha256": _sha256(row_table),
                    },
                    "inputs": {
                        "hkl": str(hkl.resolve()),
                        "hkl_sha256": _sha256(hkl),
                        "hkl_transform": candidate,
                        "res": str(model.resolve()),
                        "res_sha256": _sha256(model),
                        "model_tag": f"original;hkl_transform={candidate}",
                    },
                    "config": production_config(alternate_hand=False),
                    "row_selection": {
                        "selected_rows": len(split_payload["membership"]),
                        "work_rows": work_rows,
                        "free_rows": free_rows,
                        "transform_hand_split_contract": split_binding,
                    },
                    "best_native": {
                        "native_weight": {"a": 0.0, "b": 0.0},
                        "nuisance": frozen_nuisance,
                        "scale_fit_split": "work",
                    },
                    "bloch_kernel_contract": {
                        "schema_version": 1,
                        "n_basis": 20,
                        "n_points": 10,
                        "matrix_convention": "j_minus_i",
                        "test_fixture": True,
                    },
                    "selection_metric": "work_born_wR2",
                }
            )
            + "\n"
        )
        replay = replay_transform_work_born(summary_path)
        metric = replay["work_born_metrics"]
        job = replay["job_contract"]
        records.append(
            {
            "transform": candidate,
            "status": "ok",
            "score": replay["score"],
            "selection_target": "matched_born_work_only_hand_invariant_v1",
            "coordinate_hand_sensitive_terms_used": False,
            "free_rows_used_for_transform_selection": False,
            "work_born_wR2_all": metric["wR2_all"],
            "work_born_R1_all": metric["R1_all"],
            "work_born_R1_gt": metric["R1_gt"],
            "work_born_CC": metric["CC"],
            "artifact_score_replay": replay,
            "engine_trial_selection_metric": "work_born_wR2",
            "engine_trial_selection_metric_requested": "work_born_wR2",
            "nuisance_trial_count": 1,
            "prediction_geometry_trial_count": 1,
            "n_basis": 20,
            "n_points": 10,
            "max_rows": 0,
            "frozen_nuisance_contract": job["declared_nuisance_contract"],
            "frozen_nuisance_contract_sha256": job[
                "declared_nuisance_contract_sha256"
            ],
            }
        )
    selected_record = next(
        record
        for record in records
        if record["transform"] == info.canonical
    )
    info = parse_hkl_transform_info(transform)
    path.write_text(
        json.dumps(
            {
                "selected": {
                    **selected_record,
                    "transform": info.canonical,
                    "transform_determinant": info.determinant,
                    "status": "ok",
                    "selection_target": "matched_born_work_only_hand_invariant_v1",
                    "engine_trial_selection_metric": "work_born_wR2",
                    "nuisance_trial_count": 1,
                    "prediction_geometry_trial_count": 1,
                    "frozen_nuisance_contract": selected_record[
                        "frozen_nuisance_contract"
                    ],
                    "frozen_nuisance_contract_sha256": selected_record[
                        "frozen_nuisance_contract_sha256"
                    ],
                    "source_orientation_contract": {
                        "status": "right_handed",
                        "source_basis_determinant": 720.0,
                        "transformed_basis_determinant": 720.0,
                        "right_handed_source": True,
                        "right_handed_transformed": True,
                        "absolute_claim_geometry_ready": True,
                    },
                },
                "dataset": dataset,
                "input_identity": identity,
                "candidate_generation": candidate_generation,
                "ranking_tie_break": {
                    "policy": "fixed_full24_proper_priority_then_canonical_v1",
                    "full24_proper_priority": list(
                        ALL_PROPER_SIGNED_PERMUTATIONS
                    ),
                    "metadata_order_used_for_ties_or_rerank_admission": False,
                },
                "coarse_records": records,
                "transform_hand_split_contract": {
                    **split_validation,
                    "path": str(split.resolve()),
                    "sha256": _sha256(split),
                },
                "selection_target": "matched_born_work_only_hand_invariant_v1",
                "known_configuration_used_for_candidate_generation_or_ranking": False,
                "metadata_candidate_elimination_used": False,
            }
        )
        + "\n"
    )
    return identity


def _write_decisive_hand_decision(
    path: Path,
    *,
    transform: str,
    selected_hand: str,
    dataset_identity: dict[str, str],
) -> None:
    split_payload = json.loads(
        Path(dataset_identity["split_contract_path"]).read_text()
    )
    original_dir = path.with_name(path.stem + "_original")
    alternate_dir = path.with_name(path.stem + "_alternate")
    original_dir.mkdir()
    alternate_dir.mkdir()
    model = Path(dataset_identity["source_model"])
    hkl = Path(dataset_identity["source_hkl"])
    split = Path(dataset_identity["split_contract_path"])
    frozen_nuisance = cli.transform_scout_nuisance(
        SimpleNamespace(
            fast=False,
            thickness_grid="1200",
            thickness_grad_grid="0.0",
            frame_offset_grid="0.0",
            group_width_grid="frame",
            detector_scale_bins="0",
        )
    )
    frozen_nuisance.update(
        {
            "group_width_deg": 1.0,
            "sg_jitter_points": 1,
            "sg_jitter_student_df": 4.0,
            "sg_jitter_width_factor": 3.0,
            "dose_b_ref_frame": float("nan"),
            "matrix_convention": "j_minus_i",
            "native_wght_a": 0.0,
            "native_wght_b": 0.0,
            "scale_smoothness": 0.0,
        }
    )
    nuisance_to_grid = {
        "thickness_A": "thickness_grid",
        "thickness_model": "thickness_model_grid",
        "thickness_grad_frac": "thickness_grad_grid",
        "thickness_sigma_frac": "thickness_sigma_frac_grid",
        "frame_offset": "frame_offset_grid",
        "group_width_deg": "group_width_grid",
        "detector_scale_bins": "detector_scale_bins",
        "sigma_a": "sigma_a_grid",
        "sigma_b_corr": "sigma_b_corr_grid",
        "sigma_c_peak": "sigma_c_peak_grid",
        "sg_jitter_mode": "sg_jitter_mode_grid",
        "sg_jitter_sigma0": "sg_jitter_sigma0_grid",
        "sg_jitter_sigma_d_inv": "sg_jitter_sigma_d_inv_grid",
        "sg_jitter_sigma_frame": "sg_jitter_sigma_frame_grid",
        "absorption_mode": "absorption_mode_grid",
        "absorption_mu": "absorption_mu_grid",
        "absorption_frac": "absorption_frac_grid",
        "absorption_mu_g2": "absorption_mu_g2_grid",
        "absorption_mu_theta2": "absorption_mu_theta2_grid",
        "absorption_tds_b": "absorption_tds_b_grid",
        "dose_b_mode": "dose_b_mode_grid",
        "dose_b_per_frame": "dose_b_per_frame_grid",
        "row_resolution_b": "row_resolution_b_grid",
    }

    def production_config(*, alternate_hand: bool) -> dict[str, object]:
        return {
            **{
                option: frozen_nuisance[field]
                for field, option in nuisance_to_grid.items()
            },
            "split_contract": str(split.resolve()),
            "alternate_hand": alternate_hand,
            "n_basis": 20,
            "n_points": 10,
            "max_rows": 0,
            "split_seed": 42,
            "free_fraction": 0.10,
        }

    source_rows = []
    source_hkl = hkl
    for line in source_hkl.read_text().splitlines():
        if not line.strip() or line.startswith("!"):
            continue
        values = line.split()
        source_rows.append(
            (
                int(values[0]),
                int(values[1]),
                int(values[2]),
                float(values[3]),
            )
        )
    fields = [
        "row_index",
        "h",
        "k",
        "l",
        "target_I",
        "sigma_eff",
        "xds_sigma",
        "group",
        "detector_bin",
        "split",
        "dyn_raw",
        "born_raw",
        "dyn_scaled",
        "born_scaled",
    ]

    def write_branch(
        out_dir: Path, *, favored: bool, alternate_hand: bool
    ) -> None:
        table = out_dir / "native_dyn_calc.csv"
        with table.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row_index, (h, k, ell, target) in enumerate(source_rows):
                error = 0.01 if favored else 0.25
                pattern = -1.0 if row_index % 2 else 1.0
                prediction = target * (1.0 + error * pattern)
                writer.writerow(
                    {
                        "row_index": row_index,
                        "h": h,
                        "k": k,
                        "l": ell,
                        "target_I": target,
                        "sigma_eff": 5.0,
                        "xds_sigma": 5.0,
                        "group": 0,
                        "detector_bin": 0,
                        "split": (
                            "free"
                            if split_payload["membership"][row_index]
                            else "work"
                        ),
                        "dyn_raw": prediction,
                        "born_raw": target,
                        "dyn_scaled": prediction,
                        "born_scaled": target,
                    }
                )
        (out_dir / "summary.json").write_text(
            json.dumps(
                {
                    "artifacts": {
                        "native_row_table": str(table.resolve()),
                        "native_row_table_sha256": _sha256(table),
                    },
                    "inputs": {
                        "hkl": str(hkl.resolve()),
                        "hkl_sha256": _sha256(hkl),
                        "hkl_transform": transform,
                        "res": str(model.resolve()),
                        "res_sha256": _sha256(model),
                        "model_tag": (
                            "alternate_fractional_xyz"
                            if alternate_hand
                            else "original"
                        )
                        + f";hkl_transform={transform}",
                    },
                    "config": production_config(
                        alternate_hand=alternate_hand
                    ),
                    "row_selection": {
                        "selected_rows": len(split_payload["membership"]),
                        "work_rows": int(
                            sum(
                                value == 0
                                for value in split_payload["membership"]
                            )
                        ),
                        "free_rows": int(
                            sum(
                                value == 1
                                for value in split_payload["membership"]
                            )
                        ),
                        "transform_hand_split_contract": {
                            "sha256": _sha256(split),
                            "membership_sha256": split_payload[
                                "membership_sha256"
                            ],
                        },
                    },
                    "best_native": {
                        "nuisance": frozen_nuisance,
                        "scale_fit_split": "work",
                    },
                    "selection_metric": "adaptive",
                }
            )
            + "\n"
        )

    write_branch(
        original_dir,
        favored=selected_hand == "original",
        alternate_hand=False,
    )
    write_branch(
        alternate_dir,
        favored=selected_hand == "alternate",
        alternate_hand=True,
    )
    for out_dir, work_wr2 in (
        (original_dir, 0.20),
        (alternate_dir, 0.22),
    ):
        trial_path = out_dir / "nuisance_trials.csv"
        with trial_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[*frozen_nuisance, "work_dyn_wR2"],
            )
            writer.writeheader()
            writer.writerow({**frozen_nuisance, "work_dyn_wR2": work_wr2})
    nuisance_selection = cli.select_branch_symmetric_nuisance(
        original_dir / "summary.json",
        alternate_dir / "summary.json",
    )
    evidence = build_common_sigma_hand_evidence(
        original_dir / "summary.json",
        alternate_dir / "summary.json",
        minimum_abs_adjusted_z=2.0,
    )
    delta = float(evidence["score_delta_original_minus_alternate"])
    path.write_text(
        json.dumps(
            {
                "selected_hand": selected_hand,
                "quality_gate_status": "passed",
                "score_delta_original_minus_alternate": delta,
                "accept_delta": 0.015,
                "comparison_contract": {
                    "hkl_transform": transform,
                    "dataset_identity": dataset_identity,
                    "row_selection_match": True,
                    "positive_matched_free_rows": True,
                    "finite_raw_free_metrics": True,
                    "branch_symmetric_nuisance_selection": nuisance_selection,
                    "common_sigma_hand_evidence": evidence,
                },
            }
        )
        + "\n"
    )


@pytest.mark.parametrize(
    ("transform", "determinant", "paired"),
    [
        ("identity", 1, "-h,-k,-l"),
        ("l,-k,h", 1, "-l,k,-h"),
        ("-h,k,l", -1, "h,-k,-l"),
        ("-k,-h,l", -1, "k,h,-l"),
    ],
)
def test_transform_metadata_has_exact_determinant_and_sign_pair(
    transform: str,
    determinant: int,
    paired: str,
) -> None:
    info = parse_hkl_transform_info(transform)

    assert info.determinant == determinant
    assert info.sign_paired_transform == paired
    assert parse_hkl_transform_info(paired).determinant == -determinant
    assert parse_hkl_transform_info(info.proper_canonical_transform).proper


def test_auto_and_common_are_proper_only_unless_diagnostic_policy_is_explicit() -> None:
    automatic = expand_transform_candidates("-h,k,l", "auto")
    common = expand_transform_candidates("identity", "common")
    diagnostic = expand_transform_candidates(
        "identity",
        "common",
        parity_policy="all-diagnostic",
    )

    assert automatic
    assert all(parse_hkl_transform_info(item).proper for item in automatic)
    assert all(parse_hkl_transform_info(item).proper for item in common)
    assert any(not parse_hkl_transform_info(item).proper for item in diagnostic)
    with pytest.raises(ValueError, match="improper transform"):
        expand_transform_candidates("identity", "auto;-h,k,l")
    with pytest.raises(ValueError, match="improper fixed transform"):
        expand_transform_candidates("-h,k,l", "off")
    assert expand_transform_candidates(
        "-h,k,l",
        "off",
        parity_policy="all-diagnostic",
    ) == ["-h,k,l"]


def test_trusted_improper_transform_requires_explicit_diagnostic_policy(
    tmp_path: Path,
) -> None:
    hkl = tmp_path / "INTEGRATE.HKL"
    hkl.write_text("!UNIT_CELL_A-AXIS= 1 0 0\n")
    ds = DatasetSpec(name="dataset", hkl=str(hkl), hkl_transform="-h,k,l")
    prep = PrepConfig(structure=str(tmp_path / "model.res"), datasets=[ds])
    proper_only = type(
        "Args",
        (),
        {
            "transform_parity_policy": "proper-only",
            "trust_hkl_transform": True,
        },
    )()
    with pytest.raises(ValueError, match="canonicalize-hand"):
        cli.choose_transform_by_scout(
            prep,
            ds,
            proper_only,
            "python",
            tmp_path / "logs",
            tmp_path / "proper_only",
        )

    diagnostic = type(
        "Args",
        (),
        {
            "transform_parity_policy": "all-diagnostic",
            "trust_hkl_transform": True,
        },
    )()
    selected, payload = cli.choose_transform_by_scout(
        prep,
        ds,
        diagnostic,
        "python",
        tmp_path / "logs",
        tmp_path / "diagnostic",
    )
    assert selected.hkl_transform == "-h,k,l"
    assert (
        payload["selected_transform_claim_contract"]["configuration_status"]
        == "parity_unresolved"
    )
    assert (
        payload["selected_transform_claim_contract"][
            "absolute_configuration_claim_allowed"
        ]
        is False
    )


def test_transform_ranking_is_invariant_to_swapped_input_hand_dynamic_blocks() -> None:
    # Deliberate input inversion can exchange or greatly alter the dynamical
    # scout blocks.  The matched-Born target and chosen operator must not move.
    original = {
        "identity": _summary(born_wr2=0.30, dyn_wr2=0.10),
        "h,-k,-l": _summary(born_wr2=0.20, dyn_wr2=0.50),
    }
    inverted = {
        "identity": _summary(born_wr2=0.30, dyn_wr2=0.50),
        "h,-k,-l": _summary(born_wr2=0.20, dyn_wr2=0.10),
    }

    original_scores = {
        transform: score_transform_summary(summary)["score"]
        for transform, summary in original.items()
    }
    inverted_scores = {
        transform: score_transform_summary(summary)["score"]
        for transform, summary in inverted.items()
    }

    assert original_scores == inverted_scores
    assert min(original_scores, key=original_scores.get) == "h,-k,-l"
    assert min(inverted_scores, key=inverted_scores.get) == "h,-k,-l"
    assert all(
        score_transform_summary(summary)["selection_target"]
        == "matched_born_work_only_hand_invariant_v1"
        for summary in original.values()
    )


def test_transform_ranking_fails_when_work_born_metrics_are_missing() -> None:
    summary = _summary(born_wr2=0.2, dyn_wr2=0.1)
    del summary["best_native"]["metrics"]["work"]

    score = score_transform_summary(summary)

    assert score["score"] == float("inf")
    assert score["free_rows_used_for_transform_selection"] is False


def test_claim_contract_fails_closed_without_external_parity_calibration() -> None:
    proper = absolute_configuration_claim_contract("h,-k,-l")
    improper = absolute_configuration_claim_contract("-h,k,l")
    calibrated = absolute_configuration_claim_contract(
        "h,-k,-l",
        externally_calibrated_parity=True,
        calibration_reference="calibration.json#sha256=abc",
        calibration_validation_status="validated",
    )

    assert proper["configuration_status"] == "conditional_inversion_preference"
    assert proper["absolute_configuration_claim_allowed"] is False
    assert improper["configuration_status"] == "parity_unresolved"
    assert improper["absolute_configuration_claim_allowed"] is False
    assert calibrated["configuration_status"] == "absolute_configuration_calibrated"
    assert calibrated["absolute_configuration_claim_allowed"] is True


def test_bare_external_calibration_and_wrong_source_orientation_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="missing required field"):
        validate_external_parity_calibration(
            {},
            setup_id="setup",
            proper_transform="identity",
            transform_decision_sha256="transform-hash",
            matched_hand_decision_sha256="hand-hash",
            matched_selected_hand="original",
        )

    hkl = tmp_path / "INTEGRATE.HKL"
    hkl.write_text(
        "\n".join(
            [
                "!UNIT_CELL_A-AXIS= 8 0 0",
                "!UNIT_CELL_B-AXIS= 0 9 0",
                "!UNIT_CELL_C-AXIS= 0 0 10",
                "   1   0   0  100.00   10.00",
                "",
            ]
        )
    )
    proper = source_orientation_contract(hkl, "identity")
    improper = source_orientation_contract(hkl, "-h,k,l")

    assert proper["source_basis_determinant"] == pytest.approx(720.0)
    assert proper["transformed_basis_determinant"] == pytest.approx(720.0)
    assert proper["absolute_claim_geometry_ready"] is True
    assert improper["source_basis_determinant"] == pytest.approx(720.0)
    assert improper["transformed_basis_determinant"] == pytest.approx(-720.0)
    assert improper["absolute_claim_geometry_ready"] is False


def test_canonicalize_hand_cli_materializes_exact_res_and_audited_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.res"
    source_hkl = tmp_path / "source.hkl"
    output = tmp_path / "canonical.res"
    output_hkl = tmp_path / "canonical.hkl"
    manifest = tmp_path / "canonical.manifest.json"
    decision = tmp_path / "matched_hand.json"
    transform_decision = tmp_path / "transform_decision.json"
    source.write_text(_p41_res())
    source_hkl.write_text(_hklf4_text())
    decision.write_text(
        json.dumps(
            {
                "selected_hand": "original",
                "comparison_contract": {"hkl_transform": "-h,k,l"},
            }
        )
        + "\n"
    )
    transform_decision.write_text(
        json.dumps(
            {
                "selected": {"transform": "-h,k,l"},
                "selection_target": "matched_born_work_only_hand_invariant_v1",
                "known_configuration_used_for_candidate_generation_or_ranking": False,
                "metadata_candidate_elimination_used": False,
            }
        )
        + "\n"
    )
    args = cli.build_parser().parse_args(
        [
            "canonicalize-hand",
            "--input-res",
            str(source),
            "--input-hkl",
            str(source_hkl),
            "--hkl-transform=-h,k,l",
            "--output-res",
            str(output),
            "--output-hkl",
            str(output_hkl),
            "--manifest",
            str(manifest),
            "--matched-hand-decision",
            str(decision),
            "--transform-decision",
            str(transform_decision),
        ]
    )

    assert args.func(args) == 0
    payload = json.loads(manifest.read_text())
    original_atom = parse_model(source)["atoms"][0]
    canonical_atom = parse_model(output)["atoms"][0]
    assert canonical_atom["x"] == pytest.approx(1.0 - original_atom["x"])
    assert canonical_atom["y"] == pytest.approx(1.0 - original_atom["y"])
    assert canonical_atom["z"] == pytest.approx(1.0 - original_atom["z"])
    assert "EQIV $1 -y,x,z-1/4" in output.read_text()
    assert payload["input"]["transform"]["determinant"] == -1
    assert payload["output"]["transform"]["canonical"] == "h,-k,-l"
    assert payload["output"]["transform"]["determinant"] == 1
    assert payload["coordinate_inversion_applied"] is True
    assert payload["hklf4_gauge_pair_audit"]["reflection_count_preserved"] is True
    assert payload["hklf4_gauge_pair_audit"]["observation_payload_preserved"] is True
    source_lines = source_hkl.read_text().splitlines()
    output_lines = output_hkl.read_text().splitlines()
    assert [line[:12] for line in output_lines[:2]] == [
        f"{-1:4d}{2:4d}{-3:4d}",
        f"{4:4d}{-5:4d}{6:4d}",
    ]
    assert [line[12:] for line in output_lines] == [line[12:] for line in source_lines]
    assert payload["matched_hand_decision"]["sha256"]
    assert (
        payload["configuration_claim_contract"]["configuration_status"]
        == "conditional_inversion_preference"
    )

    # A deliberate second application is an audited no-op, not another
    # inversion: the canonical bytes and proper transform remain fixed.
    second_res = tmp_path / "canonical_second.res"
    second_hkl = tmp_path / "canonical_second.hkl"
    second_manifest = tmp_path / "canonical_second.manifest.json"
    second_args = cli.build_parser().parse_args(
        [
            "canonicalize-hand",
            "--input-res",
            str(output),
            "--input-hkl",
            str(output_hkl),
            "--hkl-transform=h,-k,-l",
            "--output-res",
            str(second_res),
            "--output-hkl",
            str(second_hkl),
            "--manifest",
            str(second_manifest),
            "--allow-proper-noop",
        ]
    )
    assert second_args.func(second_args) == 0
    assert second_res.read_bytes() == output.read_bytes()
    assert second_hkl.read_bytes() == output_hkl.read_bytes()
    assert (
        json.loads(second_manifest.read_text())["operation"]
        == "audited_proper_noop_copy"
    )


def test_canonicalize_hand_rejects_proper_input_without_explicit_noop(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.res"
    source_hkl = tmp_path / "source.hkl"
    source.write_text(_p41_res())
    source_hkl.write_text(_hklf4_text())
    proper_args = cli.build_parser().parse_args(
        [
            "canonicalize-hand",
            "--input-res",
            str(source),
            "--input-hkl",
            str(source_hkl),
            "--hkl-transform=identity",
            "--output-res",
            str(tmp_path / "proper.res"),
            "--output-hkl",
            str(tmp_path / "proper.hkl"),
        ]
    )
    with pytest.raises(ValueError, match="already proper"):
        proper_args.func(proper_args)

def test_reserved_calibrant_is_created_then_applied_to_different_validation(
    tmp_path: Path,
) -> None:
    calibrant_transform = tmp_path / "calibrant_transform.json"
    calibrant_hand = tmp_path / "calibrant_hand.json"
    calibration = tmp_path / "setup_calibration.json"
    calibrant_identity = _write_transform_decision(
        calibrant_transform,
        "identity",
        dataset="reserved_001",
        intensity_offset=0.0,
        setup_id="falcon4_linear",
        physical_crystal_id="physical_reserved_001",
        acquisition_fingerprint="a" * 64,
    )
    _write_decisive_hand_decision(
        calibrant_hand,
        transform="identity",
        selected_hand="original",
        dataset_identity=calibrant_identity,
    )
    create_args = cli.build_parser().parse_args(
        [
            "create-parity-calibration",
            "--setup-id",
            "falcon4_linear",
            "--calibrant-dataset-id",
            "reserved_001",
            "--calibrant-physical-crystal-id",
            "physical_reserved_001",
            "--calibrant-acquisition-fingerprint",
            "a" * 64,
            "--setup-manifest-sha256",
            "b" * 64,
            "--transform-decision",
            str(calibrant_transform),
            "--matched-hand-decision",
            str(calibrant_hand),
            "--expected-known-configuration",
            "known_L_reference",
            "--expected-known-branch",
            "alternate",
            "--reserved-from-validation",
            "--out",
            str(calibration),
        ]
    )
    assert create_args.func(create_args) == 0
    calibration_payload = json.loads(calibration.read_text())
    # The calibrant discovered a reversed setup sign rather than being forced
    # to match the expected branch.
    assert calibration_payload["observed_branch"] == "original"
    assert calibration_payload["expected_known_branch"] == "alternate"
    assert calibration_payload["setup_parity_multiplier"] == -1

    validation_transform = tmp_path / "validation_transform.json"
    validation_hand = tmp_path / "validation_hand.json"
    applied = tmp_path / "validation_applied_calibration.json"
    validation_identity = _write_transform_decision(
        validation_transform,
        "h,-k,-l",
        dataset="validation_002",
        intensity_offset=1000.0,
        setup_id="falcon4_linear",
        physical_crystal_id="physical_validation_002",
        acquisition_fingerprint="c" * 64,
    )
    _write_decisive_hand_decision(
        validation_hand,
        transform="h,-k,-l",
        selected_hand="alternate",
        dataset_identity=validation_identity,
    )
    apply_args = cli.build_parser().parse_args(
        [
            "apply-parity-calibration",
            "--calibration",
            str(calibration),
            "--setup-id",
            "falcon4_linear",
            "--validation-dataset-id",
            "validation_002",
            "--validation-physical-crystal-id",
            "physical_validation_002",
            "--validation-acquisition-fingerprint",
            "c" * 64,
            "--setup-manifest-sha256",
            "b" * 64,
            "--transform-decision",
            str(validation_transform),
            "--matched-hand-decision",
            str(validation_hand),
            "--out",
            str(applied),
        ]
    )
    assert apply_args.func(apply_args) == 0
    applied_payload = json.loads(applied.read_text())
    assert applied_payload["reserved_calibrant_dataset_id"] == "reserved_001"
    assert applied_payload["validation_dataset_id"] == "validation_002"
    assert applied_payload["setup_parity_multiplier"] == -1
    assert applied_payload["validation_branch_sign"] == -1
    assert (
        applied_payload["calibrated_coordinate_hand_factor"] == 1
    )
    assert (
        applied_payload["configuration_relative_to_validation_input_coordinates"]
        == "as_supplied"
    )
    assert (
        applied_payload["configuration_claim_contract"][
            "absolute_configuration_claim_allowed"
        ]
        is True
    )


def test_weak_default_original_hand_cannot_create_a_calibration(
    tmp_path: Path,
) -> None:
    transform = tmp_path / "transform.json"
    hand = tmp_path / "weak_hand.json"
    identity = _write_transform_decision(
        transform,
        dataset="reserved",
        intensity_offset=0.0,
    )
    _write_decisive_hand_decision(
        hand,
        transform="identity",
        selected_hand="original",
        dataset_identity=identity,
    )
    payload = json.loads(hand.read_text())
    payload["score_delta_original_minus_alternate"] = -0.001
    hand.write_text(json.dumps(payload) + "\n")
    args = cli.build_parser().parse_args(
        [
            "create-parity-calibration",
            "--setup-id",
            "setup",
            "--calibrant-dataset-id",
            "reserved",
            "--calibrant-physical-crystal-id",
            "physical_reserved",
            "--calibrant-acquisition-fingerprint",
            "a" * 64,
            "--setup-manifest-sha256",
            "b" * 64,
            "--transform-decision",
            str(transform),
            "--matched-hand-decision",
            str(hand),
            "--expected-known-configuration",
            "known",
            "--expected-known-branch",
            "original",
            "--reserved-from-validation",
            "--out",
            str(tmp_path / "must_not_exist.json"),
        ]
    )
    with pytest.raises(ValueError, match="not decisive enough"):
        args.func(args)


def test_full24_metric_filter_is_cell_only_and_retains_orthorhombic_klein_four(
    tmp_path: Path,
) -> None:
    assert len(ALL_PROPER_SIGNED_PERMUTATIONS) == 24
    assert all(
        parse_hkl_transform_info(transform).determinant == 1
        for transform in ALL_PROPER_SIGNED_PERMUTATIONS
    )
    model = tmp_path / "orthorhombic.res"
    first_hkl = tmp_path / "first.INTEGRATE.HKL"
    second_hkl = tmp_path / "second.INTEGRATE.HKL"
    model.write_text(_orthorhombic_res())
    _write_calibration_xds(
        first_hkl,
        intensity_offset=0.0,
        cell=(8.0, 9.0, 12.0),
    )
    _write_calibration_xds(
        second_hkl,
        intensity_offset=10000.0,
        cell=(8.0, 9.0, 12.0),
    )

    first, first_contract = metric_filter_signed_axis_transforms(
        first_hkl,
        model,
        ALL_PROPER_SIGNED_PERMUTATIONS,
        relative_tolerance=0.02,
    )
    second, second_contract = metric_filter_signed_axis_transforms(
        second_hkl,
        model,
        ALL_PROPER_SIGNED_PERMUTATIONS,
        relative_tolerance=0.02,
    )

    assert set(first) == {
        "identity",
        "h,-k,-l",
        "-h,k,-l",
        "-h,-k,l",
    }
    assert first == second
    assert first_contract["generated_candidate_count"] == 24
    assert first_contract["retained_candidate_count"] == 4
    assert first_contract["uses_intensities"] is False
    assert first_contract["uses_atomic_coordinates_or_hand"] is False
    assert first_contract["uses_atom_types_or_known_chemistry"] is False
    assert first_contract["uses_known_configuration"] is False
    assert [
        (
            row["transform"],
            row["status"],
            row["relative_frobenius_mismatch"],
        )
        for row in first_contract["records"]
    ] == [
        (
            row["transform"],
            row["status"],
            row["relative_frobenius_mismatch"],
        )
        for row in second_contract["records"]
    ]


@pytest.mark.parametrize(
    "source_cell",
    [
        (5.896, 6.957, 21.156, 89.498, 89.712, 89.196),
        (5.840, 7.050, 21.336, 88.660, 90.442, 90.060),
    ],
)
def test_metric_filter_recovers_xds_p1_integration_basis_for_higher_symmetry_model(
    tmp_path: Path,
    source_cell: tuple[float, float, float, float, float, float],
) -> None:
    model = tmp_path / "p212121.res"
    hkl = tmp_path / "INTEGRATE.HKL"
    model.write_text(_p212121_res_with_cell((6.880, 21.080, 5.940)))
    _write_distorted_p1_xds(hkl, cell=source_cell)

    retained, contract = metric_filter_signed_axis_transforms(
        hkl,
        model,
        ALL_PROPER_SIGNED_PERMUTATIONS,
        relative_tolerance=0.02,
    )

    assert set(retained) == {"k,l,h", "-k,-l,h", "-k,l,-h", "k,-l,-h"}
    assert contract["filter_mode"] == "xds_p1_distortion_fallback"
    assert contract["source_space_group_number"] == 1
    assert contract["model_symmetry_order"] == 4
    assert contract["xds_p1_distortion_fallback"]["applicable"] is True
    assert contract["xds_p1_distortion_fallback"]["applied"] is True
    assert all(
        row["retention_mode"] == "xds_p1_distortion_fallback"
        for row in contract["records"]
        if row["status"] == "retained"
    )
    assert contract["uses_intensities"] is False
    assert contract["uses_atomic_coordinates_or_hand"] is False
    assert contract["uses_atom_types_or_known_chemistry"] is False
    assert contract["uses_known_configuration"] is False


def test_metric_filter_p1_fallback_still_rejects_incompatible_cell(
    tmp_path: Path,
) -> None:
    model = tmp_path / "p212121.res"
    hkl = tmp_path / "INTEGRATE.HKL"
    model.write_text(_p212121_res_with_cell((6.880, 21.080, 5.940)))
    _write_distorted_p1_xds(
        hkl,
        cell=(4.0, 10.0, 30.0, 86.0, 93.0, 88.0),
    )

    with pytest.raises(
        ValueError,
        match="automatic XDS P1 fallback requires",
    ):
        metric_filter_signed_axis_transforms(
            hkl,
            model,
            ALL_PROPER_SIGNED_PERMUTATIONS,
            relative_tolerance=0.02,
        )


def test_metric_filter_names_dials_dataset_and_does_not_apply_xds_p1_fallback(
    tmp_path: Path,
) -> None:
    model = tmp_path / "p212121.res"
    hkl = tmp_path / "INTEGRATE.HKL"
    model.write_text(_p212121_res_with_cell((6.880, 21.080, 5.940)))
    _write_distorted_p1_xds(
        hkl,
        cell=(5.896, 6.957, 21.156, 89.498, 89.712, 89.196),
    )
    hkl.write_text(
        hkl.read_text().replace(
            "!X-RAY_WAVELENGTH=", "!GROND_SOURCE= DIALS\n!X-RAY_WAVELENGTH=", 1
        )
    )

    with pytest.raises(ValueError) as error:
        metric_filter_signed_axis_transforms(
            hkl,
            model,
            ALL_PROPER_SIGNED_PERMUTATIONS,
            relative_tolerance=0.02,
            dataset_label="crystal_07",
        )

    message = str(error.value)
    assert "dataset 'crystal_07'" in message
    assert "DIALS bridge" in message
    assert "source cell=" in message
    assert "model cell=" in message
    assert "--trust-hkl-transform" in message


def test_candidate_union_split_keeps_duplicate_symmetry_and_friedel_rows_together() -> None:
    raw = np.asarray(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, -1],
            [1, 1, 1],
            [2, 1, 0],
        ],
        dtype=int,
    )
    symmetry_ops = [
        {
            "rot": [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            "trans": [0.0, 0.0, 0.0],
        }
    ]
    result = candidate_invariant_split_membership(
        raw,
        symmetry_ops,
        ["identity", "h,-k,-l", "-h,k,-l", "-h,-k,l"],
        free_fraction=0.25,
        seed=17,
    )
    groups = result["orbit_group_by_row"]
    membership = result["membership"]
    assert groups[0] == groups[1] == groups[2] == groups[3]
    assert groups[4] == groups[5]
    for group_id in np.unique(groups):
        assert len(np.unique(membership[groups == group_id])) == 1


def test_candidate_union_split_unions_orbits_across_different_candidates() -> None:
    raw = np.asarray(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=int,
    )
    result = candidate_invariant_split_membership(
        raw,
        [],
        ["identity", "k,h,-l"],
        free_fraction=0.25,
        seed=9,
    )
    # identity(row 0) and k,h,-l(row 1) are the same transformed
    # reflection even though neither individual candidate merges rows 0/1.
    assert result["orbit_group_by_row"][0] == result["orbit_group_by_row"][1]
    assert result["membership"][0] == result["membership"][1]


def test_transform_scout_nuisance_is_one_geometry_only_trial_and_born_ranked() -> None:
    args = SimpleNamespace(
        fast=False,
        thickness_grid="1000, 1200 1400",
        thickness_grad_grid="0.0,0.4",
        frame_offset_grid="auto,-0.2,0.0,0.2",
        group_width_grid="auto,frame,2frame",
        detector_scale_bins="0,8",
    )
    nuisance = cli.transform_scout_nuisance(args)
    assert nuisance["thickness_A"] == "1000"
    assert nuisance["thickness_grad_frac"] == "0"
    assert nuisance["frame_offset"] == "0.0"
    assert nuisance["group_width_deg"] == "frame"
    assert nuisance["detector_scale_bins"] == "0"
    assert all("," not in str(value) and " " not in str(value) for value in nuisance.values())

    def trial(*, born: float, dyn: float) -> dict:
        return {
            "metrics": {
                "work": {
                    "born": {"wR2_all": born},
                    "dyn": {"wR2_all": dyn, "R1_all": dyn},
                },
                "free": {"dyn": {"wR2_all": dyn}},
                "all": {"dyn": {"wR2_all": dyn, "R1_all": dyn}},
            }
        }

    original = [trial(born=0.20, dyn=0.40), trial(born=0.25, dyn=0.10)]
    inverted = [trial(born=0.20, dyn=0.10), trial(born=0.25, dyn=0.40)]
    assert rank_trials(original, "work_born_wR2")[0] is original[0]
    assert rank_trials(inverted, "work_born_wR2")[0] is inverted[0]


def test_calibration_rejects_restricted_candidate_universe(
    tmp_path: Path,
) -> None:
    transform = tmp_path / "transform.json"
    hand = tmp_path / "hand.json"
    identity = _write_transform_decision(
        transform,
        dataset="reserved",
        intensity_offset=0.0,
    )
    _write_decisive_hand_decision(
        hand,
        transform="identity",
        selected_hand="original",
        dataset_identity=identity,
    )
    payload = json.loads(transform.read_text())
    payload["candidate_generation"]["policy"] = "custom_or_restricted_candidates_v1"
    payload["candidate_generation"]["calibration_grade_candidate_generation"] = False
    transform.write_text(json.dumps(payload) + "\n")
    args = cli.build_parser().parse_args(
        [
            "create-parity-calibration",
            "--setup-id",
            "setup",
            "--calibrant-dataset-id",
            "reserved",
            "--calibrant-physical-crystal-id",
            "physical_reserved",
            "--calibrant-acquisition-fingerprint",
            "a" * 64,
            "--setup-manifest-sha256",
            "b" * 64,
            "--transform-decision",
            str(transform),
            "--matched-hand-decision",
            str(hand),
            "--expected-known-configuration",
            "known",
            "--expected-known-branch",
            "original",
            "--reserved-from-validation",
            "--out",
            str(tmp_path / "must_not_exist.json"),
        ]
    )
    with pytest.raises(ValueError, match="24 determinant-positive"):
        args.func(args)


def test_calibration_rejects_cli_identity_invented_after_selection(
    tmp_path: Path,
) -> None:
    transform = tmp_path / "transform.json"
    hand = tmp_path / "hand.json"
    identity = _write_transform_decision(
        transform,
        dataset="reserved",
        intensity_offset=0.0,
    )
    _write_decisive_hand_decision(
        hand,
        transform="identity",
        selected_hand="original",
        dataset_identity=identity,
    )
    args = cli.build_parser().parse_args(
        [
            "create-parity-calibration",
            "--setup-id",
            "setup",
            "--calibrant-dataset-id",
            "reserved",
            "--calibrant-physical-crystal-id",
            "invented_after_selection",
            "--calibrant-acquisition-fingerprint",
            "a" * 64,
            "--setup-manifest-sha256",
            "b" * 64,
            "--transform-decision",
            str(transform),
            "--matched-hand-decision",
            str(hand),
            "--expected-known-configuration",
            "known",
            "--expected-known-branch",
            "original",
            "--reserved-from-validation",
            "--out",
            str(tmp_path / "must_not_exist.json"),
        ]
    )
    with pytest.raises(ValueError, match="identity frozen before"):
        args.func(args)


def test_calibration_rebuild_rejects_rehashed_arbitrary_split(
    tmp_path: Path,
) -> None:
    transform = tmp_path / "transform.json"
    _write_transform_decision(
        transform,
        dataset="reserved",
        intensity_offset=0.0,
    )
    decision = json.loads(transform.read_text())
    split_path = Path(decision["input_identity"]["split_contract_path"])
    split = json.loads(split_path.read_text())
    inverted = [1 - int(value) for value in split["membership"]]
    split["membership"] = inverted
    split["membership_sha256"] = hashlib.sha256(
        np.asarray(inverted, dtype=np.uint8).tobytes()
    ).hexdigest()
    split["work_row_count"] = int(sum(value == 0 for value in inverted))
    split["free_row_count"] = int(sum(value == 1 for value in inverted))
    split_without_hash = dict(split)
    split_without_hash.pop("contract_payload_sha256", None)
    split["contract_payload_sha256"] = _json_sha256(split_without_hash)
    split_path.write_text(json.dumps(split, indent=2) + "\n")
    decision["input_identity"]["split_contract_sha256"] = _sha256(split_path)
    decision["input_identity"]["split_membership_sha256"] = split[
        "membership_sha256"
    ]
    decision["transform_hand_split_contract"]["sha256"] = _sha256(split_path)
    decision["transform_hand_split_contract"]["membership_sha256"] = split[
        "membership_sha256"
    ]
    decision["transform_hand_split_contract"][
        "contract_payload_sha256"
    ] = split["contract_payload_sha256"]
    transform.write_text(json.dumps(decision) + "\n")
    with pytest.raises(
        ValueError,
        match="cannot be reproduced|work-only matched-Born",
    ):
        cli._fixed_transform_decision_binding(
            str(transform),
            require_proper_geometry=True,
            require_calibration_grade=True,
        )


def test_calibration_recomputes_transform_score_from_bound_rows(
    tmp_path: Path,
) -> None:
    transform = tmp_path / "transform.json"
    _write_transform_decision(
        transform,
        dataset="reserved",
        intensity_offset=0.0,
    )
    decision = json.loads(transform.read_text())
    decision["coarse_records"][0]["score"] = (
        float(decision["coarse_records"][0]["score"]) + 0.1
    )
    transform.write_text(json.dumps(decision) + "\n")
    with pytest.raises(ValueError, match="successfully score every"):
        cli._fixed_transform_decision_binding(
            str(transform),
            require_proper_geometry=True,
            require_calibration_grade=True,
        )


def test_calibration_recomputes_hand_common_sigma_and_adjusted_z(
    tmp_path: Path,
) -> None:
    transform = tmp_path / "transform.json"
    hand = tmp_path / "hand.json"
    identity = _write_transform_decision(
        transform,
        dataset="reserved",
        intensity_offset=0.0,
    )
    _write_decisive_hand_decision(
        hand,
        transform="identity",
        selected_hand="original",
        dataset_identity=identity,
    )
    payload = json.loads(hand.read_text())
    payload["comparison_contract"]["common_sigma_hand_evidence"][
        "branch_aligned_adjusted_z"
    ] = 999.0
    hand.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="common-sigma hand evidence mismatch"):
        cli._fixed_hand_decision_binding(
            str(hand),
            "identity",
            require_decisive=True,
        )


def test_calibration_rejects_candidate_specific_fidelity_after_rehash(
    tmp_path: Path,
) -> None:
    transform = tmp_path / "transform.json"
    _write_transform_decision(
        transform,
        dataset="reserved",
        intensity_offset=0.0,
    )
    decision = json.loads(transform.read_text())
    record = decision["coarse_records"][1]
    binding = record["artifact_score_replay"]["artifact_binding"]
    summary_path = Path(binding["summary_path"])
    summary = json.loads(summary_path.read_text())
    summary["config"]["n_basis"] = 999
    summary_path.write_text(json.dumps(summary) + "\n")
    binding["summary_sha256"] = _sha256(summary_path)
    transform.write_text(json.dumps(decision) + "\n")

    with pytest.raises(
        ValueError,
        match="successfully score every|fidelity and row-support",
    ):
        cli._fixed_transform_decision_binding(
            str(transform),
            require_proper_geometry=True,
            require_calibration_grade=True,
        )


def test_branch_symmetric_nuisance_is_invariant_to_real_coordinate_inversion(
    tmp_path: Path,
) -> None:
    transform = tmp_path / "transform.json"
    hand = tmp_path / "hand.json"
    identity = _write_transform_decision(
        transform,
        dataset="crystal",
        intensity_offset=0.0,
    )
    _write_decisive_hand_decision(
        hand,
        transform="identity",
        selected_hand="original",
        dataset_identity=identity,
    )
    original_dir = hand.with_name(hand.stem + "_original")
    alternate_dir = hand.with_name(hand.stem + "_alternate")

    def rewrite_grid(
        directory: Path,
        *,
        first_score: float,
        second_score: float,
    ) -> None:
        trial_path = directory / "nuisance_trials.csv"
        with trial_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        first_row = dict(rows[0])
        first_row["thickness_A"] = "1000"
        first_row["work_dyn_wR2"] = f"{first_score:.17g}"
        second_row = dict(rows[0])
        second_row["thickness_A"] = "1400"
        second_row["work_dyn_wR2"] = f"{second_score:.17g}"
        with trial_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows([first_row, second_row])

    rewrite_grid(original_dir, first_score=0.10, second_score=0.24)
    rewrite_grid(alternate_dir, first_score=0.52, second_score=0.25)
    as_supplied = cli.select_branch_symmetric_nuisance(
        original_dir / "summary.json",
        alternate_dir / "summary.json",
    )
    assert as_supplied["selected_nuisance"]["dose_b_ref_frame"] is None
    # The public/replay payload must be valid RFC 8259 JSON.  Candidate
    # comparison and hashing still use the explicit canonical string "nan".
    json.dumps(as_supplied, allow_nan=False)
    assert any(
        record["canonical_nuisance"]["dose_b_ref_frame"] == "nan"
        for record in as_supplied["paired_candidate_records"]
    )

    inverted_root = tmp_path / "inverted_materialization"
    inverted_model = cli.make_alternate_hand_model(
        sys.executable,
        str(identity["source_model"]),
        inverted_root,
        tmp_path / "logs",
        False,
    )
    assert _sha256(inverted_model) != identity["source_model_sha256"]

    inverted_original = tmp_path / "inverted_original"
    inverted_alternate = tmp_path / "inverted_alternate"
    shutil.copytree(original_dir, inverted_original)
    shutil.copytree(alternate_dir, inverted_alternate)
    for directory in (inverted_original, inverted_alternate):
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["inputs"]["res"] = str(inverted_model.resolve())
        summary["inputs"]["res_sha256"] = _sha256(inverted_model)
        summary_path.write_text(json.dumps(summary) + "\n")
    # Coordinate inversion exchanges the two semantic physical branches.
    rewrite_grid(inverted_original, first_score=0.52, second_score=0.25)
    rewrite_grid(inverted_alternate, first_score=0.10, second_score=0.24)
    inverted = cli.select_branch_symmetric_nuisance(
        inverted_original / "summary.json",
        inverted_alternate / "summary.json",
    )

    assert (
        as_supplied["selected_candidate_sha256"]
        == inverted["selected_candidate_sha256"]
    )
    assert as_supplied["selected_nuisance"]["thickness_A"] == pytest.approx(
        1400.0
    )
    assert inverted["selected_nuisance"]["thickness_A"] == pytest.approx(
        1400.0
    )
    supplied_record = next(
        record
        for record in as_supplied["paired_candidate_records"]
        if record["candidate_sha256"]
        == as_supplied["selected_candidate_sha256"]
    )
    inverted_record = next(
        record
        for record in inverted["paired_candidate_records"]
        if record["candidate_sha256"]
        == inverted["selected_candidate_sha256"]
    )
    assert supplied_record["original_work_dyn_wR2"] == pytest.approx(
        inverted_record["alternate_work_dyn_wR2"]
    )
    assert supplied_record["alternate_work_dyn_wR2"] == pytest.approx(
        inverted_record["original_work_dyn_wR2"]
    )
