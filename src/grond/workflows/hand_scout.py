"""Compare coordinate hands on the same scout contract and combine per-crystal evidence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import try_read_json
from grond.hashing import sha256_file
from grond.io.config import DatasetSpec, PrepConfig


def _dataset_decision_identity(prep: PrepConfig, ds: DatasetSpec) -> dict[str, object]:
    """Bind transform and hand decisions to one exact dataset/model/split."""

    hkl_path = Path(ds.hkl).expanduser().resolve()
    model_path = Path(prep.structure).expanduser().resolve()
    split_path = Path(ds.split_contract).expanduser().resolve() if ds.split_contract else None
    split_payload = try_read_json(split_path) if split_path is not None and split_path.is_file() else {}
    split_payload = split_payload if isinstance(split_payload, Mapping) else {}
    return {
        "dataset": ds.name,
        "setup_id": str(prep.parity_setup_id).strip(),
        "setup_manifest_sha256": str(prep.parity_setup_manifest_sha256).strip().lower(),
        "physical_crystal_id": str(ds.physical_crystal_id).strip(),
        "acquisition_fingerprint": str(ds.acquisition_fingerprint).strip().lower(),
        "source_hkl": str(hkl_path),
        "source_hkl_sha256": sha256_file(hkl_path) if hkl_path.is_file() else "",
        "source_model": str(model_path),
        "source_model_sha256": (sha256_file(model_path) if model_path.is_file() else ""),
        "split_contract_path": str(split_path) if split_path is not None else "",
        "split_contract_sha256": (sha256_file(split_path) if split_path is not None and split_path.is_file() else ""),
        "split_membership_sha256": str(split_payload.get("membership_sha256") or ""),
    }
