"""Prep files: the structure, the XDS datasets and their per-dataset options for a GROND run.

A prep file is JSON. ``write_datasets_csv`` writes the same datasets, with their nuisance
parameters, as a spreadsheet-friendly table.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from grond.hkl_transform import parse_hkl_transform_info

DATASET_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
TARGET_MODES = {"iobs", "iobs_over_rlp", "iobs_times_peak", "recorded_proxy"}
DATASET_OPTION_KEYS = {
    "acquisition_fingerprint",
    "baseline_hkl",
    "dmin",
    "geometry_percentile",
    "hkl_transform",
    "include_nonpositive",
    "min_d_spacing",
    "min_peak",
    "physical_crystal_id",
    "target_mode",
}
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
TRUE_TEXT = {"1", "true", "yes", "on"}
FALSE_TEXT = {"0", "false", "no", "off"}


def validate_hkl_transform(value: str) -> None:
    """Validate a signed permutation of the crystallographic h, k, l axes."""

    try:
        parse_hkl_transform_info(value)
    except ValueError as exc:
        raise ValueError(
            "hkl_transform must be 'identity' or a signed permutation such as 'l,-k,h'"
        ) from exc


def parse_bool_text(value: str, *, field: str) -> bool:
    """Parse yes/no text (1/0, true/false, yes/no, on/off) as a bool; raise ValueError naming ``field`` otherwise."""

    text = str(value).strip().lower()
    if text in TRUE_TEXT:
        return True
    if text in FALSE_TEXT:
        return False
    raise ValueError(f"{field} must be one of {sorted(TRUE_TEXT | FALSE_TEXT)}, got {value!r}")


@dataclass
class DatasetSpec:
    """One XDS dataset and the options GROND needs to process it."""

    name: str
    hkl: str
    xds_dir: str = ""
    baseline_hkl: str = ""
    hkl_transform: str = "identity"
    min_d_spacing: float = 0.70
    min_peak: float = 90.0
    target_mode: str = "iobs"
    include_nonpositive: bool = False
    geometry_percentile: float = 99.0
    split_contract: str = ""
    physical_crystal_id: str = ""
    acquisition_fingerprint: str = ""

    def __post_init__(self) -> None:
        """Check the dataset name, HKL path, transform, target mode and numeric limits; normalize the ID fields."""

        if not DATASET_NAME_RE.fullmatch(self.name):
            raise ValueError(
                "dataset name must start with a letter or number and contain only "
                "letters, numbers, dots, underscores, or hyphens"
            )
        if not str(self.hkl).strip():
            raise ValueError(f"dataset {self.name!r} has no HKL path")
        validate_hkl_transform(self.hkl_transform)
        if self.target_mode not in TARGET_MODES:
            raise ValueError(
                f"dataset {self.name!r} target_mode must be one of {sorted(TARGET_MODES)}"
            )
        d_min = float(self.min_d_spacing)
        peak = float(self.min_peak)
        percentile = float(self.geometry_percentile)
        if not math.isfinite(d_min) or d_min <= 0.0:
            raise ValueError(f"dataset {self.name!r} min_d_spacing must be > 0")
        if not math.isfinite(peak) or not 0.0 <= peak <= 100.0:
            raise ValueError(f"dataset {self.name!r} min_peak must be in [0, 100]")
        if not math.isfinite(percentile) or not 0.0 < percentile <= 100.0:
            raise ValueError(f"dataset {self.name!r} geometry_percentile must be in (0, 100]")
        acquisition = str(self.acquisition_fingerprint).strip().lower()
        if acquisition and not SHA256_RE.fullmatch(acquisition):
            raise ValueError(
                f"dataset {self.name!r} acquisition_fingerprint must be a SHA-256 hex digest"
            )
        self.physical_crystal_id = str(self.physical_crystal_id).strip()
        self.acquisition_fingerprint = acquisition


@dataclass
class PrepConfig:
    """A reusable run-preparation record for one structure and one or more datasets."""

    structure: str
    datasets: list[DatasetSpec]
    baseline_hkl: str = ""
    notes: str = ""
    default_hkl_transform: str = "identity"
    geometry_percentile: float = 99.0
    parity_setup_id: str = ""
    parity_setup_manifest_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the structure path, unique dataset names, default transform and percentile;
        normalize the setup fields.
        """

        if not str(self.structure).strip():
            raise ValueError("structure path must not be empty")
        if not self.datasets:
            raise ValueError("at least one dataset is required")
        names = [dataset.name for dataset in self.datasets]
        folded_names = [name.casefold() for name in names]
        duplicates = sorted(
            {
                names[index]
                for index, folded in enumerate(folded_names)
                if folded_names.count(folded) > 1
            }
        )
        if duplicates:
            raise ValueError(f"dataset names must be unique; repeated: {', '.join(duplicates)}")
        validate_hkl_transform(self.default_hkl_transform)
        percentile = float(self.geometry_percentile)
        if not math.isfinite(percentile) or not 0.0 < percentile <= 100.0:
            raise ValueError("geometry_percentile must be in (0, 100]")
        self.parity_setup_id = str(self.parity_setup_id).strip()
        setup_hash = str(self.parity_setup_manifest_sha256).strip().lower()
        if setup_hash and not SHA256_RE.fullmatch(setup_hash):
            raise ValueError(
                "parity_setup_manifest_sha256 must be a SHA-256 hex digest"
            )
        self.parity_setup_manifest_sha256 = setup_hash


def resolve_hkl(path_text: str) -> tuple[str, str]:
    """Resolve a user path into an HKL file path and its XDS directory."""

    path = Path(path_text).expanduser()
    if path.is_dir():
        for name in ("INTEGRATE.HKL", "XDS_ASCII.HKL"):
            candidate = path / name
            if candidate.exists():
                return str(candidate.resolve()), str(path.resolve())
        return str((path / "INTEGRATE.HKL").resolve()), str(path.resolve())
    return str(path.resolve()), str(path.parent.resolve())


def parse_dataset_spec(text: str, *, default_hkl_transform: str = "identity", geometry_percentile: float = 99.0) -> DatasetSpec:
    """Parse a command-line dataset specification into a structured dataset record."""

    if "=" not in text:
        raise ValueError("--dataset must look like name=/path/to/xds_dir,dmin=0.70,hkl_transform=identity")
    name, rest = text.split("=", 1)
    raw_parts = [p.strip() for p in rest.split(",") if p.strip()]
    parts: list[str] = []
    for item in raw_parts:
        if not parts or "=" in item:
            parts.append(item)
        else:
            parts[-1] += "," + item
    if not parts:
        raise ValueError(f"dataset {name!r} has no path")
    hkl, xds_dir = resolve_hkl(parts[0])
    values: dict[str, str] = {}
    for item in parts[1:]:
        if "=" not in item:
            raise ValueError(f"dataset option {item!r} must be key=value")
        key, value = item.split("=", 1)
        values[key.strip().replace("-", "_")] = value.strip()
    unknown = sorted(set(values) - DATASET_OPTION_KEYS)
    if unknown:
        raise ValueError(f"unknown dataset option(s): {', '.join(unknown)}")
    return DatasetSpec(
        name=name.strip(),
        hkl=hkl,
        xds_dir=xds_dir,
        baseline_hkl=str(Path(values["baseline_hkl"]).expanduser().resolve()) if values.get("baseline_hkl") else "",
        hkl_transform=values.get("hkl_transform", default_hkl_transform),
        min_d_spacing=float(values.get("dmin", values.get("min_d_spacing", 0.70))),
        min_peak=float(values.get("min_peak", 90.0)),
        target_mode=values.get("target_mode", "iobs"),
        include_nonpositive=parse_bool_text(
            values.get("include_nonpositive", "false"),
            field="include_nonpositive",
        ),
        geometry_percentile=float(values.get("geometry_percentile", geometry_percentile)),
        physical_crystal_id=values.get("physical_crystal_id", ""),
        acquisition_fingerprint=values.get("acquisition_fingerprint", ""),
    )


def load_prep(path: Path) -> PrepConfig:
    """Load a GROND prep JSON file from disk."""

    data = json.loads(path.read_text())
    return PrepConfig(
        structure=data["structure"],
        baseline_hkl=data.get("baseline_hkl", ""),
        notes=data.get("notes", ""),
        default_hkl_transform=data.get("default_hkl_transform", "identity"),
        geometry_percentile=float(data.get("geometry_percentile", 99.0)),
        parity_setup_id=data.get("parity_setup_id", ""),
        parity_setup_manifest_sha256=data.get(
            "parity_setup_manifest_sha256", ""
        ),
        metadata=data.get("metadata", {}),
        datasets=[DatasetSpec(**row) for row in data.get("datasets", [])],
    )


def write_prep(path: Path, prep: PrepConfig) -> None:
    """Write a GROND prep JSON file with stable indentation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(prep), indent=2) + "\n")


def write_datasets_csv(path: Path, prep: PrepConfig, nuisance_by_name: dict[str, dict] | None = None) -> None:
    """Write a spreadsheet-friendly dataset and nuisance-parameter table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    nuisance_by_name = nuisance_by_name or {}
    fields = [
        "include",
        "name",
        "dataset_role",
        "dataset_weight",
        "endpoint_weight",
        "objective_rows",
        "hkl",
        "baseline_hkl",
        "hkl_transform",
        "parity_setup_id",
        "parity_setup_manifest_sha256",
        "physical_crystal_id",
        "acquisition_fingerprint",
        "split_contract",
        "min_d_spacing",
        "min_peak",
        "target_mode",
        "include_nonpositive",
        "geometry_percentile",
        "thickness_A",
        "thickness_model",
        "thickness_grad_frac",
        "thickness_sigma_frac",
        "frame_offset",
        "group_width_deg",
        "detector_scale_bins",
        "sigma_a",
        "sigma_b_corr",
        "sigma_c_peak",
        "sg_jitter_mode",
        "sg_jitter_sigma0",
        "sg_jitter_sigma_d_inv",
        "sg_jitter_sigma_frame",
        "sg_jitter_points",
        "absorption_mode",
        "absorption_mu",
        "dose_b_mode",
        "dose_b_per_frame",
        "row_resolution_b",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ds in prep.datasets:
            nui = nuisance_by_name.get(ds.name, {})
            writer.writerow(
                {
                    "include": str(nui.get("include", True)).lower(),
                    "name": ds.name,
                    "dataset_role": nui.get("dataset_role", "primary"),
                    "dataset_weight": nui.get("dataset_weight", nui.get("raw_row_weight", 1.0)),
                    "endpoint_weight": nui.get("endpoint_weight", nui.get("dataset_weight", nui.get("raw_row_weight", 1.0))),
                    "objective_rows": nui.get("objective_rows", ""),
                    "hkl": ds.hkl,
                    "baseline_hkl": ds.baseline_hkl,
                    "hkl_transform": ds.hkl_transform,
                    "parity_setup_id": prep.parity_setup_id,
                    "parity_setup_manifest_sha256": (
                        prep.parity_setup_manifest_sha256
                    ),
                    "physical_crystal_id": ds.physical_crystal_id,
                    "acquisition_fingerprint": ds.acquisition_fingerprint,
                    "split_contract": ds.split_contract,
                    "min_d_spacing": ds.min_d_spacing,
                    "min_peak": ds.min_peak,
                    "target_mode": ds.target_mode,
                    "include_nonpositive": str(ds.include_nonpositive).lower(),
                    "geometry_percentile": ds.geometry_percentile,
                    "thickness_A": nui.get("thickness_A", ""),
                    "thickness_model": nui.get("thickness_model", ""),
                    "thickness_grad_frac": nui.get("thickness_grad_frac", ""),
                    "thickness_sigma_frac": nui.get("thickness_sigma_frac", ""),
                    "frame_offset": nui.get("frame_offset", ""),
                    "group_width_deg": nui.get("group_width_deg", ""),
                    "detector_scale_bins": nui.get("detector_scale_bins", ""),
                    "sigma_a": nui.get("sigma_a", ""),
                    "sigma_b_corr": nui.get("sigma_b_corr", ""),
                    "sigma_c_peak": nui.get("sigma_c_peak", ""),
                    "sg_jitter_mode": nui.get("sg_jitter_mode", ""),
                    "sg_jitter_sigma0": nui.get("sg_jitter_sigma0", ""),
                    "sg_jitter_sigma_d_inv": nui.get("sg_jitter_sigma_d_inv", ""),
                    "sg_jitter_sigma_frame": nui.get("sg_jitter_sigma_frame", ""),
                    "sg_jitter_points": nui.get("sg_jitter_points", ""),
                    "absorption_mode": nui.get("absorption_mode", ""),
                    "absorption_mu": nui.get("absorption_mu", ""),
                    "dose_b_mode": nui.get("dose_b_mode", ""),
                    "dose_b_per_frame": nui.get("dose_b_per_frame", ""),
                    "row_resolution_b": nui.get("row_resolution_b", ""),
                }
            )
