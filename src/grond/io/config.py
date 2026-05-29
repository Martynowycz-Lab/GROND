from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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


@dataclass
class PrepConfig:
    """A reusable run-preparation record for one structure and one or more datasets."""

    structure: str
    datasets: list[DatasetSpec]
    baseline_hkl: str = ""
    notes: str = ""
    default_hkl_transform: str = "identity"
    geometry_percentile: float = 99.0
    metadata: dict[str, Any] = field(default_factory=dict)


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
    return DatasetSpec(
        name=name.strip(),
        hkl=hkl,
        xds_dir=xds_dir,
        baseline_hkl=str(Path(values["baseline_hkl"]).expanduser().resolve()) if values.get("baseline_hkl") else "",
        hkl_transform=values.get("hkl_transform", default_hkl_transform),
        min_d_spacing=float(values.get("dmin", values.get("min_d_spacing", 0.70))),
        min_peak=float(values.get("min_peak", 90.0)),
        target_mode=values.get("target_mode", "iobs"),
        include_nonpositive=values.get("include_nonpositive", "false").lower() in {"1", "true", "yes", "on"},
        geometry_percentile=float(values.get("geometry_percentile", geometry_percentile)),
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
        "absorption_frac",
        "absorption_mu_g2",
        "absorption_mu_theta2",
        "absorption_tds_b",
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
                    "absorption_frac": nui.get("absorption_frac", ""),
                    "absorption_mu_g2": nui.get("absorption_mu_g2", ""),
                    "absorption_mu_theta2": nui.get("absorption_mu_theta2", ""),
                    "absorption_tds_b": nui.get("absorption_tds_b", ""),
                    "dose_b_mode": nui.get("dose_b_mode", ""),
                    "dose_b_per_frame": nui.get("dose_b_per_frame", ""),
                    "row_resolution_b": nui.get("row_resolution_b", ""),
                }
            )
