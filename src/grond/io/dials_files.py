"""Read DIALS integrated.refl and integrated.expt files without DIALS.

A reflection file (DIALS 2.0 and later) is msgpack: ["dials::af::reflection_table", 1, {"identifiers", "nrows",
"data"}], where each column is [type name, [rows, little-endian bytes]]. An experiment list is JSON. ``read_dials``
returns the arrays and metadata that the DIALS bridge converts. The values are those that DIALS itself reads from
the files; the static A matrix can differ from dxtbx ``Crystal.get_A()`` in the last bit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import gemmi
import msgpack
import numpy as np

# dials.array_family.flex.reflection_table.flags; a flag is set when all its bits are set
FLAGS = {
    "dont_integrate": 1 << 7, "integrated_sum": 1 << 8, "integrated_prf": 1 << 9, "overloaded": 1 << 10,
    "in_powder_ring": 1 << 13, "foreground_includes_bad_pixels": 1 << 14, "background_includes_bad_pixels": 1 << 15,
    "includes_bad_pixels": 0xC000, "bad_shoebox": 0xFC00, "failed_during_background_modelling": 1 << 18,
    "failed_during_summation": 1 << 19, "outlier_in_scaling": 1 << 23, "excluded_for_scaling": 1 << 24,
    "bad_for_scaling": 0x1C00000, "scaled": 1 << 26,
}
COLUMN_TYPES = {
    "double": ("<f8", ()), "int": ("<i4", ()), "std::size_t": ("<u8", ()), "bool": ("?", ()),
    "vec2<double>": ("<f8", (2,)), "vec3<double>": ("<f8", (3,)), "mat3<double>": ("<f8", (9,)),
    "int6": ("<i4", (6,)), "cctbx::miller::index<>": ("<i4", (3,)),
}
# The class names of the profile models that DIALS reads back (dials.extensions)
PROFILE_CLASSES = {"gaussian_rs": "Model", "ellipsoid": "EllipsoidProfileModel"}
PI_180 = math.atan(1.0) / 45.0  # scitbx constants::pi_180; dxtbx keeps scan angles in radians


def read_reflections(path: Path) -> tuple[int, dict[str, np.ndarray], set[str]]:
    """Row count, numeric columns and all column names of a DIALS reflection file (shoeboxes are not read)."""

    try:
        table = msgpack.unpackb(Path(path).read_bytes(), raw=False, strict_map_key=False)
    except (msgpack.ExtraData, msgpack.FormatError, msgpack.StackError, ValueError) as exc:
        raise ValueError(f"{path} is not a DIALS reflection file (DIALS 2.0 or later)") from exc
    if not (isinstance(table, list) and len(table) == 3 and table[0] == "dials::af::reflection_table"):
        raise ValueError(f"{path} is not a DIALS reflection file (DIALS 2.0 or later)")
    nrows = int(table[2]["nrows"])
    columns = {}
    for name, (type_name, (rows, blob)) in table[2]["data"].items():
        if type_name in COLUMN_TYPES:
            dtype, shape = COLUMN_TYPES[type_name]
            columns[name] = np.frombuffer(blob, dtype=dtype).reshape((int(rows),) + shape)
    return nrows, columns, set(table[2]["data"])


def _compose(local: dict, parent: tuple) -> tuple:
    """Lab frame (fast, slow, origin) of a panel or group from its frame local to the parent group."""

    fast, slow, normal, origin = parent
    to_lab = np.column_stack((fast, slow, normal))
    return (to_lab @ np.asarray(local["fast_axis"], float), to_lab @ np.asarray(local["slow_axis"], float),
            origin + to_lab @ np.asarray(local["origin"], float))


def _parent_frames(group: dict, frame: tuple, frames: dict) -> None:
    """Map each panel index to the lab frame of its parent group."""

    fast, slow, origin = frame
    parent = (fast, slow, np.cross(fast, slow), origin)
    for child in group.get("children", []):
        if "panel" in child:
            frames[child["panel"]] = parent
        else:
            _parent_frames(child, _compose(child, parent), frames)


def _panel0(detector: dict | None) -> dict:
    """The first panel in the lab frame, as dxtbx gives it."""

    if not detector or not detector.get("panels"):
        return {}
    root = detector.get("hierarchy") or {}
    frame = tuple(np.asarray(root.get(key, default), float)
                  for key, default in (("fast_axis", [1, 0, 0]), ("slow_axis", [0, 1, 0]), ("origin", [0, 0, 0])))
    frames: dict = {}
    _parent_frames(root, frame, frames)
    panel = detector["panels"][0]
    fast, slow, origin = _compose(panel, frames.get(0, (frame[0], frame[1], np.cross(frame[0], frame[1]), frame[2])))
    return {
        "n_panels": len(detector["panels"]),
        "panel0_name": str(panel.get("name", "")),
        "origin_mm": [float(v) for v in origin],
        "fast_axis": [float(v) for v in fast],
        "slow_axis": [float(v) for v in slow],
        "pixel_size_mm": [float(v) for v in panel["pixel_size"]],
        "image_size_px": [int(v) for v in panel["image_size"]],
        "distance_mm": float(np.dot(origin, np.cross(fast, slow))),
        "gain": float(panel.get("gain", 1.0)),
    }


def _oscillation(scan: dict) -> list[float]:
    """(start, width) in degrees, as dxtbx ``Scan.get_oscillation()`` gives it."""

    if "oscillation" in scan:  # files from before the scan properties table
        return [float(v) for v in scan["oscillation"]]
    starts = [float(v) * PI_180 for v in scan["properties"]["oscillation"]]
    if len(starts) > 1:
        return [starts[0] / PI_180, (starts[1] - starts[0]) / PI_180]
    return [starts[0] / PI_180, float(scan["properties"]["oscillation_width"][0]) * PI_180 / PI_180]


def _space_group(hall: str) -> tuple[str, int | None]:
    """Name and number of a Hall symbol. The name is written as cctbx writes it ("R -3 c :H")."""

    group = gemmi.find_spacegroup_by_ops(gemmi.symops_from_hall(hall))
    if group is None:
        return f"Hall: {hall}", None
    return group.xhm().replace(":", " :"), int(group.number)


def _unit_cell(a: list[float], b: list[float], c: list[float]) -> list[float]:
    """Cell parameters from the real-space axes, with the arithmetic of dxtbx ``Crystal.set_unit_cell``."""

    def dot(u, v):
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

    def length(u):
        return math.sqrt(dot(u, u))

    def angle(u, v):
        return math.acos(dot(u, v) / (length(u) * length(v))) / PI_180

    return [length(a), length(b), length(c), angle(b, c), angle(c, a), angle(a, b)]


def _profile(profile: dict | None) -> dict:
    """The profile model as DIALS reads it back: its class name and ``to_dict()``.

    DIALS keeps the gaussian_rs widths in radians, so their degrees come back through ``* pi / 180 * 180 / pi``.
    """

    if profile is None:
        return {"model": None, "parameters": None}
    name = profile.get("__id__")
    if name != "gaussian_rs":
        return {"model": PROFILE_CLASSES.get(name, name), "parameters": profile}

    def degrees(value):
        if isinstance(value, list):
            return [v * math.pi / 180.0 * 180.0 / math.pi for v in value]
        return value * math.pi / 180.0 * 180.0 / math.pi

    parameters = {"__id__": name, "n_sigma": profile["n_sigma"], "sigma_b": degrees(profile["sigma_b"]),
                  "sigma_m": degrees(profile["sigma_m"])}
    return {"model": PROFILE_CLASSES[name], "parameters": parameters}


def read_dials(refl: Path, expt: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read one integrated.refl/integrated.expt pair into (arrays, metadata) for the DIALS bridge.

    Missing observed centroids (xyzobs.px.value) are replaced by the calculated ones (xyzcal.px). Several
    experiments or a scan-varying goniometer setting rotation stop the conversion.
    """

    nrows, cols, names = read_reflections(refl)
    if "miller_index" not in cols:
        raise ValueError(f"{refl} has no miller_index column. Give integrated.refl or scaled.refl.")
    experiments = json.loads(Path(expt).read_text())
    count = len(experiments.get("experiment", []))
    if count != 1:
        raise ValueError(
            f"GROND import-dials expects one experiment per dataset; {expt} has {count}. "
            "Split multi-experiment DIALS outputs into one GROND dataset per experiment first."
        )
    exp = experiments["experiment"][0]

    def model(kind):
        index = exp.get(kind)
        return experiments[kind][index] if index is not None and experiments.get(kind) else None

    def column(name, default, dtype=float, shape=()):
        if name in cols:
            return cols[name].astype(dtype)
        return np.full((nrows,) + shape, default, dtype=dtype)

    flags = cols.get("flags", np.zeros(nrows, dtype=np.uint64))
    xyzcal = column("xyzcal.px", np.nan, shape=(3,))
    xyzobs = column("xyzobs.px.value", np.nan, shape=(3,))
    missing = ~np.all(np.isfinite(xyzobs), axis=1)
    xyzobs[missing] = xyzcal[missing]
    arrays = {
        "hkl": cols["miller_index"].astype(np.int32),
        "i_sum": column("intensity.sum.value", np.nan),
        "var_sum": column("intensity.sum.variance", np.nan),
        "i_prf": column("intensity.prf.value", np.nan),
        "var_prf": column("intensity.prf.variance", np.nan),
        "i_scale": column("intensity.scale.value", np.nan),
        "var_scale": column("intensity.scale.variance", np.nan),
        "inverse_scale_factor": column("inverse_scale_factor", np.nan),
        "inverse_scale_factor_variance": column("inverse_scale_factor_variance", np.nan),
        "lp": column("lp", 1.0),
        "qe": column("qe", 1.0) if "qe" in cols else column("dqe", 1.0),
        "partiality": column("partiality", 1.0),
        "zeta": column("zeta", np.nan),
        "d": column("d", np.nan),
        "bbox": column("bbox", -1, np.int32, (6,)),
        "xyzcal": xyzcal,
        "xyzobs": xyzobs,
        "panel": column("panel", 0, np.int32),
        "id": column("id", 0, np.int32),
    }
    for name in ("integrated_sum", "integrated_prf", "dont_integrate", "overloaded", "in_powder_ring",
                 "excluded_for_scaling", "bad_for_scaling", "outlier_in_scaling", "scaled",
                 "background_includes_bad_pixels", "foreground_includes_bad_pixels", "includes_bad_pixels",
                 "bad_shoebox", "failed_during_background_modelling", "failed_during_summation"):
        bits = np.uint64(FLAGS[name])
        arrays[name] = (flags & bits) == bits

    crystal, beam, goniometer, scan = model("crystal"), model("beam"), model("goniometer"), model("scan")
    if crystal is None or beam is None:
        raise ValueError(f"{expt} has no crystal or beam model. Give integrated.expt or scaled.expt.")
    axes = [crystal[f"real_space_{axis}"] for axis in "abc"]
    a_static = np.linalg.inv(np.asarray(axes, dtype=float))
    a_scan_points = [np.asarray(a, float).reshape(3, 3).tolist() for a in crystal.get("A_at_scan_points", [])]
    if not a_scan_points:
        a_scan_points = [a_static.tolist()]
    space_group, space_group_number = _space_group(crystal["space_group_hall_symbol"])

    direction = [float(v) for v in beam["direction"]]
    norm = math.sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
    direction = [v / norm for v in direction]
    wavelength = float(beam["wavelength"])
    identity = np.eye(3).ravel()
    if goniometer is None:
        datum = axis = [1.0, 0.0, 0.0]
        fixed = setting = identity
        setting_points = 0
    else:
        datum = [float(v) for v in goniometer["rotation_axis"]]
        fixed = np.asarray(goniometer.get("fixed_rotation", identity), float)
        setting = np.asarray(goniometer.get("setting_rotation", identity), float)
        axis = [float(setting[3 * i] * datum[0] + setting[3 * i + 1] * datum[1] + setting[3 * i + 2] * datum[2])
                for i in range(3)]
        setting_points = len(goniometer.get("setting_rotation_at_scan_points") or [])
    if setting_points:
        raise ValueError(
            "GROND import-dials does not support scan-varying goniometer setting rotations; "
            f"the experiment has {setting_points} setting-rotation scan points. "
            "Export a static setting rotation or split the experiment before conversion."
        )
    if scan is not None:
        image_range = [int(v) for v in scan["image_range"]]
        array_range = [image_range[0] - 1, image_range[1]]
        oscillation = _oscillation(scan)
    else:
        image_range, array_range, oscillation = [1, nrows], [0, nrows], [0.0, 1.0]
    meta = {
        "source_refl": str(refl),
        "source_expt": str(expt),
        "row_count": nrows,
        "experiment_count": 1,
        "experiment_identifier": str(exp.get("identifier", "") or ""),
        "space_group": space_group,
        "space_group_number": space_group_number,
        "unit_cell": _unit_cell(*axes),
        "A_static": a_static.tolist(),
        "A_scan_points": a_scan_points,
        "num_scan_points": len(a_scan_points),
        "beam": {"wavelength_A": wavelength, "s0": [-v / wavelength for v in direction],
                 "sample_to_source_direction": direction},
        "goniometer": {"rotation_axis": axis, "rotation_axis_datum": datum,
                       "fixed_rotation": np.reshape(fixed, (3, 3)).tolist(),
                       "setting_rotation": np.reshape(setting, (3, 3)).tolist(),
                       "setting_rotation_scan_points": setting_points},
        "scan": {"image_range": image_range, "array_range": array_range, "oscillation": oscillation},
        "detector": _panel0(model("detector")),
        "profile": _profile(model("profile")),
        "frame_sliced_shoebox": {"available": "frame_sliced_shoebox" in names, "used_for_intensity": False,
                                 "status": "recorded_for_future_rocking_profile_import"},
    }
    return arrays, meta
