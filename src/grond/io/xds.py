# ruff: noqa: E741
# h, k, l retain standard crystallographic Miller-index notation.
"""Signed XDS-like observations and their oriented scan geometry.

PETS and DIALS compatibility metadata remain explicit; ordinary integrated
observations are never silently mapped to an asymmetric unit here."""

from __future__ import annotations

import json
import math
from pathlib import Path

import gemmi
import numpy as np

from grond.bloch.numerics import cell_from_real_matrix, parse_hkl_transform


def rotation_matrix_from_header(header: dict[str, str], key: str, path: Path) -> np.ndarray:
    """Parse an optional proper 3x3 rotation matrix from an XDS-like header."""

    text = str(header.get(key, "")).strip()
    if not text:
        return np.eye(3, dtype=float)
    values = np.asarray([float(value) for value in text.split()], dtype=float)
    if values.shape != (9,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{key} must contain nine finite values in {path}")
    matrix = values.reshape(3, 3)
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError(f"{key} is not an orthogonal rotation matrix in {path}")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{key} is not a proper rotation matrix in {path}")
    return matrix


class XDSData:
    def __init__(self, hkl_path: Path, hkl_transform: str | None = None):
        self.path = Path(hkl_path)
        self.header = {}
        self.hkl_transform_spec = "identity"
        self.hkl_transform_matrix = np.eye(3, dtype=int)
        rows = []
        with open(self.path) as f:
            for line in f:
                if line.startswith("!"):
                    if "=" in line:
                        key, val = line[1:].split("=", 1)
                        self.header[key.strip()] = val.strip()
                    continue
                p = line.split()
                if len(p) < 11:
                    continue
                row = [float(x) for x in p[:21]]
                if len(row) == 12:
                    # XDS_ASCII.HKL (MERGE=FALSE) keeps signed, unmerged rows
                    # but only writes H K L I SIGMA XD YD ZD RLP PEAK CORR PSI.
                    # Pad to the INTEGRATE.HKL layout so downstream geometry and
                    # target code can use either file without changing row logic.
                    h, k, l, iobs, sig, xd, yd, zd, rlp, peak, corr, psi = row
                    row = [
                        h,
                        k,
                        l,
                        iobs,
                        sig,
                        xd,
                        yd,
                        zd,
                        rlp,
                        peak,
                        corr,
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        psi,
                        0.0,
                    ]
                elif len(row) < 21:
                    continue
                rows.append(tuple(row))
        if not rows:
            raise ValueError(f"no numeric reflection rows found in {self.path}")
        arr = np.asarray(rows, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 21:
            raise ValueError(f"unexpected reflection-table shape {arr.shape} in {self.path}")
        if not np.all(arr[:, :3] == np.rint(arr[:, :3])):
            raise ValueError(f"non-integer Miller indices found in {self.path}")
        self.hkl = arr[:, :3].astype(int)
        self.iobs = arr[:, 3].astype(float)
        self.sigma = arr[:, 4].astype(float)
        self.xcal = arr[:, 5].astype(float)
        self.ycal = arr[:, 6].astype(float)
        self.zcal = arr[:, 7].astype(float)
        self.rlp = arr[:, 8].astype(float)
        self.peak = arr[:, 9].astype(float)
        self.corr = arr[:, 10].astype(float)
        self.maxc = arr[:, 11].astype(float)
        self.xobs = arr[:, 12].astype(float)
        self.yobs = arr[:, 13].astype(float)
        self.zobs = arr[:, 14].astype(float)
        # INTEGRATE.HKL uses an all-zero observed centroid when XOBS/YOBS/ZOBS
        # are unavailable. It is not a measurement at the detector origin.
        # Normalize that sentinel once so geometry diagnostics leave the row
        # unscored while XCAL/YCAL/ZCAL remain available to the Bloch model.
        missing_observed_centroid = (self.xobs == 0.0) & (self.yobs == 0.0) & (self.zobs == 0.0)
        self.xobs[missing_observed_centroid] = np.nan
        self.yobs[missing_observed_centroid] = np.nan
        self.zobs[missing_observed_centroid] = np.nan
        self.alf0 = arr[:, 15].astype(float)
        self.bet0 = arr[:, 16].astype(float)
        self.alf1 = arr[:, 17].astype(float)
        self.bet1 = arr[:, 18].astype(float)
        self.psi = arr[:, 19].astype(float)
        self.iseg = arr[:, 20].astype(int)

        required = {
            "wavelength": ("X-RAY_WAVELENGTH", "WAVELENGTH"),
            "starting frame": ("STARTING_FRAME",),
            "starting angle": ("STARTING_ANGLE",),
            "oscillation range": ("OSCILLATION_RANGE",),
            "rotation axis": ("ROTATION_AXIS",),
            "incident beam direction": ("INCIDENT_BEAM_DIRECTION",),
            "unit-cell constants": ("UNIT_CELL_CONSTANTS",),
            "unit-cell a axis": ("UNIT_CELL_A-AXIS",),
            "unit-cell b axis": ("UNIT_CELL_B-AXIS",),
            "unit-cell c axis": ("UNIT_CELL_C-AXIS",),
        }
        missing = [
            label
            for label, aliases in required.items()
            if not any(str(self.header.get(alias, "")).strip() for alias in aliases)
        ]
        if missing:
            raise ValueError(f"missing required XDS header fields in {self.path}: {', '.join(missing)}")

        wavelength_text = self.header.get("X-RAY_WAVELENGTH") or self.header["WAVELENGTH"]
        self.wavelength = float(wavelength_text.split()[0])
        if not math.isfinite(self.wavelength) or self.wavelength <= 0.0:
            raise ValueError(f"XDS wavelength must be finite and positive in {self.path}")
        self.k = 1.0 / self.wavelength
        self.starting_frame = float(self.header["STARTING_FRAME"].split()[0])
        self.starting_angle = float(self.header["STARTING_ANGLE"].split()[0])
        self.oscillation_range = float(self.header["OSCILLATION_RANGE"].split()[0])
        if not all(math.isfinite(v) for v in (self.starting_frame, self.starting_angle, self.oscillation_range)):
            raise ValueError(f"XDS scan metadata must be finite in {self.path}")
        if self.oscillation_range == 0.0:
            raise ValueError(f"XDS oscillation range must be nonzero in {self.path}")
        axis_text = self.header.get(
            "GROND_ROTATION_AXIS_DATUM",
            self.header.get("DIALS_ROTATION_AXIS_DATUM", self.header["ROTATION_AXIS"]),
        )
        self.axis = np.array([float(x) for x in axis_text.split()[:3]], dtype=float)
        axis_norm = float(np.linalg.norm(self.axis))
        if self.axis.shape != (3,) or not np.all(np.isfinite(self.axis)) or axis_norm <= 0.0:
            raise ValueError(f"XDS rotation axis must be a finite nonzero 3-vector in {self.path}")
        self.axis = self.axis / axis_norm
        rotation_axis_lab = np.array([float(x) for x in self.header["ROTATION_AXIS"].split()[:3]], dtype=float)
        rotation_axis_lab_norm = float(np.linalg.norm(rotation_axis_lab))
        if (
            rotation_axis_lab.shape != (3,)
            or not np.all(np.isfinite(rotation_axis_lab))
            or rotation_axis_lab_norm <= 0.0
        ):
            raise ValueError(f"XDS rotation axis must be a finite nonzero 3-vector in {self.path}")
        self.rotation_axis_lab = rotation_axis_lab / rotation_axis_lab_norm
        fixed_rotation_key = (
            "GROND_FIXED_ROTATION" if self.header.get("GROND_FIXED_ROTATION") else "DIALS_FIXED_ROTATION"
        )
        setting_rotation_key = (
            "GROND_SETTING_ROTATION" if self.header.get("GROND_SETTING_ROTATION") else "DIALS_SETTING_ROTATION"
        )
        self.fixed_rotation = rotation_matrix_from_header(self.header, fixed_rotation_key, self.path)
        self.setting_rotation = rotation_matrix_from_header(self.header, setting_rotation_key, self.path)
        self.axis_cross = np.array(
            [
                [0.0, -self.axis[2], self.axis[1]],
                [self.axis[2], 0.0, -self.axis[0]],
                [-self.axis[1], self.axis[0], 0.0],
            ],
            dtype=float,
        )
        self.axis_cross2 = self.axis_cross @ self.axis_cross
        self.axis_eye = np.eye(3, dtype=float)
        beam_dir = np.array([float(x) for x in self.header["INCIDENT_BEAM_DIRECTION"].split()[:3]], dtype=float)
        beam_norm = float(np.linalg.norm(beam_dir))
        if beam_dir.shape != (3,) or not np.all(np.isfinite(beam_dir)) or beam_norm <= 0.0:
            raise ValueError(f"XDS incident beam direction must be a finite nonzero 3-vector in {self.path}")
        self.s0 = self.k * beam_dir / beam_norm
        cell_vals = [float(x) for x in self.header["UNIT_CELL_CONSTANTS"].split()[:6]]
        self.cell = gemmi.UnitCell(*cell_vals)
        va = np.array([float(x) for x in self.header["UNIT_CELL_A-AXIS"].split()[:3]], dtype=float)
        vb = np.array([float(x) for x in self.header["UNIT_CELL_B-AXIS"].split()[:3]], dtype=float)
        vc = np.array([float(x) for x in self.header["UNIT_CELL_C-AXIS"].split()[:3]], dtype=float)
        self.real_matrix = np.stack([va, vb, vc], axis=1)
        if not np.all(np.isfinite(self.real_matrix)):
            raise ValueError(f"XDS unit-cell axes contain non-finite values in {self.path}")
        try:
            self.A = np.linalg.inv(self.real_matrix).T
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"XDS unit-cell axes are singular in {self.path}") from exc
        self.A_at_scan_points = None
        self.a_scan_points_path = None
        scan_a_file = self.header.get("DIALS_A_AT_SCAN_POINTS_FILE", "").split()
        if scan_a_file:
            p = Path(scan_a_file[0])
            if not p.is_absolute():
                p = self.path.parent / p
            self.a_scan_points_path = p.resolve()
            mats = []
            with p.open() as f:
                for line_number, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        vals = [float(x) for x in line.split()]
                    except ValueError as exc:
                        raise ValueError(
                            f"DIALS A scan-point line {line_number} in {p} contains nonnumeric values"
                        ) from exc
                    if len(vals) != 9:
                        raise ValueError(
                            f"DIALS A scan-point line {line_number} in {p} must contain exactly nine values"
                        )
                    matrix = np.asarray(vals, dtype=float).reshape(3, 3)
                    if not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) <= 1e-15:
                        raise ValueError(f"DIALS A scan-point line {line_number} in {p} must be finite and nonsingular")
                    mats.append(matrix)
            if not mats:
                raise ValueError(f"declared DIALS A scan-point file {p} contains no matrices")
            self.A_at_scan_points = np.asarray(mats, dtype=float)
        self.pets_exact_geometry_path = None
        self.pets_exact_a_center = None
        self.pets_integration_semiangle_deg = None
        pets_geometry_file = self.header.get("PETS_EXACT_GEOMETRY_FILE", "").split()
        if pets_geometry_file:
            p = Path(pets_geometry_file[0])
            if not p.is_absolute():
                p = self.path.parent / p
            self.pets_exact_geometry_path = p.resolve()
            with np.load(self.pets_exact_geometry_path, allow_pickle=False) as loaded:
                required = {"a_center", "integration_semiangle_deg", "source_zone_id"}
                missing = sorted(required.difference(loaded.files))
                if missing:
                    raise ValueError(f"PETS exact geometry is missing arrays: {', '.join(missing)}")
                centers = np.asarray(loaded["a_center"], dtype=float)
                semiangles = np.asarray(loaded["integration_semiangle_deg"], dtype=float)
                zone_ids = np.asarray(loaded["source_zone_id"], dtype=np.int64)
            if centers.ndim != 3 or centers.shape[1:] != (3, 3):
                raise ValueError("PETS exact A centers must have shape (N,3,3)")
            if semiangles.shape != (len(centers),) or zone_ids.shape != (len(centers),):
                raise ValueError("PETS exact geometry arrays must have one row per zone")
            if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(semiangles)) or np.any(semiangles <= 0.0):
                raise ValueError("PETS exact geometry must be finite with positive semiangles")
            if not np.array_equal(zone_ids, np.arange(1, len(zone_ids) + 1)):
                raise ValueError("PETS exact geometry source zone IDs must be consecutive from one")
            self.pets_exact_a_center = centers
            self.pets_integration_semiangle_deg = semiangles
        self.variance_model = tuple(float(v) for v in self.header.get("VARIANCE_MODEL", "1.0 0.0").split()[:2])
        self.apply_hkl_transform(hkl_transform)

    def apply_hkl_transform(self, spec: str | None):
        T, canonical = parse_hkl_transform(spec)
        self.hkl_transform_spec = canonical
        self.hkl_transform_matrix = T
        self.raw_hkl = self.hkl.copy()
        self.raw_A = np.asarray(self.A, dtype=float).copy()
        self.raw_real_matrix = np.asarray(self.real_matrix, dtype=float).copy()
        self.raw_cell = gemmi.UnitCell(
            self.cell.a,
            self.cell.b,
            self.cell.c,
            self.cell.alpha,
            self.cell.beta,
            self.cell.gamma,
        )
        self.axis_derived_source_cell = cell_from_real_matrix(self.raw_real_matrix)
        self.axis_derived_source_cell_basis = "UNIT_CELL_[ABC]-AXIS"
        self.working_cell_basis = "axis_derived_oriented_model_basis"
        # T is an exact signed permutation, so T^-1 = T.T.  Apply that
        # algebra directly rather than introducing avoidable floating matrix
        # inversions.  UNIT_CELL_CONSTANTS and the oriented axes are rounded
        # independently by XDS-like writers; the Ewald geometry uses A, so the
        # working cell for every candidate must come from those same axes.
        # raw_cell remains the unedited UNIT_CELL_CONSTANTS provenance.
        inv_t = T.T.astype(float)
        self.hkl = (self.raw_hkl @ T.T).astype(int)
        self.A = self.raw_A @ inv_t
        self.real_matrix = self.raw_real_matrix @ inv_t
        self.cell = cell_from_real_matrix(self.real_matrix)
        if self.A_at_scan_points is not None:
            self.A_at_scan_points = np.einsum(
                "nij,jk->nik",
                np.asarray(self.A_at_scan_points, dtype=float),
                inv_t,
            )
        if self.pets_exact_a_center is not None:
            self.pets_exact_a_center = np.einsum(
                "nij,jk->nik",
                np.asarray(self.pets_exact_a_center, dtype=float),
                inv_t,
            )
        self.header["HKL_TRANSFORM_TO_MODEL_BASIS"] = canonical
        self.header["UNIT_CELL_CONSTANTS_MODEL_BASIS"] = (
            f"{self.cell.a:.6f} {self.cell.b:.6f} {self.cell.c:.6f} "
            f"{self.cell.alpha:.6f} {self.cell.beta:.6f} {self.cell.gamma:.6f}"
        )

    def omega_from_z(self, z):
        return self.starting_angle + (float(z) - self.starting_frame) * self.oscillation_range

    def z_from_omega(self, omega):
        return self.starting_frame + (float(omega) - self.starting_angle) / self.oscillation_range

    def A_for_omega(self, omega):
        if self.A_at_scan_points is None:
            return self.A
        scan_index = self.z_from_omega(omega) - self.starting_frame
        if scan_index <= 0:
            return self.A_at_scan_points[0]
        if scan_index >= len(self.A_at_scan_points) - 1:
            return self.A_at_scan_points[-1]
        lo = int(math.floor(scan_index))
        frac = float(scan_index) - lo
        return (1.0 - frac) * self.A_at_scan_points[lo] + frac * self.A_at_scan_points[lo + 1]

    def integration_profile_provenance(self):
        """Return profile/mosaic metadata without treating it as fitted physics.

        XDS reflecting range and beam divergence describe the integration
        profile, not a uniquely separable crystal mosaicity.  DIALS profile
        dictionaries are likewise retained in their native representation.
        The values are therefore suitable as reported provenance or bounded
        prior centres, but are never activated implicitly by the Bloch engine.
        """

        def first_float(key):
            text = str(self.header.get(key, "") or "").strip()
            if not text:
                return None
            try:
                value = float(text.split()[0])
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None

        profile_json = str(self.header.get("DIALS_PROFILE_PARAMETERS_JSON", "") or "").strip()
        dials_profile = None
        if profile_json:
            try:
                dials_profile = json.loads(profile_json)
            except json.JSONDecodeError:
                dials_profile = {"status": "unparsed", "raw": profile_json}
        reflecting = first_float("REFLECTING_RANGE_E.S.D.")
        divergence = first_float("BEAM_DIVERGENCE_E.S.D.")
        dials_sigma_m_values = []
        if isinstance(dials_profile, dict) and dials_profile.get("__id__") == "gaussian_rs":
            raw_sigma_m = dials_profile.get("sigma_m")
            if not isinstance(raw_sigma_m, (list, tuple)):
                raw_sigma_m = [raw_sigma_m]
            for raw_value in raw_sigma_m:
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value >= 0.0:
                    dials_sigma_m_values.append(value)
        if dials_sigma_m_values:
            prior_sigma = float(np.median(dials_sigma_m_values))
            prior_source = "DIALS GaussianRS sigma_m (degrees)"
        else:
            prior_sigma = reflecting
            prior_source = "XDS REFLECTING_RANGE_E.S.D. (degrees)"
        return {
            "source": str(self.header.get("GROND_SOURCE", "XDS") or "XDS").strip(),
            "profile_fitting": str(self.header.get("PROFILE_FITTING", "") or "").strip() or None,
            "xds": {
                "reflecting_range_esd_deg": reflecting,
                "beam_divergence_esd_deg": divergence,
            },
            "dials": {
                "profile_model": str(self.header.get("DIALS_PROFILE_MODEL", "") or "").strip() or None,
                "profile_parameters": dials_profile,
                "gaussian_rs_sigma_m_deg": dials_sigma_m_values or None,
                "frame_sliced_shoebox_available": (
                    str(self.header.get("DIALS_FRAME_SLICED_SHOEBOX_AVAILABLE", "FALSE")).strip().upper() == "TRUE"
                ),
            },
            "optional_prior": {
                "projected_mosaicity_sigma_deg": prior_sigma,
                "source": prior_source,
                "status": "provenance_only_not_activated",
                "warning": (
                    "The recorded integration-profile width is a prior centre, not an automatic "
                    "Bloch nuisance value. XDS reflecting range can contain multiple broadening "
                    "terms, while DIALS GaussianRS sigma_m remains tied to its integration model."
                ),
            },
        }

    def recorded_proxy(self):
        return self.iobs * np.maximum(self.peak, 0.0) / 100.0 / np.maximum(self.rlp, 1e-12)
