# ruff: noqa: E741
# h, k, l retain standard crystallographic Miller-index notation.
"""Signed XDS-like observations and their oriented scan geometry.

DIALS compatibility metadata remain explicit; ordinary integrated
observations are never silently mapped to an asymmetric unit here."""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

import gemmi
import numpy as np

from grond.bloch.numerics import cell_from_real_matrix, parse_hkl_transform, sha256


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


# Programs whose frame coordinate (ZCAL, ZOBS, ZD) follows the XDS convention: image n spans z in [n - 1, n], so
# the spindle angle at z is STARTING_ANGLE + (z - STARTING_FRAME + 1) * OSCILLATION_RANGE.  XDS INTEGRATE and
# CORRECT write it; dials.export (format=xds_ascii) writes DIALS array z unchanged beside STARTING_FRAME =
# image_range[0], which is the same convention, and dials.import_xds reads ZCAL back as DIALS array z.
XDS_FRAME_CONVENTION_WRITERS = ("INTEGRATE", "CORRECT", "XSCALE", "DIALS.EXPORT")
ZCAL_CONVENTIONS = ("xds", "scan_start")


def resolve_zcal_frame_convention(header: dict[str, str], provenance: list[str]) -> tuple[str, str]:
    """Return how the file's frame coordinate maps to the spindle angle, and why.

    ``xds``: image n spans z in [n - 1, n]; phi(z) = STARTING_ANGLE + (z - STARTING_FRAME + 1) * OSCILLATION_RANGE.
    ``scan_start``: phi(z) = STARTING_ANGLE + (z - STARTING_FRAME) * OSCILLATION_RANGE, i.e. z = STARTING_FRAME is
    the start of the scan.  GROND's own writers use it: the DIALS bridge writes the dxtbx array z with
    STARTING_FRAME = array_range[0], and the portable row contract writes its own starting frame.  A file without a
    recognised writer keeps this legacy mapping; ``!GROND_ZCAL_CONVENTION=`` states it explicitly.
    """

    explicit = str(header.get("GROND_ZCAL_CONVENTION", "") or "").strip().lower()
    if explicit:
        if explicit not in ZCAL_CONVENTIONS:
            raise ValueError(f"GROND_ZCAL_CONVENTION must be one of {', '.join(ZCAL_CONVENTIONS)}; got {explicit!r}")
        return explicit, "declared by !GROND_ZCAL_CONVENTION="
    grond_source = str(header.get("GROND_SOURCE", "") or "").strip()
    generated_by = str(header.get("GENERATED_BY", "") or "").strip()
    file_format = str(header.get("FORMAT", "") or "").strip()
    if grond_source or generated_by.upper().startswith("GROND") or file_format.upper().startswith("GROND"):
        writer = grond_source or generated_by or file_format
        return "scan_start", f"GROND-written file ({writer}); z = STARTING_FRAME is the scan start"
    for line in provenance:
        words = line.split()
        program = words[2].upper() if len(words) >= 3 and [w.lower() for w in words[:2]] == ["generated", "by"] else ""
        if program in XDS_FRAME_CONVENTION_WRITERS:
            return "xds", f"written by {words[2]}; image n spans z in [n - 1, n]"
    return "scan_start", "no recognised writer line; legacy mapping z = STARTING_FRAME at the scan start"


class XDSData:
    """Signed, unmerged XDS observations with the scan geometry read from the file header.

    Reads INTEGRATE.HKL (21 columns) or unmerged XDS_ASCII.HKL (12 columns, padded to the
    INTEGRATE.HKL layout). Angles are in degrees; ZCAL is a fractional frame number, and the
    rotation angle is STARTING_ANGLE + (z - zcal_angle_origin) * OSCILLATION_RANGE, where
    zcal_angle_origin is STARTING_FRAME - 1 in the XDS convention (image n spans z n-1..n) and
    STARTING_FRAME for GROND-written files (resolve_zcal_frame_convention).
    """

    def __init__(self, hkl_path: Path, hkl_transform: str | None = None):
        """Read the rows and header of ``hkl_path``, then reindex with ``hkl_transform`` (identity by default).

        Header extensions written by GROND's DIALS bridge (scan-point A matrices,
        goniometer fixed and setting rotations) are loaded when present. An
        all-zero observed centroid is stored as NaN.
        """

        self.path = Path(hkl_path)
        self._source_files = []
        self.header = {}
        self.hkl_transform_spec = "identity"
        self.hkl_transform_matrix = np.eye(3, dtype=int)
        self.provenance_lines = []
        rows = []
        with io.StringIO(self._read_source_file("hkl", self.path).decode()) as f:
            for line in f:
                if line.startswith("!"):
                    if line[1:].strip().lower().startswith("generated by"):
                        self.provenance_lines.append(line[1:].strip())
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
        self.zcal_frame_convention, self.zcal_frame_convention_reason = resolve_zcal_frame_convention(
            self.header, self.provenance_lines
        )
        # The frame coordinate at which the spindle angle equals STARTING_ANGLE.
        self.zcal_angle_origin = (
            self.starting_frame - 1.0 if self.zcal_frame_convention == "xds" else self.starting_frame
        )
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
            with io.StringIO(self._read_source_file("scan_matrix", self.a_scan_points_path).decode()) as f:
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
        self.variance_model = tuple(float(v) for v in self.header.get("VARIANCE_MODEL", "1.0 0.0").split()[:2])
        self.apply_hkl_transform(hkl_transform)
        self._source_files = tuple(self._source_files)

    def _read_source_file(self, label: str, path: Path) -> bytes:
        """Read one input snapshot and bind its hash to the bytes being parsed."""

        path = path.resolve()
        source_bytes = path.read_bytes()
        self._source_files.append((label, path, hashlib.sha256(source_bytes).hexdigest()))
        return source_bytes

    def verify_source_files(self) -> dict[str, str]:
        """Return loaded file hashes, refusing changed or missing source inputs.

        A cache must not bind previously loaded rows or geometry to replacement
        files. Re-read XDSData after an intentional input change.
        """

        hashes = {}
        for label, path, loaded_hash in self._source_files:
            try:
                current_hash = sha256(path)
            except OSError as exc:
                raise ValueError(f"loaded {label} source is no longer readable: {path}; reload XDSData") from exc
            if current_hash != loaded_hash:
                raise ValueError(f"loaded {label} source changed after reading: {path}; reload XDSData")
            hashes[label] = loaded_hash
        return hashes

    def apply_hkl_transform(self, spec: str | None):
        """Reindex the rows with a signed HKL permutation and move A, the real axes and the cell with them.

        The unindexed values are kept as ``raw_hkl``, ``raw_A``, ``raw_real_matrix`` and
        ``raw_cell``.
        """

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
        self.header["HKL_TRANSFORM_TO_MODEL_BASIS"] = canonical
        self.header["UNIT_CELL_CONSTANTS_MODEL_BASIS"] = (
            f"{self.cell.a:.6f} {self.cell.b:.6f} {self.cell.c:.6f} "
            f"{self.cell.alpha:.6f} {self.cell.beta:.6f} {self.cell.gamma:.6f}"
        )

    def omega_from_z(self, z):
        """Return the rotation angle (degrees) at fractional frame ``z`` (ZCAL/ZOBS units) in this file's convention."""

        return self.starting_angle + (float(z) - self.zcal_angle_origin) * self.oscillation_range

    def z_from_omega(self, omega):
        """Return the fractional frame number at rotation angle ``omega`` (degrees); the inverse of ``omega_from_z``."""

        return self.zcal_angle_origin + (float(omega) - self.starting_angle) / self.oscillation_range

    def A_for_omega(self, omega):
        """Return the orientation matrix A at ``omega``, interpolating linearly between scan-
        point matrices when present.
        """

        if self.A_at_scan_points is None:
            return self.A
        # Scan point i sits at the spindle angle STARTING_ANGLE + i * OSCILLATION_RANGE.
        scan_index = self.z_from_omega(omega) - self.zcal_angle_origin
        if scan_index <= 0:
            return self.A_at_scan_points[0]
        if scan_index >= len(self.A_at_scan_points) - 1:
            return self.A_at_scan_points[-1]
        lo = int(math.floor(scan_index))
        frac = float(scan_index) - lo
        return (1.0 - frac) * self.A_at_scan_points[lo] + frac * self.A_at_scan_points[lo + 1]

    def recorded_proxy(self):
        """Return IOBS * (PEAK/100) / RLP, an estimate of the recorded (uncorrected) intensity of each row."""

        return self.iobs * np.maximum(self.peak, 0.0) / 100.0 / np.maximum(self.rlp, 1e-12)
