"""Validate model conventions and prepare legacy-compatible model inputs."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from grond.cli_support.execution import run_logged
from grond.cli_support.values import DEFAULT_ELECTRON_WAVELENGTH_A, try_read_json
from grond.hashing import sha256_file
from grond.paths import engine_script
from grond.workflows.decision import write_json


def _cif_number(text: str) -> int | None:
    """Parse a CIF integer field that may carry uncertainty notation."""

    cleaned = str(text or "").strip().strip("'\"")
    if not cleaned or cleaned in {".", "?"}:
        return None
    cleaned = cleaned.split("(", 1)[0]
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _is_number(text: str) -> bool:
    """Return true when text can be parsed as a floating-point number."""

    try:
        float(str(text).replace("D", "E").replace("d", "e"))
        return True
    except ValueError:
        return False


def _shelx_cell_wavelength_status(value: object) -> dict[str, object]:
    """Classify a SHELX CELL wavelength for GROND's electron endpoint contract."""

    try:
        wavelength = float(str(value).replace("D", "E").replace("d", "e"))
    except (TypeError, ValueError):
        return {
            "status": "warning",
            "wavelength_A": None,
            "endpoint_wavelength_A": DEFAULT_ELECTRON_WAVELENGTH_A,
            "message": "CELL wavelength is not numeric; endpoint INS files will use the GROND electron default.",
        }
    if not math.isfinite(wavelength) or wavelength <= 0.0 or abs(wavelength - 1.0) < 1e-6 or wavelength >= 0.1:
        return {
            "status": "warning",
            "wavelength_A": wavelength,
            "endpoint_wavelength_A": DEFAULT_ELECTRON_WAVELENGTH_A,
            "message": "CELL wavelength is a placeholder or non-electron value; endpoint INS files will use the GROND electron default.",
        }
    return {
        "status": "ok",
        "wavelength_A": wavelength,
        "endpoint_wavelength_A": wavelength,
        "message": "CELL wavelength is in the expected electron range.",
    }


def check_sfac_contract(structure: str | Path) -> dict[str, object]:
    """Check whether a model declares explicit numeric SFAC rows for GROND.

    SHELXL accepts element-only lines such as ``SFAC C H N O`` by using its
    built-in scattering table.  GROND's Bloch calculator needs explicit
    electron-scattering coefficient rows, so fail this case early and clearly.
    """

    path = Path(structure).expanduser()
    report: dict[str, object] = {
        "schema_version": 1,
        "structure": str(path),
        "status": "ok",
        "ok": True,
        "default_electron_wavelength_A": DEFAULT_ELECTRON_WAVELENGTH_A,
        "sfac_rows": [],
        "messages": [],
    }
    messages: list[dict[str, str]] = []
    sfac_rows: list[dict[str, object]] = []
    endpoint_contract: dict[str, object] = {
        "hklf": None,
        "hklf_status": "missing",
        "stripped_endpoint_commands": [],
        "unsupported_features": [],
    }
    suffix = path.suffix.lower()
    if suffix == ".cif":
        try:
            embedded_shelx = _extract_cif_shelx_block(path)
        except Exception as exc:
            report.update(
                status="error",
                ok=False,
                messages=[
                    {
                        "severity": "error",
                        "message": f"could not read CIF model: {exc}",
                    }
                ],
            )
            return report
        if embedded_shelx:
            with tempfile.TemporaryDirectory(prefix="grond_sfac_check_") as tmp:
                embedded_path = Path(tmp) / "embedded.res"
                embedded_path.write_text(embedded_shelx)
                embedded_report = check_sfac_contract(embedded_path)
            embedded_report["structure"] = str(path)
            embedded_report["cif_model_contract"] = "embedded_shelx"
            embedded_report.setdefault("messages", []).insert(
                0,
                {
                    "severity": "info",
                    "message": "Checked the complete SHELX instruction block embedded in the CIF.",
                },
            )
            return embedded_report
        try:
            from grond.engine.bloch_engine import parse_cif

            model = parse_cif(path)
        except (OSError, RuntimeError, ValueError) as exc:
            report.update(
                status="error",
                ok=False,
                messages=[
                    {
                        "severity": "error",
                        "message": f"CIF is not a usable electron-scattering model: {exc}",
                    }
                ],
            )
            return report
        report.update(
            status="warning",
            ok=True,
            cif_model_contract="explicit_electron_coefficients",
            sfac_rows=[
                {
                    "element": str(row["element"]),
                    "kind": "numeric_coefficients",
                    "coefficient_count": len(row.get("raw", [])),
                }
                for row in model["sfac"]
            ],
            cell={
                "status": "warning",
                "wavelength_A": None,
                "endpoint_wavelength_A": DEFAULT_ELECTRON_WAVELENGTH_A,
                "message": "CIF has no SHELX CELL wavelength; generated endpoint files use the GROND electron default.",
            },
            messages=[
                {
                    "severity": "warning",
                    "message": "The CIF has explicit electron coefficients and is usable by the lower-level Bloch parser, but GROND run needs a complete embedded SHELX instruction block for model-update and endpoint files. Provide that block or use a RES/INS model.",
                }
            ],
        )
        return report
    if suffix not in {".res", ".ins"}:
        report.update(
            status="warning",
            ok=True,
            messages=[{"severity": "warning", "message": "SFAC check only audits SHELX .res/.ins models."}],
        )
        return report
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return {
            **report,
            "status": "error",
            "ok": False,
            "messages": [{"severity": "error", "message": f"could not read model: {exc}"}],
        }

    cell_seen = False
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(("!", "#")):
            continue
        parts = line.split()
        head = parts[0].upper()
        if head == "ACTA":
            endpoint_contract["stripped_endpoint_commands"].append({"line": raw, "command": "ACTA"})
            messages.append(
                {
                    "severity": "warning",
                    "message": "ACTA is accepted in the input model but stripped from GROND endpoint files because it can interfere with zero-cycle/check runs and is not needed for corrected-HKL refinement.",
                }
            )
            continue
        if head == "HKLF":
            hklf = parts[1] if len(parts) > 1 else ""
            endpoint_contract["hklf"] = hklf
            if hklf == "4":
                endpoint_contract["hklf_status"] = "ok"
            elif hklf == "3":
                endpoint_contract["hklf_status"] = "source_amplitudes_endpoint_forced_hklf4"
                messages.append(
                    {
                        "severity": "warning",
                        "message": "Input model uses HKLF 3 amplitudes. GROND can read the model, but every corrected endpoint is written as HKLF 4 intensities.",
                    }
                )
            else:
                endpoint_contract["hklf_status"] = "unsupported"
                endpoint_contract["unsupported_features"].append({"line": raw, "feature": f"HKLF {hklf or '?'}"})
                messages.append(
                    {
                        "severity": "error",
                        "message": f"HKLF {hklf or '?'} is not supported. GROND currently writes corrected intensity endpoints as HKLF 4; HKLF 5/twin endpoints are a current limitation.",
                    }
                )
            continue
        if head in {"BASF", "TWIN", "SWAT"}:
            endpoint_contract["unsupported_features"].append({"line": raw, "feature": head})
            messages.append(
                {
                    "severity": "error",
                    "message": f"{head} is not supported by GROND corrected-HKLF4 endpoints yet. Twin/BASF/SWAT refinement is a current limitation.",
                }
            )
            continue
        if head == "SQUEEZE" or "SQUEEZE" in line.upper():
            endpoint_contract["unsupported_features"].append({"line": raw, "feature": "SQUEEZE"})
            messages.append(
                {
                    "severity": "error",
                    "message": "SQUEEZE/solvent-mask corrected SHELX models are not supported by GROND corrected-HKLF4 endpoints yet.",
                }
            )
            continue
        if head == "CELL":
            cell_seen = True
            if len(parts) >= 2:
                status = _shelx_cell_wavelength_status(parts[1])
                report["cell"] = status
                if status["status"] != "ok":
                    messages.append({"severity": "warning", "message": str(status["message"])})
            else:
                messages.append({"severity": "error", "message": "CELL line is missing a wavelength."})
            continue
        if head != "SFAC":
            if head == "END":
                break
            continue
        if len(parts) < 2:
            sfac_rows.append({"line": raw, "status": "error", "reason": "empty_sfac"})
            messages.append({"severity": "error", "message": "SFAC line is empty."})
            continue
        if len(parts) >= 3 and _is_number(parts[2]):
            element = parts[1].capitalize()
            coeff_count = len(parts) - 2
            row_status = "ok" if coeff_count >= 8 else "error"
            row = {
                "line": raw,
                "status": row_status,
                "kind": "numeric_coefficients",
                "element": element,
                "coefficient_count": coeff_count,
            }
            sfac_rows.append(row)
            if coeff_count < 8:
                messages.append(
                    {
                        "severity": "error",
                        "message": f"SFAC {element} has {coeff_count} numeric coefficients; GROND needs explicit electron rows with at least 8 numbers.",
                    }
                )
            continue
        elements = [token.capitalize() for token in parts[1:]]
        sfac_rows.append({"line": raw, "status": "error", "kind": "element_only", "elements": elements})
        messages.append(
            {
                "severity": "error",
                "message": "Element-only SHELX SFAC lines use built-in SHELXL factors. Replace them with explicit electron scattering coefficient rows before running GROND.",
            }
        )
    if not cell_seen:
        messages.append({"severity": "error", "message": "No CELL line was found."})
    if not sfac_rows:
        messages.append({"severity": "error", "message": "No SFAC rows were found."})
    report["sfac_rows"] = sfac_rows
    report["endpoint_contract"] = endpoint_contract
    report["messages"] = messages
    has_error = any(item.get("severity") == "error" for item in messages) or any(
        row.get("status") == "error" for row in sfac_rows
    )
    has_warning = any(item.get("severity") == "warning" for item in messages)
    report["ok"] = not has_error
    report["status"] = "error" if has_error else ("warning" if has_warning else "ok")
    return report


def absolute_structure_policy(structure: str) -> dict:
    """Decide whether an alternate hand check makes sense for a model."""

    path = Path(structure)
    suffix = path.suffix.lower()
    policy = {
        "structure": str(path),
        "run_hand_control": True,
        "run_absolute_structure_control": True,
        "reason": "alternate hand check may be informative for acentric models",
    }
    try:
        if suffix in {".res", ".ins"}:
            shelx_lines = path.read_text(errors="replace").splitlines()
            chiv_lines = [
                line.strip()
                for line in shelx_lines
                if line.strip().split() and line.strip().split()[0].upper() == "CHIV"
            ]
            if chiv_lines:
                policy.update(
                    run_hand_control=False,
                    run_absolute_structure_control=False,
                    reason=(
                        "SHELX model contains signed CHIV restraints; automatic hand inversion is disabled "
                        "because GROND does not yet transform CHIV targets"
                    ),
                    unsupported_hand_records=chiv_lines,
                )
                return policy
            for line in shelx_lines:
                parts = line.strip().split()
                if parts and parts[0].upper() == "LATT" and len(parts) > 1:
                    latt = int(float(parts[1]))
                    policy["shelx_latt"] = latt
                    if latt > 0:
                        policy.update(
                            run_hand_control=False,
                            run_absolute_structure_control=False,
                            reason="SHELX LATT is centrosymmetric; alternate hand check is not meaningful",
                        )
                    else:
                        policy.update(
                            run_hand_control=True,
                            run_absolute_structure_control=True,
                            reason="SHELX LATT is acentric; alternate hand check is applicable",
                        )
                    return policy
            policy["reason"] = "no SHELX LATT record found; keeping alternate hand control enabled"
            return policy
        if suffix == ".cif":
            import gemmi

            block = gemmi.cif.read_file(str(path)).sole_block()
            number = _cif_number(
                block.find_value("_space_group_IT_number") or block.find_value("_symmetry_Int_Tables_number") or ""
            )
            if number is not None:
                sg = gemmi.find_spacegroup_by_number(number)
                if sg is not None:
                    policy["space_group_number"] = number
                    policy["space_group_hm"] = str(sg.hm)
                    if sg.is_centrosymmetric():
                        policy.update(
                            run_hand_control=False,
                            run_absolute_structure_control=False,
                            reason="CIF space group is centrosymmetric; alternate hand check is not meaningful",
                        )
                    else:
                        policy.update(
                            run_hand_control=True,
                            run_absolute_structure_control=True,
                            reason="CIF space group is acentric; alternate hand check is applicable",
                        )
                    return policy
            for tag in ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"):
                col = block.find_loop(tag)
                if not col:
                    continue
                loop = col.get_loop()
                tags = [str(t) for t in loop.tags]
                tag_i = tags.index(tag)
                for i in range(loop.length()):
                    op = gemmi.Op(str(loop[i, tag_i]).strip().strip("'\""))
                    den = int(op.DEN)
                    inv = [[-den, 0, 0], [0, -den, 0], [0, 0, -den]]
                    if [list(row) for row in op.rot] == inv:
                        policy.update(
                            run_hand_control=False,
                            run_absolute_structure_control=False,
                            reason="CIF symmetry operators include inversion; alternate hand check is not meaningful",
                        )
                        return policy
            return policy
    except Exception as exc:
        policy.update(
            reason=f"could not determine hand-control symmetry policy ({exc}); keeping alternate hand control enabled"
        )
        return policy
    return policy


def _clean_embedded_shelx(text: str) -> str:
    """Normalize a SHELX instruction block extracted from a CIF text field."""

    lines = [line.rstrip() for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and lines[0].strip() == ";":
        lines.pop(0)
    while lines and lines[-1].strip() == ";":
        lines.pop()
    if not any(line.strip().upper() == "END" for line in lines):
        lines.append("END")
    return "\n".join(lines) + "\n"


def _extract_cif_shelx_block(path: Path) -> str:
    """Extract an embedded SHELX .res/.ins block from an CIF."""

    import gemmi

    block = gemmi.cif.read_file(str(path)).sole_block()
    iucr_shelx_tag = "_iucr_" + "ref" + "ine_instructions_details"
    for tag in (iucr_shelx_tag, "_shelx_res_file"):
        value = block.find_value(tag)
        if value and "CELL" in str(value).upper() and "SFAC" in str(value).upper():
            return _clean_embedded_shelx(str(value))
    return ""


def prepare_structure_for_legacy(structure: str, out_dir: Path) -> tuple[str, dict]:
    """Return a SHELX-compatible structure path for engine regularization scripts.

    The Bloch calculator can read CIF coordinates directly, but map generation
    and the SHELXL handoff preserve and rewrite SHELX records.
    Some CIF exports carry exact SHELX instructions in a text block;
    extracting that block is safer than rebuilding AFIX/WGHT/SFAC records.
    """
    src = Path(structure).expanduser().resolve()
    if src.suffix.lower() != ".cif":
        return str(src), {"source": str(src), "prepared": False}
    prepared_dir = out_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    dst = prepared_dir / f"{src.stem}.res"
    shelx = _extract_cif_shelx_block(src)
    if not shelx:
        raise RuntimeError(
            f"{src} is a CIF, but no embedded SHELX model-update block was found. "
            "Provide a .res/.ins model or a CIF that embeds SHELX instruction text."
        )
    dst.write_text(shelx)
    return str(dst), {
        "source": str(src),
        "prepared": True,
        "prepared_structure": str(dst),
        "method": "embedded_shelx_block",
    }


def is_shelx_model(path: str) -> bool:
    """Return true when a model path is a SHELX .res or .ins file."""

    return Path(path).suffix.lower() in {".res", ".ins"}


def make_noh_model(python: str, structure: str, out_dir: Path, logs: Path, dry_run: bool) -> Path:
    """Create or reuse a hydrogen-stripped model variant."""

    src = Path(structure)
    dst = out_dir / "input" / f"{src.stem}_noH{src.suffix or '.res'}"
    if dst.exists():
        return dst
    cmd = [python, str(engine_script("make_res_variants.py")), "--input", str(src), "--output", str(dst), "--remove-h"]
    run_logged(cmd, logs, "make_noH_model", dry_run=dry_run)
    return dst


def make_alternate_hand_model(python: str, structure: str, out_dir: Path, logs: Path, dry_run: bool) -> Path:
    """Create or content-validate the alternate-hand model variant."""

    src = Path(structure).expanduser().resolve()
    dst = out_dir / "input" / f"{src.stem}_alternate_hand{src.suffix or '.res'}"
    manifest_path = dst.with_suffix(dst.suffix + ".source.json")
    expected_manifest = {
        "schema_version": 1,
        "variant": "alternate_hand",
        "materialization_contract": "invert_atoms_and_shelx_symmetry_translations_v1",
        "source_path": str(src),
        "source_sha256": sha256_file(src),
    }
    recorded = try_read_json(manifest_path)
    output_matches = bool(
        dst.is_file() and recorded.get("output_sha256") and recorded.get("output_sha256") == sha256_file(dst)
    )
    if output_matches and all(recorded.get(key) == value for key, value in expected_manifest.items()):
        return dst
    if not dry_run:
        dst.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    cmd = [
        python,
        str(engine_script("make_res_variants.py")),
        "--input",
        str(src),
        "--output",
        str(dst),
        "--alternate-hand",
    ]
    run_logged(cmd, logs, "make_alternate_hand_model", dry_run=dry_run)
    if not dry_run:
        if not dst.is_file() or dst.stat().st_size <= 0:
            raise RuntimeError(f"alternate-hand model materialization did not produce {dst}")
        write_json(manifest_path, {**expected_manifest, "output_sha256": sha256_file(dst)})
    return dst
