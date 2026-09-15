import csv
import json
import math
from pathlib import Path

import gemmi
import numpy as np
import pytest

import grond.maps.native_dynamic as native_dynamic
from grond.maps.native_dynamic import NativeDynamicMapOptions, write_native_dynamic_map_package


def _periodic_fractional_distance(a, b) -> float:
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    delta -= np.rint(delta)
    return float(np.linalg.norm(delta))


def test_native_dynamic_map_package_uses_row_level_dynamical_coefficients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fcalc_calls: list[int] = []
    original_fcalc = native_dynamic.fcalc_res_grid

    def counted_fcalc(model, h, k, ell):
        fcalc_calls.append(len(np.asarray(h)))
        return original_fcalc(model, h, k, ell)

    monkeypatch.setattr(native_dynamic, "fcalc_res_grid", counted_fcalc)
    model = tmp_path / "model.res"
    model.write_text(
        "\n".join(
            [
                "TITL native map smoke",
                "CELL 0.025079 5 6 7 90 90 90",
                "LATT -1",
                "SFAC C 1 0 0 0 0 0 0 0 0",
                "UNIT 1",
                "C1 1 0 0 0 1 0.05",
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )
    rows = tmp_path / "native_dyn_calc.csv"
    rows.write_text(
        "\n".join(
            [
                "h,k,l,d_A,target_I,sigma_eff,dyn_raw,born_scaled,q_dyn_over_born,dyn_scaled,split",
                "1,0,0,5.0,4.0,1.0,2.0,3.0,1.5,4.0,work",
                "1,0,0,5.0,4.0,1.0,2.0,3.0,1.5,4.0,free",
            ]
        )
        + "\n"
    )
    out = tmp_path / "native_maps"
    manifest = write_native_dynamic_map_package(
        row_table=rows,
        model_path=model,
        out_dir=out,
        options=NativeDynamicMapOptions(grid_spacing_A=1.0),
    )

    assert manifest["status"] == "ok"
    assert manifest["summary"]["merged_reflections"] == 1
    assert Path(manifest["outputs"]["coefficients_csv"]).exists()
    assert Path(manifest["outputs"]["coefficients_hkl"]).exists()
    assert Path(manifest["outputs"]["dynamic_cif"]).exists()
    assert Path(manifest["maps"]["fo"]["map"]).exists()
    assert Path(manifest["maps"]["fofc"]["map"]).exists()
    assert Path(manifest["maps"]["2fofc"]["map"]).exists()

    coeff_text = Path(manifest["outputs"]["coefficients_csv"]).read_text()
    assert "F_dyn_obs" in coeff_text
    assert "phase_calc_deg" in coeff_text
    assert "mean_q_dyn_over_born" in coeff_text
    assert "mean_row_scale" in coeff_text
    cif_text = Path(manifest["outputs"]["dynamic_cif"]).read_text()
    cif_block = gemmi.cif.read_string(cif_text).sole_block()
    assert gemmi.cif.as_string(cif_block.find_value("_space_group_name_H-M_alt")) == "P 1"
    assert "_grond_dynamic_map_formula" in cif_text
    assert "_refln_F_dyn_obs" in cif_text
    assert "_refln_grond_mean_q_dyn_over_born" in cif_text

    ccp4 = gemmi.read_ccp4_map(manifest["maps"]["fo"]["map"])
    assert ccp4.grid.nu > 0
    saved = json.loads((out / "native_dynamic_maps_manifest.json").read_text())
    assert saved["outputs"]["dynamic_cif"] == manifest["outputs"]["dynamic_cif"]
    # One vector call for input rows, one for merged orbits, and one for each
    # of the three map coefficient kinds.  Never regress to one call per
    # reflection or symmetry mate.
    assert fcalc_calls == [2, 1, 1, 1, 1]


def test_native_dynamic_fo_map_peaks_at_model_atom_not_inverted_hand(tmp_path: Path):
    model = tmp_path / "off_origin_atom.res"
    atom_frac = np.asarray([0.20, 0.30, 0.40], dtype=float)
    model.write_text(
        "TITL native map Fourier-sign regression\n"
        "CELL 0.025079 10 10 10 90 90 90\n"
        "LATT -1\n"
        "SFAC C 1 0 0 0 0 0 0 0 0\n"
        "UNIT 1\n"
        "C1 1 0.20 0.30 0.40 1 0.0\n"
        "HKLF 4\n"
        "END\n"
    )
    rows = tmp_path / "native_dyn_calc.csv"
    records = ["h,k,l,d_A,target_I,sigma_eff,dyn_scaled,split"]
    for h in range(-4, 5):
        for k in range(-4, 5):
            for ell in range(-4, 5):
                if h == k == ell == 0:
                    continue
                records.append(f"{h},{k},{ell},1.0,1.0,0.1,1.0,work")
    rows.write_text("\n".join(records) + "\n")

    manifest = write_native_dynamic_map_package(
        row_table=rows,
        model_path=model,
        out_dir=tmp_path / "native_maps",
        options=NativeDynamicMapOptions(grid_spacing_A=0.5),
    )

    ccp4 = gemmi.read_ccp4_map(manifest["maps"]["fo"]["map"])
    ccp4.setup(float("nan"))
    data = np.asarray(ccp4.grid, dtype=np.float64)
    peak_index = np.asarray(np.unravel_index(int(np.nanargmax(data)), data.shape), dtype=float)
    peak_frac = peak_index / np.asarray(data.shape, dtype=float)
    inverted_frac = (-atom_frac) % 1.0

    assert _periodic_fractional_distance(peak_frac, atom_frac) < 0.08
    assert _periodic_fractional_distance(peak_frac, atom_frac) < _periodic_fractional_distance(peak_frac, inverted_frac)
    assert manifest["maps"]["fo"]["normalization"] == ("sum(F exp(-2*pi*i*h.x)) / cell_volume via numpy fftn")


def test_native_dynamic_map_merges_model_symmetry_and_projects_friedel_pairs(tmp_path: Path):
    """A real map retains dynamical Friedel information until Hermitian projection."""

    model = tmp_path / "p212121.res"
    model.write_text(
        "TITL native map symmetry regression\n"
        "CELL 0.025079 10 11 12 90 90 90\n"
        "LATT -1\n"
        "SYMM 0.5-X,-Y,0.5+Z\n"
        "SYMM -X,0.5+Y,0.5-Z\n"
        "SYMM 0.5+X,0.5-Y,-Z\n"
        "SFAC C 1 0 0 0 0 0 0 0 0\n"
        "UNIT 1\n"
        "C1 1 0.17 0.29 0.41 1 0.02\n"
        "HKLF 4\n"
        "END\n"
    )
    rows = tmp_path / "native_dyn_calc.csv"
    rows.write_text(
        "\n".join(
            [
                "h,k,l,d_A,target_I,sigma_eff,dyn_scaled,split",
                # The first two rows are P212121 equivalents.  The third is
                # their independent Friedel mate with deliberately different
                # observed intensity, as dynamical scattering permits.
                "1,2,3,1.0,4.0,0.5,4.0,work",
                "-1,-2,3,1.0,9.0,0.5,4.0,work",
                "-1,-2,-3,1.0,6.0,0.5,4.0,free",
            ]
        )
        + "\n"
    )

    manifest = write_native_dynamic_map_package(
        row_table=rows,
        model_path=model,
        out_dir=tmp_path / "native_maps",
        options=NativeDynamicMapOptions(grid_spacing_A=0.5),
    )

    summary = manifest["summary"]
    map_meta = manifest["maps"]["fo"]
    assert summary["exact_signed_reflections_before_symmetry_merge"] == 3
    assert summary["model_symmetry_orbits_after_merge"] == 2
    assert summary["signed_reflections_collapsed_by_model_symmetry"] == 1
    assert summary["independent_friedel_pairs_retained"] == 1
    assert map_meta["independent_friedel_coefficient_pairs_projected"] >= 1
    assert map_meta["friedel_preprojection_absolute_rms"] > 0.0
    assert map_meta["imaginary_rms_before_real_projection"] < 1e-10


def test_native_dynamic_cif_normalizes_placeholder_wavelength(tmp_path: Path):
    model = tmp_path / "placeholder.res"
    model.write_text(
        "\n".join(
            [
                "TITL native map smoke",
                "CELL 1 5 6 7 90 90 90",
                "LATT -1",
                "SFAC C 1 0 0 0 0 0 0 0 0",
                "UNIT 1",
                "C1 1 0 0 0 1 0.05",
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )
    rows = tmp_path / "native_dyn_calc.csv"
    rows.write_text(
        "\n".join(
            [
                "h,k,l,d_A,target_I,sigma_eff,dyn_scaled,split",
                "1,0,0,5.0,4.0,1.0,4.0,work",
            ]
        )
        + "\n"
    )

    manifest = write_native_dynamic_map_package(
        row_table=rows,
        model_path=model,
        out_dir=tmp_path / "native_maps",
        options=NativeDynamicMapOptions(grid_spacing_A=1.0),
    )

    cif_text = Path(manifest["outputs"]["dynamic_cif"]).read_text()
    assert "_diffrn_radiation_wavelength 0.025079" in cif_text


def test_native_dynamic_map_package_accepts_labeled_multi_dataset_rows(tmp_path: Path):
    model = tmp_path / "model.res"
    model.write_text(
        "\n".join(
            [
                "TITL native map multi smoke",
                "CELL 0.025079 5 6 7 90 90 90",
                "LATT -1",
                "SFAC C 1 0 0 0 0 0 0 0 0",
                "UNIT 1",
                "C1 1 0 0 0 1 0.05",
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )
    row_header = "h,k,l,d_A,target_I,sigma_eff,dyn_scaled,split\n"
    rows_a = tmp_path / "a_native_dyn_calc.csv"
    rows_b = tmp_path / "b_native_dyn_calc.csv"
    rows_a.write_text(row_header + "1,0,0,5.0,4.0,1.0,4.0,work\n")
    rows_b.write_text(row_header + "1,0,0,5.0,9.0,1.0,4.0,free\n")

    manifest = write_native_dynamic_map_package(
        row_table=[rows_a, rows_b],
        row_table_labels=["crystal_a", "crystal_b"],
        model_path=model,
        out_dir=tmp_path / "native_maps",
        options=NativeDynamicMapOptions(grid_spacing_A=1.0),
    )

    assert manifest["status"] == "ok"
    assert manifest["summary"]["row_table_count"] == 2
    assert manifest["summary"]["input_rows"] == 2
    assert manifest["summary"]["merged_reflections"] == 1
    assert manifest["summary"]["datasets"] == ["crystal_a", "crystal_b"]
    coeff_text = Path(manifest["outputs"]["coefficients_csv"]).read_text()
    assert "datasets,dataset_count" in coeff_text
    assert "crystal_a,crystal_b" in coeff_text
    cif_text = Path(manifest["outputs"]["dynamic_cif"]).read_text()
    assert "_grond_dynamic_map_source_row_table_count 2" in cif_text
    assert "_grond_dynamic_map_row_table_dataset" in cif_text
    assert "_refln_grond_dataset_count" in cif_text


def test_native_dynamic_multi_maps_apply_dataset_policy_once(tmp_path: Path):
    model = tmp_path / "model.res"
    model.write_text(
        "\n".join(
            [
                "TITL native map policy contract",
                "CELL 0.025079 5 6 7 90 90 90",
                "LATT -1",
                "SFAC C 1 0 0 0 0 0 0 0 0",
                "UNIT 1",
                "C1 1 0 0 0 1 0.05",
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )
    header = (
        "h,k,l,d_A,target_I,sigma_eff,sigma_eff_original,dyn_scaled,split,"
        "dataset_role,raw_row_weight,policy_sigma_weight_applied\n"
    )
    tables = []
    specifications = [
        ("primary", "1,0,0,5.0,4.0,2.0,,4.0,work,primary,1.0,false\n"),
        # sigma_eff is already weighted in this copied table.  The map reader
        # must restore 2.0, then apply 0.25 once to obtain 4.0 (not 8.0).
        ("downweight", "1,0,0,5.0,4.0,4.0,2.0,4.0,work,downweight,0.25,true\n"),
        ("complete", "1,0,0,5.0,400.0,1.0,1.0,4.0,work,completeness_only,0.25,false\n"),
        ("excluded", "1,0,0,5.0,400.0,1.0,1.0,4.0,work,exclude,1.0,false\n"),
        ("zero", "1,0,0,5.0,400.0,1.0,1.0,4.0,work,primary,0.0,false\n"),
        ("invalid", "1,0,0,5.0,400.0,1.0,1.0,4.0,work,primary,not-a-weight,false\n"),
    ]
    for label, row in specifications:
        table = tmp_path / f"{label}_native_dyn_calc.csv"
        table.write_text(header + row)
        tables.append(table)

    manifest = write_native_dynamic_map_package(
        row_table=tables,
        row_table_labels=[label for label, _row in specifications],
        model_path=model,
        out_dir=tmp_path / "native_maps",
        options=NativeDynamicMapOptions(grid_spacing_A=1.0),
    )

    assert manifest["status"] == "ok"
    summary = manifest["summary"]
    assert summary["input_rows"] == 6
    assert summary["policy_eligible_rows"] == 2
    assert summary["valid_positive_rows"] == 2
    assert summary["rejected_rows"] == 4
    assert summary["rejected_after_policy_rows"] == 0
    policy = summary["dataset_policy"]
    assert policy["source_rows"] == 6
    assert policy["included_rows"] == 2
    assert policy["excluded_rows"] == 4
    assert policy["excluded_completeness_only_rows"] == 1
    assert policy["excluded_explicit_role_rows"] == 1
    assert policy["excluded_nonpositive_raw_row_weight_rows"] == 1
    assert policy["excluded_invalid_raw_row_weight_rows"] == 1
    assert policy["downweighted_rows"] == 1
    assert policy["sigma_eff_reset_from_original_rows"] == 1
    assert policy["raw_row_weight_applied_rows"] == 1

    with Path(manifest["outputs"]["coefficients_csv"]).open(newline="") as handle:
        coefficient = next(csv.DictReader(handle))
    assert int(coefficient["n_rows"]) == 2
    assert coefficient["datasets"] == "downweight,primary"
    assert int(coefficient["dataset_count"]) == 2
    assert float(coefficient["sigF_dyn_obs"]) / float(coefficient["F_dyn_calc"]) == pytest.approx(1.0 / math.sqrt(20.0))


def test_native_dynamic_maps_report_all_rows_removed_by_policy(tmp_path: Path):
    model = tmp_path / "model.res"
    model.write_text(
        "TITL native map excluded rows\n"
        "CELL 0.025079 5 6 7 90 90 90\n"
        "LATT -1\n"
        "SFAC C 1 0 0 0 0 0 0 0 0\n"
        "UNIT 1\n"
        "C1 1 0 0 0 1 0.05\n"
        "HKLF 4\n"
        "END\n"
    )
    rows = tmp_path / "native_dyn_calc.csv"
    rows.write_text(
        "h,k,l,d_A,target_I,sigma_eff,dyn_scaled,split,dataset_role,raw_row_weight\n"
        "1,0,0,5.0,4.0,1.0,4.0,work,completeness_only,1.0\n"
    )

    manifest = write_native_dynamic_map_package(
        row_table=rows,
        model_path=model,
        out_dir=tmp_path / "native_maps",
        options=NativeDynamicMapOptions(grid_spacing_A=1.0),
    )

    assert manifest["status"] == "no_coefficients"
    assert manifest["summary"]["input_rows"] == 1
    assert manifest["summary"]["policy_eligible_rows"] == 0
    assert manifest["summary"]["dataset_policy"]["excluded_completeness_only_rows"] == 1
