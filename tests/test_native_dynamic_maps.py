import json
from pathlib import Path

import gemmi

from grond.maps.native_dynamic import NativeDynamicMapOptions, write_native_dynamic_map_package


def test_native_dynamic_map_package_uses_row_level_dynamical_coefficients(tmp_path: Path):
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
    assert "_grond_dynamic_map_formula" in cif_text
    assert "_refln_F_dyn_obs" in cif_text
    assert "_refln_grond_mean_q_dyn_over_born" in cif_text

    ccp4 = gemmi.read_ccp4_map(manifest["maps"]["fo"]["map"])
    assert ccp4.grid.nu > 0
    saved = json.loads((out / "native_dynamic_maps_manifest.json").read_text())
    assert saved["outputs"]["dynamic_cif"] == manifest["outputs"]["dynamic_cif"]


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
