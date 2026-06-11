from pathlib import Path

from grond.legacy.engine.combine_native_nuisance_runs import (
    endpoint_hkl_path,
    estimate_overlap_scales,
    merge_hkl_rows,
    safe_fixed_width_factor,
    scale_hkl_rows,
)


def test_overlap_scales_put_datasets_on_common_intensity_scale():
    rows_by_dataset = {
        "d1": [(1, 0, 0, 10.0, 1.0), (0, 1, 0, 20.0, 1.0)],
        "d2": [(1, 0, 0, 20.0, 1.0), (0, 1, 0, 40.0, 1.0)],
    }
    result = estimate_overlap_scales(rows_by_dataset, iterations=4)
    scales = result["scales"]
    d1_scaled = scale_hkl_rows(rows_by_dataset["d1"], scales["d1"])
    d2_scaled = scale_hkl_rows(rows_by_dataset["d2"], scales["d2"])
    assert abs(d1_scaled[0][3] - d2_scaled[0][3]) < 1e-6
    assert abs(d1_scaled[1][3] - d2_scaled[1][3]) < 1e-6
    assert result["overlap_counts"]["d1"] == 2
    assert result["overlap_counts"]["d2"] == 2


def test_endpoint_hkl_path_finds_b_zero_noexti_directory(tmp_path: Path):
    endpoint = tmp_path / "endpoint_lam_0p9_B_0p0_noexti"
    endpoint.mkdir()
    hkl = endpoint / "dyn_nuisance_corrected_unmerged.hkl"
    hkl.write_text("   1   0   0  100.00   10.00\n   0   0   0    0.00    0.00\n")
    assert endpoint_hkl_path(tmp_path, "0.9") == hkl


def test_merge_hkl_rows_can_inflate_disagreeing_replicates():
    rows = [(1, 0, 0, 10.0, 1.0), (1, 0, 0, 14.0, 1.0)]

    formal, formal_meta = merge_hkl_rows(rows, sigma_policy="formal")
    overdispersed, overdispersed_meta = merge_hkl_rows(rows, sigma_policy="chisq_overdispersion")

    assert formal[0][:4] == overdispersed[0][:4]
    assert abs(formal[0][4] - (0.5**0.5)) < 1e-12
    assert abs(overdispersed[0][4] - 2.0) < 1e-12
    assert formal_meta["inflated_rows"] == 0
    assert overdispersed_meta["inflated_rows"] == 1
    assert overdispersed_meta["rows_with_multiple_observations"] == 1


def test_safe_fixed_width_factor_keeps_shelxl_fields_in_range():
    assert safe_fixed_width_factor([(1, 0, 0, 100.0, 5.0)]) == 1.0
    factor = safe_fixed_width_factor([(1, 0, 0, 12000.0, 5.0)])
    assert 0.0 < factor < 1.0
    assert abs(12000.0 * factor - 9000.0) < 1e-9
