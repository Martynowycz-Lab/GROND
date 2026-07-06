from pathlib import Path

from grond.legacy.engine.combine_native_nuisance_runs import (
    canonical_model_friedel,
    canonical_model_symmetry,
    endpoint_hkl_path,
    endpoint_key_function,
    estimate_overlap_scales,
    merge_hkl_rows,
    run_shelxl,
    safe_fixed_width_factor,
    scale_hkl_rows,
    write_no_exti_ins,
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


def test_overlap_scales_can_use_model_symmetry_key():
    rows_by_dataset = {
        "d1": [(1, 2, 0, 10.0, 1.0)],
        "d2": [(2, 1, 0, 20.0, 1.0)],
    }
    swap_hk = [
        {"rot": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        {"rot": [[0, 1, 0], [1, 0, 0], [0, 0, 1]]},
    ]
    def key_for(h_index: int, k_index: int, l_index: int):
        return canonical_model_friedel(h_index, k_index, l_index, swap_hk)

    default = estimate_overlap_scales(rows_by_dataset, iterations=1)
    symmetry_aware = estimate_overlap_scales(rows_by_dataset, iterations=2, canonicalizer=key_for)

    assert default["overlap_counts"]["d1"] == 0
    assert symmetry_aware["overlap_counts"]["d1"] == 1
    assert symmetry_aware["overlap_counts"]["d2"] == 1


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


def test_merge_hkl_rows_can_merge_model_symmetry_before_endpoint_refinement():
    swap_hk = [
        {"rot": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        {"rot": [[0, 1, 0], [1, 0, 0], [0, 0, 1]]},
    ]
    rows = [(1, 2, 0, 10.0, 1.0), (2, 1, 0, 14.0, 1.0)]
    key_for = endpoint_key_function("model_symmetry", swap_hk)

    exact, _exact_meta = merge_hkl_rows(rows, sigma_policy="formal")
    merged, meta = merge_hkl_rows(rows, sigma_policy="formal", canonicalizer=key_for)

    assert len(exact) == 2
    assert len(merged) == 1
    assert merged[0][:3] == canonical_model_symmetry(1, 2, 0, swap_hk)
    assert abs(merged[0][3] - 12.0) < 1e-12
    assert abs(merged[0][4] - (0.5**0.5)) < 1e-12
    assert meta["rows_with_multiple_observations"] == 1


def test_model_symmetry_merge_preserves_friedel_opposites_until_requested():
    identity = [{"rot": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]
    rows = [(1, 0, 0, 10.0, 1.0), (-1, 0, 0, 14.0, 1.0)]

    symmetry_rows, _ = merge_hkl_rows(rows, sigma_policy="formal", canonicalizer=endpoint_key_function("model_symmetry", identity))
    friedel_rows, _ = merge_hkl_rows(rows, sigma_policy="formal", canonicalizer=endpoint_key_function("model_symmetry_friedel", identity))

    assert len(symmetry_rows) == 2
    assert len(friedel_rows) == 1
    assert friedel_rows[0][:3] == canonical_model_friedel(1, 0, 0, identity)


def test_safe_fixed_width_factor_keeps_shelxl_fields_in_range():
    assert safe_fixed_width_factor([(1, 0, 0, 100.0, 5.0)]) == 1.0
    factor = safe_fixed_width_factor([(1, 0, 0, 12000.0, 5.0)])
    assert 0.0 < factor < 1.0
    assert abs(12000.0 * factor - 9000.0) < 1e-9


def test_combined_endpoint_writer_forces_zero_wght_and_ls1(tmp_path: Path):
    src = tmp_path / "model.res"
    dst = tmp_path / "multi_dyn_corrected.ins"
    src.write_text(
        "\n".join(
            [
                "TITL combined",
                "CELL 1.0 5.0 6.0 7.0 90 90 90",
                "SFAC C 1 0 1 0 1 0 1 0",
                "WGHT 0.200000 1.000000",
                "EXTI 0.123",
                "CGLS 10",
                "FVAR 1.0",
                "HKLF 3",
                "END",
                "WGHT 0.1111 0.2222",
            ]
        )
        + "\n"
    )

    write_no_exti_ins(src, dst, "multi_dyn_corrected.hkl", wght=(0.0, 0.0), ls_cycles=1)

    text = dst.read_text()
    assert "WGHT      0.0000      0.0000" in text
    assert "WGHT 0.200000" not in text
    assert "WGHT 0.1111" not in text
    assert "EXTI" not in text
    assert "CGLS 1" in text
    assert "CGLS 10" not in text
    assert "HKLF 4" in text


def test_combined_run_shelxl_freeze_zero_survives_multiple_passes(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    stem = "multi_dyn_corrected"
    src = tmp_path / "model.res"
    src.write_text(
        "\n".join(
            [
                "TITL combined",
                "CELL 0.025079 5.0 6.0 7.0 90 90 90",
                "SFAC C 1 0 1 0 1 0 1 0",
                "WGHT 0.200000 1.000000",
                "FVAR 1.0",
                "L.S. 10",
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )
    write_no_exti_ins(src, work / f"{stem}.ins", f"{stem}.hkl", wght=(0.0, 0.0), ls_cycles=1)
    (work / f"{stem}.hkl").write_text("   1   0   0  100.00   10.00\n   0   0   0    0.00    0.00\n")
    fake_shelxl = tmp_path / "fake_shelxl"
    fake_shelxl.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import sys
stem = sys.argv[1]
ins = Path(stem + ".ins").read_text(errors="replace")
wght = next((line for line in ins.splitlines() if line.strip().upper().startswith("WGHT")), "WGHT 9.0000 9.0000")
Path(stem + ".lst").write_text("\\n".join([
    wght,
    " 1 Unique reflections, of which 0 suppressed",
    " wR2 = 0.1234, GooF = S = 0.987, Restrained GooF = 0.987 for all data",
    " R1 = 0.0500 for 1 Fo > 4sig(Fo) and 0.0600 for all 1 data",
    " Recommended weighting scheme:  WGHT    0.3210    0.6540",
]) + "\\n")
Path(stem + ".res").write_text("\\n".join([
    "TITL combined output",
    "CELL 0.025079 5.0 6.0 7.0 90 90 90",
    "SFAC C 1 0 1 0 1 0 1 0",
    wght,
    "FVAR 1.0",
    "L.S. 10",
    "HKLF 4",
    "END",
    "WGHT 0.3210 0.6540",
]) + "\\n")
"""
    )
    fake_shelxl.chmod(0o755)

    result = run_shelxl(work, stem, str(fake_shelxl), iterations=2, wght_mode="freeze-zero", endpoint_ls_cycles=1)

    assert result["status"] == "ok"
    assert result["wght_mode"] == "freeze-zero"
    assert result["active_wght_a"] == 0.0
    assert result["active_wght_b"] == 0.0
    text = (work / f"{stem}.res").read_text()
    assert "WGHT      0.0000      0.0000" in text
    assert "WGHT 0.3210" not in text
    assert "L.S. 1" in text
    assert "L.S. 10" not in text
