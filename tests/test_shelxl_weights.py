import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "grond" / "legacy" / "engine"))

from grond.legacy.engine.native_row_regularize import (
    normalize_shelx_cell_wavelength,
    parse_int_grid,
    parse_lst,
    parse_recommended_wght,
    parse_shelx_wght_terms,
    promote_recommended_wght,
    write_active_wght,
    write_endpoint_ins,
)


def test_parse_int_grid_accepts_detector_scale_candidates():
    assert parse_int_grid("0,8,12") == [0, 8, 12]
    assert parse_int_grid("8 12") == [8, 12]


def test_parse_recommended_wght_uses_last_scheme(tmp_path: Path):
    lst = tmp_path / "trial.lst"
    lst.write_text(
        "\n".join(
            [
                " Recommended weighting scheme:  WGHT    0.0835    0.0000",
                " other text",
                " Recommended weighting scheme:  WGHT    0.0720    1.2500",
            ]
        )
    )

    assert parse_recommended_wght(lst) == (0.0720, 1.2500)


def test_parse_recommended_wght_single_term_does_not_consume_exti(tmp_path: Path):
    lst = tmp_path / "trial.lst"
    lst.write_text(" Recommended weighting scheme:  WGHT    1E-4\nEXTI 1320.486328\n")

    assert parse_recommended_wght(lst) == (0.0001, 0.0)


def test_parse_shelx_wght_terms_stays_on_one_line():
    assert parse_shelx_wght_terms("WGHT    0.100000") == [0.1]
    assert parse_shelx_wght_terms("WGHT    0.100000 EXTI") == [0.1]
    assert parse_shelx_wght_terms("WGHT    1E-4    -2.5e+1    0.3") == [0.0001, -25.0, 0.3]


def test_parse_lst_records_active_wght(tmp_path: Path):
    lst = tmp_path / "trial.lst"
    lst.write_text(
        "\n".join(
            [
                " WGHT    0.126534",
                " 4462 Reflections read, of which 17 rejected",
                " 1900 Unique reflections, of which 0 suppressed",
                " R(int) = 0.1106 R(sigma) = 0.3908",
                " wR2 = 0.1220, GooF = S = 0.384, Restrained GooF = 0.374 for all data",
                " R1 = 0.0432 for 684 Fo > 4sig(Fo) and 0.1046 for all 1900 data",
            ]
        )
        + "\n"
    )

    parsed = parse_lst(lst)

    assert parsed["active_wght_a"] == 0.126534
    assert parsed["active_wght_b"] == 0.0
    assert parsed["wR2_all"] == 0.122


def test_parse_lst_one_term_wght_does_not_consume_following_exti(tmp_path: Path):
    lst = tmp_path / "trial.lst"
    lst.write_text(
        "\n".join(
            [
                " WGHT    0.100000",
                " EXTI 1320.486328",
                " wR2 = 0.1220, GooF = S = 0.384, Restrained GooF = 0.374 for all data",
            ]
        )
        + "\n"
    )

    parsed = parse_lst(lst)

    assert parsed["active_wght_a"] == 0.1
    assert parsed["active_wght_b"] == 0.0
    assert parsed["active_wght_terms"] == [0.1]


def test_promote_recommended_wght_replaces_active_and_strips_post_end_tail(tmp_path: Path):
    ins = tmp_path / "model.ins"
    ins.write_text(
        "\n".join(
            [
                "TITL trial",
                "WGHT    0.200000",
                "FVAR       1.0",
                "C1 1 0 0 0 11 0.1",
                "HKLF 4",
                "END",
                "WGHT      0.0834      0.0000",
                "Q1 1 0 0 0 11 0.05",
            ]
        )
        + "\n"
    )

    summary = promote_recommended_wght(ins, (0.0720, 0.0))

    assert summary == {"promoted": True, "active_wght_a": 0.072, "active_wght_b": 0.0}
    text = ins.read_text()
    assert "WGHT      0.0720      0.0000" in text
    assert "WGHT    0.200000" not in text
    assert "Q1 " not in text
    assert text.rstrip().endswith("END")


def test_promote_recommended_wght_inserts_before_fvar_when_missing(tmp_path: Path):
    ins = tmp_path / "model.ins"
    ins.write_text("TITL trial\nFVAR 1.0\nHKLF 4\nEND\n")

    promote_recommended_wght(ins, (0.05, 0.0))

    assert ins.read_text().splitlines() == [
        "TITL trial",
        "WGHT      0.0500      0.0000",
        "FVAR 1.0",
        "HKLF 4",
        "END",
    ]


def test_write_endpoint_ins_forces_one_shelxl_cycle_and_electron_wavelength(tmp_path: Path):
    src = tmp_path / "model.res"
    dst = tmp_path / "model.ins"
    src.write_text(
        "\n".join(
            [
                "TITL trial",
                "CELL 1.0 5.0 6.0 7.0 90 90 90",
                "SFAC C 1 0 1 0 1 0 1 0",
                "CGLS 10",
                'REM <HklSrc "%./old.hkl">',
                "EXTI 0.1",
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )

    write_endpoint_ins(src, dst, hkl_name="dyn_nuisance_corrected.hkl")

    text = dst.read_text()
    assert "CELL 0.025079 5.0 6.0 7.0 90 90 90" in text
    assert "CGLS 1" in text
    assert "CGLS 10" not in text
    assert 'REM <HklSrc "%./dyn_nuisance_corrected.hkl">' in text
    assert "EXTI" not in text


def test_write_endpoint_ins_can_write_diagnostic_shelxl_cycle_count(tmp_path: Path):
    src = tmp_path / "model.res"
    dst = tmp_path / "model.ins"
    src.write_text(
        "\n".join(
            [
                "TITL trial",
                "CELL 0.02508 5.0 6.0 7.0 90 90 90",
                "SFAC C 1 0 1 0 1 0 1 0",
                "L.S. 1",
                "HKLF 4",
                "END",
            ]
        )
        + "\n"
    )

    write_endpoint_ins(src, dst, hkl_name="dyn_nuisance_corrected.hkl", ls_cycles=20)

    text = dst.read_text()
    assert "L.S. 20" in text
    assert "L.S. 1" not in text


def test_write_endpoint_ins_strips_acta_and_forces_hklf4(tmp_path: Path):
    src = tmp_path / "model.res"
    dst = tmp_path / "model.ins"
    src.write_text(
        "\n".join(
            [
                "TITL trial",
                "CELL 0.02508 5.0 6.0 7.0 90 90 90",
                "SFAC C 1 0 1 0 1 0 1 0",
                "ACTA",
                "L.S. 0",
                "HKLF 3",
                "END",
            ]
        )
        + "\n"
    )

    write_endpoint_ins(src, dst, hkl_name="dyn_nuisance_corrected.hkl", wght=(0.0, 0.0))

    lines = dst.read_text().splitlines()
    assert "ACTA" not in lines
    assert "L.S. 1" in lines
    assert "HKLF 3" not in lines
    assert "HKLF 4" in lines
    assert lines.count("HKLF 4") == 1


def test_write_endpoint_ins_rejects_unsupported_shelx_endpoint_features(tmp_path: Path):
    for feature in ["HKLF 5", "TWIN 0 1 0 1 0 0 0 0 -1", "BASF 0.2", "SWAT 0.1 1.0", "REM PLATON SQUEEZE applied"]:
        src = tmp_path / "model.res"
        dst = tmp_path / "model.ins"
        src.write_text(
            "\n".join(
                [
                    "TITL trial",
                    "CELL 0.02508 5.0 6.0 7.0 90 90 90",
                    "SFAC C 1 0 1 0 1 0 1 0",
                    feature,
                    "END",
                ]
            )
            + "\n"
        )

        try:
            write_endpoint_ins(src, dst)
        except ValueError as exc:
            assert any(word in str(exc) for word in ["HKLF", "TWIN", "BASF", "SWAT", "SQUEEZE"])
        else:
            raise AssertionError(f"{feature!r} was not rejected")


def test_write_endpoint_ins_can_force_zero_weight_and_strip_tail(tmp_path: Path):
    src = tmp_path / "model.res"
    dst = tmp_path / "model.ins"
    src.write_text(
        "\n".join(
            [
                "TITL trial",
                "CELL 1.0 5.0 6.0 7.0 90 90 90",
                "SFAC C 1 0 1 0 1 0 1 0",
                "WGHT    0.126534",
                "L.S. 10",
                "FVAR 1.0",
                "HKLF 4",
                "END",
                "WGHT      0.0834      0.0000",
                "Q1 1 0 0 0 11 0.05",
            ]
        )
        + "\n"
    )

    write_endpoint_ins(src, dst, hkl_name="dyn_nuisance_corrected.hkl", wght=(0.0, 0.0))

    text = dst.read_text()
    assert "WGHT      0.0000      0.0000" in text
    assert "WGHT    0.126534" not in text
    assert "WGHT      0.0834" not in text
    assert "Q1 " not in text
    assert "L.S. 1" in text
    assert "L.S. 10" not in text
    assert text.rstrip().endswith("END")


def test_write_active_wght_normalizes_res_for_reopen(tmp_path: Path):
    res = tmp_path / "model.res"
    res.write_text("TITL trial\nWGHT 0.2\nFVAR 1.0\nHKLF 4\nEND\nWGHT 0.1 0\n")

    summary = write_active_wght(res, 0.0, 0.0)

    assert summary["written"] is True
    assert res.read_text().splitlines() == [
        "TITL trial",
        "WGHT      0.0000      0.0000",
        "FVAR 1.0",
        "HKLF 4",
        "END",
    ]


def test_write_endpoint_ins_inserts_one_cycle_when_missing(tmp_path: Path):
    src = tmp_path / "model.res"
    dst = tmp_path / "model.ins"
    src.write_text("TITL trial\nCELL 0.02508 5 6 7 90 90 90\nSFAC C 1 0 1 0 1 0 1 0\nHKLF 4\nEND\n")

    write_endpoint_ins(src, dst)

    assert dst.read_text().splitlines()[3] == "L.S. 1"


def test_normalize_shelx_cell_wavelength_preserves_electron_value():
    assert normalize_shelx_cell_wavelength("CELL 0.02508 5 6 7 90 90 90") == "CELL 0.02508 5 6 7 90 90 90"
