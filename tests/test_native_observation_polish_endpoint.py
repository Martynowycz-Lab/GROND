from pathlib import Path

import pytest

from grond.legacy.engine import native_observation_polish_endpoint as polish


def test_observation_polish_scales_merged_and_unmerged_hkl_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {
            "h": "1",
            "k": "0",
            "l": "0",
            "target_I": "150000",
            "sigma_eff": "120000",
            "dyn_raw": "1",
            "born_raw": "1",
            "dyn_scaled": "1",
        }
    ]
    source_res = tmp_path / "source.res"
    source_res.write_text("TITL test\nEND\n")

    monkeypatch.setattr(
        polish,
        "write_no_exti_ins",
        lambda _source, output, **_kwargs: output.write_text("TITL test\nEND\n"),
    )

    def fake_shelxl(out_dir: Path, stem: str, **_kwargs):
        (out_dir / f"{stem}.res").write_text("TITL test\nEND\n")
        (out_dir / f"{stem}.lst").write_text("0 atoms may be split and 0 atoms NPD\n")
        return {"returncode": 0}

    monkeypatch.setattr(polish, "run_shelxl", fake_shelxl)

    result = polish.write_endpoint(
        tmp_path / "endpoint",
        rows,
        additive=[0.0],
        log_mult=[0.0],
        res_path=source_res,
        lambda_q=1.0,
        hkl_format="shelx_fixed",
        rescale="none",
        make_maps=False,
        shelxl_iterations=1,
    )

    assert result["hkl_safe_factor"] == pytest.approx(0.6)
    for name in (
        "dyn_observation_polished.hkl",
        "dyn_observation_polished_unmerged.hkl",
    ):
        lines = (tmp_path / "endpoint" / name).read_text().splitlines()
        assert lines
        assert all(len(line) == 28 for line in lines)
