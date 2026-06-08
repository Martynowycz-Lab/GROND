from __future__ import annotations

from pathlib import Path

from grond.refine.endpoints import ENDPOINTS, endpoint_report, olex2_command_script, render_custom_command


def test_endpoint_registry_declares_default_and_adapters():
    assert ENDPOINTS["shelxl"].status == "implemented_default"
    assert ENDPOINTS["olex2_app"].status == "supported_external_app"
    assert ENDPOINTS["bloch_only"].status == "implemented_no_refiner"
    assert ENDPOINTS["native_gradient"].status == "implemented_diagnostic"
    assert ENDPOINTS["custom"].status == "adapter_template"


def test_endpoint_report_sees_custom_command_template():
    report = endpoint_report(
        {
            "endpoints": {
                "custom": {
                    "command": "my_refiner --model {model} --hkl {reflections} --cycles {cycles} --out {out_dir}",
                    "output_model": "{out_dir}/refined.cif",
                }
            }
        }
    )
    custom = next(row for row in report["endpoints"] if row["name"] == "custom")
    assert custom["available"] is True
    assert custom["configured_command_template"] is True


def test_endpoint_report_resolves_olex2_app_directory(tmp_path: Path):
    app_dir = tmp_path / "olex2.app" / "Contents" / "MacOS"
    (app_dir / "lib").mkdir(parents=True)
    (app_dir / "cctbx" / "cctbx_build" / "lib").mkdir(parents=True)
    console = app_dir / "olex2c_exe"
    console.write_text("#!/bin/sh\n")
    console.chmod(0o755)

    report = endpoint_report({"tools": {"olex2_app_dir": str(app_dir)}})
    olex2 = next(row for row in report["endpoints"] if row["name"] == "olex2_app")
    assert olex2["available"] is True
    assert olex2["executable"]["resolved"] == str(console.resolve())
    assert olex2["olex2"]["dyld_library_path"]


def test_olex2_command_script_uses_one_gauss_newton_cycle():
    script = olex2_command_script(model_name="model.res", cycles=1)
    assert "reap model.res" in script
    assert "olex2.refine" in script
    assert "Gauss-Newton" in script
    assert "max_cycles',1" in script


def test_render_custom_endpoint_command():
    cmd = render_custom_command(
        "my_refiner --model {model} --hkl {reflections} --cycles {cycles} --out {out_dir}",
        model="start.res",
        reflections="corrected.hkl",
        out_dir="endpoint",
        cycles=1,
    )
    assert cmd == ["my_refiner", "--model", "start.res", "--hkl", "corrected.hkl", "--cycles", "1", "--out", "endpoint"]
