from __future__ import annotations

from grond.refine.endpoints import ENDPOINTS, endpoint_report, render_custom_command


def test_endpoint_registry_declares_default_and_adapters():
    assert ENDPOINTS["shelxl"].status == "implemented_default"
    assert ENDPOINTS["bloch_only"].status == "implemented_no_refiner"
    assert ENDPOINTS["native_gradient"].status == "implemented_diagnostic"
    assert ENDPOINTS["phenix_refine"].status == "macromolecular_adapter"
    assert ENDPOINTS["refmac5"].status == "macromolecular_adapter"
    assert ENDPOINTS["servalcat"].status == "macromolecular_adapter"
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


def test_endpoint_report_sees_macromolecular_command_template():
    report = endpoint_report(
        {
            "endpoints": {
                "phenix_refine": {
                    "command": "phenix.refine {mtz} {model} main.number_of_macro_cycles={cycles}",
                    "output_model": "{out_dir}/refined.pdb",
                }
            }
        }
    )
    phenix = next(row for row in report["endpoints"] if row["name"] == "phenix_refine")
    assert phenix["available"] is True
    assert phenix["configured_command_template"] is True


def test_render_custom_endpoint_command():
    cmd = render_custom_command(
        "my_refiner --model {model} --hkl {reflections} --cycles {cycles} --out {out_dir}",
        model="start.res",
        reflections="corrected.hkl",
        out_dir="endpoint",
        cycles=1,
    )
    assert cmd == ["my_refiner", "--model", "start.res", "--hkl", "corrected.hkl", "--cycles", "1", "--out", "endpoint"]
