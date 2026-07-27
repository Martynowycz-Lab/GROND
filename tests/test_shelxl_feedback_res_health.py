import argparse
import sys
from pathlib import Path

import pytest

from grond.legacy.engine import native_observation_polish_endpoint as observation_polish
from grond.legacy.engine import native_row_regularize as nr
from grond.legacy.engine import scale_contract_replay as replay
from grond.legacy.engine import shelxl_feedback_loop as fb


def valid_res_text() -> str:
    return (
        "TITL ok\n"
        "CELL 0.02508 5.0 6.0 7.0 90 90 90\n"
        "SFAC C 1 0 1 0 1 0 1 0\n"
        "FVAR 1\n"
        "C1 1 0.10 0.20 0.30 11.0 0.05\n"
        "HKLF 4\n"
        "END\n"
    )


def test_feedback_cycles_share_hash_validated_prediction_cache(tmp_path: Path):
    cycle = tmp_path / "feedback" / "cycle_03_N107Q20"
    recompute = tmp_path / "feedback" / "final_row_recompute_N107Q20"

    assert fb.feedback_prediction_cache_dir(cycle) == fb.feedback_prediction_cache_dir(recompute)
    assert fb.feedback_prediction_cache_dir(cycle) == tmp_path / "feedback" / "cache" / "shared"


def test_observation_polish_default_preserves_native_shelxl_resolution():
    assert observation_polish.DEFAULT_SHELXL == nr.SHELXL


def test_feedback_native_command_forwards_explicit_non_path_shelxl(tmp_path: Path):
    explicit_shelxl = tmp_path / "private-ccp4" / "shelxl"
    args = argparse.Namespace(
        python="python",
        hkl=str(tmp_path / "rows.hkl"),
        baseline_hkl="",
        hkl_transform="identity",
        min_d_spacing=0.7,
        topup_d_min=0.7,
        min_peak=0.0,
        geometry_percentile=99.0,
        lambda_grid="1.0",
        shelxl=str(explicit_shelxl),
        shelxl_iterations=1,
        endpoint_ls_cycles=1,
        endpoint_wght_mode="freeze-zero",
        post_bloch_sigma_policy="uniform_z_rms",
        post_bloch_sigma_holdout_fraction=0.2,
        post_bloch_sigma_min_scale=0.25,
        post_bloch_sigma_max_scale=4.0,
        rescale="median-iobs",
        workers=2,
        chunksize=1,
        cache_compression="none",
        compute_backend="scipy",
        row_plan_cache="off",
        jax_batch_size=0,
        jax_matrix_bucket_size=0,
        bethe_strong_count=48,
        bethe_variant="schur_target",
        bethe_promote_denom=0.005,
        bethe_denom_eps=1e-5,
        max_rows=0,
    )
    source_summary = {
        "best_native": {
            "nuisance": {
                "thickness_A": 1200.0,
                "thickness_model": "uniform_gradient",
                "thickness_grad_frac": 0.4,
            }
        }
    }

    command = fb.native_command(
        args,
        source_summary,
        tmp_path / "model.res",
        tmp_path / "cycle_01",
        n_basis=24,
        n_points=10,
    )

    assert command.count("--shelxl") == 1
    assert command[command.index("--shelxl") + 1] == str(explicit_shelxl)


def test_native_worker_uses_explicit_non_path_shelxl(tmp_path: Path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    marker = work / "explicit-shelxl-ran"
    explicit_shelxl = tmp_path / "private-ccp4" / "shelxl"
    explicit_shelxl.parent.mkdir()
    explicit_shelxl.write_text(
        "#!/bin/sh\n"
        f"printf invoked > {str(marker)!r}\n"
        "exit 7\n"
    )
    explicit_shelxl.chmod(0o755)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(nr, "SHELXL", "")

    assert nr.configure_shelxl_executable(str(explicit_shelxl)) == str(explicit_shelxl)
    result = nr.run_shelxl(work, "explicit_endpoint", iterations=1, wght_mode="inherit")

    assert marker.read_text() == "invoked"
    assert result["returncode"] == 7
    assert result["status"] == "shelxl_failed"


def test_native_endpoint_requires_launchable_shelxl_but_row_only_does_not(tmp_path: Path):
    missing = tmp_path / "not-installed" / "shelxl"
    non_executable = tmp_path / "private-ccp4" / "shelxl"
    non_executable.parent.mkdir()
    non_executable.write_text("#!/bin/sh\nexit 0\n")

    for unavailable in (missing, non_executable):
        with pytest.raises(SystemExit, match="missing or not executable"):
            nr.require_endpoint_shelxl(
                str(unavailable),
                skip_endpoint_refinement=False,
            )
        nr.require_endpoint_shelxl(
            str(unavailable),
            skip_endpoint_refinement=True,
        )

    with pytest.raises(RuntimeError, match="refusing to mark"):
        nr.fail_if_shelxl_became_unavailable(
            [{"shelxl": {"status": "missing_shelxl"}}],
            context="test endpoint",
        )
    nr.fail_if_shelxl_became_unavailable(
        [{"shelxl": {"status": "ok", "returncode": 0}}],
        context="test endpoint",
    )


def test_feedback_endpoint_res_rejects_empty_direct_endpoint(tmp_path: Path):
    run = tmp_path / "cycle"
    direct = run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    direct.parent.mkdir(parents=True)
    direct.write_text("")

    with pytest.raises(FileNotFoundError, match="empty_res"):
        fb.endpoint_res(run)


def test_feedback_endpoint_res_skips_empty_and_uses_valid_fallback(tmp_path: Path):
    run = tmp_path / "cycle"
    direct = run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    fallback = run / "endpoint_lam_0p9_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    direct.parent.mkdir(parents=True)
    fallback.parent.mkdir(parents=True)
    direct.write_text("")
    fallback.write_text(valid_res_text())
    fallback.with_suffix(".lst").write_text("     0  atoms may be split and     0  atoms NPD\n")

    assert fb.endpoint_res(run) == fallback


def test_feedback_endpoint_res_uses_exact_summary_selected_lambda(tmp_path: Path):
    run = tmp_path / "cycle"
    stale = run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    selected = run / "endpoint_lam_0p8_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    stale.parent.mkdir(parents=True)
    selected.parent.mkdir(parents=True)
    stale.write_text(valid_res_text().replace("TITL ok", "TITL STALE_LAMBDA_ONE"))
    selected.write_text(valid_res_text().replace("TITL ok", "TITL CURRENT_LAMBDA_POINT_EIGHT"))
    selected.with_suffix(".lst").write_text("     0  atoms may be split and     0  atoms NPD\n")
    health = nr.shelx_res_health(selected, selected.with_suffix(".lst"))
    summary = {
        "best_endpoint": {
            "lambda_q": 0.8,
            "endpoint_b": 0.0,
            "keep_exti": False,
            "shelxl": {
                "status": "ok",
                "lst": str(selected.with_suffix(".lst")),
                "res_health": health,
            },
        }
    }

    assert fb.endpoint_res(run, summary=summary) == selected

    # Replacing the selected bytes after the summary was written must not make
    # the stale lambda=1 file eligible and must not reuse the changed file.
    selected.write_text(valid_res_text().replace("TITL ok", "TITL REPLACED_AFTER_SUMMARY"))
    with pytest.raises(FileNotFoundError, match="hash no longer matches"):
        fb.endpoint_res(run, summary=summary)


def test_feedback_endpoint_res_rejects_incomplete_metrics_only_lst(tmp_path: Path):
    run = tmp_path / "cycle"
    endpoint = run / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    endpoint.parent.mkdir(parents=True)
    endpoint.write_text(valid_res_text())
    endpoint.with_suffix(".lst").write_text(
        "wR2 = 0.1200, GooF = S = 1.000 for all data\n"
        "R1 = 0.0500 for 50 Fo > 4sig(Fo) and 0.0900 for all 100 data\n"
    )

    with pytest.raises(FileNotFoundError, match="safe_for_feedback"):
        fb.endpoint_res(run)


def test_structural_only_input_health_ignores_unrelated_sibling_lst(tmp_path: Path):
    model = tmp_path / "selected_input.res"
    model.write_text(valid_res_text())
    model.with_suffix(".lst").write_text(
        " -0.0010  -0.0010  -0.0010   C1   ** NON POSITIVE DEFINITE **\n"
        "     0  atoms may be split and     1  atoms NPD\n"
    )

    endpoint_health = nr.shelx_res_health(model)
    structural_health = nr.shelx_res_health(model, require_lst=False)

    assert endpoint_health["safe_for_feedback"] is False
    assert structural_health["safe_for_feedback"] is True
    assert structural_health["lst_required"] is False
    assert structural_health["pathology_status"] == "not_applicable_structural_only"
    assert structural_health["lst"] == ""


def test_native_run_shelxl_marks_empty_res_invalid(tmp_path: Path, monkeypatch):
    fake_shelxl = tmp_path / "fake_shelxl.py"
    fake_shelxl.write_text(
        "import pathlib, sys\n"
        "stem = sys.argv[1]\n"
        "pathlib.Path(stem + '.lst').write_text('')\n"
        "pathlib.Path(stem + '.res').write_text('')\n"
    )
    fake_shelxl.chmod(fake_shelxl.stat().st_mode | 0o111)
    work = tmp_path / "work"
    work.mkdir()

    # run_shelxl calls [SHELXL, stem], so wrap the fake script with a small
    # executable shim that inserts the script path before the stem argument.
    shim = tmp_path / "shelxl_shim"
    shim.write_text(f"#!/bin/sh\nexec {sys.executable!r} {str(fake_shelxl)!r} \"$@\"\n")
    shim.chmod(shim.stat().st_mode | 0o111)
    monkeypatch.setattr(nr, "SHELXL", str(shim))

    result = nr.run_shelxl(work, "dyn_nuisance_corrected", iterations=2)

    assert result["status"] == "invalid_res_output"
    assert result["invalid_res_reason"] == "empty_res"
    assert result["res_health"]["ok"] is False
    assert result["iterations"] == 2
    assert len(result["iteration_history"]) == 1


def test_native_run_shelxl_rejects_header_only_res_even_with_success_metrics(
    tmp_path: Path, monkeypatch
):
    fake_shelxl = tmp_path / "header_only_shelxl.py"
    fake_shelxl.write_text(
        "import pathlib, sys\n"
        "stem = sys.argv[1]\n"
        "pathlib.Path(stem + '.lst').write_text('''\n"
        " wR2 = 0.1000, GooF = S = 1.000, Restrained GooF = 1.000 for all data\n"
        " R1 = 0.0500 for 1 Fo > 4sig(Fo) and 0.0800 for all 1 data\n"
        "''')\n"
        "pathlib.Path(stem + '.res').write_text('TITL truncated\\nCELL 0.02508 5 6 7 90 90 90\\nSFAC C\\nEND\\n')\n"
    )
    fake_shelxl.chmod(0o755)
    shim = tmp_path / "header_only_shelxl"
    shim.write_text(f"#!/bin/sh\nexec {sys.executable!r} {str(fake_shelxl)!r} \"$@\"\n")
    shim.chmod(0o755)
    work = tmp_path / "header_only_work"
    work.mkdir()
    monkeypatch.setattr(nr, "SHELXL", str(shim))

    result = nr.run_shelxl(work, "dyn_nuisance_corrected", iterations=2)

    assert result["status"] == "invalid_res_output"
    assert result["invalid_res_reason"] == "invalid_structure_model"
    assert result["res_health"]["ok"] is False
    assert "SFAC" in result["res_health"]["error"]
    assert result["selected_iteration"] is None


def test_native_run_shelxl_rejects_structural_model_without_statistics(
    tmp_path: Path, monkeypatch
):
    fake_shelxl = tmp_path / "no_stats_shelxl.py"
    fake_shelxl.write_text(
        "import pathlib, sys\n"
        "stem = sys.argv[1]\n"
        "pathlib.Path(stem + '.lst').write_text('SHELXL ended without final statistics\\n')\n"
        f"pathlib.Path(stem + '.res').write_text({valid_res_text()!r})\n"
    )
    fake_shelxl.chmod(0o755)
    shim = tmp_path / "no_stats_shelxl"
    shim.write_text(f"#!/bin/sh\nexec {sys.executable!r} {str(fake_shelxl)!r} \"$@\"\n")
    shim.chmod(0o755)
    work = tmp_path / "no_stats_work"
    work.mkdir()
    monkeypatch.setattr(nr, "SHELXL", str(shim))

    result = nr.run_shelxl(work, "dyn_nuisance_corrected", iterations=2)

    assert result["status"] == "invalid_shelxl_statistics"
    assert "complete finite R1/wR2/N" in result["invalid_res_reason"]
    assert result["res_health"]["ok"] is True
    assert result["selected_iteration"] is None


def test_native_run_shelxl_rejects_complete_metrics_without_pathology_footer(
    tmp_path: Path, monkeypatch
):
    fake_shelxl = tmp_path / "no_pathology_footer_shelxl.py"
    fake_shelxl.write_text(
        "import pathlib, sys\n"
        "stem = sys.argv[1]\n"
        "pathlib.Path(stem + '.lst').write_text(\n"
        "    ' wR2 = 0.1000, GooF = S = 1.000, Restrained GooF = 1.000 for all data\\n'\n"
        "    ' R1 = 0.0500 for 1 Fo > 4sig(Fo) and 0.0800 for all 1 data\\n'\n"
        ")\n"
        f"pathlib.Path(stem + '.res').write_text({valid_res_text()!r})\n"
    )
    fake_shelxl.chmod(0o755)
    shim = tmp_path / "no_pathology_footer_shelxl"
    shim.write_text(f"#!/bin/sh\nexec {sys.executable!r} {str(fake_shelxl)!r} \"$@\"\n")
    shim.chmod(0o755)
    work = tmp_path / "no_pathology_footer_work"
    work.mkdir()
    monkeypatch.setattr(nr, "SHELXL", str(shim))

    result = nr.run_shelxl(work, "dyn_nuisance_corrected", iterations=1)

    assert result["status"] == "incomplete_shelxl_diagnostics"
    assert "conclusive final NPD/split" in result["invalid_res_reason"]
    assert result["res_health"]["pathology_status"] == "incomplete"
    assert result["selected_iteration"] is None


@pytest.mark.parametrize(
    ("lst_text", "expected_status"),
    [
        ("SHELXL ended without final statistics\n", "invalid_shelxl_statistics"),
        (
            " wR2 = 0.1000, GooF = S = 1.000, Restrained GooF = 1.000 for all data\n"
            " R1 = 0.0500 for 1 Fo > 4sig(Fo) and 0.0800 for all 1 data\n",
            "incomplete_shelxl_diagnostics",
        ),
    ],
)
def test_scale_replay_preserves_non_npd_rejection_status(
    tmp_path: Path,
    lst_text: str,
    expected_status: str,
) -> None:
    fake_shelxl = tmp_path / f"{expected_status}_replay.py"
    fake_shelxl.write_text(
        "import pathlib, sys\n"
        "stem = sys.argv[1]\n"
        f"pathlib.Path(stem + '.lst').write_text({lst_text!r})\n"
        f"pathlib.Path(stem + '.res').write_text({valid_res_text()!r})\n"
    )
    fake_shelxl.chmod(0o755)
    shim = tmp_path / f"{expected_status}_replay"
    shim.write_text(f"#!/bin/sh\nexec {sys.executable!r} {str(fake_shelxl)!r} \"$@\"\n")
    shim.chmod(0o755)
    work = tmp_path / f"{expected_status}_replay_work"
    work.mkdir()
    (work / "dyn_scale_replay.ins").write_text(valid_res_text())

    result = replay.run_shelxl_replay(
        work,
        "dyn_scale_replay",
        str(shim),
        iterations=1,
        wght_mode="freeze-zero",
    )

    assert result["status"] == expected_status
    assert result["has_npd"] is False
    assert result["res_health"]["safe_for_feedback"] is False
    assert result["selected_iteration"] is None


def test_native_run_shelxl_does_not_accept_stale_outputs_after_failure(tmp_path: Path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    stem = "dyn_nuisance_corrected"
    (work / f"{stem}.ins").write_text(valid_res_text())
    (work / f"{stem}.res").write_text(valid_res_text())
    (work / f"{stem}.lst").write_text("wR2 = 0.1\n")
    failing = tmp_path / "failing_shelxl"
    failing.write_text("#!/bin/sh\nexit 7\n")
    failing.chmod(0o755)
    monkeypatch.setattr(nr, "SHELXL", str(failing))

    result = nr.run_shelxl(work, stem, iterations=1)

    assert result["status"] == "shelxl_failed"
    assert result["returncode"] == 7
    assert result.get("res_normalized_for_reopen") is not True
    assert not (work / f"{stem}.res").exists()
    assert not (work / f"{stem}.lst").exists()


def test_native_run_shelxl_marks_npd_res_pathological(tmp_path: Path, monkeypatch):
    fake_shelxl = tmp_path / "fake_shelxl.py"
    fake_shelxl.write_text(
        "import pathlib, sys\n"
        "stem = sys.argv[1]\n"
        "pathlib.Path(stem + '.lst').write_text('''\\n"
        "  -0.0010  -0.0010  -0.0010   O001   ** NON POSITIVE DEFINITE **\\n"
        " ** Warning:     0  atoms may be split and     1  atoms NPD **\\n"
        "''')\n"
        "pathlib.Path(stem + '.res').write_text('TITL npd\\nCELL 0.02508 5 6 7 90 90 90\\nSFAC C 1 0 1 0 1 0 1 0\\nFVAR 1\\nC1 1 0.10 0.20 0.30 11.0 0.05\\nHKLF 4\\nEND\\n')\n"
    )
    fake_shelxl.chmod(fake_shelxl.stat().st_mode | 0o111)
    work = tmp_path / "work"
    work.mkdir()
    shim = tmp_path / "shelxl_shim"
    shim.write_text(f"#!/bin/sh\nexec {sys.executable!r} {str(fake_shelxl)!r} \"$@\"\n")
    shim.chmod(shim.stat().st_mode | 0o111)
    monkeypatch.setattr(nr, "SHELXL", str(shim))

    result = nr.run_shelxl(work, "dyn_nuisance_corrected", iterations=3)

    assert result["status"] == "adp_pathology"
    assert result["invalid_res_reason"] == "adp_pathology"
    assert result["npd_count"] == 1
    assert result["npd_atoms"] == ["O001"]
    assert result["has_npd"] is True
    assert result["res_health"]["ok"] is True
    assert result["res_health"]["safe_for_feedback"] is False
    assert result["iterations"] == 3
    assert len(result["iteration_history"]) == 1


def test_native_run_shelxl_recovers_first_npd_step_with_bounded_damp(tmp_path: Path, monkeypatch):
    fake_shelxl = tmp_path / "damp_sensitive_shelxl.py"
    fake_shelxl.write_text(
        "import pathlib, re, sys\n"
        "stem = sys.argv[1]\n"
        "ins = pathlib.Path(stem + '.ins').read_text()\n"
        "match = re.search(r'^DAMP\\s+([0-9.eE+-]+)', ins, flags=re.MULTILINE | re.IGNORECASE)\n"
        "damp = float(match.group(1)) if match else 0.0\n"
        "safe = damp >= 1000.0\n"
        "npd = 0 if safe else 11\n"
        "pathology = (\n"
        "    f' ** Warning:     0  atoms may be split and     {npd}  atoms NPD **\\n'\n"
        ")\n"
        "pathlib.Path(stem + '.lst').write_text(\n"
        "    pathology\n"
        "    + ' 100 Unique reflections, of which 0 suppressed\\n'\n"
        "    + ' wR2 = 0.2365, GooF = S = 1.043, Restrained GooF = 1.043 for all data\\n'\n"
        "    + ' R1 = 0.0948 for 50 Fo > 4sig(Fo) and 0.1299 for all 100 data\\n'\n"
        ")\n"
        "pathlib.Path(stem + '.res').write_text(ins.replace('TITL start', f'TITL damp {damp:g}'))\n"
        "pathlib.Path(stem + '.fcf').write_text(f'DAMP {damp:g}\\n')\n"
    )
    fake_shelxl.chmod(fake_shelxl.stat().st_mode | 0o111)
    shim = tmp_path / "shelxl_shim"
    shim.write_text(f"#!/bin/sh\nexec {sys.executable!r} {str(fake_shelxl)!r} \"$@\"\n")
    shim.chmod(shim.stat().st_mode | 0o111)
    work = tmp_path / "work"
    work.mkdir()
    stem = "dyn_nuisance_corrected"
    (work / f"{stem}.ins").write_text(
        "TITL start\n"
        "CELL 0.02508 5 6 7 90 90 90\n"
        "SFAC C 1 0 1 0 1 0 1 0\n"
        "WGHT 0 0\n"
        "FVAR 1\n"
        "C1 1 0.10 0.20 0.30 11.0 0.05\n"
        "CGLS 1\n"
        "HKLF 4\n"
        "END\n"
    )
    (work / f"{stem}.hkl").write_text("   1   0   0  100.00   10.00\n   0   0   0    0.00    0.00\n")
    monkeypatch.setattr(nr, "SHELXL", str(shim))

    result = nr.run_shelxl(work, stem, iterations=1)

    assert result["status"] == "ok"
    assert result["has_npd"] is False
    assert result["npd_count"] == 0
    assert result["R1_all"] == pytest.approx(0.1299)
    assert result["wR2_all"] == pytest.approx(0.2365)
    recovery = result["npd_recovery"]
    assert recovery["status"] == "accepted_first_safe_damped_step"
    assert recovery["accepted_damp"] == pytest.approx(1000.0)
    assert [attempt["damp"] for attempt in recovery["attempts"]] == [
        None,
        100.0,
        300.0,
        1000.0,
    ]
    assert [attempt["npd_count"] for attempt in recovery["attempts"]] == [11, 11, 11, 0]
    for attempt in recovery["attempts"]:
        assert Path(attempt["res"]).exists()
        assert Path(attempt["lst"]).exists()
        assert Path(attempt["fcf"]).exists()
        assert attempt["res_health"]["path"] == attempt["res"]
        assert attempt["res_health"]["lst"] == attempt["lst"]
    assert "DAMP 1000 15" in (work / f"{stem}.res").read_text()
    assert result["res_health"]["safe_for_feedback"] is True


def test_native_run_shelxl_reports_split_warning_without_blocking_feedback(tmp_path: Path, monkeypatch):
    fake_shelxl = tmp_path / "fake_shelxl.py"
    fake_shelxl.write_text(
        "import pathlib, sys\n"
        "stem = sys.argv[1]\n"
        "pathlib.Path(stem + '.lst').write_text('''\\n"
        " ** Warning:     2  atoms may be split and     0  atoms NPD **\\n"
        " wR2 = 0.1220, GooF = S = 1.010, Restrained GooF = 1.000 for all data\\n"
        " R1 = 0.0432 for 684 Fo > 4sig(Fo) and 0.1046 for all 1900 data\\n"
        "''')\n"
        "pathlib.Path(stem + '.res').write_text('TITL split\\nCELL 0.02508 5 6 7 90 90 90\\nSFAC C 1 0 1 0 1 0 1 0\\nFVAR 1\\nC1 1 0.10 0.20 0.30 11.0 0.05\\nHKLF 4\\nEND\\n')\n"
    )
    fake_shelxl.chmod(fake_shelxl.stat().st_mode | 0o111)
    work = tmp_path / "work"
    work.mkdir()
    shim = tmp_path / "shelxl_shim"
    shim.write_text(f"#!/bin/sh\nexec {sys.executable!r} {str(fake_shelxl)!r} \"$@\"\n")
    shim.chmod(shim.stat().st_mode | 0o111)
    monkeypatch.setattr(nr, "SHELXL", str(shim))

    result = nr.run_shelxl(work, "dyn_nuisance_corrected", iterations=3)

    assert result["status"] == "ok"
    assert result["split_count"] == 2
    assert result["npd_count"] == 0
    assert result["has_split_warnings"] is True
    assert result["res_health"]["safe_for_feedback"] is True
