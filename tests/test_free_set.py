"""--free-set: off (default) refines the published endpoint against work and free reflections; on keeps them out."""

import argparse

from grond.cli_options.parser import build_parser
from grond.validation.endpoint_free import flagged_validation_lines, write_hklf4_lines
from grond.workflows.all_data_refinement import all_data_lines, run_all_data_refinement, scale_hklf_record


def test_free_flags_are_removed_and_every_reflection_kept(tmp_path):
    work = ["   1   2   3  100.00    2.00", "   2   0   0   50.00    1.50"]
    free = ["   1   1   1   20.00    1.00"]
    write_hklf4_lines(tmp_path / "validation.hkl", flagged_validation_lines(work, free), batch_column=True)
    lines, n_work, n_free = all_data_lines(tmp_path / "validation.hkl")
    assert (n_work, n_free) == (2, 1)
    assert lines == [*work, *free]


def test_hklf_scale_undoes_the_extra_fixed_width_factor(tmp_path):
    ins = tmp_path / "all_data.ins"
    ins.write_text("FVAR 1.0\nHKLF 4\nEND\n")
    scale_hklf_record(ins, 1.0 / 0.5)
    assert "HKLF 4 2" in ins.read_text()


def test_free_set_is_off_by_default_and_on_keeps_the_endpoint():
    args = build_parser().parse_args(["auto", "--prep", "prep.json", "--out-dir", "out"])
    assert args.free_set == "off"
    assert run_all_data_refinement(None, {}, argparse.Namespace(free_set="on"))["status"] == "skipped_free_set_on"
    # a namespace without the option (continuations, older callers) keeps the free set out
    assert run_all_data_refinement(None, {}, argparse.Namespace())["status"] == "skipped_free_set_on"
