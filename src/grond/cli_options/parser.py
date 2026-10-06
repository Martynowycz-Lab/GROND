"""Build the public command grammar in its established display order."""

from __future__ import annotations

import argparse

from .diagnostic_commands import register_diagnostics
from .endpoint_commands import register_check_shelx, register_finish, register_push, register_validate_endpoint
from .figure_maps import register_figure_maps
from .input_commands import register_inputs
from .integration_commands import register_integration
from .manual_commands import register_manual
from .never_h import register_never_h
from .protein_commands import register_protein
from .run_commands import register_runs
from .scout_commands import register_scouts, register_thickness
from .splits_commands import register_splits
from .validation_commands import register_ls1, register_runtime


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level GROND argument parser."""

    parser = argparse.ArgumentParser(
        prog="GROND",
        description="GROND: Global Regularization Of Non-kinematical Diffraction.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    register_manual(sub)
    register_inputs(sub)
    register_protein(sub)
    register_integration(sub)
    register_scouts(sub)
    register_splits(sub)
    register_runs(sub)
    register_diagnostics(sub)
    register_thickness(sub)
    register_ls1(sub)
    register_push(sub)
    register_never_h(sub)
    register_finish(sub)
    register_figure_maps(sub)
    register_check_shelx(sub)
    register_validate_endpoint(sub)
    register_runtime(sub)
    return parser
