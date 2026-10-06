"""Register shared run options without changing their help order or defaults."""

from __future__ import annotations

import argparse

from .forensics import add_forensic_args
from .run_decisions import add_decision_controls
from .run_endpoints import add_endpoints, add_final_polish
from .run_inputs import add_execution, add_inputs, add_observations_and_outputs
from .run_physics import add_backend, add_basis, add_geometry, add_nuisance
from .run_refinement import (
    add_dataset_policy,
    add_resolution,
    add_scale_polish,
    add_scales_and_screening,
    add_structure_updates,
)


def add_common_run_args(ap: argparse.ArgumentParser, *, expose_lambda_controls: bool = True) -> None:
    """Attach the shared run/auto options to an argparse subcommand."""

    add_inputs(ap)
    add_execution(ap)
    add_backend(ap)
    add_decision_controls(ap)
    add_observations_and_outputs(ap)
    add_scales_and_screening(ap)
    add_basis(ap)
    add_structure_updates(ap)
    add_geometry(ap)
    add_endpoints(ap, expose_lambda_controls=expose_lambda_controls)
    add_final_polish(ap)
    add_scale_polish(ap)
    add_resolution(ap)
    add_dataset_policy(ap)
    add_nuisance(ap)
    add_forensic_args(ap, expose_lambda_controls=expose_lambda_controls)
