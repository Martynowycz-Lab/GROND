"""Explicit implementation files used by the prediction-source fingerprint.

Keep this list explicit. Moving numerical code without hashing its new owner
can make an old cached prediction look current after a later helper edit.
"""
from pathlib import Path

REGULARIZATION_SOURCE_PATHS = (
    'refine/feedback_trajectory.py',
    'refine/native_output.py',
    'regularize/evaluation.py',
    'regularize/row_output.py',
    'regularize/grid.py',
    'regularize/arguments_native.py',
    'regularize/arguments_combination.py',
    'refine/arguments_feedback.py',
    'regularize/source.py',
    'refine/feedback_contract.py',
    'refine/shelx_input.py',
    'refine/shelxl_diagnostics.py',
    'refine/shelxl_iterations.py',
    'regularize/contracts.py',
    'regularize/multidataset.py',
    'regularize/observations.py',
    'regularize/prediction_cache.py',
    'regularize/scales.py',
    'regularize/selection.py',
    'engine/combine_native_nuisance_runs.py',
    'engine/native_row_regularize.py',
    'engine/shelxl_feedback_loop.py',
)


def regularization_source_files() -> tuple[Path, ...]:
    """Return the absolute paths of REGULARIZATION_SOURCE_PATHS in the installed grond package."""

    package_root = Path(__file__).resolve().parents[1]
    return tuple(package_root / relative for relative in REGULARIZATION_SOURCE_PATHS)
