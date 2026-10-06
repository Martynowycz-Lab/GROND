"""Explicit implementation files used by the prediction-source fingerprint.

Keep this list explicit. Moving numerical code without hashing its new owner
can make an old cached prediction look current after a later helper edit.
"""
from pathlib import Path

REGULARIZATION_SOURCE_PATHS = (
    'refine/feedback_trajectory.py',
    'refine/native_output.py',
    'refine/native_objective.py',
    'regularize/evaluation.py',
    'regularize/row_output.py',
    'regularize/grid.py',
    'regularize/arguments_native.py',
    'regularize/arguments_compression.py',
    'regularize/arguments_combination.py',
    'regularize/arguments_thickness.py',
    'refine/arguments_feedback.py',
    'refine/arguments_native.py',
    'refine/arguments_multidataset.py',
    'regularize/source.py',
    'refine/datasets.py',
    'refine/feedback_contract.py',
    'refine/geometry_restraints.py',
    'refine/parameters.py',
    'refine/shelx_input.py',
    'refine/shelxl_diagnostics.py',
    'refine/shelxl_iterations.py',
    'regularize/compression_policy.py',
    'regularize/compression_topology.py',
    'regularize/compression_weights.py',
    'regularize/contracts.py',
    'regularize/multidataset.py',
    'regularize/observations.py',
    'regularize/prediction_cache.py',
    'regularize/scales.py',
    'regularize/selection.py',
    'regularize/thickness_response.py',
    'legacy/engine/combine_native_nuisance_runs.py',
    'legacy/engine/continuous_thickness_polish.py',
    'legacy/engine/native_multi_structure_update.py',
    'legacy/engine/native_row_regularize.py',
    'legacy/engine/native_structure_update.py',
    'legacy/engine/scale_compression_polish.py',
    'legacy/engine/shelxl_feedback_loop.py',
)


def regularization_source_files() -> tuple[Path, ...]:
    """Return the absolute paths of REGULARIZATION_SOURCE_PATHS in the installed grond package."""

    package_root = Path(__file__).resolve().parents[1]
    return tuple(package_root / relative for relative in REGULARIZATION_SOURCE_PATHS)
