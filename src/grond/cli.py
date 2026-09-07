"""Public GROND entry point and compatibility imports.

Command handlers, option registration, workflows and reports have separate
owners. Keep this module free of workflow state and numerical decisions.
"""

from __future__ import annotations

import argparse as argparse
import compileall as compileall
import csv as csv
import hashlib as hashlib
import json as json
import math as math
import os as os
import re as re
import shlex as shlex
import shutil as shutil
import subprocess as subprocess
import sys as sys
import tempfile as tempfile
import threading as threading
import time as time
from collections.abc import Iterable as Iterable, Mapping as Mapping
from concurrent.futures import ThreadPoolExecutor as ThreadPoolExecutor, as_completed as as_completed
from dataclasses import asdict as asdict
from pathlib import Path as Path

from grond import __version__ as __version__
from grond.accelerators import probe_accelerators as probe_accelerators, probe_bloch_ops as probe_bloch_ops
from grond.cli_options.common import (
    add_common_run_args as add_common_run_args,
)
from grond.cli_options.forensics import (
    add_forensic_args as add_forensic_args,
)
from grond.cli_options.parser import (
    build_parser as build_parser,
)
from grond.cli_support.entrypoint import (
    main as main,
)
from grond.cli_support.execution import (
    _estimated_forensic_steps as _estimated_forensic_steps,
    _estimated_resolution_steps as _estimated_resolution_steps,
    _estimated_sg_window_scout_steps as _estimated_sg_window_scout_steps,
    _percent as _percent,
    _print_stage_metrics as _print_stage_metrics,
    _stage_output_dir as _stage_output_dir,
    configure_progress as configure_progress,
    estimate_run_steps as estimate_run_steps,
    run_logged as run_logged,
    transform_scout_job_count as transform_scout_job_count,
    transform_scout_semaphore as transform_scout_semaphore,
    transform_scout_worker_count as transform_scout_worker_count,
)
from grond.cli_support.tools import (
    add_tool_config_args as add_tool_config_args,
    apply_tool_config as apply_tool_config,
    choose_python as choose_python,
    default_shelxl_executable as default_shelxl_executable,
    external_tool_statuses as external_tool_statuses,
    has_runtime_deps as has_runtime_deps,
    raise_for_missing_external_tools as raise_for_missing_external_tools,
    required_external_tool_errors as required_external_tool_errors,
    shelxl_is_available as shelxl_is_available,
)
from grond.cli_support.values import (
    DEFAULT_ELECTRON_WAVELENGTH_A as DEFAULT_ELECTRON_WAVELENGTH_A,
    DEFAULT_MAX_SUPPORT_SG as DEFAULT_MAX_SUPPORT_SG,
    FULL_Q_LAMBDA as FULL_Q_LAMBDA,
    XDS_TARGET_MODE_LABELS as XDS_TARGET_MODE_LABELS,
    XDS_TARGET_MODES as XDS_TARGET_MODES,
    _json_contract_equal as _json_contract_equal,
    centered_grid as centered_grid,
    format_numeric_grid as format_numeric_grid,
    parse_int_grid as parse_int_grid,
    parse_name_grid as parse_name_grid,
    parse_numeric_grid as parse_numeric_grid,
    read_json as read_json,
    try_read_json as try_read_json,
)
from grond.commands.doctor import (
    _doctor_readiness_failures as _doctor_readiness_failures,
    cmd_doctor as cmd_doctor,
)
from grond.commands.finish import (
    cmd_finish as cmd_finish,
)
from grond.commands.forensics import (
    cmd_calibrate_endpoint as cmd_calibrate_endpoint,
    cmd_forensics as cmd_forensics,
)
from grond.commands.hand import (
    _apply_source_physics_to_hand_args as _apply_source_physics_to_hand_args,
    _source_summary_hand_dataset as _source_summary_hand_dataset,
    cmd_hand_check as cmd_hand_check,
)
from grond.commands.hydrogen_audit import (
    cmd_audit_hydrogen_omit as cmd_audit_hydrogen_omit,
)
from grond.commands.integration_imports import (
    cmd_apply_diffbloch_model as cmd_apply_diffbloch_model,
    cmd_export_pets as cmd_export_pets,
    cmd_import_dials as cmd_import_dials,
    cmd_import_pets as cmd_import_pets,
)
from grond.commands.parity import (
    cmd_apply_parity_calibration as cmd_apply_parity_calibration,
    cmd_canonicalize_hand as cmd_canonicalize_hand,
    cmd_create_parity_calibration as cmd_create_parity_calibration,
)
from grond.commands.pets_evaluate import (
    cmd_pets_evaluate as cmd_pets_evaluate,
)
from grond.commands.preparation import (
    cmd_prep as cmd_prep,
    prep_from_args as prep_from_args,
)
from grond.commands.protein import (
    _format_protein_output_model_template as _format_protein_output_model_template,
    cmd_export_mtz as cmd_export_mtz,
    cmd_make_free_flags as cmd_make_free_flags,
    cmd_protein_ls1 as cmd_protein_ls1,
)
from grond.commands.push import (
    cmd_push_endpoint as cmd_push_endpoint,
    cmd_push_endpoint_multi as cmd_push_endpoint_multi,
)
from grond.commands.raw_polish import (
    cmd_raw_polish as cmd_raw_polish,
)
from grond.commands.run import (
    _configure_static_pets_auto as _configure_static_pets_auto,
    _finalize_quick_stats as _finalize_quick_stats,
    _validated_h_omit_mode as _validated_h_omit_mode,
    _write_run_completion_artifacts as _write_run_completion_artifacts,
    cmd_auto as cmd_auto,
    cmd_run as cmd_run,
)
from grond.commands.scale import (
    cmd_scale_polish as cmd_scale_polish,
    cmd_scale_replay as cmd_scale_replay,
)
from grond.commands.scouting import (
    _fill_common_run_defaults as _fill_common_run_defaults,
    cmd_scout as cmd_scout,
)
from grond.commands.target_ablation import (
    cmd_target_ablation as cmd_target_ablation,
)
from grond.commands.thickness import (
    cmd_thickness_scout as cmd_thickness_scout,
)
from grond.commands.utilities import (
    cmd_endpoints as cmd_endpoints,
    cmd_init_config as cmd_init_config,
    cmd_manual as cmd_manual,
    cmd_row_template as cmd_row_template,
    cmd_row_validate as cmd_row_validate,
    cmd_sfac_check as cmd_sfac_check,
)
from grond.commands.validation import (
    cmd_compile as cmd_compile,
    cmd_validate as cmd_validate,
    cmd_validate_ls1 as cmd_validate_ls1,
)
from grond.hkl_transform import (
    absolute_configuration_claim_contract as absolute_configuration_claim_contract,
    parse_hkl_transform_info as parse_hkl_transform_info,
    source_orientation_contract as source_orientation_contract,
    validate_external_parity_calibration as validate_external_parity_calibration,
)
from grond.io.config import (
    DatasetSpec as DatasetSpec,
    PrepConfig as PrepConfig,
    load_prep as load_prep,
    parse_dataset_spec as parse_dataset_spec,
    resolve_hkl as resolve_hkl,
    write_datasets_csv as write_datasets_csv,
    write_prep as write_prep,
)
from grond.io.structure_contract import (
    _cif_number as _cif_number,
    _clean_embedded_shelx as _clean_embedded_shelx,
    _extract_cif_shelx_block as _extract_cif_shelx_block,
    _is_number as _is_number,
    _shelx_cell_wavelength_status as _shelx_cell_wavelength_status,
    absolute_structure_policy as absolute_structure_policy,
    check_sfac_contract as check_sfac_contract,
    estimate_model_nonh_atoms as estimate_model_nonh_atoms,
    infer_cbf_dir_from_hkl as infer_cbf_dir_from_hkl,
    is_shelx_model as is_shelx_model,
    make_alternate_hand_model as make_alternate_hand_model,
    make_noh_model as make_noh_model,
    prepare_structure_for_legacy as prepare_structure_for_legacy,
)
from grond.io.tool_config import (
    configured_tools as configured_tools,
    find_tool_config as find_tool_config,
    load_tool_config as load_tool_config,
    resolve_executable as resolve_executable,
    tool_env_updates as tool_env_updates,
    tool_statuses as tool_statuses,
    write_config_template as write_config_template,
)
from grond.legacy.engine.shelx_diagnostics import (
    parse_shelx_lst_metrics as parse_shelx_lst_metrics,
    parse_shelx_lst_pathologies as parse_shelx_lst_pathologies,
)
from grond.macromolecule import (
    FREE_FLAG_LABEL as FREE_FLAG_LABEL,
    export_corrected_mtz as export_corrected_mtz,
    infer_cell_and_spacegroup as infer_cell_and_spacegroup,
    make_free_flags as make_free_flags,
    propagate_free_flags as propagate_free_flags,
    read_free_flags as read_free_flags,
    read_table as read_macromolecule_table,  # noqa: F401 - historical CLI import alias
    render_protein_endpoint_command as render_protein_endpoint_command,
    run_protein_endpoint as run_protein_endpoint,
    sha256_file as sha256_file,
    write_table as write_macromolecule_table,  # noqa: F401 - historical CLI import alias
)
from grond.maps.native_dynamic import (
    NativeDynamicMapOptions as NativeDynamicMapOptions,
    write_native_dynamic_map_package as write_native_dynamic_map_package,
)
from grond.paths import LEGACY_ROOT as LEGACY_ROOT, engine_script as engine_script
from grond.progress import ProgressState as ProgressState, format_seconds as format_seconds
from grond.refine.endpoints import (
    configured_endpoint_blocks as configured_endpoint_blocks,
    endpoint_report as endpoint_report,
)
from grond.reports.bloch_diagnostics import (
    _write_bloch_diagnostics_report as _write_bloch_diagnostics_report,
    _write_bloch_diagnostics_reports as _write_bloch_diagnostics_reports,
)
from grond.reports.distributions import (
    _quick_csv_numeric_audit as _quick_csv_numeric_audit,
    _quick_distribution_row as _quick_distribution_row,
    _quick_numeric_distribution as _quick_numeric_distribution,
    _quick_quantile as _quick_quantile,
    _quick_residual_z_summary as _quick_residual_z_summary,
)
from grond.reports.endpoints import (
    _chooser_diagnostic_endpoint as _chooser_diagnostic_endpoint,
    _chooser_publication_endpoint as _chooser_publication_endpoint,
    _endpoint_calibration_publication_endpoint as _endpoint_calibration_publication_endpoint,
    _normalize_publication_endpoint_block as _normalize_publication_endpoint_block,
    _quick_endpoint_file_block as _quick_endpoint_file_block,
)
from grond.reports.hand import (
    _quick_final_elastic_hand_aggregate as _quick_final_elastic_hand_aggregate,
    _quick_final_elastic_hand_zkp as _quick_final_elastic_hand_zkp,
    _quick_final_hand_summary as _quick_final_hand_summary,
    _quick_hand_block_t_style as _quick_hand_block_t_style,
    _quick_hand_t_style_text as _quick_hand_t_style_text,
    _quick_multi_hand_console_rows as _quick_multi_hand_console_rows,
    _quick_multi_hand_diagnostic as _quick_multi_hand_diagnostic,
    _quick_multi_hand_evidence_lines as _quick_multi_hand_evidence_lines,
)
from grond.reports.nuisance import (
    _quick_dataset_nuisance_table_lines as _quick_dataset_nuisance_table_lines,
    _quick_thickness_text as _quick_thickness_text,
)
from grond.reports.progress import (
    _quick_progress_finish_info as _quick_progress_finish_info,
    _quick_progress_source_roots as _quick_progress_source_roots,
    _quick_progress_timings as _quick_progress_timings,
)
from grond.reports.push import (
    write_multi_push_endpoint_report as write_multi_push_endpoint_report,
    write_push_endpoint_report as write_push_endpoint_report,
)
from grond.reports.quick_stats import (
    write_quick_stats as write_quick_stats,
)
from grond.reports.shelxl import (
    _QUICK_SHELX_FLOAT_TOKEN_RE as _QUICK_SHELX_FLOAT_TOKEN_RE,
    _quick_baseline_shelxl as _quick_baseline_shelxl,
    _quick_endpoint_scale_text as _quick_endpoint_scale_text,
    _quick_goof_warning as _quick_goof_warning,
    _quick_iter_recommended_wght_terms as _quick_iter_recommended_wght_terms,
    _quick_lst_diagnostics as _quick_lst_diagnostics,
    _quick_parse_wght_terms as _quick_parse_wght_terms,
    _quick_selected_feedback_shelxl as _quick_selected_feedback_shelxl,
    _quick_shelxl_details as _quick_shelxl_details,
    _quick_shelxl_issue_text as _quick_shelxl_issue_text,
    _quick_shelxl_weight_text as _quick_shelxl_weight_text,
)
from grond.reports.source_controls import (
    _quick_delta_percent as _quick_delta_percent,
    _quick_flat_hand_t_style_text as _quick_flat_hand_t_style_text,
    _quick_row_weight_definition as _quick_row_weight_definition,
    _quick_source_shelx_control_text as _quick_source_shelx_control_text,
    _quick_source_shelx_controls as _quick_source_shelx_controls,
    _quick_transform_summary as _quick_transform_summary,
)
from grond.reports.table1 import (
    _quick_complete_jana_metrics as _quick_complete_jana_metrics,
    _quick_final_jana_row_summary as _quick_final_jana_row_summary,
    _table1_jana_record as _table1_jana_record,
    _table1_pair as _table1_pair,
    _table1_shelxl_record as _table1_shelxl_record,
    _write_final_statistics_table1 as _write_final_statistics_table1,
)
from grond.reports.terminal import (
    _format_terminal_finish_summary as _format_terminal_finish_summary,
)
from grond.reports.values import (
    _quick_best_endpoint as _quick_best_endpoint,
    _quick_best_native as _quick_best_native,
    _quick_born_metric as _quick_born_metric,
    _quick_combined_native_metrics as _quick_combined_native_metrics,
    _quick_count_pair as _quick_count_pair,
    _quick_dataset_nuisance_contracts as _quick_dataset_nuisance_contracts,
    _quick_dyn_metric as _quick_dyn_metric,
    _quick_final_run as _quick_final_run,
    _quick_float as _quick_float,
    _quick_fraction as _quick_fraction,
    _quick_int as _quick_int,
    _quick_load_run_summary as _quick_load_run_summary,
    _quick_metric as _quick_metric,
    _quick_nuisance as _quick_nuisance,
    _quick_number as _quick_number,
    _quick_percent as _quick_percent,
)
from grond.runtime import (
    apply_thread_limits as apply_thread_limits,
    auto_chunksize as auto_chunksize,
    auto_worker_count as auto_worker_count,
)
from grond.workflows.checkpoints import (
    _checkpoint_dataset_contract as _checkpoint_dataset_contract,
    _checkpoint_path as _checkpoint_path,
    _summary_section as _summary_section,
    write_continue_checkpoint as write_continue_checkpoint,
)
from grond.workflows.conventional_baseline import (
    _endpoint_baseline_no_harm_gate as _endpoint_baseline_no_harm_gate,
    _fresh_baseline_shelxl as _fresh_baseline_shelxl,
    _provenance_file_is_current as _provenance_file_is_current,
    _publication_baseline_hkl as _publication_baseline_hkl,
    _validated_fresh_conventional_baseline as _validated_fresh_conventional_baseline,
    run_fresh_conventional_baseline as run_fresh_conventional_baseline,
)
from grond.workflows.dataset_gates import (
    all_primary_policy as all_primary_policy,
    apply_dataset_policy_to_nuisance as apply_dataset_policy_to_nuisance,
    dataset_policy_kwargs as dataset_policy_kwargs,
    policy_active_runs as policy_active_runs,
    require_materialized_dataset_policy as require_materialized_dataset_policy,
    require_policy_active_runs as require_policy_active_runs,
    run_dataset_policy_gate as run_dataset_policy_gate,
)
from grond.workflows.dataset_policy import (
    build_dataset_policy as build_dataset_policy,
    policy_by_name as policy_by_name,
    write_policy_artifacts as write_policy_artifacts,
    write_policy_weighted_runs as write_policy_weighted_runs,
)
from grond.workflows.decision import (
    ALL_PROPER_SIGNED_PERMUTATIONS as ALL_PROPER_SIGNED_PERMUTATIONS,
    apply_preflight as apply_preflight,
    expand_transform_candidates as expand_transform_candidates,
    preflight_dataset as preflight_dataset,
    safe_float as safe_float,
    safe_label as safe_label,
    score_native_summary as score_native_summary,
    score_transform_summary as score_transform_summary,
    should_stop_for_bad_data as should_stop_for_bad_data,
    thickness_polish_nuisance as thickness_polish_nuisance,
    write_json as write_json,
)
from grond.workflows.decision_bindings import (
    _external_parity_calibration_binding as _external_parity_calibration_binding,
    _fixed_hand_decision_binding as _fixed_hand_decision_binding,
    _require_matching_decision_identities as _require_matching_decision_identities,
    _validated_decision_identity as _validated_decision_identity,
    _validated_sha256_text as _validated_sha256_text,
)
from grond.workflows.endpoint_candidates import (
    _endpoint_calibration_chooser_candidate as _endpoint_calibration_chooser_candidate,
    _multi_feedback_endpoint_candidates as _multi_feedback_endpoint_candidates,
    _publication_model_endpoint_candidate as _publication_model_endpoint_candidate,
)
from grond.workflows.endpoint_chooser import (
    endpoint_artifact_provenance as endpoint_artifact_provenance,
    endpoint_lst_metric_consistency as endpoint_lst_metric_consistency,
    endpoint_promotion_field_contract as endpoint_promotion_field_contract,
    write_publication_endpoint_choice as write_publication_endpoint_choice,
)
from grond.workflows.endpoint_commands import (
    shelxl_feedback_cmd as shelxl_feedback_cmd,
    structure_cmd as structure_cmd,
)
from grond.workflows.endpoint_selection import (
    _endpoint_calibration_lambda as _endpoint_calibration_lambda,
    _multi_endpoint_calibration_not_applicable as _multi_endpoint_calibration_not_applicable,
    _multi_hand_publication_blockers as _multi_hand_publication_blockers,
    _recorded_native_row_tables as _recorded_native_row_tables,
    run_endpoint_calibration as run_endpoint_calibration,
    run_publication_endpoint_chooser as run_publication_endpoint_chooser,
    run_publication_report as run_publication_report,
    run_scale_window_polish as run_scale_window_polish,
)
from grond.workflows.final_hand_selection import (
    run_multi_final_elastic_hand_selection as run_multi_final_elastic_hand_selection,
)
from grond.workflows.finish_maps import (
    _finish_list6_fcf as _finish_list6_fcf,
    _finish_native_model_contract as _finish_native_model_contract,
    _finish_native_row_tables as _finish_native_row_tables,
    _finish_run_dir as _finish_run_dir,
    _publication_endpoint_from_result as _publication_endpoint_from_result,
    _shelx2map_executable as _shelx2map_executable,
    _shelx_fcf_list_code as _shelx_fcf_list_code,
    _write_finish_list6_ins as _write_finish_list6_ins,
    run_finish_endpoint_maps as run_finish_endpoint_maps,
    run_finish_maps as run_finish_maps,
    run_finish_native_dynamic_maps as run_finish_native_dynamic_maps,
)
from grond.workflows.forensic_gates import (
    forensic_command_plan as forensic_command_plan,
    observation_polish_cmd as observation_polish_cmd,
    run_forensic_gate as run_forensic_gate,
    run_resolution_limit_gate as run_resolution_limit_gate,
    should_run_observation_polish as should_run_observation_polish,
    summarize_forensic_gate as summarize_forensic_gate,
)
from grond.workflows.hand_bindings import (
    _native_physics_config_contract as _native_physics_config_contract,
    _policy_selected_table_binding as _policy_selected_table_binding,
    _summary_hkl_binding as _summary_hkl_binding,
    _summary_model_binding as _summary_model_binding,
    _summary_native_row_table_binding as _summary_native_row_table_binding,
)
from grond.workflows.hand_row_contracts import (
    _args_with_recorded_native_contract as _args_with_recorded_native_contract,
    _native_split_dyn_wr2 as _native_split_dyn_wr2,
    _paired_row_split_contract as _paired_row_split_contract,
    _pinned_nuisance_mismatches as _pinned_nuisance_mismatches,
    _positive_summary_int as _positive_summary_int,
    _row_contract_nullable_numeric_fields as _row_contract_nullable_numeric_fields,
    _validated_row_selection_counts as _validated_row_selection_counts,
    elastic_hand_nuisance as elastic_hand_nuisance,
    score_native_metrics as score_native_metrics,
)
from grond.workflows.hand_scout import (
    _dataset_decision_identity as _dataset_decision_identity,
    aggregate_shared_hand_decisions as aggregate_shared_hand_decisions,
    choose_hand_by_scout as choose_hand_by_scout,
    choose_shared_hand_by_scout as choose_shared_hand_by_scout,
    raw_free_dyn_wr2 as raw_free_dyn_wr2,
)
from grond.workflows.hand_statistics import (
    _adjusted_hand_blocks_match as _adjusted_hand_blocks_match,
    _adjusted_hand_z_block as _adjusted_hand_z_block,
    _combine_adjusted_hand_z as _combine_adjusted_hand_z,
    _hand_evidence_warnings as _hand_evidence_warnings,
    _pooled_hand_pair_statistics as _pooled_hand_pair_statistics,
    hand_diagnostic_supports_other as hand_diagnostic_supports_other,
    hand_preference_summary as hand_preference_summary,
)
from grond.workflows.model_selection import (
    best_endpoint_model as best_endpoint_model,
    best_nuisance as best_nuisance,
    dataset_baseline_hkl as dataset_baseline_hkl,
    resolve_structure_update_strategy as resolve_structure_update_strategy,
    resolved_frame_offset_candidates as resolved_frame_offset_candidates,
    shelx_model_is_usable as shelx_model_is_usable,
    shelxl_feedback_current_model as shelxl_feedback_current_model,
    shelxl_feedback_current_run as shelxl_feedback_current_run,
    shelxl_feedback_record_safe_for_reuse as shelxl_feedback_record_safe_for_reuse,
    should_run_endpoint_feedback as should_run_endpoint_feedback,
)
from grond.workflows.multi import (
    run_multi as run_multi,
)
from grond.workflows.native_commands import (
    native_stage_cmd as native_stage_cmd,
)
from grond.workflows.nuisance_policy import (
    apply_speed_profile as apply_speed_profile,
    compact_base_nuisance as compact_base_nuisance,
    effective_topup_d_min as effective_topup_d_min,
    feature_grid as feature_grid,
    feature_is_enabled as feature_is_enabled,
    first as first,
    grid_from_nuisance as grid_from_nuisance,
    grid_has_variation as grid_has_variation,
    nuisance_with_updates as nuisance_with_updates,
    prediction_cache_dir as prediction_cache_dir,
    resolve_final_n_basis as resolve_final_n_basis,
    resolved_dataset_sigma_policy as resolved_dataset_sigma_policy,
    resolved_merge_sigma_policy as resolved_merge_sigma_policy,
    resolved_post_bloch_sigma_policy as resolved_post_bloch_sigma_policy,
    resolved_row_plan_cache as resolved_row_plan_cache,
    resolved_sg_window_mode as resolved_sg_window_mode,
    transform_scout_nuisance as transform_scout_nuisance,
)
from grond.workflows.parity_evidence import (
    build_common_sigma_hand_evidence as build_common_sigma_hand_evidence,
    replay_transform_work_born as replay_transform_work_born,
    select_branch_symmetric_nuisance as select_branch_symmetric_nuisance,
)
from grond.workflows.post_feedback_hand import (
    run_multi_post_feedback_hand_diagnostic as run_multi_post_feedback_hand_diagnostic,
)
from grond.workflows.push_contracts import (
    apply_push_checkpoint as apply_push_checkpoint,
    dataset_policy_from_row_contracts as dataset_policy_from_row_contracts,
    hydrate_push_args_for_multi as hydrate_push_args_for_multi,
    override_push_native_weights as override_push_native_weights,
    prep_from_multi_checkpoint as prep_from_multi_checkpoint,
    resolve_dataset_row_contracts as resolve_dataset_row_contracts,
    resolve_dataset_scale_specs as resolve_dataset_scale_specs,
)
from grond.workflows.push_models import (
    SHELX_RESTRAINT_COMMANDS as SHELX_RESTRAINT_COMMANDS,
    is_shelx_atom_line as is_shelx_atom_line,
    prepare_push_endpoint_model as prepare_push_endpoint_model,
    shelx_label_element as shelx_label_element,
)
from grond.workflows.report_commands import (
    combine_cmd as combine_cmd,
    publication_report_cmd as publication_report_cmd,
    resolution_limit_cmd as resolution_limit_cmd,
    resolution_limit_report_args as resolution_limit_report_args,
    scale_polish_report_args as scale_polish_report_args,
)
from grond.workflows.scale_contract_selector import select_multi_scale_contract as select_multi_scale_contract
from grond.workflows.scale_materialization import (
    materialize_dataset_scale_contract as materialize_dataset_scale_contract,
    materialize_scale_contract_cmd as materialize_scale_contract_cmd,
)
from grond.workflows.scale_selection import (
    _SCALE_CONTRACT_MUTABLE_COLUMNS as _SCALE_CONTRACT_MUTABLE_COLUMNS,
    _multi_scale_candidate_runs as _multi_scale_candidate_runs,
    _policy_weight_scale_contract_runs as _policy_weight_scale_contract_runs,
    combined_scale_contract_candidate as combined_scale_contract_candidate,
    run_multi_scale_contract_selection as run_multi_scale_contract_selection,
    scale_contract_totals as scale_contract_totals,
    validate_scale_contract_run_map as validate_scale_contract_run_map,
    write_multi_scale_contract_report as write_multi_scale_contract_report,
    write_selected_scale_polish_summaries as write_selected_scale_polish_summaries,
)
from grond.workflows.sg_window_selection import (
    native_dyn_metrics as native_dyn_metrics,
    run_sg_window_scout as run_sg_window_scout,
    sg_window_score as sg_window_score,
)
from grond.workflows.shared_refinement import (
    multi_combined_endpoint_model as multi_combined_endpoint_model,
    multi_combined_endpoint_shelxl as multi_combined_endpoint_shelxl,
    run_multi_final_row_recompute as run_multi_final_row_recompute,
    run_multi_shelxl_feedback_loop as run_multi_shelxl_feedback_loop,
)
from grond.workflows.single import (
    ensure_single_final_row_model_contract as ensure_single_final_row_model_contract,
    run_feature_gate as run_feature_gate,
    run_single as run_single,
    run_single_adaptive as run_single_adaptive,
)
from grond.workflows.target_ablation import (
    _run_target_ablation_variant as _run_target_ablation_variant,
    _target_ablation_auto_command as _target_ablation_auto_command,
    _target_ablation_summary_row as _target_ablation_summary_row,
    _target_ablation_variants as _target_ablation_variants,
    _write_target_ablation_report as _write_target_ablation_report,
)
from grond.workflows.thickness_materialization import (
    continuous_thickness_cmd as continuous_thickness_cmd,
    continuous_thickness_enabled as continuous_thickness_enabled,
    continuous_thickness_seed_nuisance as continuous_thickness_seed_nuisance,
    discrete_thickness_baseline_nuisance as discrete_thickness_baseline_nuisance,
    discrete_thickness_promotion_record as discrete_thickness_promotion_record,
    materialize_continuous_nuisance_run as materialize_continuous_nuisance_run,
    numeric_nuisance_contract_matches as numeric_nuisance_contract_matches,
)
from grond.workflows.thickness_scout import (
    _float_from_mapping as _float_from_mapping,
    _logq_by_row as _logq_by_row,
    _p95 as _p95,
    _pearson as _pearson,
    _q_field_stability as _q_field_stability,
    _read_csv_dicts as _read_csv_dicts,
    _row_key as _row_key,
    _thickness_scout_variant_args as _thickness_scout_variant_args,
    _thickness_summary_row as _thickness_summary_row,
    _write_csv as _write_csv,
    _write_thickness_scout_report as _write_thickness_scout_report,
)
from grond.workflows.transform_binding import (
    _fixed_transform_decision_binding as _fixed_transform_decision_binding,
)
from grond.workflows.transform_metadata import metadata_prefilter_transforms as metadata_prefilter_transforms
from grond.workflows.transform_selection import (
    _materialize_transform_hand_split_contract as _materialize_transform_hand_split_contract,
    _selected_transform_claim_contract as _selected_transform_claim_contract,
    _transform_candidate_generation_contract as _transform_candidate_generation_contract,
    _transform_record_metadata as _transform_record_metadata,
    choose_transform_by_scout as choose_transform_by_scout,
    dataset_with_split_contract as dataset_with_split_contract,
    dataset_with_transform as dataset_with_transform,
    ranked_transform_records as ranked_transform_records,
    transform_scout_records as transform_scout_records,
)

if __name__ == "__main__":
    raise SystemExit(main())
