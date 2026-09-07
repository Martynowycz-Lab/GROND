"""Compatibility imports for the expert workflow; implementations live in manual/."""

import hashlib as hashlib
import json as json
import math as math
import os as os
import queue as queue
import shlex as shlex
import shutil as shutil
import subprocess as subprocess
import sys as sys
import threading as threading
import time as time
from collections.abc import Mapping as Mapping
from dataclasses import asdict as asdict
from pathlib import Path as Path
from typing import Any as Any

from grond.hkl_transform import parse_hkl_transform_info as parse_hkl_transform_info
from grond.io.config import (
    PrepConfig as PrepConfig,
    load_prep as load_prep,
    parse_dataset_spec as parse_dataset_spec,
    write_prep as write_prep,
)
from grond.legacy.engine.shelx_diagnostics import (
    parse_shelx_lst_metrics as parse_shelx_lst_metrics,
    parse_shelx_lst_pathologies as parse_shelx_lst_pathologies,
)
from grond.manual.entrypoint import (
    manual_status as manual_status,
    run_manual_stage as run_manual_stage,
)
from grond.manual.execution import (
    _child_base as _child_base,
    _command_elapsed as _command_elapsed,
    _command_text as _command_text,
    _manual_stream_line as _manual_stream_line,
    _run_stage_command as _run_stage_command,
    _tool_args as _tool_args,
    _verbose_child_output as _verbose_child_output,
)
from grond.manual.feedback import (
    _manual_checkpoint as _manual_checkpoint,
    manual_feedback as manual_feedback,
    manual_scales as manual_scales,
)
from grond.manual.finish import (
    manual_finish as manual_finish,
)
from grond.manual.hand import (
    _hand_evidence as _hand_evidence,
    manual_hand as manual_hand,
)
from grond.manual.reporting import (
    _finite_float as _finite_float,
    _manual_markdown_summary as _manual_markdown_summary,
    _manual_row_statistics as _manual_row_statistics,
    _manual_scale_summary as _manual_scale_summary,
    _number as _number,
    _percent as _percent,
    _selected_hand_summary as _selected_hand_summary,
)
from grond.manual.scouting import (
    _summary_bindings as _summary_bindings,
    _validate_reused_summary_binding as _validate_reused_summary_binding,
    manual_scout as manual_scout,
    manual_thickness as manual_thickness,
)
from grond.manual.setup import (
    manual_setup as manual_setup,
)
from grond.manual.state import (
    SCHEMA_VERSION as SCHEMA_VERSION,
    STAGE_ORDER as STAGE_ORDER,
    WORKFLOW_KIND as WORKFLOW_KIND,
    _load_workflow as _load_workflow,
    _next_command as _next_command,
    _print_handoff as _print_handoff,
    _read_json as _read_json,
    _record_stage as _record_stage,
    _require_stage as _require_stage,
    _save_workflow as _save_workflow,
    _sha256 as _sha256,
    _stage as _stage,
    _summary_nuisance as _summary_nuisance,
    _workflow_path as _workflow_path,
    _write_json as _write_json,
)
from grond.maps.native_dynamic import (
    NativeDynamicMapOptions as NativeDynamicMapOptions,
    write_native_dynamic_map_package as write_native_dynamic_map_package,
)
