"""``grond figure-maps``: every map of a finished run boxed around one reference model placement.

The run directory is only read; everything goes to a new ``--out-dir``.  See
:mod:`grond.workflows.figure_maps` and docs/OUTPUTS.md (Figure maps).
"""

from __future__ import annotations

import argparse
import json
import sys


def cmd_figure_maps(args: argparse.Namespace) -> int:
    """Run ``grond figure-maps`` and print the manifest or a summary; return 0 when its status is ok, else 1."""

    from grond.workflows.figure_maps import run_figure_maps

    args.argv = list(sys.argv[1:])
    manifest = run_figure_maps(args)
    if getattr(args, "json", False):
        print(json.dumps(manifest, indent=2, allow_nan=True))
    else:
        boxes = manifest.get("model_box") or {}
        print(f"Figure maps: {manifest['out_dir']}/FIGURE_MAPS_MANIFEST.json ({manifest['status']})", flush=True)
        print(f"  reference model: {manifest['reference_model']}", flush=True)
        for key, report in (boxes.get("placements") or {}).items():
            if report.get("status") == "aligned":
                print(
                    f"  {key}: {report['operation']['triplet']}, RMSD {report['rmsd_A']:.4f} A "
                    f"over {report['matched_atoms']} non-H atoms",
                    flush=True,
                )
            else:
                print(f"  {key}: {report.get('status')}: {report.get('error', '')}", flush=True)
        written = [item for item in (boxes.get("maps") or {}).values()
                   if item.get("status") in {"ok", "ok_own_model_box"}]
        print(f"  model-box maps written: {len(written)} of {len(boxes.get('maps') or {})}", flush=True)
        for warning in manifest.get("warnings") or []:
            print(f"  warning: {warning}", flush=True)
    return 0 if manifest.get("status") in {"ok", "ok_with_refused_placements"} else 1
