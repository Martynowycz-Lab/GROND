"""Public GROND entry point (``grond`` and ``GROND`` console scripts).

Command handlers, option registration, workflows and reports have separate owners under ``grond.commands``,
``grond.cli_options``, ``grond.workflows`` and ``grond.reports``.
"""

from __future__ import annotations

from grond.cli_support.entrypoint import main as main

if __name__ == "__main__":
    raise SystemExit(main())
