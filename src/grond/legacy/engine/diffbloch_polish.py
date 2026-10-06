#!/usr/bin/env python3
"""Engine entry of AUTO's optional last stage: ``grond push-endpoint --final-polish diffbloch ...``.

Run by path like every other AUTO stage, so the stage does not depend on grond being installed for the
worker interpreter; the arguments are those of ``grond push-endpoint``.
"""

from __future__ import annotations

import sys
from pathlib import Path

if not __package__:
    # Direct engine execution must import this checkout, not another installed GROND.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from grond.cli_support.entrypoint import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["push-endpoint", *sys.argv[1:]]))
