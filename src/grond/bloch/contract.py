from __future__ import annotations

from grond.legacy.engine.bloch_engine import Config, frozen_kernel_contract


def default_kernel_contract() -> dict:
    """Return the frozen numerical-kernel contract used by GROND."""

    return frozen_kernel_contract(Config())
