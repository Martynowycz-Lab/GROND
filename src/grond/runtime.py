from __future__ import annotations

import math
import os
from collections.abc import MutableMapping


def auto_worker_count(requested: int = 0, fraction: float = 0.85) -> int:
    """Return a conservative multiprocessing worker count.

    A requested positive value is honored exactly. Otherwise GROND uses a
    fraction of detected logical CPU cores. The default fraction is 85%, which
    leaves headroom for the OS, filesystem, SHELXL/map helpers, and interactive
    monitoring while still using large Linux servers effectively.
    """

    if requested and requested > 0:
        return int(requested)
    cores = os.cpu_count() or 1
    try:
        frac = float(fraction)
    except (TypeError, ValueError):
        frac = 0.85
    frac = min(1.0, max(0.05, frac))
    return max(1, min(cores, int(math.floor(cores * frac))))


def auto_chunksize(requested: int = 0, workers: int = 1) -> int:
    """Choose a row-task chunksize for variable-cost Bloch predictions."""

    if requested and requested > 0:
        return int(requested)
    if workers >= 64:
        return 1
    if workers >= 16:
        return 2
    return 4


def apply_thread_limits(env: MutableMapping[str, str] | None = None) -> MutableMapping[str, str]:
    """Pin BLAS/OpenMP helper threads inside each worker process.

    The row-level Bloch calculation parallelizes over Python worker processes.
    Letting each process spawn BLAS/OpenMP threads oversubscribes both laptops
    and high-core Linux servers, so these are defaults rather than hard
    overrides. Users can still set them explicitly before launching GROND.
    """

    env = env if env is not None else os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(name, "1")
    return env
