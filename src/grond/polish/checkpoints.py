"""Completed row-chunk evaluations, bound to their inputs and numerical implementation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import scipy


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def evaluate_chunk_cached(task: dict) -> dict:
    """Reuse only a complete, hash-verified evaluation of exactly this task.

    Workers save each completed chunk before returning it. An interrupted multi-hour evaluation can therefore
    resume on the same parameter vector. Changing a potential, derivative, plan, numerical setting or kernel
    creates a different entry. A corrupt completed entry is an error, never a silently accepted prediction.
    """

    from grond.bloch import thickness
    from grond.polish import crystals, propagator

    work = dict(task)
    directory = Path(work.pop("cache_directory"))
    inputs = {key: digest(Path(work[key])) for key in ("chunk", "u", "d_re", "d_im", "lookup") if work.get(key)}
    sources = {Path(p).name: digest(Path(p)) for p in (__file__, crystals.__file__, propagator.__file__, thickness.__file__)}
    # The same long thickness arrays occur in every task of an evaluation. Bind their complete serialized values
    # by hash, rather than storing tens of thousands of redundant numbers in each chunk's marker.
    task_hash = hashlib.sha256(json.dumps(work, sort_keys=True).encode()).hexdigest()
    binding = {"schema": 2, "task_sha256": task_hash, "input_paths": {key: work[key] for key in inputs},
               "inputs": inputs, "sources": sources,
               "numpy": np.__version__, "scipy": scipy.__version__}
    key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    target = directory / key
    manifest = target / "COMPLETE.json"
    arrays = target / "arrays.npz"
    if manifest.exists():
        record = json.loads(manifest.read_text())
        if record["binding"] != binding or digest(arrays) != record["array_sha256"]:
            raise ValueError(f"invalid evaluation checkpoint: {target}")
        with np.load(arrays, allow_pickle=False) as saved:
            result = {name: saved[name].copy() for name in saved.files}
        result.update(record["scalars"])
        return result
    result = crystals.evaluate_chunk(work)
    target.mkdir(parents=True, exist_ok=True)
    temporary = target / f"arrays.{os.getpid()}.npz"
    np.savez_compressed(temporary, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    os.replace(temporary, arrays)
    record = {"binding": binding, "array_sha256": digest(arrays),
              "scalars": {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}}
    temporary = target / f"complete.{os.getpid()}.json"
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, manifest)
    return result
