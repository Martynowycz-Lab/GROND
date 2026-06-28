from __future__ import annotations

import json
import platform
import subprocess
import sys

PROBE_CODE = r"""
import importlib
import json
import os
import platform
import sys

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
mods = {}
for name in ["numpy", "scipy", "gemmi"]:
    try:
        mod = importlib.import_module(name)
        mods[name] = {"available": True, "version": str(getattr(mod, "__version__", "installed"))}
    except Exception as exc:
        mods[name] = {"available": False, "error": f"{type(exc).__name__}: {str(exc)[:240]}"}

jax_info = {"available": False}
try:
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    jax_info = {
        "available": True,
        "default_backend": str(jax.default_backend()),
        "devices": [str(d) for d in jax.devices()],
        "x64": bool(jax.config.jax_enable_x64),
    }
except Exception as exc:
    jax_info = {
        "available": False,
        "optional": True,
        "status": "not_installed_optional",
        "message": "JAX is not required for production GROND. The portable NumPy/SciPy Bloch solver is the reference path.",
        "import_error": f"{type(exc).__name__}: {str(exc)[:500]}",
    }

print(json.dumps({
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "machine": platform.machine(),
    "packages": mods,
    "optional_packages": {"jax": jax_info},
    "jax": jax_info,
    "engine": {
        "current_core": "numpy_scipy_cpu",
        "optional_backends": ["jax", "jax_auto", "jax_cpu", "jax_mps", "jax_cuda", "jax_cpu_padded", "jax_cuda_padded"],
        "reason": "The portable NumPy/SciPy Bloch solver remains the production reference; JAX backends are experimental and must pass complex eig/eigh agreement checks before use. Padded JAX backends bucket matrix dimensions to reduce recompilation on CPU/CUDA.",
    },
}, indent=2))
"""


BLOCH_OPS_PROBE_CODE = r"""
import json
import os
import time

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
out = {}
try:
    import numpy as np
    from scipy.linalg import eig, eigh

    H = np.eye(4, dtype=np.complex128) + (0.01 + 0.02j) * np.ones((4, 4), dtype=np.complex128)
    Hh = H + H.conj().T
    rec = {}
    for name, func, mat in [("eig", eig, H), ("eigh", eigh, Hh)]:
        try:
            t = time.perf_counter()
            vals, _vecs = func(mat)
            rec[name] = {"ok": True, "sec": time.perf_counter() - t, "dtype": str(vals.dtype)}
        except Exception as exc:
            rec[name] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
    out["scipy_cpu"] = rec
except Exception as exc:
    out["scipy_cpu"] = {"import_error": f"{type(exc).__name__}: {str(exc)[:500]}"}

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    H = jnp.asarray(np.eye(4, dtype=np.complex128) + (0.01 + 0.02j) * np.ones((4, 4), dtype=np.complex128))
    Hh = H + jnp.conj(H.T)
    rec = {
        "default_backend": str(jax.default_backend()),
        "devices": [str(d) for d in jax.devices()],
    }
    for name, func, mat in [("eig", jnp.linalg.eig, H), ("eigh", jnp.linalg.eigh, Hh)]:
        try:
            t = time.perf_counter()
            vals, _vecs = func(mat)
            vals.block_until_ready()
            rec[name] = {"ok": True, "sec": time.perf_counter() - t, "dtype": str(vals.dtype)}
        except Exception as exc:
            rec[name] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
    out["jax"] = rec
except Exception as exc:
    out["jax"] = {
        "available": False,
        "optional": True,
        "status": "not_installed_optional",
        "message": "JAX is optional and experimental; SciPy eig/eigh is the production Bloch kernel.",
        "import_error": f"{type(exc).__name__}: {str(exc)[:500]}",
    }

print(json.dumps(out, indent=2))
"""


def probe_accelerators(python: str | None = None, timeout: int = 20) -> dict:
    """Report the portable Python packages GROND needs to run."""

    exe = python or sys.executable
    try:
        proc = subprocess.run([exe, "-c", PROBE_CODE], check=False, text=True, capture_output=True, timeout=timeout)
    except Exception as exc:
        return {
            "python": exe,
            "probe_error": f"{type(exc).__name__}: {exc}",
            "platform": platform.platform(),
            "engine": {"current_core": "numpy_scipy_cpu", "status": "unknown"},
        }
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        data = {
            "python": exe,
            "probe_error": "dependency probe did not return JSON",
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        }
    data["returncode"] = proc.returncode
    if proc.stderr.strip():
        data["stderr"] = proc.stderr.strip()[-2000:]
    return data


def probe_bloch_ops(python: str | None = None, timeout: int = 30) -> dict:
    """Check that SciPy can run the complex eigensolvers used by the Bloch kernel."""

    exe = python or sys.executable
    try:
        proc = subprocess.run([exe, "-c", BLOCH_OPS_PROBE_CODE], check=False, text=True, capture_output=True, timeout=timeout)
    except Exception as exc:
        return {"python": exe, "probe_error": f"{type(exc).__name__}: {exc}"}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        data = {
            "python": exe,
            "probe_error": "Bloch operation probe did not return JSON",
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        }
    data["python"] = exe
    data["returncode"] = proc.returncode
    if proc.stderr.strip():
        data["stderr"] = proc.stderr.strip()[-2000:]
    return data
