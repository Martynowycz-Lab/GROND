# Development

Run the public checks before committing:

```bash
python -m pip install -e ".[dev]"
python -m ruff check src/grond tests --no-cache
python -m pytest
python -m grond validate
python -m grond doctor --probe-bloch
```

Keep public modules in the domain folders. Historical experiments should not be added to the package tree unless they are callable, documented, tested, and free of dataset-specific assumptions.

## Release Smoke Test

Before tagging a release, build the package from a clean tree and inspect the artifacts. A clean tree avoids stale editable-install metadata such as `src/*.egg-info`.

```bash
tmp=$(mktemp -d)
export GROND_BUILD_AUDIT_OUT=$(mktemp -d)
git archive HEAD | tar -x -C "$tmp"
python -m build --no-isolation --sdist --wheel --outdir "$GROND_BUILD_AUDIT_OUT" "$tmp"
python - <<'PY'
from pathlib import Path
import tarfile
import zipfile
import os

artifacts = Path(os.environ["GROND_BUILD_AUDIT_OUT"])
bad_terms = ("__pycache__", ".pyc", ".npz", ".npy", "legacy/engine/cache", ".ruff_cache", ".pytest_cache")
for artifact in sorted(artifacts.iterdir()):
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as zf:
            names = zf.namelist()
    elif artifact.name.endswith(".tar.gz"):
        with tarfile.open(artifact) as tf:
            names = tf.getnames()
    else:
        continue
    bad = [name for name in names if any(term in name for term in bad_terms)]
    print(f"{artifact.name}: {len(names)} files, {len(bad)} unwanted generated files")
    if bad:
        raise SystemExit("\\n".join(bad[:20]))
PY
```

The wheel and source distribution should contain docs, examples, schemas, tests, and package files. They should not contain bytecode, local caches, generated prediction arrays, or benchmark cache files.
