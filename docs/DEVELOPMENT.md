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
python scripts/check_release_artifacts.py "$GROND_BUILD_AUDIT_OUT"
```

The wheel and source distribution should contain docs, examples, schemas, tests, and package files. They should not contain bytecode, local caches, generated prediction arrays, benchmark cache files, off-repo writing workspaces, private machine paths, or stale public wording.
