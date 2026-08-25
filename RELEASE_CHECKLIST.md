# Release Checklist

GROND releases should come from a clean `main` commit that has passed the public checks on every supported Python version.

Before tagging, confirm that `pyproject.toml`, `src/grond/__init__.py`, `CITATION.cff`, and the newest changelog heading carry the same version. Update user-facing behavior in the README and focused guides, and make sure the limitations, data policy, acknowledgments, and security instructions are still accurate. Confirm that the tracked tree contains only package source, tests, documentation, synthetic examples, schemas, and release tooling; experiment workspaces, foreign direct-method projects, article-drafting material, raw data, run outputs, and caches must remain outside it. Do not tag an unreleased changelog heading.

Run the normal checks:

```bash
python -m pip install -e ".[dev]"
python -m ruff check src/grond tests scripts --no-cache
python -m pytest
python -m grond validate
python -m grond doctor --probe-bloch --strict
```

Build from a clean archive so editable-install metadata and local caches cannot leak into the artifacts:

```bash
python -m pip install --upgrade 'setuptools>=77' wheel build
tmp=$(mktemp -d)
out=$(mktemp -d)
git archive HEAD | tar -x -C "$tmp"
python -m build --no-isolation --sdist --wheel --outdir "$out" "$tmp"
python "$tmp/scripts/check_release_artifacts.py" "$out"
python "$tmp/scripts/smoke_install_wheel.py" "$out"
```

Inspect the wheel metadata, console entry points, license, schema file, and source archive contents. Confirm that no experiment workspace, foreign direct-method project, article-drafting material, raw data, local path, generated array, run output, or cache is present. Compare the release commit with `origin/main`, then create an annotated `vX.Y.Z` tag and GitHub release from that exact commit. Record the tag or commit in any calculation that must be reproduced later.
