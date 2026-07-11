# Development

Run the public checks before committing:

```bash
python -m pip install -e ".[dev]"
python -m ruff check src/grond tests scripts --no-cache
python -m pytest
python -m grond validate
python -m grond doctor --probe-bloch
```

Keep public modules in the domain folders. Standalone experiments, hardware-result dumps, dataset workspaces, and article-drafting material do not belong in this repository. Promote a result only after it has become callable package behavior, a focused dataset-neutral regression test, or portable documentation.

The test suite also checks the public package boundary. Documented GROND commands must still parse, documented repository paths must point to real files, tracked text must not contain private machine paths, and the tree must not gain experiment workspaces, foreign-project checkouts, article-drafting material, raw-data directories, caches, or non-Python frontend package artifacts. These checks are deliberately strict because a new user should be able to read the docs, install the wheel, and understand that GROND is a self-contained Python crystallography package.

## Release Smoke Test

The complete human checklist is in [RELEASE_CHECKLIST.md](../RELEASE_CHECKLIST.md).

Before tagging a release, build the package from a clean tree and inspect the artifacts. A clean tree avoids stale editable-install metadata such as `src/*.egg-info`.

```bash
tmp=$(mktemp -d)
export GROND_BUILD_AUDIT_OUT=$(mktemp -d)
git archive HEAD | tar -x -C "$tmp"
python -m build --no-isolation --sdist --wheel --outdir "$GROND_BUILD_AUDIT_OUT" "$tmp"
python "$tmp/scripts/check_release_artifacts.py" "$GROND_BUILD_AUDIT_OUT"
python "$tmp/scripts/smoke_install_wheel.py" "$GROND_BUILD_AUDIT_OUT"
```

The wheel and source distribution should contain docs, examples, schemas, tests, and package files. They should not contain bytecode, local caches, generated prediction arrays, benchmark cache files, off-repo writing workspaces, private machine paths, or stale public wording.
The wheel should also install into a fresh virtual environment and expose both `grond` and `GROND` console commands.
