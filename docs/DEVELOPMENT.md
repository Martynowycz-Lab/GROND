# Development

Run the public checks before committing:

```bash
python -m pip install -e ".[dev]"
python -m ruff check src/grond tests scripts --no-cache
python -m pytest
python -m grond validate
python -m grond doctor --probe-bloch --strict
```

Keep public modules in the domain folders. Standalone experiments, hardware-result dumps, dataset workspaces, and article-drafting material do not belong in this repository. Promote a result only after it has become callable package behavior, a focused dataset-neutral regression test, or portable documentation.

The test suite also checks the public package boundary. Documented GROND commands must still parse, documented repository paths must point to real files, tracked text must not contain private machine paths, and the tree must not gain experiment workspaces, foreign-project checkouts, article-drafting material, raw-data directories, caches, or non-Python frontend package artifacts. These checks are deliberately strict because a new user should be able to read the docs, install the wheel, and understand that GROND is a self-contained Python crystallography package.

## Test ownership

Tests are grouped by physics, input contracts, hand evidence, regularization, refinement, workflows, reporting, package boundaries and CLI behavior. Every domain runs in normal CI; none is an opt-in historical benchmark. Use `python -m pytest tests/physics` or another domain path for a focused check, then run the whole suite after changes across domains. CLI cases are divided by command and reporting responsibility.

Import repository paths from `tests._support.REPO_ROOT` and shared synthetic builders from their explicit test helper module. Do not add the engine directory to `sys.path` or import a package module under a second bare name. When mocking a collaborator, patch the module where the tested function resolves that name. The `grond.cli` compatibility reexport does not change a function's defining globals.

Keep scientific tests distinct from development experiment archives. A regression test should protect an invariant, a meaningful boundary or a known failure with portable synthetic inputs. A private real-data replay establishes a different kind of evidence and belongs in the experiment catalog. [BENCHMARKS.md](BENCHMARKS.md) records the retained and rejected approaches without shipping sample data.

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

The source distribution should contain docs, examples, schemas, tests, and package files. The wheel contains the installable package, its runtime schema, and distribution metadata. They should not contain bytecode, local caches, generated prediction arrays, benchmark cache files, off-repo writing workspaces, private machine paths, or stale public wording.
The wheel should also install into a fresh virtual environment and expose both `grond` and `GROND` console commands.
