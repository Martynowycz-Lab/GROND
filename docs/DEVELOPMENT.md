# Development

Run the checks before committing:

```bash
python -m pip install -e ".[dev]"
python -m ruff check src/grond tests scripts --no-cache
python -m pytest
python -m grond validate
python -m grond doctor --probe-bloch --strict
```

Keep modules in their domain folders ([CODE_MAP.md](CODE_MAP.md)). Experiments, hardware benchmarks, data sets and writing material do not belong in the repository. A change belongs here when it is package behaviour, a regression test with synthetic inputs, or documentation.

## Tests

`tests/` holds flat `test_*.py` files, one per area (physics, thickness distributions, mosaic spread, nested selection, free set and held-out statistics, HKL writing, CLI). For a focused check run one file, for example `python -m pytest tests/test_physics.py`, then run the whole suite.

A test protects an invariant, a boundary or a known failure, with small synthetic inputs. Take repository paths from `tests._support.REPO_ROOT` and shared builders from their helper module. Do not add the engine directory to `sys.path` or import a package module under a second name. When mocking a collaborator, patch it in the module where the tested function looks it up.

## Release smoke test

Before tagging a release, build from a clean tree and inspect the artifacts. A clean tree avoids stale editable-install metadata such as `src/*.egg-info`.

```bash
tmp=$(mktemp -d)
export GROND_BUILD_AUDIT_OUT=$(mktemp -d)
git archive HEAD | tar -x -C "$tmp"
python -m build --no-isolation --sdist --wheel --outdir "$GROND_BUILD_AUDIT_OUT" "$tmp"
python "$tmp/scripts/check_release_artifacts.py" "$GROND_BUILD_AUDIT_OUT"
python "$tmp/scripts/smoke_install_wheel.py" "$GROND_BUILD_AUDIT_OUT"
```

`check_release_artifacts.py` checks that the sdist and the wheel contain the required files and no bytecode, caches, generated prediction arrays, private machine paths or excluded material. The sdist contains the docs, examples, schemas, tests and the package; the wheel contains the package, its runtime schema and metadata. `smoke_install_wheel.py` installs the wheel into a fresh virtual environment and runs its `grond` and `GROND` commands.
