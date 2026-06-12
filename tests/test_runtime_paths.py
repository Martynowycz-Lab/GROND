from pathlib import Path

from grond.paths import PACKAGE_ROOT, runtime_cache_dir


def test_runtime_cache_dir_honors_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GROND_CACHE_DIR", str(tmp_path / "grond-cache"))

    assert runtime_cache_dir("native", "ug") == tmp_path / "grond-cache" / "native" / "ug"


def test_runtime_cache_dir_default_is_not_inside_package(monkeypatch):
    monkeypatch.delenv("GROND_CACHE_DIR", raising=False)

    default = runtime_cache_dir("native_row_regularize")

    assert default.is_absolute()
    assert PACKAGE_ROOT not in default.parents
