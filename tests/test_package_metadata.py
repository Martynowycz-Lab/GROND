import importlib.metadata
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_license_metadata_is_mit():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    installed = importlib.metadata.metadata("grond")

    assert pyproject["project"]["license"] == "MIT"
    assert installed.get("License-Expression") == "MIT"
    assert (ROOT / "LICENSE").read_text().startswith("MIT License")
    assert "license: MIT" in (ROOT / "CITATION.cff").read_text()


def test_citation_title_matches_package_title():
    citation = (ROOT / "CITATION.cff").read_text()
    package_title = (ROOT / "src" / "grond" / "__init__.py").read_text()

    assert 'title: "GROND: Global Regularization Of Non-kinematical Diffraction"' in citation
    assert "Global Regularization Of Non-kinematical Diffraction" in package_title
