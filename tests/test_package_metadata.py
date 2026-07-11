import importlib.metadata
import re
import shlex
import subprocess
import tomllib
from pathlib import Path

from grond.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cfg",
    ".cff",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
DOC_COMMAND_FILES = (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")))
ROOT_DOCUMENT_FILES = (
    ROOT / "ACKNOWLEDGMENTS.md",
    ROOT / "CHANGELOG.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "DATA_POLICY.md",
    ROOT / "LIMITATIONS.md",
    ROOT / "PROVENANCE.md",
    ROOT / "RELEASE_CHECKLIST.md",
    ROOT / "SECURITY.md",
)
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
CODE_SPAN_RE = re.compile(r"`([^`]+)`")
REPO_PATH_PREFIXES = (".github/", "docs/", "examples/", "schemas/", "scripts/", "src/", "tests/")
REPO_ROOT_FILES = {
    "ACKNOWLEDGMENTS.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "DATA_POLICY.md",
    "LICENSE",
    "LIMITATIONS.md",
    "MANIFEST.in",
    "PROVENANCE.md",
    "README.md",
    "RELEASE_CHECKLIST.md",
    "SECURITY.md",
    "constraints.txt",
    "environment.yml",
    "pyproject.toml",
}
NON_PYTHON_PACKAGE_NAMES = {
    "package.json",
    "package-lock.json",
    "pn" + "pm-lock.yaml",
    "ya" + "rn.lock",
}
NON_PYTHON_PACKAGE_SUFFIXES = {".js", ".jsx", ".ts", ".tsx"}


def tracked_files() -> list[Path]:
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [Path(line) for line in out.splitlines() if (ROOT / line).is_file()]


def fenced_grond_commands() -> list[tuple[Path, int, str]]:
    commands: list[tuple[Path, int, str]] = []
    for path in DOC_COMMAND_FILES:
        lines = path.read_text().splitlines()
        in_fence = False
        block: list[str] = []
        block_start = 0
        for line_number, line in enumerate(lines, start=1):
            if line.startswith("```"):
                if not in_fence:
                    in_fence = True
                    block = []
                    block_start = line_number + 1
                    continue
                commands.extend(commands_from_block(path, block_start, block))
                in_fence = False
                continue
            if in_fence:
                block.append(line)
    return commands


def commands_from_block(path: Path, block_start: int, block: list[str]) -> list[tuple[Path, int, str]]:
    commands: list[tuple[Path, int, str]] = []
    index = 0
    while index < len(block):
        line = block[index].strip()
        if not line.startswith("GROND "):
            index += 1
            continue
        command_parts: list[str] = []
        command_line = block_start + index
        while index < len(block):
            part = block[index].strip()
            continued = part.endswith("\\")
            command_parts.append(part[:-1].strip() if continued else part)
            index += 1
            if not continued:
                break
        commands.append((path.relative_to(ROOT), command_line, " ".join(command_parts)))
    return commands


def local_markdown_links() -> list[tuple[Path, int, str]]:
    links: list[tuple[Path, int, str]] = []
    for path in DOC_COMMAND_FILES + ROOT_DOCUMENT_FILES:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            for match in MARKDOWN_LINK_RE.finditer(line):
                target = match.group(1).strip()
                if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target = target.split("#", 1)[0].strip()
                if target.startswith("<") and target.endswith(">"):
                    target = target[1:-1]
                if " " in target:
                    target = target.split(" ", 1)[0]
                links.append((path.relative_to(ROOT), line_number, target))
    return links


def documented_repo_paths() -> list[tuple[Path, int, str]]:
    paths: list[tuple[Path, int, str]] = []
    for path in DOC_COMMAND_FILES + ROOT_DOCUMENT_FILES:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            for match in CODE_SPAN_RE.finditer(line):
                target = match.group(1).strip().rstrip(".,:;")
                if not target or any(char in target for char in "*${}"):
                    continue
                if target.startswith(REPO_PATH_PREFIXES) or target in REPO_ROOT_FILES:
                    paths.append((path.relative_to(ROOT), line_number, target))
    return paths


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


def test_release_version_metadata_matches():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_version = str(pyproject["project"]["version"])
    init_text = (ROOT / "src" / "grond" / "__init__.py").read_text()
    citation_text = (ROOT / "CITATION.cff").read_text()
    changelog_text = (ROOT / "CHANGELOG.md").read_text()

    init_match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_text, re.MULTILINE)
    citation_match = re.search(r"^version:\s*['\"]?([^'\"\s]+)", citation_text, re.MULTILINE)

    assert init_match is not None
    assert citation_match is not None
    assert init_match.group(1) == package_version
    assert citation_match.group(1) == package_version
    assert f"## {package_version}" in changelog_text


def test_project_urls_point_to_canonical_repository():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    urls = pyproject["project"]["urls"]

    assert urls["Homepage"] == "https://github.com/Martynowycz-Lab/GROND"
    assert urls["Repository"] == urls["Homepage"]
    assert urls["Issues"] == f"{urls['Homepage']}/issues"
    assert urls["Changelog"].endswith("/blob/main/CHANGELOG.md")


def test_no_tracked_external_workspaces():
    forbidden_roots = {
        "exper" + "iments",
        "open" + "dm",
        "open_direct_" + "methods",
        "open-direct-" + "methods",
        "pa" + "per",
        "manu" + "script",
        "raw_" + "data",
        "raw-" + "data",
        "cache",
        "outputs",
        "grond_out",
    }
    offenders = [str(path) for path in tracked_files() if path.parts and path.parts[0].casefold() in forbidden_roots]

    assert offenders == []


def test_tracked_text_is_portable():
    banned = (
        "/Users/" + "Mike",
        "/data/Data1/users/" + "mike",
        "/data/Data1/" + "mike",
    )
    offenders = []
    for rel_path in tracked_files():
        if rel_path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = (ROOT / rel_path).read_text(errors="ignore")
        for token in banned:
            if token in text:
                offenders.append(f"{rel_path}: {token}")

    assert offenders == []


def test_no_tracked_non_python_frontend_package_artifacts():
    offenders = []
    for rel_path in tracked_files():
        parts = set(rel_path.parts)
        if "node_modules" in parts:
            offenders.append(str(rel_path))
            continue
        if rel_path.name in NON_PYTHON_PACKAGE_NAMES or rel_path.suffix.lower() in NON_PYTHON_PACKAGE_SUFFIXES:
            offenders.append(str(rel_path))

    assert offenders == []


def test_public_text_avoids_disfavored_dataset_labels():
    banned = ("Micro" + "ED", "3D " + "ED", "3D-" + "ED")
    offenders = []
    for rel_path in tracked_files():
        if rel_path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = (ROOT / rel_path).read_text(errors="ignore")
        for token in banned:
            if token in text:
                offenders.append(f"{rel_path}: {token}")

    assert offenders == []


def test_development_check_surface_matches_ci():
    development_text = (ROOT / "docs" / "DEVELOPMENT.md").read_text()
    ci_text = (ROOT / ".github" / "workflows" / "tests.yml").read_text()
    shared_commands = (
        "python -m ruff check src/grond tests scripts --no-cache",
        "python -m pytest",
        "python -m grond validate",
        "python -m grond doctor --probe-bloch",
        "python -m build",
        "check_release_artifacts.py",
        "smoke_install_wheel.py",
    )

    for command in shared_commands:
        assert command in development_text
        assert command in ci_text


def test_python_support_surface_matches_ci_matrix():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    ci_text = (ROOT / ".github" / "workflows" / "tests.yml").read_text()
    readme_text = (ROOT / "README.md").read_text()
    install_text = (ROOT / "docs" / "INSTALL.md").read_text()
    versions = [
        classifier.rsplit("::", 1)[1].strip()
        for classifier in pyproject["project"]["classifiers"]
        if classifier.startswith("Programming Language :: Python :: 3.")
    ]

    assert versions == ["3.11", "3.12", "3.13"]
    for version in versions:
        assert f'"{version}"' in ci_text
        assert version in readme_text
        assert version in install_text
    assert pyproject["project"]["requires-python"] == ">=3.11,<3.14"


def test_documented_local_links_resolve():
    links = local_markdown_links()
    failures = []

    for path, line_number, target in links:
        resolved = (ROOT / path).parent / target
        if not resolved.exists():
            failures.append(f"{path}:{line_number}: missing {target}")

    assert len(links) >= 10
    assert failures == []


def test_documented_repo_paths_resolve():
    paths = documented_repo_paths()
    failures = []

    for path, line_number, target in paths:
        if not (ROOT / target).exists():
            failures.append(f"{path}:{line_number}: missing {target}")

    assert len(paths) >= 15
    assert failures == []


def test_documented_grond_commands_still_parse():
    parser = build_parser()
    commands = fenced_grond_commands()
    failures = []

    for path, line_number, command in commands:
        try:
            parser.parse_args(shlex.split(command)[1:])
        except SystemExit as exc:
            failures.append(f"{path}:{line_number}: exited {exc.code}: {command}")

    assert len(commands) >= 30
    assert failures == []
