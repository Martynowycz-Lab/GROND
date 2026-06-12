import importlib.metadata
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


def tracked_files() -> list[Path]:
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [Path(line) for line in out.splitlines()]


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


def test_no_tracked_paper_workspace():
    paper_dir = "pa" + "per" + "/"
    assert not any(str(path).startswith(paper_dir) for path in tracked_files())


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


def test_development_lint_surface_matches_ci():
    lint_command = "python -m ruff check src/grond tests scripts --no-cache"

    assert lint_command in (ROOT / "docs" / "DEVELOPMENT.md").read_text()
    assert lint_command in (ROOT / ".github" / "workflows" / "tests.yml").read_text()


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
