"""Shared repository location; tests must not depend on their folder depth."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
