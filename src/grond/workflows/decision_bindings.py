"""Verify saved hand decisions against immutable acquisition, model and row identities."""

from __future__ import annotations

import re


def _validated_sha256_text(value: object, *, label: str) -> str:
    """Validate a caller-supplied immutable external provenance fingerprint."""

    text = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{label} must be exactly 64 hexadecimal SHA-256 characters")
    return text
