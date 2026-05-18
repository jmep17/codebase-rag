#!/usr/bin/env python3
"""Validate commit messages against a small Conventional Commits profile."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOWED_TYPES = {
    "build",
    "chore",
    "ci",
    "docs",
    "feat",
    "fix",
    "perf",
    "refactor",
    "revert",
    "style",
    "test",
}

HEADER_RE = re.compile(
    r"^(?P<type>[a-z]+)(?:\([a-z0-9][a-z0-9._-]*\))?(?P<breaking>!)?: (?P<subject>.+)$"
)


def _read_header(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def validate_header(header: str) -> list[str]:
    errors: list[str] = []
    if not header:
        return ["commit message is empty"]

    if header.startswith(("Merge ", "Revert ")):
        return []

    match = HEADER_RE.match(header)
    if not match:
        return [
            "header must match: <type>[optional scope][!]: <description>",
            "example: feat(tui): add retrieval inspector",
        ]

    commit_type = match.group("type")
    subject = match.group("subject")
    if commit_type not in ALLOWED_TYPES:
        allowed = ", ".join(sorted(ALLOWED_TYPES))
        errors.append(f"unsupported type {commit_type!r}; allowed types: {allowed}")
    if len(header) > 100:
        errors.append("header must be 100 characters or less")
    if subject[0].isupper():
        errors.append("description should start lowercase")
    if subject.endswith("."):
        errors.append("description should not end with a period")
    if not subject.strip():
        errors.append("description must not be empty")
    return errors


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_commit_msg.py <commit-msg-file>", file=sys.stderr)
        return 2

    path = Path(argv[1])
    if not path.is_file():
        print(f"commit message file not found: {path}", file=sys.stderr)
        return 2

    header = _read_header(path)
    errors = validate_header(header)
    if not errors:
        return 0

    print("invalid commit message:", file=sys.stderr)
    print(f"  {header or '(empty)'}", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
