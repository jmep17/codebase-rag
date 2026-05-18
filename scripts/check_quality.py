#!/usr/bin/env python3
"""Run local quality checks used by hooks and CI."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

PYTHON_FILES = [
    "codebase_rag/__main__.py",
    "codebase_rag/chat.py",
    "codebase_rag/tools.py",
    "codebase_rag/index.py",
    "codebase_rag/web.py",
    "codebase_rag/gitops.py",
    "codebase_rag/providers.py",
    "codebase_rag/audit.py",
    "codebase_rag/serve.py",
    "codebase_rag/tui.py",
    "scripts/check_commit_msg.py",
    "scripts/check_quality.py",
    "scripts/install_hooks.py",
]


def _run(argv: list[str]) -> int:
    print("+ " + " ".join(argv))
    proc = subprocess.run(argv, check=False)
    return proc.returncode


def _ruff_available(py: str) -> bool:
    proc = subprocess.run(
        [py, "-m", "ruff", "--version"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def _parse_python(root: Path) -> int:
    for rel in PYTHON_FILES:
        path = root / rel
        if not path.exists():
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print("AST parse: OK")
    return 0


def main(argv: list[str]) -> int:
    fix = "--fix" in argv
    root = Path(__file__).resolve().parents[1]
    py = sys.executable

    rc = _parse_python(root)
    if rc:
        return rc
    if not _ruff_available(py):
        print(
            "Ruff is required for lint/format checks. Install with "
            "`python -m pip install -e .[dev]`.",
            file=sys.stderr,
        )
        return 1

    format_cmd = [py, "-m", "ruff", "format"]
    if not fix:
        format_cmd.append("--check")
    lint_cmd = [py, "-m", "ruff", "check"]
    if fix:
        lint_cmd.append("--fix")
    lint_cmd.append(".")

    for cmd in (format_cmd, lint_cmd):
        rc = _run(cmd)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
