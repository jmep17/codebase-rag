#!/usr/bin/env python3
"""Configure this checkout to use the repository's committed git hooks."""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    hooks = root / ".githooks"
    subprocess.run(
        ["git", "config", "core.hooksPath", str(hooks.relative_to(root))],
        cwd=root,
        check=True,
    )
    print(f"git hooks enabled from {hooks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
