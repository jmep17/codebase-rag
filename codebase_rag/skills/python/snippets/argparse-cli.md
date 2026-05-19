---
id: "argparse-cli"
title: "argparse CLI entrypoint"
triggers: ["argparse", "cli", "command line", "subcommand"]
file_globs: ["*.py"]
packages: []
max_tokens: 320
---

Return an exit code from `main`, keep parsing near the entrypoint, and make errors actionable.

```python
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="example-app")
    parser.add_argument("path", help="Path to process")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose:
        print(f"processing {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```
