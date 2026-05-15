"""CLI entry point for codebase-rag."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import chat as chat_mod
from . import index as index_mod

DEFAULT_DB = Path.home() / ".codebase-rag" / "db"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="codebase-rag",
        description="Local codebase RAG with Ollama + mistral-nemo.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_index = subparsers.add_parser("index", help="Index a codebase directory.")
    p_index.add_argument("path", type=Path, help="Root directory to index.")
    p_index.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help=f"Database path (default: {DEFAULT_DB})."
    )

    p_chat = subparsers.add_parser("chat", help="Start an interactive chat session.")
    p_chat.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help=f"Database path (default: {DEFAULT_DB})."
    )
    p_chat.add_argument(
        "--show-context",
        action="store_true",
        help="Print the file paths and line ranges retrieved for each question.",
    )

    args = parser.parse_args()

    if args.command == "index":
        if not args.path.exists():
            print(f"Path does not exist: {args.path}", file=sys.stderr)
            sys.exit(1)
        index_mod.build_index(args.path, args.db)
    elif args.command == "chat":
        if not args.db.exists():
            print(
                f"No index found at {args.db}. Run `codebase-rag index <path>` first.",
                file=sys.stderr,
            )
            sys.exit(1)
        chat_mod.chat_loop(args.db, show_context=args.show_context)


if __name__ == "__main__":
    main()
