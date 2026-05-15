"""CLI entry point for codebase-rag."""

from __future__ import annotations

import argparse
import os
import subprocess
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

    p_index = subparsers.add_parser("index", help="Index a codebase directory (upserts).")
    p_index.add_argument("path", type=Path, help="Root directory to index.")
    p_index.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help=f"Database path (default: {DEFAULT_DB})."
    )
    p_index.add_argument(
        "--exclude",
        "-x",
        action="append",
        default=[],
        metavar="GLOB",
        help="Glob pattern to exclude (matched against relative path and bare filename). Repeatable.",
    )

    p_reindex = subparsers.add_parser(
        "reindex",
        help="Wipe the existing index and rebuild from scratch (drops stale chunks).",
    )
    p_reindex.add_argument("path", type=Path, help="Root directory to index.")
    p_reindex.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help=f"Database path (default: {DEFAULT_DB})."
    )
    p_reindex.add_argument(
        "--exclude",
        "-x",
        action="append",
        default=[],
        metavar="GLOB",
        help="Glob pattern to exclude (matched against relative path and bare filename). Repeatable.",
    )

    p_stats = subparsers.add_parser(
        "stats",
        help="List indexed projects, or detail one (--root) of them.",
    )
    p_stats.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help=f"Database path (default: {DEFAULT_DB})."
    )
    p_stats.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Show stats for a single project root. If omitted, lists all indexed projects.",
    )

    p_search = subparsers.add_parser(
        "search", help="Run a one-shot semantic search within a project (default: current directory)."
    )
    p_search.add_argument("query", type=str, help="Search query.")
    p_search.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help=f"Database path (default: {DEFAULT_DB})."
    )
    p_search.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root whose index to query (default: current working directory).",
    )
    p_search.add_argument(
        "--top-k", "-k", type=int, default=5, help="Number of chunks to return (default: 5)."
    )
    p_search.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        metavar="GLOB",
        help="Restrict results to chunks whose path matches this glob (e.g. 'src/*.py').",
    )
    p_search.add_argument(
        "--headers-only",
        action="store_true",
        help="Print only file:line headers, no chunk content.",
    )

    p_show = subparsers.add_parser(
        "show",
        help="Print every indexed chunk for files matching a path glob within a project.",
    )
    p_show.add_argument("file", type=str, metavar="GLOB", help="Path glob, e.g. 'src/auth.py' or 'src/*.py'.")
    p_show.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help=f"Database path (default: {DEFAULT_DB})."
    )
    p_show.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root whose index to query (default: current working directory).",
    )

    p_notes = subparsers.add_parser(
        "notes",
        help="View or edit per-project notes (stored outside the repo; auto-injected into chat).",
    )
    p_notes.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root these notes belong to (default: current working directory).",
    )
    notes_group = p_notes.add_mutually_exclusive_group()
    notes_group.add_argument("--edit", action="store_true", help="Open notes in $EDITOR (default vi).")
    notes_group.add_argument(
        "--set", dest="set_text", type=str, metavar="TEXT",
        help="Replace notes with this text. Use '-' to read from stdin.",
    )
    notes_group.add_argument(
        "--append", dest="append_text", type=str, metavar="TEXT",
        help="Append a line of text to notes. Use '-' to read from stdin.",
    )
    notes_group.add_argument("--clear", action="store_true", help="Delete the notes file.")

    p_addref = subparsers.add_parser(
        "add-reference",
        help="Index an external directory as reference material for a project.",
    )
    p_addref.add_argument("source", type=Path, help="Directory of reference docs to index.")
    p_addref.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label for this reference set (default: basename of the source directory).",
    )
    p_addref.add_argument(
        "--for-project",
        type=Path,
        default=Path.cwd(),
        help="Project root these references belong to (default: current working directory).",
    )
    p_addref.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help=f"Database path (default: {DEFAULT_DB})."
    )
    p_addref.add_argument(
        "--exclude", "-x", action="append", default=[], metavar="GLOB",
        help="Glob pattern to exclude inside the reference source. Repeatable.",
    )

    p_rmref = subparsers.add_parser(
        "remove-reference",
        help="Remove a labeled reference set from a project.",
    )
    p_rmref.add_argument("label", type=str, help="Label of the reference set to remove.")
    p_rmref.add_argument(
        "--for-project",
        type=Path,
        default=Path.cwd(),
        help="Project root the references belong to (default: current working directory).",
    )
    p_rmref.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help=f"Database path (default: {DEFAULT_DB})."
    )

    p_chat = subparsers.add_parser("chat", help="Start an interactive chat session.")
    p_chat.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help=f"Database path (default: {DEFAULT_DB})."
    )
    p_chat.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory the model may read/write via tools (default: current working directory).",
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
        index_mod.build_index(args.path, args.db, extra_excludes=args.exclude)
    elif args.command == "reindex":
        if not args.path.exists():
            print(f"Path does not exist: {args.path}", file=sys.stderr)
            sys.exit(1)
        index_mod.reset_index(args.db, args.path)
        index_mod.build_index(args.path, args.db, extra_excludes=args.exclude)
    elif args.command == "stats":
        index_mod.stats(args.db, root=args.root)
    elif args.command == "search":
        if not args.root.exists():
            print(f"Root does not exist: {args.root}", file=sys.stderr)
            sys.exit(1)
        index_mod.search(
            args.db,
            args.query,
            root=args.root.resolve(),
            top_k=args.top_k,
            file_pattern=args.file,
            headers_only=args.headers_only,
        )
    elif args.command == "show":
        if not args.root.exists():
            print(f"Root does not exist: {args.root}", file=sys.stderr)
            sys.exit(1)
        index_mod.show_file(args.db, args.file, root=args.root.resolve())
    elif args.command == "notes":
        _handle_notes(args)
    elif args.command == "add-reference":
        if not args.source.exists():
            print(f"Reference source does not exist: {args.source}", file=sys.stderr)
            sys.exit(1)
        if not args.for_project.exists():
            print(f"Project root does not exist: {args.for_project}", file=sys.stderr)
            sys.exit(1)
        label = args.label or args.source.resolve().name
        index_mod.add_reference(
            args.source.resolve(),
            args.for_project.resolve(),
            args.db,
            label=label,
            extra_excludes=args.exclude,
        )
    elif args.command == "remove-reference":
        if not args.for_project.exists():
            print(f"Project root does not exist: {args.for_project}", file=sys.stderr)
            sys.exit(1)
        index_mod.remove_reference(args.db, args.for_project.resolve(), args.label)
    elif args.command == "chat":
        if not args.db.exists():
            print(
                f"No index found at {args.db}. Run `codebase-rag index <path>` first.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.root.exists():
            print(f"Root does not exist: {args.root}", file=sys.stderr)
            sys.exit(1)
        chat_mod.agent_loop(
            args.db, root=args.root.resolve(), show_context=args.show_context
        )


def _handle_notes(args: argparse.Namespace) -> None:
    if not args.root.exists():
        print(f"Project root does not exist: {args.root}", file=sys.stderr)
        sys.exit(1)
    root = args.root.resolve()

    if args.clear:
        if index_mod.clear_notes(root):
            print(f"Cleared notes for {root}.")
        else:
            print(f"No notes to clear for {root}.")
        return

    if args.set_text is not None:
        text = sys.stdin.read() if args.set_text == "-" else args.set_text
        path = index_mod.write_notes(root, text)
        print(f"Notes saved to {path}.")
        return

    if args.append_text is not None:
        addition = sys.stdin.read() if args.append_text == "-" else args.append_text
        existing = index_mod.read_notes(root)
        if existing and not existing.endswith("\n"):
            existing += "\n"
        path = index_mod.write_notes(root, existing + addition.rstrip("\n") + "\n")
        print(f"Appended to {path}.")
        return

    if args.edit:
        notes_path = index_mod.project_meta_dir(root) / "notes.md"
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        notes_path.touch(exist_ok=True)
        editor = os.environ.get("EDITOR", "vi")
        try:
            subprocess.run([editor, str(notes_path)], check=False)
            print(f"Notes file: {notes_path}")
        except FileNotFoundError:
            print(f"Could not launch '{editor}'. Notes file is at: {notes_path}")
        return

    # Default: print existing notes
    existing = index_mod.read_notes(root)
    if not existing:
        print(f"(no notes for {root})")
        print(f"File would be: {index_mod.project_meta_dir(root) / 'notes.md'}")
        return
    print(existing.rstrip("\n"))


if __name__ == "__main__":
    main()
