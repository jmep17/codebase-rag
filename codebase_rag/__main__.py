"""CLI entry point for codebase-rag."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import audit as audit_mod
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

    p_audit = subparsers.add_parser(
        "audit",
        help="Show the per-project audit log of tool calls and slash commands.",
    )
    p_audit.add_argument(
        "--root", type=Path, default=Path.cwd(),
        help="Project root whose audit log to show (default: current working directory).",
    )
    p_audit.add_argument("--tool", type=str, default=None, help="Filter by tool name (e.g. grep, edit_file).")
    p_audit.add_argument("--event", type=str, default=None, help="Filter by event type (tool_call, tool_result, slash_command, session_start, session_end).")
    p_audit.add_argument("--since", type=str, default=None, help="e.g. 'today', 'yesterday', '15m', '2h', '7d', or ISO date.")
    p_audit.add_argument("--limit", type=int, default=50, help="Max number of events to show (default: 50). 0 = all.")
    p_audit.add_argument("--pretty", action="store_true", help="Pretty-print each entry over multiple lines.")

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
        "--model",
        type=str,
        default=None,
        help=(
            "Ollama chat model to use. Overrides the CODEBASE_RAG_CHAT_MODEL env var "
            "and the built-in default (mistral-nemo). Examples: qwen3:8b, "
            "qwen2.5-coder:32b, llama3.3:70b."
        ),
    )
    p_chat.add_argument(
        "--show-context",
        action="store_true",
        help="Print the file paths and line ranges retrieved for each question.",
    )
    p_chat.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "Print detailed timing and token-rate stats for every inference, plus full "
            "tool results (instead of the 200-char preview)."
        ),
    )
    p_chat.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue the project's last conversation. Conversations auto-save after "
            "every turn; --resume picks up where you left off. Use :forget mid-chat "
            "to delete the saved conversation."
        ),
    )
    p_chat.add_argument(
        "--read-only",
        action="store_true",
        help=(
            "Disable file-modifying tools for this session. write_file, edit_file, "
            "and run_shell are not exposed to the model — only read_file and grep. "
            "Useful for exploratory Q&A chats and as a guardrail against prompt injection."
        ),
    )
    p_chat.add_argument(
        "--allow-shell",
        action="store_true",
        help=(
            "Enable the run_shell tool (model-driven) and the :run slash command "
            "(user-driven). Model calls always prompt for confirmation before "
            "executing. Commands run via shlex.split (no shell expansion). "
            "Ignored if --read-only is also set."
        ),
    )
    p_chat.add_argument(
        "--shell-timeout",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Per-command timeout for run_shell / :run (default: 30 seconds).",
    )
    p_chat.add_argument(
        "--shell-runner",
        type=str,
        default="host",
        metavar="host|docker:IMAGE",
        help=(
            "Where to execute run_shell / :run commands. 'host' (default) runs "
            "directly on the host with scrubbed env. 'docker:<image>' runs each "
            "command inside a transient container with the project root mounted "
            "at /work, no host filesystem visible, and a minimal env. Example: "
            "--shell-runner docker:python:3.13-slim."
        ),
    )
    p_chat.add_argument(
        "--shell-network",
        type=str,
        default="none",
        choices=["none", "bridge", "host"],
        help=(
            "Docker network policy when --shell-runner=docker:... is set. "
            "'none' (default) = no network; 'bridge' = standard docker bridge; "
            "'host' = host networking (least isolation). Ignored when runner is host."
        ),
    )
    p_chat.add_argument(
        "--confirm-writes",
        action="store_true",
        help=(
            "Prompt before every model-driven write_file or edit_file. Shows a "
            "preview (truncated diff for edits, first 15 lines for writes); 'd' or "
            "'f' reveal the full version. Useful for first-time chats against "
            "unfamiliar code, or as a guardrail against prompt injection."
        ),
    )
    p_chat.add_argument(
        "--allow-web",
        action="store_true",
        help=(
            "Enable web_search (via SearXNG) and web_fetch tools, plus :search and "
            ":fetch slash commands. Requires SEARXNG_URL env var (e.g. "
            "http://localhost:8080) — run SearXNG locally for full privacy."
        ),
    )
    p_chat.add_argument(
        "--web-allow",
        action="append",
        default=[],
        metavar="HOST_GLOB",
        help=(
            "Host glob to allow for web_fetch (repeatable). e.g. 'docs.python.org', "
            "'*.readthedocs.io'. If unset, any host is allowed (subject to --web-block)."
        ),
    )
    p_chat.add_argument(
        "--web-block",
        action="append",
        default=[],
        metavar="HOST_GLOB",
        help="Host glob to block for web_fetch (repeatable). Blocklist takes precedence over allowlist.",
    )
    p_chat.add_argument(
        "--architect-model",
        type=str,
        default=None,
        metavar="MODEL",
        help=(
            "Enable architect-coder split. The architect (this model) sees the same "
            "retrieved context and produces a numbered plan; the coder (--model) then "
            "executes via tool calls. Useful pairing: --architect-model qwen3:30b-a3b "
            "--model qwen2.5-coder:7b. Default off (single-model flow)."
        ),
    )
    p_chat.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["ollama", "anthropic"],
        help=(
            "Chat provider. 'ollama' (default) runs against the local Ollama daemon. "
            "'anthropic' calls api.anthropic.com — requires `pip install -e .[cloud]` "
            "and ANTHROPIC_API_KEY env var. Embeddings stay on Ollama regardless."
        ),
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
    elif args.command == "audit":
        _handle_audit(args)
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
        searxng_url = os.environ.get("SEARXNG_URL", "")
        if args.allow_web and not searxng_url:
            print(
                "error: --allow-web requires SEARXNG_URL env var.\n"
                "  Quick start: docker run -d -p 8080:8080 --name searxng searxng/searxng\n"
                "  Then:        export SEARXNG_URL=http://localhost:8080",
                file=sys.stderr,
            )
            sys.exit(1)
        api_key = None
        if args.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print(
                    "error: --provider anthropic requires ANTHROPIC_API_KEY env var.\n"
                    "  Get one at https://console.anthropic.com",
                    file=sys.stderr,
                )
                sys.exit(1)
        chat_mod.agent_loop(
            args.db,
            root=args.root.resolve(),
            show_context=args.show_context,
            model=args.model,
            verbose=args.verbose,
            resume=args.resume,
            read_only=args.read_only,
            allow_shell=args.allow_shell,
            shell_timeout=args.shell_timeout,
            shell_runner=args.shell_runner,
            shell_network=args.shell_network,
            confirm_writes=args.confirm_writes,
            allow_web=args.allow_web,
            web_allow=tuple(args.web_allow),
            web_block=tuple(args.web_block),
            searxng_url=searxng_url,
            architect_model=args.architect_model,
            provider_name=args.provider,
            api_key=api_key,
        )


def _handle_audit(args: argparse.Namespace) -> None:
    if not args.root.exists():
        print(f"Project root does not exist: {args.root}", file=sys.stderr)
        sys.exit(1)
    root = args.root.resolve()
    since = None
    if args.since:
        try:
            since = audit_mod.parse_since(args.since)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
    rows = audit_mod.tail_audit(
        index_mod.project_meta_dir(root),
        since=since,
        tool=args.tool,
        event=args.event,
        limit=args.limit,
    )
    if not rows:
        print(f"(no matching events in audit log for {root})")
        return
    for row in rows:
        if args.pretty:
            print(json.dumps(row, indent=2))
            print()
        else:
            print(json.dumps(row, separators=(",", ":")))


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
