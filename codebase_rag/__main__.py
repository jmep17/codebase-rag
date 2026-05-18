"""CLI entry point for codebase-rag."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
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

    p_doctor = subparsers.add_parser(
        "doctor",
        help="Check local setup: project root, index DB, Ollama, optional extras, and tool runners.",
    )
    p_doctor.add_argument(
        "--db", type=Path, default=DEFAULT_DB, help=f"Database path (default: {DEFAULT_DB})."
    )
    p_doctor.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root to check for an existing index (default: current working directory).",
    )
    p_doctor.add_argument(
        "--model",
        type=str,
        default=None,
        help="Chat model to check in Ollama (default: CODEBASE_RAG_CHAT_MODEL or built-in default).",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prompt before every model-driven write_file or edit_file (default: on). "
            "Shows a preview (truncated diff for edits, first 15 lines for writes); "
            "'d' or 'f' reveal the full version. Use --no-confirm-writes to restore "
            "the old auto-apply behavior."
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
    p_chat.add_argument(
        "--tui",
        action="store_true",
        help=(
            "Launch the Textual TUI instead of the line-oriented chat. "
            "Requires `pip install -e .[tui]`. All other chat flags apply; "
            "retrieval, streaming, tool calls, and write/shell confirmation "
            "modals all render in the TUI."
        ),
    )

    p_serve = subparsers.add_parser(
        "serve",
        help="Start the local HTTP/WebSocket backend for the browser/desktop app.",
    )
    p_serve.add_argument(
        "--host",
        type=str,
        default=os.environ.get("CODEBASE_RAG_SERVE_HOST", "127.0.0.1"),
        help=(
            "Bind address (default: 127.0.0.1; env: CODEBASE_RAG_SERVE_HOST). "
            "Anything other than loopback prints a loud warning at startup."
        ),
    )
    p_serve.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CODEBASE_RAG_SERVE_PORT", "8723")),
        help="Bind port (default: 8723; env: CODEBASE_RAG_SERVE_PORT).",
    )
    p_serve.add_argument(
        "--db", type=Path, default=DEFAULT_DB,
        help=f"Database path (default: {DEFAULT_DB}).",
    )
    p_serve.add_argument(
        "--root", type=Path, default=Path.cwd(),
        help=(
            "Default project root for chat sessions when the WS client "
            "doesn't pick one (default: current working directory)."
        ),
    )
    p_serve.add_argument(
        "--static", type=Path, default=None, metavar="DIR",
        help=(
            "Serve a built SPA bundle at /. Typically web/dist after `pnpm build`. "
            "Static assets are anonymously readable; the SPA must include "
            "?token=… when calling /api/*."
        ),
    )
    p_serve.add_argument(
        "--reuse-token", action="store_true",
        help=(
            "Keep the existing serve.token instead of rotating on every start. "
            "Useful for desktop wrappers that respawn the sidecar."
        ),
    )
    p_serve.add_argument(
        "--token-file", type=Path, default=None, metavar="PATH",
        help="Override token file location (default: <meta-dir>/serve.token).",
    )
    p_serve.add_argument(
        "--quiet", action="store_true",
        help="Suppress uvicorn access logs.",
    )
    p_serve.add_argument(
        "--model", type=str, default=None,
        help="Chat model for WS sessions (default: env CODEBASE_RAG_CHAT_MODEL or mistral-nemo).",
    )
    p_serve.add_argument(
        "--provider", type=str, default="ollama",
        choices=["ollama", "anthropic"],
        help="Chat provider for WS sessions (default: ollama).",
    )
    p_serve.add_argument(
        "--architect-model", type=str, default=None, metavar="MODEL",
        help="Optional architect model (split architect/coder flow).",
    )
    p_serve.add_argument("--read-only", action="store_true")
    p_serve.add_argument("--allow-shell", action="store_true")
    p_serve.add_argument(
        "--shell-runner", type=str, default="host", metavar="host|docker:IMAGE",
    )
    p_serve.add_argument(
        "--shell-network", type=str, default="none",
        choices=["none", "bridge", "host"],
    )
    p_serve.add_argument("--shell-timeout", type=float, default=30.0)
    p_serve.add_argument(
        "--confirm-writes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prompt before every model-driven write_file or edit_file in WS sessions "
            "(default: on). Use --no-confirm-writes to auto-apply writes/edits."
        ),
    )
    p_serve.add_argument("--allow-web", action="store_true")
    p_serve.add_argument(
        "--web-allow", action="append", default=[], metavar="HOST_GLOB",
    )
    p_serve.add_argument(
        "--web-block", action="append", default=[], metavar="HOST_GLOB",
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
    elif args.command == "doctor":
        _handle_doctor(args)
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
        if not args.root.exists():
            print(f"Root does not exist: {args.root}", file=sys.stderr)
            sys.exit(1)
        if not args.db.exists():
            print(
                f"No index found at {args.db}. Run `codebase-rag index <path>` first.",
                file=sys.stderr,
            )
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
        if args.tui:
            try:
                from . import tui as tui_mod
            except ImportError:
                print(
                    "error: --tui requires `pip install -e .[tui]` (Textual).",
                    file=sys.stderr,
                )
                sys.exit(1)
            tui_mod.run_tui(
                db_path=args.db,
                root=args.root.resolve(),
                model=args.model,
                provider_name=args.provider,
                api_key=api_key,
                show_context=args.show_context,
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
            )
            return
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
    elif args.command == "serve":
        try:
            from . import serve as serve_mod
        except ImportError as e:
            print(
                "error: `serve` requires `pip install -e .[serve]` "
                "(starlette, uvicorn, websockets).\n"
                f"  underlying: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.root.exists():
            print(f"Root does not exist: {args.root}", file=sys.stderr)
            sys.exit(1)
        if not args.db.exists():
            print(
                f"No index found at {args.db}. "
                f"Run `codebase-rag index <path>` first.",
                file=sys.stderr,
            )
            sys.exit(1)
        api_key = None
        if args.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print(
                    "error: --provider anthropic requires ANTHROPIC_API_KEY env var.",
                    file=sys.stderr,
                )
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
        # Try to lazy-import starlette/uvicorn now so a missing extra fails
        # before we issue the token and print the banner.
        try:
            import starlette  # noqa: F401
            import uvicorn  # noqa: F401
        except ImportError as e:
            print(
                "error: `serve` requires `pip install -e .[serve]` "
                "(starlette, uvicorn, websockets).\n"
                f"  underlying: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
        serve_mod.run(
            host=args.host,
            port=args.port,
            db_path=args.db,
            root=args.root.resolve(),
            static_dir=args.static.resolve() if args.static else None,
            reuse_token=args.reuse_token,
            token_file=args.token_file,
            quiet=args.quiet,
            chat_defaults=dict(
                model=args.model,
                provider_name=args.provider,
                api_key=api_key,
                read_only=args.read_only,
                allow_shell=args.allow_shell,
                shell_runner=args.shell_runner,
                shell_network=args.shell_network,
                shell_timeout=args.shell_timeout,
                confirm_writes=args.confirm_writes,
                allow_web=args.allow_web,
                web_allow=tuple(args.web_allow),
                web_block=tuple(args.web_block),
                searxng_url=searxng_url,
                architect_model=args.architect_model,
            ),
        )


def _has_modules(*names: str) -> tuple[bool, list[str]]:
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    return not missing, missing


def _doctor_row(status: str, label: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")


def _ollama_model_names() -> list[str]:
    import ollama

    data = ollama.list()
    raw_models = data.get("models", []) if isinstance(data, dict) else getattr(data, "models", [])
    names: list[str] = []
    for model in raw_models:
        if isinstance(model, dict):
            name = model.get("model") or model.get("name")
        else:
            name = getattr(model, "model", None) or getattr(model, "name", None)
        if name:
            names.append(str(name))
    return names


def _handle_doctor(args: argparse.Namespace) -> None:
    failures = 0
    warnings = 0
    root = args.root.resolve()
    chat_model = chat_mod._resolve_model(args.model)

    print(f"codebase-rag doctor")
    print(f"Project: {root}")
    print(f"Database: {args.db}")
    print()

    if root.exists() and root.is_dir():
        _doctor_row("ok", "project root", str(root))
    else:
        failures += 1
        _doctor_row("fail", "project root", "path does not exist or is not a directory")

    if args.db.exists():
        try:
            import chromadb

            client = chromadb.PersistentClient(
                path=str(args.db), settings=index_mod.CHROMA_SETTINGS,
            )
            collection_name = index_mod.collection_name_for(root)
            collections = client.list_collections()
            if any(col.name == collection_name for col in collections):
                collection = client.get_collection(collection_name)
                _doctor_row(
                    "ok",
                    "project index",
                    f"{collection.count()} chunks in {collection_name}",
                )
            else:
                warnings += 1
                _doctor_row(
                    "warn",
                    "project index",
                    "no collection for this root; run `codebase-rag index .`",
                )
        except Exception as e:
            failures += 1
            _doctor_row("fail", "index database", f"{type(e).__name__}: {e}")
    else:
        warnings += 1
        _doctor_row("warn", "index database", "not found; run `codebase-rag index <path>`")

    try:
        models = _ollama_model_names()
        _doctor_row("ok", "ollama daemon", f"{len(models)} model(s) visible")
        for model in (chat_model, index_mod.EMBEDDING_MODEL):
            if model in models:
                _doctor_row("ok", f"ollama model {model}")
            else:
                warnings += 1
                _doctor_row("warn", f"ollama model {model}", f"run `ollama pull {model}`")
    except Exception as e:
        failures += 1
        _doctor_row("fail", "ollama daemon", f"{type(e).__name__}: {e}")

    extras = [
        ("web extra", ("httpx", "trafilatura")),
        ("cloud extra", ("anthropic",)),
        ("tui extra", ("textual",)),
        ("serve extra", ("starlette", "uvicorn", "websockets")),
    ]
    for label, modules in extras:
        ok, missing = _has_modules(*modules)
        if ok:
            _doctor_row("ok", label, "installed")
        else:
            warnings += 1
            _doctor_row("warn", label, f"missing {', '.join(missing)}")

    searxng_url = os.environ.get("SEARXNG_URL", "")
    if searxng_url:
        _doctor_row("ok", "SEARXNG_URL", searxng_url)
    else:
        warnings += 1
        _doctor_row("warn", "SEARXNG_URL", "unset; --allow-web will fail until configured")

    docker = shutil.which("docker")
    if docker:
        _doctor_row("ok", "docker", docker)
    else:
        warnings += 1
        _doctor_row("warn", "docker", "not on PATH; docker shell runner unavailable")

    print()
    if failures:
        print(f"doctor finished with {failures} failure(s) and {warnings} warning(s).")
        sys.exit(1)
    print(f"doctor finished with {warnings} warning(s).")


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
