"""Local HTTP/WebSocket transport for codebase-rag.

Pure transport: bridges existing modules (`chat.agent_turn`, `index._search_hits`,
`audit.tail_audit`, `tools.resolve_safe`, ...) to JSON over HTTP and WebSocket.
Never reimplements indexing, retrieval, or agent logic. Lazy-imports
starlette/uvicorn so the default install pays nothing for the [serve] extra.

Bind defaults to 127.0.0.1:8723. Every request requires `Authorization: Bearer
<token>` (the WS also accepts `?token=` because browsers can't set headers on
`new WebSocket()`). The token is a 256-bit random string written to the per-project
meta dir (`~/.codebase-rag/meta/<sha>/serve.token`, mode 0600) on every start.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb

from . import audit as audit_mod
from . import chat as chat_mod
from . import diagnostics as diagnostics_mod
from . import index as index_mod
from .tools import MAX_READ_BYTES, resolve_safe

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8723
TOKEN_FILE_NAME = "serve.token"
CORS_ORIGIN_RE = r"^http://(127\.0\.0\.1|localhost)(:\d+)?$"
_VERSION = "0.1.0"
_STARTED_AT_ISO = datetime.now(timezone.utc).isoformat(timespec="seconds")

# WS close codes: 1xxx are reserved, 4xxx are app-defined.
WS_CLOSE_AUTH = 4401
WS_CLOSE_NOT_FOUND = 4404
WS_CLOSE_INIT_FAILED = 4500


# ---------------------------------------------------------------------------
# Token lifecycle
# ---------------------------------------------------------------------------


def _token_path(meta_dir: Path) -> Path:
    return meta_dir / TOKEN_FILE_NAME


def issue_token(meta_dir: Path) -> str:
    """Generate a fresh 256-bit token, persist to meta_dir/serve.token (mode 0600)."""
    meta_dir.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    path = _token_path(meta_dir)
    tmp = path.with_suffix(".token.tmp")
    tmp.write_text(token, encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)
    return token


def read_token(meta_dir: Path) -> str | None:
    path = _token_path(meta_dir)
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Project enumeration (read-only over Chroma)
# ---------------------------------------------------------------------------


def _resolve_project(db_path: Path, slug: str) -> tuple[Path, str] | None:
    """slug (12-char digest) → (root, collection_name) or None."""
    if not db_path.exists() or not slug:
        return None
    try:
        client = chromadb.PersistentClient(
            path=str(db_path),
            settings=index_mod.CHROMA_SETTINGS,
        )
    except Exception:
        return None
    for col in client.list_collections():
        if not col.name.startswith("cbr_"):
            continue
        root_str = (col.metadata or {}).get("root")
        if not root_str:
            continue
        root = Path(root_str)
        if index_mod._root_digest(root) == slug:
            return root, col.name
    return None


def _project_summary(col, root: Path, *, with_notes: bool = False) -> dict:
    s = index_mod._collection_summary(col)
    out: dict = {
        "slug": index_mod._root_digest(root),
        "root": str(root),
        "collection": col.name,
        "total_chunks": s["total"],
        "project_chunks": s["project_chunks"],
        "project_files": s["project_files"],
        "references": {
            label: {"chunks": sum(files.values()), "files": len(files)}
            for label, files in s["references"].items()
        },
        "top_project_files": s["top_project_files"],
    }
    notes = index_mod.read_notes(root)
    out["has_notes"] = bool(notes.strip())
    if with_notes and notes:
        excerpt = notes.strip().splitlines()
        out["notes_excerpt"] = "\n".join(excerpt[:5])[:500]
    meta_dir = index_mod.project_meta_dir(root)
    rows = audit_mod.tail_audit(meta_dir, limit=1)
    out["last_active"] = rows[-1]["ts"] if rows else None
    out["last_session"] = rows[-1].get("session") if rows else None
    return out


def _list_projects(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    try:
        client = chromadb.PersistentClient(
            path=str(db_path),
            settings=index_mod.CHROMA_SETTINGS,
        )
    except Exception:
        return []
    out: list[dict] = []
    for col in client.list_collections():
        if not col.name.startswith("cbr_"):
            continue
        root_str = (col.metadata or {}).get("root")
        if not root_str:
            continue
        try:
            out.append(_project_summary(col, Path(root_str)))
        except Exception:
            continue
    out.sort(key=lambda p: (p.get("last_active") or "", p["root"]), reverse=True)
    return out


def _list_sessions(meta_dir: Path) -> list[dict]:
    """Compose per-session metadata from the audit log."""
    rows = audit_mod.tail_audit(meta_dir, limit=0)  # 0 = all
    by_session: dict[str, dict] = {}
    for row in rows:
        sid = row.get("session")
        if not sid:
            continue
        ev = row.get("event")
        bucket = by_session.setdefault(
            sid,
            {
                "session_id": sid,
                "started_at": None,
                "ended_at": None,
                "model": None,
                "provider": None,
                "events": 0,
                "tool_calls": 0,
            },
        )
        bucket["events"] += 1
        if ev == "tool_call":
            bucket["tool_calls"] += 1
        elif ev == "session_start":
            bucket["started_at"] = row.get("ts")
            bucket["model"] = row.get("model")
            bucket["provider"] = row.get("provider")
        elif ev == "session_end":
            bucket["ended_at"] = row.get("ts")
            bucket["reason"] = row.get("reason")
    sessions = list(by_session.values())
    sessions.sort(key=lambda s: s.get("started_at") or "", reverse=True)
    return sessions


# ---------------------------------------------------------------------------
# Event serialization (agent_turn tuple → JSON envelope)
# ---------------------------------------------------------------------------


def _make_confirm_preview(tname: str, args: dict, session: chat_mod.ChatSession) -> dict:
    if tname == "edit_file":
        old = (args.get("old_string") or "").splitlines()
        new = (args.get("new_string") or "").splitlines()
        return {
            "kind": "edit_file",
            "path": args.get("path", ""),
            "old_string": args.get("old_string", ""),
            "new_string": args.get("new_string", ""),
            "old_preview": old[:8],
            "old_more": max(0, len(old) - 8),
            "new_preview": new[:8],
            "new_more": max(0, len(new) - 8),
        }
    if tname == "write_file":
        content = args.get("content", "") or ""
        lines = content.splitlines()
        return {
            "kind": "write_file",
            "path": args.get("path", ""),
            "content_preview": lines[:15],
            "total_lines": len(lines),
            "bytes": len(content.encode("utf-8")),
        }
    if tname == "create_project":
        files = args.get("files")
        if isinstance(files, list):
            paths = [item.get("path", "?") for item in files[:12] if isinstance(item, dict)]
            total_files = len(files)
        else:
            paths = ["README.md", ".gitignore"]
            total_files = 2
        return {
            "kind": "create_project",
            "project_path": args.get("project_path", ""),
            "description": args.get("description", ""),
            "file_paths": paths,
            "total_files": total_files,
            "overwrite": bool(args.get("overwrite")),
        }
    if tname == "run_shell":
        return {
            "kind": "run_shell",
            "command": args.get("command", ""),
            "runner": session.shell_runner,
            "network": session.shell_network,
            "timeout": session.shell_timeout,
        }
    return {"kind": tname, "args": args}


def _safe_tool_calls(tool_calls: list) -> list:
    out: list[dict] = []
    for tc in tool_calls or []:
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        out.append({"name": fn.get("name", ""), "arguments": args or {}})
    return out


def _serialize_event(
    event: tuple, session: chat_mod.ChatSession, turn_id: str, call_counter: list[int]
) -> dict:
    """Translate one agent_turn event into a JSON envelope.

    call_counter is a single-element mutable list; tool_call_request
    increments it, and the matching confirm / tool_result / tool_declined
    use the same value.
    """
    kind = event[0]
    if kind == "retrieved":
        _, chunks, pinned, elapsed = event
        return {
            "type": "retrieved",
            "turn": turn_id,
            "chunks": chunks,
            "pinned": pinned,
            "elapsed": elapsed,
        }
    if kind == "architect_start":
        return {"type": "architect_start", "turn": turn_id, "model": event[1]}
    if kind == "architect_error":
        return {"type": "architect_error", "turn": turn_id, "message": event[1]}
    if kind == "token":
        return {"type": "token", "turn": turn_id, "piece": event[1]}
    if kind == "inference_done":
        _, content, tool_calls, stats = event
        return {
            "type": "inference_done",
            "turn": turn_id,
            "content": content,
            "tool_calls": _safe_tool_calls(tool_calls),
            "stats": stats,
            "provider": session.provider_name,
        }
    if kind == "error":
        _, sub, msg = event
        return {"type": "error", "turn": turn_id, "subtype": sub, "message": msg}
    if kind == "empty_response":
        return {"type": "empty_response", "turn": turn_id}
    if kind == "tool_call_request":
        _, tname, args = event
        call_counter[0] += 1
        return {
            "type": "tool_call_request",
            "turn": turn_id,
            "call": f"{turn_id}:{call_counter[0]}",
            "tool": tname,
            "args": args,
        }
    if kind == "confirm":
        _, tname, args = event
        return {
            "type": "confirm",
            "turn": turn_id,
            "call": f"{turn_id}:{call_counter[0]}",
            "tool": tname,
            "args": args,
            "preview": _make_confirm_preview(tname, args, session),
        }
    if kind == "tool_declined":
        _, tname, args, declined, raw = event
        return {
            "type": "tool_declined",
            "turn": turn_id,
            "call": f"{turn_id}:{call_counter[0]}",
            "tool": tname,
            "args": args,
            "declined": declined,
            "raw": raw,
        }
    if kind == "tool_result":
        _, tname, args, raw, summary, elapsed = event
        return {
            "type": "tool_result",
            "turn": turn_id,
            "call": f"{turn_id}:{call_counter[0]}",
            "tool": tname,
            "args": args,
            "raw": raw,
            "summary": summary,
            "elapsed": elapsed,
        }
    if kind == "max_turns":
        return {"type": "max_turns", "turn": turn_id, "max_turns": event[1]}
    if kind == "turn_done":
        return {"type": "turn_done", "turn": turn_id, "stats": event[1]}
    return {"type": "unknown", "turn": turn_id, "event": kind}


# ---------------------------------------------------------------------------
# ASGI app factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    db_path: Path,
    default_root: Path | None,
    token: str,
    static_dir: Path | None = None,
    quiet: bool = False,
    chat_defaults: dict[str, Any] | None = None,
):
    """Build the Starlette ASGI app. Lazy-imports starlette so `import serve`
    itself never pulls the [serve] extra.
    """
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Mount, Route, WebSocketRoute
    from starlette.staticfiles import StaticFiles
    from starlette.websockets import WebSocketDisconnect

    chat_defaults = dict(chat_defaults or {})
    expected_token = token.encode("utf-8")

    # ----- middleware -----

    class AuthMiddleware(BaseHTTPMiddleware):
        ALLOW_ANON = {"/api/health"}

        async def dispatch(self, request, call_next):
            path = request.url.path
            if request.method == "OPTIONS":
                return await call_next(request)
            if path in self.ALLOW_ANON:
                return await call_next(request)
            if static_dir is not None and not path.startswith("/api/"):
                # Static SPA bundle is anon; the SPA reads ?token= and uses it
                # for /api/* calls.
                return await call_next(request)
            auth = request.headers.get("authorization", "")
            if not auth.lower().startswith("bearer "):
                return JSONResponse({"error": "missing bearer token"}, status_code=401)
            presented = auth[7:].strip().encode("utf-8")
            if not secrets.compare_digest(presented, expected_token):
                return JSONResponse({"error": "invalid token"}, status_code=401)
            return await call_next(request)

    # ----- HTTP handlers -----

    async def health(request):
        return JSONResponse(
            {
                "ok": True,
                "version": _VERSION,
                "providers": ["ollama", "anthropic"],
                "pid": os.getpid(),
                "started_at": _STARTED_AT_ISO,
            }
        )

    async def list_projects(request):
        projects = await asyncio.to_thread(_list_projects, db_path)
        return JSONResponse({"projects": projects})

    async def get_project(request):
        slug = request.path_params["slug"]
        resolved = await asyncio.to_thread(_resolve_project, db_path, slug)
        if resolved is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        root, col_name = resolved

        def _detail():
            client = chromadb.PersistentClient(
                path=str(db_path),
                settings=index_mod.CHROMA_SETTINGS,
            )
            col = client.get_collection(col_name)
            item = _project_summary(col, root, with_notes=True)
            item["pinned"] = []  # populated by future POST /pinned endpoint
            return item

        item = await asyncio.to_thread(_detail)
        return JSONResponse(item)

    async def list_sessions(request):
        slug = request.path_params["slug"]
        resolved = await asyncio.to_thread(_resolve_project, db_path, slug)
        if resolved is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        root, _ = resolved
        sessions = await asyncio.to_thread(
            _list_sessions,
            index_mod.project_meta_dir(root),
        )
        return JSONResponse({"sessions": sessions})

    async def get_audit(request):
        slug = request.path_params["slug"]
        resolved = await asyncio.to_thread(_resolve_project, db_path, slug)
        if resolved is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        root, _ = resolved
        since_str = request.query_params.get("since")
        since = None
        if since_str:
            try:
                since = audit_mod.parse_since(since_str)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
        try:
            limit = int(request.query_params.get("limit", "50"))
        except ValueError:
            limit = 50
        events = await asyncio.to_thread(
            audit_mod.tail_audit,
            index_mod.project_meta_dir(root),
            since=since,
            tool=request.query_params.get("tool"),
            event=request.query_params.get("event"),
            limit=limit,
        )
        return JSONResponse({"events": events})

    async def get_diagnostics(request):
        slug = request.path_params["slug"]
        resolved = await asyncio.to_thread(_resolve_project, db_path, slug)
        if resolved is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        root, _ = resolved
        try:
            limit = int(request.query_params.get("limit", "100"))
        except ValueError:
            limit = 100
        result = await asyncio.to_thread(
            diagnostics_mod.read_diagnostics,
            root,
            path=request.query_params.get("path"),
            severity=request.query_params.get("severity"),
            source=request.query_params.get("source"),
            limit=limit,
        )
        status = 200 if result.get("ok") else 500
        return JSONResponse(result, status_code=status)

    async def put_diagnostics(request):
        slug = request.path_params["slug"]
        resolved = await asyncio.to_thread(_resolve_project, db_path, slug)
        if resolved is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        root, _ = resolved
        try:
            payload = await request.json()
            result = await asyncio.to_thread(diagnostics_mod.write_diagnostics, root, payload)
        except (ValueError, json.JSONDecodeError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse(result)

    async def delete_diagnostics(request):
        slug = request.path_params["slug"]
        resolved = await asyncio.to_thread(_resolve_project, db_path, slug)
        if resolved is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        root, _ = resolved
        cleared = await asyncio.to_thread(diagnostics_mod.clear_diagnostics, root)
        return JSONResponse({"ok": True, "cleared": cleared})

    async def search(request):
        slug = request.query_params.get("project") or ""
        q = (request.query_params.get("q") or "").strip()
        if not q:
            return JSONResponse({"error": "q is required"}, status_code=400)
        try:
            k = max(1, min(100, int(request.query_params.get("k", "5"))))
        except ValueError:
            k = 5
        file_glob = request.query_params.get("file") or None
        resolved = await asyncio.to_thread(_resolve_project, db_path, slug)
        if resolved is None:
            return JSONResponse({"error": "project not found"}, status_code=404)
        root, _ = resolved
        hits = await asyncio.to_thread(
            index_mod._search_hits,
            db_path,
            q,
            root,
            top_k=k,
            file_pattern=file_glob,
        )
        return JSONResponse({"hits": hits})

    async def read_file_handler(request):
        slug = request.query_params.get("project") or ""
        rel = request.query_params.get("path") or ""
        if not rel:
            return JSONResponse({"error": "path is required"}, status_code=400)
        resolved = await asyncio.to_thread(_resolve_project, db_path, slug)
        if resolved is None:
            return JSONResponse({"error": "project not found"}, status_code=404)
        root, _ = resolved
        try:
            full = resolve_safe(root, rel)
        except PermissionError as e:
            return JSONResponse({"error": str(e)}, status_code=403)
        if not full.is_file():
            return JSONResponse({"error": f"{rel} not found"}, status_code=404)
        size = full.stat().st_size
        if size > MAX_READ_BYTES:
            return JSONResponse(
                {"error": f"{rel} too large ({size} > {MAX_READ_BYTES})"},
                status_code=413,
            )
        content = await asyncio.to_thread(
            full.read_text,
            "utf-8",
            "replace",
        )
        return JSONResponse(
            {
                "path": str(full.relative_to(root.resolve())),
                "content": content,
                "bytes": size,
                "lines": content.count("\n") + 1,
            }
        )

    async def start_index(request):
        slug = request.path_params["slug"]
        resolved = await asyncio.to_thread(_resolve_project, db_path, slug)
        if resolved is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        root, _ = resolved
        try:
            body = await request.json()
        except Exception:
            body = {}
        reindex = bool(body.get("reindex", False))
        exclude = list(body.get("exclude") or [])
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_progress(payload: dict) -> None:
            asyncio.run_coroutine_threadsafe(queue.put(payload), loop)

        async def runner():
            def do_index():
                try:
                    if reindex:
                        index_mod.reset_index(db_path, root)
                    index_mod.build_index(
                        root,
                        db_path,
                        extra_excludes=exclude,
                        on_progress=on_progress,
                    )
                except Exception as e:
                    on_progress({"phase": "error", "message": f"{type(e).__name__}: {e}"})
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            await asyncio.to_thread(do_index)

        asyncio.create_task(runner())

        async def event_stream():
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ----- WebSocket handler -----

    async def chat_ws(websocket):
        # Auth: header OR ?token= (browsers can't set headers on `new WebSocket`)
        token_q = websocket.query_params.get("token", "")
        auth_h = websocket.headers.get("authorization", "")
        if auth_h.lower().startswith("bearer "):
            presented = auth_h[7:].strip()
        else:
            presented = token_q
        if not presented or not secrets.compare_digest(
            presented.encode("utf-8"),
            expected_token,
        ):
            await websocket.close(code=WS_CLOSE_AUTH, reason="auth failed")
            return

        slug = websocket.query_params.get("project", "")
        resolved = await asyncio.to_thread(_resolve_project, db_path, slug) if slug else None
        if resolved is None and default_root is not None:
            resolved = (default_root, "")
        if resolved is None:
            await websocket.accept()
            await websocket.send_json(
                {
                    "type": "error",
                    "subtype": "no_project",
                    "message": (f"project {slug!r} not found and no default --root configured"),
                }
            )
            await websocket.close(code=WS_CLOSE_NOT_FOUND, reason="no project")
            return
        root, _ = resolved
        if not db_path.exists():
            await websocket.accept()
            await websocket.send_json(
                {
                    "type": "error",
                    "subtype": "no_index",
                    "message": f"no index at {db_path}; run `codebase-rag index .` first",
                }
            )
            await websocket.close(code=WS_CLOSE_INIT_FAILED, reason="no index")
            return

        session = await asyncio.to_thread(
            chat_mod.init_chat_session,
            db_path,
            root,
            **chat_defaults,
        )
        if session is None:
            await websocket.accept()
            await websocket.send_json(
                {
                    "type": "error",
                    "subtype": "init_failed",
                    "message": "could not init chat session (provider down? no index?)",
                }
            )
            await websocket.close(code=WS_CLOSE_INIT_FAILED, reason="init failed")
            return

        await websocket.accept()

        loop = asyncio.get_running_loop()
        outbox: asyncio.Queue = asyncio.Queue()
        confirm_inbox: asyncio.Queue = asyncio.Queue()
        state = {"turn_in_flight": False}

        def _post(env: dict) -> None:
            # Safe to call from any thread.
            try:
                asyncio.run_coroutine_threadsafe(outbox.put(env), loop)
            except RuntimeError:
                pass

        session.on_change_error = lambda p, e: _post(
            {
                "type": "warn",
                "message": f"reindex failed for {p}: {e}",
            }
        )

        await websocket.send_json(
            {
                "type": "hello",
                "session_id": session.session,
                "root": str(session.root),
                "model": session.chat_model,
                "provider": session.provider_name,
                "collection": session.collection.name,
                "capabilities": {
                    "confirm_writes": session.confirm_writes,
                    "read_only": session.read_only,
                    "allow_shell": session.allow_shell,
                    "allow_web": session.allow_web,
                    "architect_model": session.architect_model,
                    "shell_runner": session.shell_runner,
                    "shell_network": session.shell_network,
                    "shell_timeout": session.shell_timeout,
                },
                "slash_specs": [
                    {
                        "name": s.name,
                        "desc": s.desc,
                        "requires": s.requires,
                        "takes_arg": s.takes_arg,
                    }
                    for s in chat_mod.SLASH_SPECS
                ],
                "pinned": list(session.pinned_paths),
                "history_len": len(session.history),
            }
        )

        async def sender():
            while True:
                msg = await outbox.get()
                if msg is None:
                    return
                try:
                    await websocket.send_json(msg)
                except Exception:
                    return

        sender_task = asyncio.create_task(sender())

        def make_slash_confirm():
            def confirm(prompt: str) -> bool:
                try:
                    asyncio.run_coroutine_threadsafe(
                        outbox.put({"type": "slash_confirm_prompt", "prompt": prompt}),
                        loop,
                    ).result()
                    reply = asyncio.run_coroutine_threadsafe(
                        confirm_inbox.get(),
                        loop,
                    ).result()
                except Exception:
                    return False
                return bool(reply and reply.get("approved"))

            return confirm

        async def drive_turn(text: str, verbose: bool):
            turn_id = uuid.uuid4().hex[:8]
            call_counter = [0]

            def worker():
                gen = chat_mod.agent_turn(session, text, verbose=verbose)
                try:
                    event = next(gen, None)
                    while event is not None:
                        env = _serialize_event(
                            event,
                            session,
                            turn_id,
                            call_counter,
                        )
                        try:
                            asyncio.run_coroutine_threadsafe(
                                outbox.put(env),
                                loop,
                            ).result()
                        except Exception:
                            return
                        if event[0] == "confirm":
                            try:
                                reply = asyncio.run_coroutine_threadsafe(
                                    confirm_inbox.get(),
                                    loop,
                                ).result()
                            except Exception:
                                reply = None
                            if reply is None or not reply.get("approved"):
                                event = gen.send(None)
                            else:
                                event = gen.send(
                                    reply.get("args") or event[2],
                                )
                        else:
                            event = next(gen, None)
                finally:
                    try:
                        gen.close()
                    except Exception:
                        pass

            await asyncio.to_thread(worker)
            await outbox.put({"type": "turn_complete", "turn": turn_id})
            try:
                await asyncio.to_thread(
                    chat_mod._save_conversation,
                    session.root,
                    session.history,
                    session.chat_model,
                    session.pinned_paths,
                )
            except OSError:
                pass

        async def run_slash(line: str) -> bool:
            """Dispatch a slash command. Returns True if the session should exit."""
            dispatch = await asyncio.to_thread(
                chat_mod.dispatch_slash,
                session,
                line,
                confirm=make_slash_confirm(),
            )
            if dispatch is None:
                await outbox.put({"type": "warn", "message": f"not a slash command: {line!r}"})
                return False
            await outbox.put(
                {
                    "type": "slash_output",
                    "lines": [{"level": lv, "text": tx} for lv, tx in dispatch],
                }
            )
            if any(lv == "exit" for lv, _ in dispatch):
                return True
            try:
                await asyncio.to_thread(
                    chat_mod._save_conversation,
                    session.root,
                    session.history,
                    session.chat_model,
                    session.pinned_paths,
                )
            except OSError:
                pass
            return False

        try:
            while True:
                msg = await websocket.receive_json()
                mtype = msg.get("type")

                if mtype == "ping":
                    await outbox.put({"type": "pong"})
                    continue
                if mtype == "close":
                    await websocket.close(code=1000, reason="client close")
                    break
                if mtype == "confirm":
                    await confirm_inbox.put(
                        {
                            "approved": bool(msg.get("approved")),
                            "args": msg.get("args"),
                        }
                    )
                    continue
                if mtype == "slash_confirm_reply":
                    await confirm_inbox.put(
                        {
                            "approved": bool(msg.get("approved")),
                        }
                    )
                    continue
                if mtype == "abort":
                    # Best-effort: unblock the worker if it's at a confirm.
                    await confirm_inbox.put(None)
                    continue
                if mtype == "config_update":
                    flags = msg.get("flags") or {}
                    changed: dict[str, Any] = {}
                    for k in ("confirm_writes", "read_only"):
                        if k in flags:
                            setattr(session, k, bool(flags[k]))
                            changed[k] = bool(flags[k])
                    await outbox.put({"type": "config_ack", "flags": changed})
                    continue
                if mtype == "slash":
                    line = (msg.get("command") or "").strip()
                    if not line:
                        await outbox.put({"type": "warn", "message": "empty slash"})
                        continue
                    if state["turn_in_flight"]:
                        await outbox.put(
                            {
                                "type": "error",
                                "subtype": "turn_in_flight",
                                "message": "wait for the current turn",
                            }
                        )
                        continue
                    if await run_slash(line):
                        await websocket.close(code=1000, reason="slash exit")
                        break
                    continue
                if mtype == "user_input":
                    text = (msg.get("text") or "").strip()
                    if not text:
                        await outbox.put({"type": "warn", "message": "empty input"})
                        continue
                    if state["turn_in_flight"]:
                        await outbox.put(
                            {
                                "type": "error",
                                "subtype": "turn_in_flight",
                                "message": "a turn is already running",
                            }
                        )
                        continue
                    # Composer slash UX: ":add foo", "exit", "quit"
                    if text.startswith(":") or text in ("exit", "quit"):
                        dispatch = await asyncio.to_thread(
                            chat_mod.dispatch_slash,
                            session,
                            text,
                            confirm=make_slash_confirm(),
                        )
                        if dispatch is not None:
                            await outbox.put(
                                {
                                    "type": "slash_output",
                                    "lines": [{"level": lv, "text": tx} for lv, tx in dispatch],
                                }
                            )
                            if any(lv == "exit" for lv, _ in dispatch):
                                await websocket.close(code=1000, reason="slash exit")
                                break
                            try:
                                await asyncio.to_thread(
                                    chat_mod._save_conversation,
                                    session.root,
                                    session.history,
                                    session.chat_model,
                                    session.pinned_paths,
                                )
                            except OSError:
                                pass
                            continue
                    state["turn_in_flight"] = True
                    try:
                        await drive_turn(text, bool(msg.get("verbose")))
                    finally:
                        state["turn_in_flight"] = False
                    continue

                await outbox.put({"type": "warn", "message": f"unknown type {mtype!r}"})
        except WebSocketDisconnect:
            # Poison the confirm queue so a pending worker unblocks.
            confirm_inbox.put_nowait(None)
        finally:
            try:
                audit_mod.log_event(
                    session.meta_dir,
                    session.session,
                    "session_end",
                    reason="ws_closed",
                )
            except Exception:
                pass
            await outbox.put(None)
            try:
                await asyncio.wait_for(sender_task, timeout=2.0)
            except Exception:
                sender_task.cancel()

    # ----- routes + app -----

    routes = [
        Route("/api/health", health, methods=["GET"]),
        Route("/api/projects", list_projects, methods=["GET"]),
        Route("/api/projects/{slug}", get_project, methods=["GET"]),
        Route("/api/projects/{slug}/sessions", list_sessions, methods=["GET"]),
        Route("/api/projects/{slug}/audit", get_audit, methods=["GET"]),
        Route("/api/projects/{slug}/diagnostics", get_diagnostics, methods=["GET"]),
        Route("/api/projects/{slug}/diagnostics", put_diagnostics, methods=["PUT"]),
        Route("/api/projects/{slug}/diagnostics", delete_diagnostics, methods=["DELETE"]),
        Route("/api/projects/{slug}/index", start_index, methods=["POST"]),
        Route("/api/search", search, methods=["GET"]),
        Route("/api/file", read_file_handler, methods=["GET"]),
        WebSocketRoute("/api/chat", chat_ws),
    ]
    if static_dir is not None:
        routes.append(
            Mount("/", StaticFiles(directory=str(static_dir), html=True), name="spa"),
        )

    app = Starlette(
        debug=False,
        routes=routes,
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origin_regex=CORS_ORIGIN_RE,
                allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                allow_headers=["Authorization", "Content-Type"],
                allow_credentials=False,
            ),
            Middleware(AuthMiddleware),
        ],
    )
    return app


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _print_banner(
    host: str,
    port: int,
    root: Path,
    db_path: Path,
    token: str,
    token_path: Path,
    static_dir: Path | None,
    chat_defaults: dict[str, Any],
) -> None:
    is_loopback = host in ("127.0.0.1", "localhost", "::1")
    truncated = f"{token[:8]}…{token[-4:]}" if len(token) > 14 else "***"
    provider = chat_defaults.get("provider_name") or "ollama"
    model = (
        chat_defaults.get("model") or os.environ.get("CODEBASE_RAG_CHAT_MODEL") or "mistral-nemo"
    )
    binding = f"http://{host}:{port}"
    print(f"codebase-rag serve  v{_VERSION}")
    print(
        f"  bind:     {binding}    {'(loopback — local only)' if is_loopback else '(NON-LOOPBACK)'}"
    )
    print(f"  root:     {root}")
    print(f"  db:       {db_path}")
    print(f"  provider: {provider}   model: {model}")
    print(f"  token:    {truncated}    (file: {token_path}, mode 0600)")
    print(f"  static:   {static_dir or '(none — REST + WS only)'}")
    if not is_loopback:
        print()
        print(
            "  ‼ BINDING NON-LOOPBACK. Your codebase is reachable from every "
            "device on this network."
        )
        print("  ‼ Token + provider keys may be at risk. Ctrl-C now if unintended.")
    print()
    print(
        "  ⚠ Anyone with the token can read this codebase. Rotate by restarting (or --reuse-token)."
    )
    print()
    print(f"  open: {binding}/?token={token}")
    if provider == "anthropic":
        print()
        print("  ⚠ Provider: anthropic — chat content WILL leave your machine (api.anthropic.com).")
        print("    Embeddings remain local via Ollama.")
    sys.stdout.flush()


def run(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    db_path: Path,
    root: Path,
    static_dir: Path | None = None,
    reuse_token: bool = False,
    token_file: Path | None = None,
    quiet: bool = False,
    chat_defaults: dict[str, Any] | None = None,
) -> None:
    """Issue a token (or reuse existing), print the banner, launch uvicorn."""
    import uvicorn

    root = root.resolve()
    if token_file is not None:
        token_file = token_file.resolve()
        meta_dir = token_file.parent
        target_path = token_file
    else:
        meta_dir = index_mod.project_meta_dir(root)
        target_path = _token_path(meta_dir)

    existing = None
    if reuse_token:
        try:
            existing = target_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            existing = None
    if existing:
        token = existing
    else:
        if token_file is not None:
            meta_dir.mkdir(parents=True, exist_ok=True)
            token = secrets.token_urlsafe(32)
            tmp = token_file.with_suffix(token_file.suffix + ".tmp")
            tmp.write_text(token, encoding="utf-8")
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, token_file)
        else:
            token = issue_token(meta_dir)

    app = create_app(
        db_path=db_path,
        default_root=root,
        token=token,
        static_dir=static_dir,
        quiet=quiet,
        chat_defaults=chat_defaults,
    )

    _print_banner(
        host,
        port,
        root,
        db_path,
        token,
        target_path,
        static_dir,
        chat_defaults or {},
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="warning" if quiet else "info",
        access_log=not quiet,
    )
