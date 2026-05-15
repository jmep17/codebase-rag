"""Tool implementations for the agent loop. All paths sandboxed under root."""

from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path
from typing import Callable

from .index import _find_nested_repos, _load_ignore_file, iter_source_files

MAX_READ_BYTES = 200_000
GREP_MAX_RESULTS = 300
GREP_MAX_FILE_BYTES = 1_000_000


def resolve_safe(root: Path, requested: str) -> Path:
    root = root.resolve()
    if Path(requested).is_absolute():
        candidate = Path(requested).resolve()
    else:
        candidate = (root / requested).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise PermissionError(f"path {candidate} is outside root {root}")
    return candidate


def read_file(root: Path, path: str) -> dict:
    p = resolve_safe(root, path)
    if not p.exists():
        return {"ok": False, "error": f"{path} does not exist"}
    if not p.is_file():
        return {"ok": False, "error": f"{path} is not a file"}
    size = p.stat().st_size
    if size > MAX_READ_BYTES:
        return {
            "ok": False,
            "error": f"{path} is {size} bytes (limit {MAX_READ_BYTES}); use retrieval instead",
        }
    content = p.read_text(encoding="utf-8", errors="replace")
    return {
        "ok": True,
        "path": str(p.relative_to(root.resolve())),
        "content": content,
        "lines": content.count("\n") + 1,
    }


def write_file(root: Path, path: str, content: str, on_change: Callable[[str], None]) -> dict:
    p = resolve_safe(root, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    actual = p.read_text(encoding="utf-8")
    rel = str(p.relative_to(root.resolve()))
    on_change(rel)
    return {
        "ok": True,
        "path": rel,
        "bytes_written": len(actual.encode("utf-8")),
        "lines": actual.count("\n") + 1,
    }


def grep(root: Path, pattern: str, file_glob: str | None = None) -> dict:
    """Regex-search every source file under `root`. Returns up to GREP_MAX_RESULTS matches."""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return {"ok": False, "error": f"invalid regex: {e}"}

    root = root.resolve()
    user_excludes = tuple(_load_ignore_file(root))
    nested = _find_nested_repos(root)

    matches: list[dict] = []
    files_scanned = 0
    truncated = False
    file_glob_lc = file_glob

    for path in iter_source_files(root, user_excludes, nested):
        rel = str(path.relative_to(root)).replace("\\", "/")
        if file_glob_lc and not (
            fnmatch.fnmatch(rel, file_glob_lc) or fnmatch.fnmatch(path.name, file_glob_lc)
        ):
            continue
        try:
            if path.stat().st_size > GREP_MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        files_scanned += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                matches.append(
                    {
                        "path": rel,
                        "line": lineno,
                        "text": line.strip()[:240],
                    }
                )
                if len(matches) >= GREP_MAX_RESULTS:
                    truncated = True
                    break
        if truncated:
            break

    result = {
        "ok": True,
        "match_count": len(matches),
        "files_scanned": files_scanned,
        "matches": matches,
    }
    if truncated:
        result["truncated"] = True
        result["note"] = (
            f"hit {GREP_MAX_RESULTS}-match cap; narrow the pattern or pass file_glob"
        )
    return result


def edit_file(
    root: Path,
    path: str,
    old_string: str,
    new_string: str,
    on_change: Callable[[str], None],
) -> dict:
    p = resolve_safe(root, path)
    if not p.exists():
        return {"ok": False, "error": f"{path} does not exist"}
    text = p.read_text(encoding="utf-8")
    count = text.count(old_string)
    if count == 0:
        return {
            "ok": False,
            "error": "old_string not found; read the file and copy the exact text including whitespace",
        }
    if count > 1:
        return {
            "ok": False,
            "error": f"old_string appears {count} times; provide more surrounding context to make it unique",
        }
    new_text = text.replace(old_string, new_string, 1)
    p.write_text(new_text, encoding="utf-8")
    rel = str(p.relative_to(root.resolve()))
    on_change(rel)
    return {
        "ok": True,
        "path": rel,
        "bytes_before": len(text.encode("utf-8")),
        "bytes_after": len(new_text.encode("utf-8")),
    }


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file inside the project root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to project root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, overwriting if it exists. Content must be complete; never use placeholders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {
                        "type": "string",
                        "description": "Full file content. Never use '...' or 'rest omitted'.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Regex-search every source file in the project. Use this for exhaustive "
                "queries — 'list every X', 'find all usages of Y', 'where is Z imported' — "
                "where the retrieved Context block alone is not enough. Returns each match "
                "with path, line number, and the matched line. Respects the project's "
                "exclude rules and skips nested git repos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Python regex. Use alternation for variants, e.g. "
                            "'fetch\\\\(|axios\\\\.|http\\\\.(get|post)'."
                        ),
                    },
                    "file_glob": {
                        "type": "string",
                        "description": (
                            "Optional glob filter (matches the relative path or filename), "
                            "e.g. 'src/**/*.ts' or '*.py'."
                        ),
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exactly one occurrence of old_string with new_string in a file. Use read_file first to copy the exact text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to replace, including whitespace. Must be unique in the file.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
]


def run_tool(name: str, args: dict, root: Path, on_change: Callable[[str], None]) -> str:
    impls = {
        "read_file": lambda: read_file(root, **args),
        "write_file": lambda: write_file(root, on_change=on_change, **args),
        "edit_file": lambda: edit_file(root, on_change=on_change, **args),
        "grep": lambda: grep(root, **args),
    }
    if name not in impls:
        return json.dumps({"ok": False, "error": f"unknown tool: {name}"})
    try:
        return json.dumps(impls[name]())
    except TypeError as e:
        return json.dumps({"ok": False, "error": f"bad arguments to {name}: {e}"})
    except Exception as e:
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})
