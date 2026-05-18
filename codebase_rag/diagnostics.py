"""Per-project IDE/LSP diagnostics cache.

Diagnostics are written by an editor bridge or CLI command and stored outside
the user's repository in the existing project metadata directory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .index import project_meta_dir

DIAGNOSTICS_FILE = "diagnostics.json"
MAX_DIAGNOSTICS = 5000

_SEVERITY_BY_LSP_VALUE = {
    1: "error",
    2: "warning",
    3: "information",
    4: "hint",
}


def diagnostics_path(root: Path) -> Path:
    return project_meta_dir(root) / DIAGNOSTICS_FILE


def _as_line_col(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [int(value[0]), int(value[1])]
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        line = value.get("line")
        character = value.get("character", value.get("col", value.get("column")))
        try:
            return [int(line), int(character)]
        except (TypeError, ValueError):
            return None
    return None


def _normalize_severity(value: Any) -> str:
    if isinstance(value, int):
        return _SEVERITY_BY_LSP_VALUE.get(value, str(value))
    text = str(value or "").strip().lower()
    if text in {"1", "error", "err"}:
        return "error"
    if text in {"2", "warning", "warn"}:
        return "warning"
    if text in {"3", "information", "info"}:
        return "information"
    if text in {"4", "hint"}:
        return "hint"
    return text or "unknown"


def _normalize_range(item: dict[str, Any]) -> dict[str, list[int] | None]:
    raw_range = item.get("range") if isinstance(item.get("range"), dict) else {}
    start = _as_line_col(raw_range.get("start")) or _as_line_col(item.get("start"))
    end = _as_line_col(raw_range.get("end")) or _as_line_col(item.get("end"))

    if start is None and ("line" in item or "line_number" in item):
        try:
            line = int(item.get("line", item.get("line_number")))
        except (TypeError, ValueError):
            line = 0
        try:
            col = int(item.get("character", item.get("column", item.get("col", 0))))
        except (TypeError, ValueError):
            col = 0
        start = [line, col]
    return {"start": start, "end": end}


def _path_from_uri_or_path(root: Path, item: dict[str, Any]) -> str | None:
    raw = item.get("path") or item.get("file") or item.get("uri") or item.get("target")
    if not isinstance(raw, str) or not raw.strip():
        return None
    raw = raw.strip()
    if raw.startswith("file://"):
        raw = raw[7:]
    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            return str(candidate.resolve().relative_to(root.resolve()))
        except ValueError:
            return None
    cleaned = str(candidate).replace("\\", "/")
    if cleaned.startswith("../") or cleaned == "..":
        return None
    return cleaned


def _normalize_diagnostic(root: Path, item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    path = _path_from_uri_or_path(root, item)
    if path is None:
        return None
    message = str(item.get("message") or item.get("text") or "").strip()
    if not message:
        return None
    out = {
        "path": path,
        "range": _normalize_range(item),
        "severity": _normalize_severity(item.get("severity")),
        "source": str(item.get("source") or "").strip(),
        "code": str(item.get("code") or "").strip(),
        "message": message,
    }
    return out


def _extract_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("diagnostics payload must be a list or object")
    items = payload.get("diagnostics", payload.get("items", payload.get("problems")))
    if isinstance(items, list):
        return items
    raise ValueError("diagnostics payload must contain a diagnostics/items/problems list")


def write_diagnostics(root: Path, payload: Any) -> dict[str, Any]:
    root = root.resolve()
    items = _extract_items(payload)
    diagnostics: list[dict[str, Any]] = []
    skipped = 0
    for item in items[:MAX_DIAGNOSTICS]:
        normalized = _normalize_diagnostic(root, item)
        if normalized is None:
            skipped += 1
            continue
        diagnostics.append(normalized)
    if len(items) > MAX_DIAGNOSTICS:
        skipped += len(items) - MAX_DIAGNOSTICS

    meta_dir = project_meta_dir(root)
    meta_dir.mkdir(parents=True, exist_ok=True)
    cache = {
        "root": str(root),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "diagnostics": diagnostics,
    }
    path = diagnostics_path(root)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return {
        "ok": True,
        "path": str(path),
        "updated_at": cache["updated_at"],
        "count": len(diagnostics),
        "skipped": skipped,
    }


def read_diagnostics(
    root: Path,
    *,
    path: str | None = None,
    severity: str | None = None,
    source: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    cache_path = diagnostics_path(root)
    if not cache_path.is_file():
        return {
            "ok": True,
            "path": str(cache_path),
            "updated_at": None,
            "count": 0,
            "diagnostics": [],
        }
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"ok": False, "error": f"could not read diagnostics cache: {e}"}

    diagnostics = list(cache.get("diagnostics") or [])
    if path:
        path_lc = path.replace("\\", "/")
        diagnostics = [d for d in diagnostics if d.get("path") == path_lc]
    if severity:
        severity_lc = severity.lower()
        diagnostics = [d for d in diagnostics if str(d.get("severity", "")).lower() == severity_lc]
    if source:
        source_lc = source.lower()
        diagnostics = [d for d in diagnostics if str(d.get("source", "")).lower() == source_lc]

    counts: dict[str, int] = {}
    for item in diagnostics:
        sev = str(item.get("severity") or "unknown")
        counts[sev] = counts.get(sev, 0) + 1

    effective_limit = max(1, min(int(limit or 100), 500))
    truncated = len(diagnostics) > effective_limit
    return {
        "ok": True,
        "path": str(cache_path),
        "updated_at": cache.get("updated_at"),
        "count": len(diagnostics),
        "counts": counts,
        "diagnostics": diagnostics[:effective_limit],
        "truncated": truncated,
    }


def clear_diagnostics(root: Path) -> bool:
    path = diagnostics_path(root)
    if path.is_file():
        path.unlink()
        return True
    return False
