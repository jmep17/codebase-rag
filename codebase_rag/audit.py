"""Per-project audit log: append-only JSON-line record of tool calls and slash commands."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

AUDIT_FILE = "audit.log"
AUDIT_ROLLOVER_BYTES = 5_000_000
ARG_LEN_REDACT_THRESHOLD = 256
SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "password",
    "secret",
    "token",
)


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower())
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def _redact(value: Any, *, key: str = "") -> Any:
    """Long strings → {"_redacted_len": N}. Recurses into dicts/lists."""
    if key and _is_sensitive_key(key):
        return {"_redacted": True}
    if isinstance(value, str):
        if len(value) > ARG_LEN_REDACT_THRESHOLD:
            return {"_redacted_len": len(value)}
        return value
    if isinstance(value, dict):
        return {k: _redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, key=key) for v in value]
    return value


def _maybe_rotate(path: Path) -> None:
    try:
        if path.stat().st_size > AUDIT_ROLLOVER_BYTES:
            backup = path.with_name(f"{path.name}.1")
            if backup.exists():
                backup.unlink()
            path.rename(backup)
    except OSError:
        pass


def log_event(meta_dir: Path, session: str, event: str, **payload: Any) -> None:
    """Append one JSON line to the project's audit.log. Never raises."""
    try:
        meta_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    path = meta_dir / AUDIT_FILE
    _maybe_rotate(path)
    line = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "session": session,
        "event": event,
        **{k: _redact(v, key=k) for k, v in payload.items()},
    }
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, separators=(",", ":")) + "\n")
    except OSError:
        pass


def parse_since(s: str) -> datetime:
    """Accept ISO timestamps and short relative offsets like '15m', '2h', '7d'."""
    s = s.strip().lower()
    m = re.fullmatch(r"(\d+)\s*([mhd])", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        secs = {"m": 60, "h": 3600, "d": 86400}[unit] * n
        return datetime.now(timezone.utc) - timedelta(seconds=secs)
    if s in {"today"}:
        now = datetime.now(timezone.utc)
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if s in {"yesterday"}:
        return datetime.now(timezone.utc) - timedelta(days=1)
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError as e:
        raise ValueError(
            f"cannot parse --since {s!r}: use ISO date, 'today', 'yesterday', or '15m'/'2h'/'7d'"
        ) from e


def tail_audit(
    meta_dir: Path,
    *,
    since: datetime | None = None,
    tool: str | None = None,
    event: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Read the project's audit log; newest-last; filtered."""
    path = meta_dir / AUDIT_FILE
    if not path.is_file():
        return []
    rows: list[dict] = []
    try:
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if tool is not None and row.get("tool") != tool:
                continue
            if event is not None and row.get("event") != event:
                continue
            if since is not None:
                ts = row.get("ts")
                try:
                    row_ts = datetime.fromisoformat(ts) if ts else None
                except ValueError:
                    row_ts = None
                if row_ts is None or row_ts < since:
                    continue
            rows.append(row)
    except OSError:
        return []
    if limit > 0:
        rows = rows[-limit:]
    return rows
