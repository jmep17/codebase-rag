"""Web search/fetch tools backed by self-hosted SearXNG.

Privacy posture: queries go only to the user's own SearXNG instance (no third
party sees the query). Fetched URLs are subject to allow / block glob patterns
on the host part. Fetched page text is cached per project for 24 hours so the
same URL across turns is a no-op.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import time
import urllib.parse
from collections.abc import Sequence
from pathlib import Path

MAX_FETCH_CHARS = 50_000
FETCH_TIMEOUT = 10.0
SEARCH_TIMEOUT = 10.0
CACHE_TTL_SECONDS = 24 * 3600


class WebToolsUnavailable(RuntimeError):
    """Raised when httpx or trafilatura isn't installed."""


def _require_deps() -> tuple:
    try:
        import httpx  # noqa: F401
    except ImportError as e:
        raise WebToolsUnavailable(
            "httpx not installed. The web tools need `pip install -e .[web]`."
        ) from e
    try:
        import trafilatura  # noqa: F401
    except ImportError as e:
        raise WebToolsUnavailable(
            "trafilatura not installed. The web tools need `pip install -e .[web]`."
        ) from e
    return None


def _host_of(url: str) -> str:
    try:
        return (urllib.parse.urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _host_allowed(
    host: str,
    allow_patterns: Sequence[str],
    block_patterns: Sequence[str],
) -> tuple[bool, str]:
    """Return (allowed, reason). Blocklist takes precedence."""
    if not host:
        return False, "missing host"
    for pat in block_patterns:
        if fnmatch.fnmatch(host, pat):
            return False, f"host {host} blocked by --web-block {pat!r}"
    if allow_patterns:
        for pat in allow_patterns:
            if fnmatch.fnmatch(host, pat):
                return True, ""
        return False, f"host {host} not in --web-allow list"
    return True, ""


def _cache_path(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def web_search(
    query: str,
    *,
    searxng_url: str,
    top_k: int = 10,
) -> dict:
    """POST a query to SearXNG, return top_k results."""
    if not query.strip():
        return {"ok": False, "error": "empty query"}
    if not searxng_url:
        return {"ok": False, "error": "SEARXNG_URL not set"}
    try:
        _require_deps()
    except WebToolsUnavailable as e:
        return {"ok": False, "error": str(e)}
    import httpx

    from .tools import wrap_untrusted

    endpoint = searxng_url.rstrip("/") + "/search"
    try:
        with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
            resp = client.get(
                endpoint,
                params={"q": query, "format": "json", "safesearch": "1"},
            )
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"search request failed: {e}", "query": query}
    if resp.status_code >= 400:
        return {
            "ok": False,
            "error": f"SearXNG returned HTTP {resp.status_code}",
            "query": query,
        }
    try:
        data = resp.json()
    except json.JSONDecodeError:
        return {"ok": False, "error": "SearXNG returned non-JSON", "query": query}
    raw_results = data.get("results") or []
    trimmed = []
    for r in raw_results[:top_k]:
        trimmed.append(
            {
                "title": (r.get("title") or "").strip(),
                "url": r.get("url") or "",
                "snippet": wrap_untrusted((r.get("content") or "").strip())
                if r.get("content")
                else "",
                "engine": r.get("engine") or "",
            }
        )
    return {
        "ok": True,
        "query": query,
        "result_count": len(trimmed),
        "results": trimmed,
    }


def web_fetch(
    url: str,
    *,
    allow_patterns: Sequence[str] = (),
    block_patterns: Sequence[str] = (),
    cache_dir: Path,
    use_cache: bool = True,
) -> dict:
    """HTTP GET `url`, extract main content via trafilatura, cap at MAX_FETCH_CHARS."""
    if not url:
        return {"ok": False, "error": "empty url"}
    host = _host_of(url)
    ok, reason = _host_allowed(host, allow_patterns, block_patterns)
    if not ok:
        return {"ok": False, "error": reason, "url": url, "host": host}
    try:
        _require_deps()
    except WebToolsUnavailable as e:
        return {"ok": False, "error": str(e), "url": url}
    import httpx
    import trafilatura

    from .tools import wrap_untrusted

    # Cache lookup
    cpath = _cache_path(cache_dir, url)
    if use_cache and cpath.is_file():
        try:
            cached = json.loads(cpath.read_text(encoding="utf-8"))
            if time.time() - cached.get("ts", 0) < CACHE_TTL_SECONDS:
                return {
                    "ok": True,
                    "url": cached.get("url", url),
                    "title": cached.get("title", ""),
                    "content": wrap_untrusted(cached.get("content", "")),
                    "status": cached.get("status", 200),
                    "cached": True,
                    "cached_at": cached.get("ts"),
                }
        except (OSError, json.JSONDecodeError):
            pass

    # Fetch
    try:
        with httpx.Client(follow_redirects=True, timeout=FETCH_TIMEOUT, max_redirects=3) as client:
            resp = client.get(url)
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"fetch failed: {e}", "url": url}
    if resp.status_code >= 400:
        return {"ok": False, "error": f"HTTP {resp.status_code}", "url": url}

    # Re-check the final host after redirects
    final_host = _host_of(str(resp.url))
    if final_host != host:
        ok, reason = _host_allowed(final_host, allow_patterns, block_patterns)
        if not ok:
            return {
                "ok": False,
                "error": f"redirect to disallowed host: {reason}",
                "url": url,
                "final_url": str(resp.url),
            }

    # Extract main content. trafilatura is forgiving with raw HTML.
    extracted = None
    try:
        extracted = trafilatura.extract(
            resp.text,
            include_links=False,
            include_comments=False,
            favor_recall=True,
        )
    except Exception:
        extracted = None
    text = extracted if extracted else resp.text
    truncated = False
    if len(text) > MAX_FETCH_CHARS:
        text = text[:MAX_FETCH_CHARS] + f"\n... [{len(text) - MAX_FETCH_CHARS} chars truncated]"
        truncated = True

    title = ""
    try:
        meta = trafilatura.extract_metadata(resp.text)
        if meta and meta.title:
            title = meta.title
    except Exception:
        pass

    # Save cache (raw content, no wrap; wrap on read)
    if use_cache:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cpath.write_text(
                json.dumps(
                    {
                        "url": str(resp.url),
                        "title": title,
                        "content": text,
                        "status": resp.status_code,
                        "ts": time.time(),
                    }
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    return {
        "ok": True,
        "url": str(resp.url),
        "title": title,
        "content": wrap_untrusted(text),
        "status": resp.status_code,
        "cached": False,
        "truncated": truncated,
    }
