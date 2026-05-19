"""Web search/fetch tools backed by self-hosted SearXNG.

Privacy posture: queries go only to the user's own SearXNG instance (no third
party sees the query). Fetched URLs are subject to allow / block glob patterns
on the host part. Fetched page text is cached per project for 24 hours so the
same URL across turns is a no-op.
"""

from __future__ import annotations

import fnmatch
import hashlib
import ipaddress
import json
import socket
import time
import urllib.parse
from collections.abc import Sequence
from pathlib import Path

MAX_FETCH_CHARS = 50_000
MAX_FETCH_BYTES = 1_000_000
FETCH_TIMEOUT = 10.0
SEARCH_TIMEOUT = 10.0
CACHE_TTL_SECONDS = 24 * 3600
MAX_REDIRECTS = 3


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
        return (urllib.parse.urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _parsed_url(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL scheme must be http or https")
    if not parsed.hostname:
        raise ValueError("URL must include a host")
    if parsed.username or parsed.password:
        raise ValueError("URL credentials are not allowed")
    # Accessing .port raises ValueError for malformed ports; force that here.
    _ = parsed.port
    return parsed


def _default_port(parsed: urllib.parse.ParseResult) -> int:
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def _blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return any(
        (
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_unspecified,
            ip.is_reserved,
        )
    )


def _validate_url_policy(
    url: str,
    *,
    allow_patterns: Sequence[str],
    block_patterns: Sequence[str],
) -> tuple[bool, str]:
    if not allow_patterns:
        return False, "at least one --web-allow host glob is required"
    try:
        parsed = _parsed_url(url)
    except ValueError as e:
        return False, str(e)
    host = (parsed.hostname or "").lower()
    ok, reason = _host_allowed(host, allow_patterns, block_patterns)
    if not ok:
        return False, reason
    try:
        infos = socket.getaddrinfo(host, _default_port(parsed), type=socket.SOCK_STREAM)
    except OSError as e:
        return False, f"could not resolve host {host}: {e}"
    seen: set[str] = set()
    for info in infos:
        ip_s = info[4][0]
        if ip_s in seen:
            continue
        seen.add(ip_s)
        try:
            ip = ipaddress.ip_address(ip_s)
        except ValueError:
            return False, f"could not parse resolved address {ip_s!r}"
        if _blocked_ip(ip):
            return False, f"host {host} resolves to blocked private/reserved address {ip}"
    if not seen:
        return False, f"host {host} did not resolve to any addresses"
    return True, ""


def _read_response_capped(resp) -> tuple[str, bool, int]:
    chunks: list[bytes] = []
    total = 0
    truncated = False
    for chunk in resp.iter_bytes():
        total += len(chunk)
        if total > MAX_FETCH_BYTES:
            remaining = MAX_FETCH_BYTES - sum(len(c) for c in chunks)
            if remaining > 0:
                chunks.append(chunk[:remaining])
            truncated = True
            break
        chunks.append(chunk)
    encoding = resp.encoding or "utf-8"
    return b"".join(chunks).decode(encoding, errors="replace"), truncated, total


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
        with httpx.Client(timeout=SEARCH_TIMEOUT, trust_env=False) as client:
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
    ok, reason = _validate_url_policy(
        url,
        allow_patterns=allow_patterns,
        block_patterns=block_patterns,
    )
    if not ok:
        return {"ok": False, "error": reason, "url": url, "host": _host_of(url)}
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
                cached_url = cached.get("url", url)
                ok, reason = _validate_url_policy(
                    cached_url,
                    allow_patterns=allow_patterns,
                    block_patterns=block_patterns,
                )
                if not ok:
                    return {
                        "ok": False,
                        "error": f"cached URL no longer allowed: {reason}",
                        "url": url,
                        "final_url": cached_url,
                    }
                return {
                    "ok": True,
                    "url": cached_url,
                    "title": cached.get("title", ""),
                    "content": wrap_untrusted(cached.get("content", "")),
                    "status": cached.get("status", 200),
                    "cached": True,
                    "cached_at": cached.get("ts"),
                }
        except (OSError, json.JSONDecodeError):
            pass

    # Fetch. Redirects are followed manually so every hop is checked before
    # the request is sent; this prevents allowed hosts from redirecting the
    # agent to loopback, link-local, or other internal addresses.
    current = url
    redirects = 0
    status_code = 0
    final_url = current
    html = ""
    response_truncated = False
    response_bytes = 0
    try:
        with httpx.Client(follow_redirects=False, timeout=FETCH_TIMEOUT, trust_env=False) as client:
            while True:
                ok, reason = _validate_url_policy(
                    current,
                    allow_patterns=allow_patterns,
                    block_patterns=block_patterns,
                )
                if not ok:
                    return {"ok": False, "error": reason, "url": url, "final_url": current}
                with client.stream("GET", current) as resp:
                    status_code = resp.status_code
                    final_url = str(resp.url)
                    if resp.status_code in {301, 302, 303, 307, 308}:
                        redirects += 1
                        if redirects > MAX_REDIRECTS:
                            return {
                                "ok": False,
                                "error": f"too many redirects (>{MAX_REDIRECTS})",
                                "url": url,
                                "final_url": current,
                            }
                        location = resp.headers.get("location")
                        if not location:
                            return {
                                "ok": False,
                                "error": f"HTTP {resp.status_code} redirect without Location",
                                "url": url,
                                "final_url": current,
                            }
                        current = urllib.parse.urljoin(current, location)
                        continue
                    html, response_truncated, response_bytes = _read_response_capped(resp)
                    break
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"fetch failed: {e}", "url": url}
    if status_code >= 400:
        return {"ok": False, "error": f"HTTP {status_code}", "url": url, "final_url": final_url}

    ok, reason = _validate_url_policy(
        final_url,
        allow_patterns=allow_patterns,
        block_patterns=block_patterns,
    )
    if not ok:
        return {"ok": False, "error": reason, "url": url, "final_url": final_url}

    # Extract main content. trafilatura is forgiving with raw HTML.
    extracted = None
    try:
        extracted = trafilatura.extract(
            html,
            include_links=False,
            include_comments=False,
            favor_recall=True,
        )
    except Exception:
        extracted = None
    text = extracted if extracted else html
    truncated = False
    if len(text) > MAX_FETCH_CHARS:
        text = text[:MAX_FETCH_CHARS] + f"\n... [{len(text) - MAX_FETCH_CHARS} chars truncated]"
        truncated = True

    title = ""
    try:
        meta = trafilatura.extract_metadata(html)
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
                        "url": final_url,
                        "title": title,
                        "content": text,
                        "status": status_code,
                        "ts": time.time(),
                    }
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    return {
        "ok": True,
        "url": final_url,
        "title": title,
        "content": wrap_untrusted(text),
        "status": status_code,
        "cached": False,
        "truncated": truncated,
        "response_bytes": response_bytes,
        "response_truncated": response_truncated,
        "redirects": redirects,
    }
