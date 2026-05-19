"""Fetch documentation URLs, convert them to Markdown, and store as references."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import socket
import subprocess
import time
import urllib.parse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import index as index_mod
from .web import WebToolsUnavailable, _host_allowed, _require_deps

FETCH_TIMEOUT = 10.0
MAX_REDIRECTS = 3
MAX_RESPONSE_BYTES = 2_000_000
MAX_MARKDOWN_CHARS = 180_000
DOCKER_TIMEOUT = 45.0


def _url_digest(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def _label_dir_name(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip()).strip(".-")
    if not slug:
        slug = "reference"
    return f"{slug[:64]}-{hashlib.sha1(label.encode('utf-8')).hexdigest()[:8]}"


def _parsed_url(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL scheme must be http or https")
    if not parsed.hostname:
        raise ValueError("URL must include a host")
    if parsed.username or parsed.password:
        raise ValueError("URL credentials are not allowed")
    return parsed


def _default_port(parsed: urllib.parse.ParseResult) -> int:
    if parsed.port:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80


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


def _read_response_capped(resp: Any) -> tuple[str, bool, int]:
    chunks: list[bytes] = []
    total = 0
    truncated = False
    for chunk in resp.iter_bytes():
        total += len(chunk)
        if total > MAX_RESPONSE_BYTES:
            remaining = MAX_RESPONSE_BYTES - sum(len(c) for c in chunks)
            if remaining > 0:
                chunks.append(chunk[:remaining])
            truncated = True
            break
        chunks.append(chunk)
    content = b"".join(chunks)
    encoding = resp.encoding or "utf-8"
    return content.decode(encoding, errors="replace"), truncated, total


def fetch_url_markdown(
    url: str,
    *,
    allow_patterns: Sequence[str],
    block_patterns: Sequence[str],
) -> dict:
    """Fetch one URL under strict policy and convert the HTML response to Markdown."""
    try:
        _require_deps()
    except WebToolsUnavailable as e:
        return {"ok": False, "error": str(e), "url": url}

    import httpx
    import trafilatura

    current = url
    redirects = 0
    html = ""
    response_truncated = False
    response_bytes = 0
    status_code = 0
    final_url = current
    try:
        with httpx.Client(
            timeout=FETCH_TIMEOUT,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            while True:
                ok, reason = _validate_url_policy(
                    current,
                    allow_patterns=allow_patterns,
                    block_patterns=block_patterns,
                )
                if not ok:
                    return {"ok": False, "error": reason, "url": url, "final_url": current}
                try:
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
                break
    except Exception as e:
        return {"ok": False, "error": f"unexpected fetch error: {e}", "url": url}

    ok, reason = _validate_url_policy(
        final_url,
        allow_patterns=allow_patterns,
        block_patterns=block_patterns,
    )
    if not ok:
        return {"ok": False, "error": reason, "url": url, "final_url": final_url}
    if status_code >= 400:
        return {"ok": False, "error": f"HTTP {status_code}", "url": url, "final_url": final_url}

    if not html.strip():
        return {"ok": False, "error": "empty response", "url": url, "final_url": final_url}

    try:
        markdown = trafilatura.extract(
            html,
            url=final_url,
            output_format="markdown",
            include_links=True,
            include_tables=True,
            include_comments=False,
            deduplicate=True,
            favor_recall=True,
        )
    except Exception as e:
        return {"ok": False, "error": f"markdown extraction failed: {e}", "url": url}
    if not markdown:
        markdown = trafilatura.html2txt(html) or html

    markdown_truncated = False
    if len(markdown) > MAX_MARKDOWN_CHARS:
        markdown = (
            markdown[:MAX_MARKDOWN_CHARS]
            + f"\n\n...[{len(markdown) - MAX_MARKDOWN_CHARS} chars truncated]"
        )
        markdown_truncated = True

    title = ""
    try:
        meta = trafilatura.extract_metadata(html, default_url=final_url)
        if meta and meta.title:
            title = meta.title
    except Exception:
        pass

    return {
        "ok": True,
        "url": url,
        "final_url": final_url,
        "title": title,
        "status": status_code,
        "redirects": redirects,
        "response_bytes": response_bytes,
        "response_truncated": response_truncated,
        "markdown": markdown,
        "markdown_truncated": markdown_truncated,
        "fetched_at": int(time.time()),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def worker_fetch_to_json(
    url: str,
    output_json: Path,
    *,
    allow_patterns: Sequence[str],
    block_patterns: Sequence[str],
) -> int:
    result = fetch_url_markdown(
        url,
        allow_patterns=allow_patterns,
        block_patterns=block_patterns,
    )
    _write_json(output_json, result)
    return 0 if result.get("ok") else 2


def _docker_argv(
    *,
    image: str,
    work_dir: Path,
    url: str,
    output_json_name: str,
    allow_patterns: Sequence[str],
    block_patterns: Sequence[str],
) -> list[str]:
    uid = os.getuid()
    gid = os.getgid()
    argv = [
        "docker",
        "run",
        "--rm",
        "--network=bridge",
        "-v",
        f"{work_dir.resolve()}:/work:rw",
        "-w",
        "/work",
        "--user",
        f"{uid}:{gid}",
        "--memory",
        "512m",
        "--cpus",
        "1",
        "--pids-limit",
        "128",
        "--read-only",
        "--tmpfs",
        "/tmp:size=64m",
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges",
        "-e",
        "HOME=/tmp",
        "-e",
        "LANG=C.UTF-8",
        image,
        "codebase-rag",
        "__url-fetch-parse",
        "--url",
        url,
        "--output-json",
        f"/work/{output_json_name}",
    ]
    for pat in allow_patterns:
        argv.extend(["--web-allow", pat])
    for pat in block_patterns:
        argv.extend(["--web-block", pat])
    return argv


def fetch_url_markdown_with_runner(
    url: str,
    *,
    runner: str,
    work_dir: Path,
    allow_patterns: Sequence[str],
    block_patterns: Sequence[str],
) -> dict:
    if runner == "host":
        return fetch_url_markdown(
            url,
            allow_patterns=allow_patterns,
            block_patterns=block_patterns,
        )
    if not runner.startswith("docker:"):
        return {"ok": False, "error": "invalid --url-runner; use host or docker:IMAGE", "url": url}
    image = runner[len("docker:") :].strip()
    if not image:
        return {"ok": False, "error": "docker runner missing image", "url": url}
    work_dir.mkdir(parents=True, exist_ok=True)
    output_name = "result.json"
    output_path = work_dir / output_name
    argv = _docker_argv(
        image=image,
        work_dir=work_dir,
        url=url,
        output_json_name=output_name,
        allow_patterns=allow_patterns,
        block_patterns=block_patterns,
    )
    try:
        proc = subprocess.run(
            argv,
            text=True,
            capture_output=True,
            timeout=DOCKER_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "error": f"docker runner failed: {e}", "url": url}
    if output_path.is_file():
        try:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = {"ok": False, "error": "docker runner wrote invalid JSON", "url": url}
    else:
        result = {
            "ok": False,
            "error": "docker runner did not write result.json",
            "url": url,
        }
    if proc.returncode != 0 and result.get("ok"):
        result = {
            "ok": False,
            "error": f"docker runner exited {proc.returncode}",
            "url": url,
        }
    if not result.get("ok"):
        stderr = (proc.stderr or "").strip()
        if stderr:
            result["docker_stderr"] = stderr[-1000:]
    result["runner"] = runner
    return result


def store_reference_markdown(
    project_root: Path,
    *,
    label: str,
    result: dict,
    runner: str,
) -> Path:
    if not result.get("ok"):
        raise ValueError(result.get("error") or "cannot store failed URL fetch")
    source_url = str(result.get("url") or "")
    final_url = str(result.get("final_url") or source_url)
    digest = _url_digest(source_url)[:16]
    ref_dir = index_mod.project_meta_dir(project_root) / "references" / _label_dir_name(label)
    ref_dir.mkdir(parents=True, exist_ok=True)
    md_path = ref_dir / f"{digest}.md"
    title = str(result.get("title") or "").strip()
    header = [
        "---",
        f"label: {json.dumps(label)}",
        f"source_url: {json.dumps(source_url)}",
        f"final_url: {json.dumps(final_url)}",
        f"title: {json.dumps(title)}",
        f"fetched_at: {int(result.get('fetched_at') or time.time())}",
        f"runner: {json.dumps(runner)}",
        "---",
        "",
    ]
    markdown = str(result.get("markdown") or "")
    md_path.write_text("\n".join(header) + markdown.rstrip() + "\n", encoding="utf-8")

    manifest_path = ref_dir / "manifest.json"
    manifest: dict[str, Any] = {"label": label, "documents": []}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
                manifest.setdefault("documents", [])
        except json.JSONDecodeError:
            pass
    docs = [d for d in manifest.get("documents", []) if d.get("source_url") != source_url]
    docs.append(
        {
            "source_url": source_url,
            "final_url": final_url,
            "title": title,
            "path": md_path.name,
            "fetched_at": result.get("fetched_at"),
            "status": result.get("status"),
            "runner": runner,
            "markdown_truncated": bool(result.get("markdown_truncated")),
            "response_truncated": bool(result.get("response_truncated")),
        }
    )
    manifest["label"] = label
    manifest["documents"] = docs
    _write_json(manifest_path, manifest)
    return ref_dir
