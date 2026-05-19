from __future__ import annotations

import socket
import unittest
from pathlib import Path
from unittest import mock

from codebase_rag import tools, url_ingest, web


def fake_getaddrinfo(ip: str):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]


class UrlPolicyTests(unittest.TestCase):
    def test_requires_allowlist(self):
        ok, reason = url_ingest._validate_url_policy(
            "https://docs.example.com/page",
            allow_patterns=(),
            block_patterns=(),
        )
        self.assertFalse(ok)
        self.assertIn("--web-allow", reason)

    def test_rejects_non_http_scheme(self):
        ok, reason = url_ingest._validate_url_policy(
            "file:///etc/passwd",
            allow_patterns=("docs.example.com",),
            block_patterns=(),
        )
        self.assertFalse(ok)
        self.assertIn("scheme", reason)

    def test_rejects_private_resolved_ip(self):
        with mock.patch("socket.getaddrinfo", return_value=fake_getaddrinfo("127.0.0.1")):
            ok, reason = url_ingest._validate_url_policy(
                "https://docs.example.com/page",
                allow_patterns=("docs.example.com",),
                block_patterns=(),
            )
        self.assertFalse(ok)
        self.assertIn("blocked", reason)

    def test_allows_matching_public_host(self):
        with mock.patch("socket.getaddrinfo", return_value=fake_getaddrinfo("93.184.216.34")):
            ok, reason = url_ingest._validate_url_policy(
                "https://docs.example.com/page",
                allow_patterns=("*.example.com",),
                block_patterns=(),
            )
        self.assertTrue(ok, reason)


class WebFetchPolicyTests(unittest.TestCase):
    def test_web_fetch_policy_requires_allowlist(self):
        ok, reason = web._validate_url_policy(
            "https://docs.example.com/page",
            allow_patterns=(),
            block_patterns=(),
        )
        self.assertFalse(ok)
        self.assertIn("--web-allow", reason)

    def test_web_fetch_policy_rejects_private_resolved_ip(self):
        with mock.patch("socket.getaddrinfo", return_value=fake_getaddrinfo("10.0.0.5")):
            ok, reason = web._validate_url_policy(
                "https://docs.example.com/page",
                allow_patterns=("docs.example.com",),
                block_patterns=(),
            )
        self.assertFalse(ok)
        self.assertIn("blocked", reason)

    def test_host_shell_runner_is_disabled(self):
        result = tools.run_shell(
            Path("."),
            "python --version",
            runner="host",
        )
        self.assertFalse(result["ok"])
        self.assertIn("host shell runner is disabled", result["error"])


class FakeResponse:
    status_code = 200
    headers = {}
    encoding = "utf-8"
    url = "https://docs.example.com/guide"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_bytes(self):
        yield b"""
        <html>
          <head><title>Guide</title></head>
          <body>
            <nav>Navigation</nav>
            <main>
              <h1>Install Guide</h1>
              <p>Use the package manager.</p>
              <pre><code>pip install demo</code></pre>
            </main>
          </body>
        </html>
        """


class FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url):
        return FakeResponse()


class UrlMarkdownTests(unittest.TestCase):
    def test_fetch_url_markdown_extracts_markdown(self):
        try:
            import httpx  # noqa: F401
            import trafilatura  # noqa: F401
        except ImportError:
            self.skipTest("web extra is not installed")

        with (
            mock.patch("socket.getaddrinfo", return_value=fake_getaddrinfo("93.184.216.34")),
            mock.patch("httpx.Client", FakeClient),
        ):
            result = url_ingest.fetch_url_markdown(
                "https://docs.example.com/guide",
                allow_patterns=("docs.example.com",),
                block_patterns=(),
            )

        self.assertTrue(result.get("ok"), result)
        self.assertIn("Install Guide", result["markdown"])
        self.assertIn("pip install demo", result["markdown"])


if __name__ == "__main__":
    unittest.main()
