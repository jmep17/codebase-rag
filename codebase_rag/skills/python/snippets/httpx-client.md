---
id: "httpx-client"
title: "HTTPX client"
triggers: ["httpx", "http client", "request", "fetch"]
file_globs: ["*.py"]
packages: ["httpx"]
max_tokens: 280
---

Use explicit timeouts and let callers decide when network access is allowed.

```python
from __future__ import annotations

import httpx


def fetch_json(url: str, *, timeout: float = 10.0) -> dict:
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()
```

Do not add background telemetry, update checks, or anonymous reporting.
