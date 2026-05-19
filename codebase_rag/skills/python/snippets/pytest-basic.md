---
id: "pytest-basic"
title: "Focused pytest test"
triggers: ["pytest", "test", "tests", "fixture"]
file_globs: ["test_*.py", "tests/*.py", "tests/**/*.py"]
packages: ["pytest"]
max_tokens: 260
---

Prefer focused behavior tests with explicit setup and direct assertions.

```python
from pathlib import Path


def test_writes_expected_text(tmp_path: Path) -> None:
    output = tmp_path / "result.txt"

    output.write_text("hello\n", encoding="utf-8")

    assert output.read_text(encoding="utf-8") == "hello\n"
```

Avoid tests that depend on the user's cwd or machine-specific absolute paths.
