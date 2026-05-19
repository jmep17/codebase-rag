---
id: "pathlib-file-io"
title: "pathlib file I/O"
triggers: ["pathlib", "read file", "write file", "file io"]
file_globs: ["*.py"]
packages: []
max_tokens: 240
---

Use `Path` at file-system boundaries and keep encoding explicit.

```python
from pathlib import Path


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
```
