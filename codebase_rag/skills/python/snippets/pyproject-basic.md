---
id: "pyproject-basic"
title: "Basic pyproject.toml"
triggers: ["pyproject", "package", "project metadata"]
file_globs: ["pyproject.toml"]
packages: []
max_tokens: 260
---

Use `pyproject.toml` as the single source for package metadata and lightweight tool config.

```toml
[project]
name = "example-app"
version = "0.1.0"
description = "Short project description"
requires-python = ">=3.10"
dependencies = []

[project.scripts]
example-app = "example_app.__main__:main"

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
```

Keep optional integrations behind extras instead of making the default install heavy.
