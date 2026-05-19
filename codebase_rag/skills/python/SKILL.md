Use established Python project practices:

- Prefer a small, explicit project layout with `pyproject.toml` for package metadata and tool configuration.
- Use `pathlib.Path` for local file paths, and pass `encoding="utf-8"` when reading or writing text.
- Keep side effects at the edge: library functions should not change cwd, mutate global state, or perform network calls unless the user asked for that behavior.
- Type public boundaries and important internal helpers. Prefer simple built-in collection types such as `list[str]` and `dict[str, Any]`.
- Raise clear exceptions or return structured error data that matches the surrounding codebase; do not swallow failures silently.
- Keep dependencies minimal. Use the standard library when it is enough, and lazy-import optional packages inside the feature that needs them.
- Add focused pytest coverage for behavior that can regress. Prefer small fixtures and direct assertions over broad snapshot-style tests.
- Follow existing repository style before introducing new abstractions, formatters, or frameworks.
