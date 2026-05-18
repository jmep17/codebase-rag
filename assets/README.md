# Logo assets

Mark concept: three stacked chunks (the index) feeding into a `>` prompt (the
agent). Reads as `≡>` — codebase on the left, chevron on the right. Palette
matches the design tokens in `design/SPEC.md` (`--accent #7aa2f7`, `--fg-1
#a7b0bf`, `--fg-0 #e6e9ef`, `--fg-2 #6b7484`).

| File | Use |
|---|---|
| `logo-mark.svg` | Monochrome mark. Uses `currentColor` — set CSS `color` (or stroke `color` in img contexts) to theme. Favicon, terminal, anywhere. |
| `logo-mark-color.svg` | Two-tone mark in brand colors. README hero, social card. |
| `logo-wordmark.svg` | Mark + `codebase-rag` wordmark in JetBrains Mono. Horizontal lockup, 360×80. |
| `banner.txt` | 5-line Unicode banner for the `chat` startup splash. |
| `banner-ascii.txt` | Pure-ASCII fallback for terminals that mangle box-drawing. |

## README hero

```markdown
<p align="center">
  <img src="assets/logo-wordmark.svg" alt="codebase-rag" height="64">
</p>
```

## Favicon

`logo-mark.svg` works as-is — modern browsers accept SVG favicons.

## Chat startup banner

In `codebase_rag/chat.py`, before the first prompt:

```python
from pathlib import Path
print(Path(__file__).parent.parent.joinpath("assets/banner.txt").read_text())
```

Wrap in ANSI if you want the chevron tinted accent-blue (`\x1b[38;2;122;162;247m`).
