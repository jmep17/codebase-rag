# codebase-rag — UI redesign spec

Three surfaces, one product. The CLI/TUI is the canonical experience. The browser app is the same product re-rendered for people who don't want a terminal. The desktop app is the browser app plus the things only a native shell can do (global hotkey, menu bar, file watcher, system permissions, OS notifications).

This document covers:

1. Design principles
2. Information architecture (what's the same across all surfaces)
3. Design tokens
4. The TUI (Textual/Rich)
5. The browser app (web)
6. The desktop app (deltas only)
7. Key flows
8. What's deliberately NOT in v1

---

## 1. Design principles

**Local-first is a feature, not fine print.** The current README hides this in a table; the UI puts it in the chrome. A persistent indicator in every surface tells the user exactly what's leaving their machine: "Local only", "+ SearXNG", "+ Anthropic". The indicator is also the entry point — clicking it shows the network surface and lets the user toggle it.

**Retrieval is visible by default.** The model is doing RAG; pretending it isn't is a regression. Every assistant turn shows which chunks were retrieved, from which files, with what scores, and — for the curious — the raw query embedding's nearest neighbors. The model's answer sits on top of an inspector, not in front of a black box.

**Every destructive action is reviewable, never silent.** Writes, edits, and shell calls produce a diff or command card with explicit Approve / Reject / Edit affordances. The model proposes; the user disposes. This is already true in `--confirm-writes`; the redesign makes it the default surface (with a "trust" toggle for users who want autonomy).

**The audit log is a first-class view, not a debug aid.** It's the receipt for the agent's behavior — and the single best artifact for explaining what happened in a session. Promote it to a top-level navigation item.

**One mental model across surfaces.** A user who learns the TUI can use the browser app without retraining. Component names, status colors, keyboard shortcuts (where the surface allows them), and slash commands are identical. The desktop adds capabilities; it doesn't change the model.

**Keyboard-first everywhere.** The CLI obviously, but the browser and desktop versions also commit to `cmd-k` as a universal entrypoint, slash commands inside the chat input (mirroring the TUI), and `j/k`/arrow navigation in lists. Mouse works; it isn't the primary input.

---

## 2. Information architecture

Every surface exposes the same five regions. Their layout differs; their contents do not.

| Region | What it shows |
|---|---|
| **Project switcher** | Indexed projects (path, chunk count, last activity). Active project highlighted. Quick-index button. |
| **Conversation** | Turn-by-turn chat. Each assistant turn carries inline retrieval, tool calls, and diffs. Slash commands work inline. |
| **Retrieval inspector** | For the focused turn: chunks retrieved, scores, file paths, line ranges, which were from project code vs. reference sets. Collapsible. |
| **Context panel** | Pins, project notes, references, current branch / dirty state, model + provider. |
| **Status bar / shield** | Network surface (Local / + SearXNG / + Anthropic), shell mode (off / host / docker), read-only flag, tool count, audit-log link. |

The Conversation region is always the primary one. Everything else expands and collapses without leaving the conversation.

---

## 3. Design tokens

A single token set powers all three surfaces. The TUI maps these to terminal colors (with a 16-color fallback); the browser/desktop use them as CSS variables.

### Color (dark theme — default)

```
--bg-0:        #0b0d10   /* page bg */
--bg-1:        #11141a   /* panel bg */
--bg-2:        #181c24   /* hover / focused panel */
--bg-3:        #232834   /* input, raised */
--border:      #2a3140
--border-soft: #1c2230

--fg-0:        #e6e9ef   /* primary text */
--fg-1:        #a7b0bf   /* secondary */
--fg-2:        #6b7484   /* tertiary / meta */
--fg-dim:      #4a5263

--accent:      #7aa2f7   /* primary action, links, focus ring */
--accent-soft: #29406b

--ok:          #9ece6a   /* success, local-only, approved */
--warn:        #e0af68   /* network surface added, confirm needed */
--danger:      #f7768e   /* writes, destructive, blocked */
--info:        #7dcfff   /* informational, audit, references */
--purple:      #bb9af7   /* tool calls */
```

### Light theme

Mirror image — `--bg-0: #ffffff`, `--fg-0: #11141a`, accents desaturated 12%. Auto-follows `prefers-color-scheme`; explicit toggle in settings.

### Type

- UI sans: `Inter, -apple-system, BlinkMacSystemFont, sans-serif`
- Mono (chat code, file paths, retrieval headers): `"JetBrains Mono", ui-monospace, SF Mono, Consolas, monospace`
- Sizes: 12 / 13 / 14 (body) / 16 (input) / 18 / 22

### Spacing

4-px grid. Components target 8/12/16/24/32.

### Status colors (semantic, used identically across surfaces)

- **Local** → `--ok` dot
- **+ Web (SearXNG)** → `--warn` dot
- **+ Anthropic** → `--danger` dot (most network exposure)
- **Tool call** → `--purple` background tint
- **Write/edit pending approval** → `--warn` border
- **Approved write** → `--ok` border
- **Rejected/blocked** → `--danger` border

---

## 4. The TUI (Textual / Rich)

The terminal UI is a full-screen Textual app launched by `codebase-rag chat` (or `codebase-rag chat --tui` if we keep the line-oriented mode as the default; recommended: `--tui` default-on for interactive TTYs, line mode for non-TTY/piped).

### Layout

```
┌ codebase-rag ─────────────────────────────────────────────────────────── ⏻ ─┐
│ ● Local  · mistral-nemo · read+write · shell:off · web:off · 4,231 chunks   │ ← top bar / shield
├─────────────┬──────────────────────────────────────────────┬────────────────┤
│ projects    │  conversation                                │  context       │
│             │                                              │                │
│ ● my-app    │  > how does auth middleware work?            │  pinned (2)    │
│   312 ch.   │                                              │   src/auth.py  │
│   refs: 1   │  ⟢ retrieved 5 chunks · 124ms                │   src/mw.ts    │
│             │     src/middleware/auth.ts:1-48    .82       │                │
│   other-p   │     src/middleware/auth.ts:48-94   .79       │  references    │
│   1.1k ch.  │     src/lib/jwt.ts:12-44           .71       │   api-spec     │
│             │     ...                                      │                │
│   [+ index] │                                              │  notes         │
│             │  ▸ The auth middleware verifies JWT tokens   │   React+Vite,  │
│             │    on every request to /api/*. It reads      │   snake_case   │
│             │    cookies via getToken() and...             │                │
│             │                                              │                │
│             │  ─ tool call: read_file(src/middleware/...)  │  git           │
│             │    230 lines · ok                            │   ● main       │
│             │                                              │   2 dirty      │
│             │  > add a docstring to verifyToken            │                │
│             │                                              │                │
│             │  ✎ wants to edit src/middleware/auth.ts      │                │
│             │  ─────────────────────────────────────────── │                │
│             │  - export function verifyToken(req) {        │                │
│             │  + /** Verify a JWT from req cookies. */     │                │
│             │  + export function verifyToken(req) {        │                │
│             │  [a]pprove  [r]eject  [e]dit  [d]iff full    │                │
│             │                                              │                │
├─────────────┴──────────────────────────────────────────────┴────────────────┤
│ > _                                                                          │ ← composer
├──────────────────────────────────────────────────────────────────────────────┤
│ ^P projects  ^R retrieval  ^L audit  ^K cmd  / slash  ?  help          ⌥+↵   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Components

- **Top bar / shield**: network state, model, mode flags, chunk count. Color of the lead dot encodes posture (`--ok`/`--warn`/`--danger`).
- **Project pane** (left, collapsible with `^P`): active project highlighted; chunks/refs/last-used metadata. `n` to index a new path.
- **Conversation pane** (center, primary): scrollable transcript. Each assistant turn is composed of cards:
  - `⟢ retrieved` — collapsible chunk list with scores. Press `r` to expand the focused turn's retrieval inspector full-screen.
  - `▸ answer` — model output, syntax-highlighted code.
  - `─ tool call` — purple-tinted, dim until expanded.
  - `✎ proposed edit` — yellow border, inline diff (first 8 lines each side), keyboard approvals.
- **Context pane** (right, collapsible): pinned files, references, notes (first 6 lines preview), git status. Click any pin to open a chunk view.
- **Composer**: multi-line input with `⌥+↵` to send, `↵` for newline. Typing `/` opens an inline slash-command picker (`:add`, `:diff`, etc.) with descriptions, the same names as today's CLI so muscle memory carries.
- **Status / hint bar**: rotating contextual hints; key bindings.

### Keyboard surface (TUI)

| Key | Action |
|---|---|
| `⌥+↵` | Send |
| `^P` | Toggle projects pane |
| `^R` | Toggle retrieval inspector for focused turn |
| `^L` | Open audit log overlay |
| `^K` | Command palette (everything you can do, fuzzy-searchable) |
| `/` (in composer) | Slash-command picker |
| `j`/`k` | Walk turns; arrow keys also work |
| `a`/`r`/`e`/`d` | Approve / reject / edit / show full diff on a pending tool card |
| `g d` | Show pending diff (mirrors `:diff`) |
| `g c` | Commit (mirrors `:commit`) |
| `?` | Help overlay |

### What stays line-oriented

Non-TTY invocation (piping into something) preserves the existing line-oriented output verbatim. `codebase-rag search`, `stats`, `audit` keep emitting plain text. The TUI is for `chat` only.

---

## 5. The browser app

A web app served by a local Python process (`codebase-rag serve`) on `127.0.0.1:8723` (or a chosen port). The UI talks to the same Python core through a small JSON-RPC layer; no separate backend.

### Why a browser version

- A non-trivial fraction of devs prefer reading diffs and chunk listings in a real browser.
- Multi-pane layouts and rich diff views are easier in HTML than in a terminal.
- Sharing a stable URL for an audit-log view ("look at what the agent did") is genuinely useful.

### Layout

Three columns, mirroring the TUI exactly.

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│  ● Local · mistral-nemo · read+write · shell:off · web:off  | shield                │ ← top bar
├─────────────┬────────────────────────────────────────────────────┬─────────────────┤
│  Projects   │  Conversation                                     │  Context        │
│  (sidebar)  │  (chat + inline retrieval + diffs)                │  pins, refs,    │
│             │                                                    │  notes, git     │
├─────────────┴────────────────────────────────────────────────────┴─────────────────┤
│  Composer (slash-aware textarea, attach-pin button)                                │
├───────────────────────────────────────────────────────────────────────────────────┤
│  Status hints · last write 2m ago · ⌘K command palette                            │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### Top-level views (left rail)

- **Chat** — primary
- **Search** — semantic-search workbench, equivalent to `codebase-rag search`
- **Index** — list/manage indexed projects, references, excludes, reindex
- **Audit** — filterable timeline of every tool call and slash command, per project
- **Settings** — model, provider, shell sandbox, web allowlist, theme

### Conversation surfaces (the meat)

Each assistant turn is a stack of cards. They are visually distinct, collapsible, and addressable (each card has a stable hash you can link to from the audit log).

- **Retrieval card** (top of turn): a horizontal list of chunk pills (`src/auth.ts:1-48 · 0.82`) — click any to open a side-by-side viewer showing the chunk in its file. A toggle reveals the full ranked list and lets you re-rank by file or score.
- **Answer card**: the model's prose and code. Code blocks have copy + open-in-editor buttons. Inline references to retrieved chunks are hyperlinked back to the retrieval card.
- **Tool-call card**: title (`read_file src/auth.ts`), latency, status, expand to see arguments and trimmed result. Web-search and web-fetch cards show the host + a warning if the host is outside the allowlist.
- **Proposed-edit card**: full file diff inline (Monaco-style), with **Approve**, **Reject**, **Open in editor**, and a "Why?" affordance that shows the retrieval that led to this edit. Approving stages the change and triggers the reindex of that file (mirroring current behavior).
- **Shell-call card**: command, runner (host vs docker:image), stdout/stderr, exit code. If `--shell-runner docker:...`, the card surfaces network mode and read-only rootfs status.

### Retrieval inspector (full view)

`Cmd+R` (or click the retrieval card header) opens a dedicated full-width pane that pins to the focused turn:

- Left: ranked chunk list with scores, file paths, source (project vs. reference set), and a tiny score sparkline.
- Right: the highest-ranked chunk rendered in its file, with adjacent chunks dimmed and other matches highlighted.
- A small toolbar: filter by file glob, hide chunks below a score threshold, jump to chunk in editor.

### Safety/shield panel (clickable from top bar)

When opened, this panel shows:

- **What's leaving your machine right now?** Live list of network destinations the current session has used or is configured to use.
- Toggles for `--allow-web`, `--allow-shell`, `--read-only`, `--confirm-writes`, `--provider`.
- The current web-allow / web-block lists, editable inline.
- Last 5 audit events as a preview; "Open audit log" link.

### Composer

A textarea with slash-command autocomplete. Typing `:` opens a picker with the same commands as the TUI. A small chip strip above the textarea shows current pins (click to drop). A keyboard-shortcut button reveals the full palette (`Cmd+K`).

### Audit view

A filterable timeline. Per-event row: timestamp, kind (`tool_call` / `slash_command` / `tool_result` / `session_*`), tool name, project, summary. Click to expand the full JSON. Filter chips: project, session, tool, event type, since.

This is a *real* page, deeply linkable, exportable to NDJSON (the existing format). It's the single most valuable artifact for explaining what an agent did.

---

## 6. The desktop app (deltas only)

Desktop is the browser app wrapped in **Tauri** (preferred over Electron — smaller, faster, fewer Node deps, ships a single binary). Tauri talks to the same local Python process over an `IPC` bridge. The UI bundle is shared with the browser version; the deltas below are what only the desktop can offer.

### Native-only features

- **Global hotkey** (default `⌥⇧Space`) — opens a floating "ask anything" command bar over whatever's frontmost. Type a question, get an answer, dismiss. The current project is inferred from the frontmost folder/Finder window or last-opened editor; user can override.
- **Menu bar item** — running status, current project, "open chat", "audit log", "quit". Click to peek at the last 3 events.
- **System notifications** — for long-running indexes ("indexed 4,231 chunks of `my-app`"), denied tool calls, completed multi-step edits.
- **File watcher integration** — when the user edits a file outside the agent, the indexer hot-updates that chunk. Status bar shows `re-indexing src/auth.ts...` for visibility.
- **OS-native file pickers** for "index a directory" / "add reference docs" — replaces the form input on the web side.
- **Deep links** — `codebase-rag://project/abc123/audit/event/...` lets the menu bar and notifications jump straight into the right view.
- **Permissions UI** — first run prompts (with platform-appropriate copy) for Disk Access (to read files), Network (for `--allow-web`/Anthropic), and Notifications. These map directly to the existing `--read-only`, `--allow-web`, `--allow-shell` flags so the desktop's permission UI and the CLI's flag set never diverge.
- **Auto-start option** — toggleable in settings; off by default.
- **Dock badge** — pending approvals count (number of writes/edits awaiting decision).

### What desktop deliberately does NOT add

- No separate code editor. We open the user's editor (`$EDITOR`, VS Code, etc.) via deep link/CLI integration.
- No model hosting in the app itself — Ollama still runs separately.
- No telemetry pipeline. The product story is local-first; we don't betray it for usage stats.

---

## 7. Key flows

### Flow A: First-time index → first chat

1. User runs `codebase-rag` with no args (or opens the desktop app for the first time).
2. Empty state: "No projects indexed yet. Pick a folder." Native file picker (desktop) or path input (browser).
3. Indexing screen shows live progress: files walked, chunks written, estimated remaining time. Cancelable.
4. On finish, project lands in the switcher, chat opens with a starter prompt ("Try: how does this codebase handle X?").

### Flow B: Ask a question, get an answer, inspect retrieval

1. User types a question. Hit send.
2. Retrieval card streams in first (within ~200ms): "Retrieved 5 chunks". Cards expand on hover.
3. Answer streams next. Inline citations link back to retrieval pills.
4. If retrieval looks wrong, user hits `^R` (or clicks the card) — full retrieval inspector opens.
5. User can re-run the same query with a `--file` filter from the inspector toolbar (sends a `:search ...` slash command equivalent).

### Flow C: Agent proposes a multi-file edit

1. User asks for a change spanning several files.
2. Assistant streams reasoning, then queues N proposed edits — each its own card.
3. Cards arrive yellow-bordered (`pending`). The status bar shows `3 edits awaiting approval` and the desktop dock badges 3.
4. User reviews each (full diff, "why?" shows the retrieval that informed it). Approves 2, rejects 1.
5. Approved cards turn green. The model is told "rejected edit to X — try a different approach" without the user retyping.
6. Touched files are re-indexed in the background; status bar shows the heartbeat.

### Flow D: Privacy posture changes mid-session

1. User clicks the shield (top bar). Panel opens showing **Local only**.
2. User toggles `--allow-web`. Confirm dialog: "This will let the agent fetch from any host. Limit to:" — text input prefilled with their `--web-allow` setting if any.
3. Shield dot turns yellow. Top bar reflows: `● + Web · mistral-nemo · ...`.
4. Audit log writes `posture_change` event.
5. If the agent attempts to `web_fetch` an out-of-allowlist host, the tool card renders red with the reason ("host blocked by --web-block 'evil.example.com'"), audit logs the denial.

### Flow E: Inspecting "what did the agent do today"

1. User opens the audit view.
2. Default filter: today, current project.
3. Timeline groups events into sessions. Each session has a header (start time, model, posture).
4. Filter chips: tool=edit_file → only file edits shown. Each row links to the conversation turn that produced it.
5. Export NDJSON for the day (already supported by the existing format).

---

## 8. What's deliberately NOT in v1

- **Multi-user / team mode.** The product is single-user, local-first. Multi-user changes the threat model fundamentally; it's a separate product.
- **In-app code editor.** Tempting; out of scope. We integrate with the user's editor.
- **Telemetry / analytics.** Hard no, see principles.
- **Marketplace of prompts / agents.** Distracts from the core RAG-over-my-code loop.
- **Cloud sync of projects/audit logs.** All artifacts stay on disk; sync would require a server we don't want to run.
- **Streaming Anthropic-style tool-use UI for non-Anthropic providers.** Ollama tool calls aren't streamed in the same way; we make the UI honest about that rather than faking it.

---

## Appendix: surface comparison at a glance

| Capability | TUI | Browser | Desktop |
|---|---|---|---|
| Chat + retrieval + diffs | ✓ | ✓ | ✓ |
| Slash commands | ✓ | ✓ | ✓ |
| Project switcher | ✓ | ✓ | ✓ |
| Audit timeline | ✓ (overlay) | ✓ (full page) | ✓ (full page) |
| Retrieval inspector | ✓ (overlay) | ✓ (full pane) | ✓ (full pane) |
| Inline diff with approve/reject | ✓ (line) | ✓ (Monaco) | ✓ (Monaco) |
| Shield / posture panel | ✓ (overlay) | ✓ (drop-down) | ✓ (drop-down + menu bar) |
| Global hotkey | — | — | ✓ |
| Menu bar item | — | — | ✓ |
| OS notifications | — | (web push, declined) | ✓ |
| File watcher hot-reindex | — | (via background server) | ✓ (surfaced) |
| Deep links | — | URLs | ✓ (`codebase-rag://`) |
