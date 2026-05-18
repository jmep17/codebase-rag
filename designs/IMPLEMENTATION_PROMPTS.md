# Implementation prompts for Claude Code

Six prompts to ship the redesign in `designs/{cli,browser,desktop}.html` against the IA in `design/SPEC.md`. One prompt per session, one branch per prompt. Run the AST smoke test from `CLAUDE.md` after each.

---

## Prompt 1 — TUI scaffold (mock data)

```
We're implementing the TUI redesign at designs/cli.html. The IA is in
design/SPEC.md §2 and §4. Read both before doing anything.

Goal: a Textual app at codebase_rag/tui.py that renders the three-pane
chat layout from designs/cli.html with mock data only. No model calls,
no retrieval, no Ollama in this PR — that's the next session.

Scope:
- Add `textual` as an optional dep under
  [project.optional-dependencies].tui in pyproject.toml. Justify briefly
  in CLAUDE.md (rich is already a transitive dep via chromadb, textual
  adds ~5MB).
- New file codebase_rag/tui.py. Class CodebaseRagApp(App). Compose: top
  shield bar, three columns (projects/conversation/context), composer,
  hint bar — match the layout in designs/cli.html, scene "chat".
- Map the design tokens from cli.html into a Textual CSS file at
  codebase_rag/tui.tcss. Truecolor for the accent green; rely on
  Textual's default 16-color fallback.
- New CLI flag in codebase_rag/__main__.py: `codebase-rag chat --tui`.
  When --tui is set, dispatch to tui.py; otherwise the existing line-
  oriented agent_loop runs unchanged.
- Render hardcoded mock data: one active project (codebase-rag), two
  sessions, the demo conversation turn from cli.html (user message →
  retrieval card → answer → read_file tool card). Pinned files and git
  status in the right rail.
- Keyboard: ^P toggles projects pane, ^R toggles context pane (right),
  :q or ctrl+c quits.

Do NOT touch:
- codebase_rag/chat.py agent_loop (still the default; --tui is opt-in)
- The existing test suite or smoke tests

Acceptance:
- `.venv/bin/codebase-rag chat --tui` launches the TUI from this repo
- All three panes render with mock content matching cli.html scene "chat"
- ^P, ^R, :q work
- Existing `.venv/bin/codebase-rag chat` works exactly as before
- The AST smoke test from CLAUDE.md passes
- pyproject.toml + CLAUDE.md mention the new textual extra

Read first: design/SPEC.md (§2 and §4), designs/cli.html (focus on the
chat scene markup), codebase_rag/__main__.py (where the chat subcommand
is wired up), codebase_rag/chat.py (to see what's there — don't refactor it).
```

---

## Prompt 2 — TUI wiring + streaming (real agent loop)

```
TUI scaffold from the previous PR is at codebase_rag/tui.py. Now wire
it to the real agent loop.

Read design/SPEC.md §4 and codebase_rag/chat.py agent_loop end-to-end
before starting.

Goal: replace the mock data with live retrieval, streaming model
responses, and tool execution. Refactor agent_loop minimally so its
core (retrieve → stream → execute tools → save) can drive either the
line-oriented sink or the TUI sink.

Approach (this is the hard call to get right):
- Extract a pure AgentTurn coroutine from agent_loop that yields events:
  ("retrieved", chunks), ("token", str), ("tool_call_request", call),
  ("tool_result", result), ("turn_done", stats). The line-oriented loop
  becomes one consumer; the TUI becomes another.
- The TUI runs the turn on Textual's worker thread and renders events
  reactively. Streaming tokens append to the current assistant Static.
- Confirmation flow (_confirm_write, _confirm_shell) becomes a protocol:
  the coroutine yields ("confirm", tname, args) and waits for the
  consumer to send back resolved args or None. Line loop keeps using
  input(); TUI shows a Textual ModalScreen matching designs/cli.html
  scenes "confirm-write" and "confirm-shell" with a/r/e/d keys.
- Pinning, project notes, git status, audit logging: unchanged — read
  from the same places agent_loop already reads.

Do NOT touch:
- The audit event format. Same names, same fields.
- The Anthropic/Ollama provider abstraction.
- The slash-command dispatch table — TUI keybinds come in the next PR.

Acceptance:
- `.venv/bin/codebase-rag chat --tui` runs a real session against local
  Ollama
- Retrieval card populates from real chunks (not mock)
- Tokens stream live into the assistant card
- An edit_file tool call pops the modal from scene "confirm-write" with
  the actual diff; approving runs it, declining returns the same
  {ok:false} payload as today
- The existing line-oriented `chat` still works identically — same
  prompts, same output, byte-for-byte where possible
- audit.log_event is called for exactly the same events as before
  (verify with `codebase-rag audit`)
- AST smoke test passes

Read first: design/SPEC.md §4, designs/cli.html (scenes chat /
confirm-write / confirm-shell), codebase_rag/chat.py (entire file),
codebase_rag/audit.py (to make sure event shapes don't drift).
```

---

## Prompt 3 — TUI overlays (palette, command palette, inspector, audit)

```
Add the overlays from designs/cli.html that didn't make the previous
PR: slash-command palette, command palette, retrieval inspector, audit
overlay.

Read design/SPEC.md §4 "Keyboard surface" and designs/cli.html scenes
"palette" and "audit".

Scope:
- `/` inside the composer opens a slash palette (Textual ModalScreen).
  Filterable. Same commands as today's chat.py (:add, :drop, :pinned,
  :search, :fetch, :run, :gitstatus, :diff, :commit, :undo, :reset,
  :forget). Show the description and required flags (--allow-web, etc.)
  for each. Enter runs the command via the same dispatch path the line
  loop uses.
- ^K opens a command palette — superset of slash, plus "switch project",
  "toggle read-only", "toggle confirm-writes", "open audit", "open
  retrieval inspector". Fuzzy filter on name + description.
- ^R opens the retrieval inspector for the focused turn: full chunk
  list with scores, file paths, source (project vs reference), and the
  highest-scoring chunk rendered in its file.
- ^L opens the audit overlay: scrollable timeline of the current
  session's events. Filter chips for tool / event-kind. Wire to
  codebase_rag/audit.py's existing reader.

Constraints:
- All overlays dismiss with Esc and trap focus while open (Textual
  ModalScreen handles this).
- Slash and command palette share one component, two configs.
- No new dependencies.

Acceptance:
- All four overlays open and dismiss with the documented keys
- Slash palette runs `:add codebase_rag/chat.py` and pins the file
  (verify in the right rail)
- ^K → "toggle read-only" flips the session flag and the top-bar
  shield updates within one render frame
- ^R shows the actual top-ranked chunk from the focused turn, not mock
- ^L pages through real audit events from this session

Read first: codebase_rag/chat.py slash-command handlers, codebase_rag/
audit.py reader functions, designs/cli.html scenes "palette" and "audit".
```

---

## Prompt 4 — `codebase-rag serve` (backend)

```
Add a `codebase-rag serve` subcommand: a local HTTP/WebSocket server
that exposes the same core to the browser app. Read design/SPEC.md §5
before doing anything.

Scope:
- New optional extra `serve` in pyproject.toml: starlette, uvicorn,
  websockets. Justify in CLAUDE.md.
- New module codebase_rag/serve.py. ASGI app. Routes:
  GET  /api/projects                → list indexed projects
  GET  /api/projects/{slug}/sessions → list saved sessions
  GET  /api/projects/{slug}/audit    → tail audit log; same filter
                                        params as the `audit` CLI cmd
  POST /api/projects/{slug}/index    → kick off index/reindex, streams
                                        progress events via SSE
  GET  /api/search?q=...&project=... → semantic search, same as
                                        `codebase-rag search`
  GET  /api/file?path=...            → read file via tools.resolve_safe
                                        (re-uses existing path safety)
  WS   /api/chat                     → bidirectional. Client sends user
                                        message + flags; server runs
                                        AgentTurn (from prompt 2) and
                                        forwards every event.
- Default bind: 127.0.0.1:8723. Refuse non-loopback unless --host is
  explicit. Print the URL on start.
- Auth: a per-process random token written to
  ~/.codebase-rag/meta/<sha>/serve.token. Required as
  `Authorization: Bearer <token>` on every request. The desktop wrapper
  reads the same file.
- Startup banner identical in spirit to the Anthropic-provider warning
  in chat.py: "serve binds 127.0.0.1:8723 — anyone with the token can
  read this codebase".

Do NOT touch:
- codebase_rag/index.py, tools.py, providers.py beyond imports. Serve
  is a transport over existing primitives.
- The CLI flag names or audit event format.

Acceptance:
- `.venv/bin/codebase-rag serve` starts on 127.0.0.1:8723, prints URL
  + token
- `curl -H "Authorization: Bearer $TOKEN" .../api/projects` returns the
  same list as `codebase-rag stats`
- A websocket client (test with `websockets` lib) can send a chat
  message and receive an event stream including retrieved chunks,
  streamed tokens, and turn_done
- Requests without the token return 401 with a clear error
- `--host 0.0.0.0` prints a louder warning before binding
- AST smoke test passes

Read first: design/SPEC.md §5, codebase_rag/__main__.py, codebase_rag/
chat.py (the AgentTurn extracted in prompt 2), codebase_rag/audit.py.
```

---

## Prompt 5 — Browser SPA frontend

```
Build the browser app frontend. Read design/SPEC.md §5 and
designs/browser.html (every scene) before starting.

Scope:
- New top-level directory `web/`. Vite + React + TypeScript + Tailwind.
  No shadcn (tokens already defined in designs/browser.html — port to
  Tailwind theme.colors). No Redux/Zustand — useReducer for chat state.
- Pages, matching designs/browser.html scenes:
    /projects/:slug/chat    → conversation
    /projects/:slug/index   → indexing
    /projects/:slug/search  → semantic search workbench
    /projects/:slug/audit   → audit timeline
    /settings               → model, provider, shield toggles
- All data via the API from prompt 4. Token read once from the URL
  (?token=…) then held in memory; never persisted to localStorage.
- Streaming chat: open the websocket from /api/chat, dispatch every
  event into the reducer. Render tool-call cards, edit/shell
  confirmation cards (Approve/Decline → callback over the socket),
  retrieval pill rows. Tokens append in place.
- ⌘K command palette — same items as the TUI's (prompt 3).
- Shield drawer (right-slide-over) matches designs/browser.html scene
  "shield".
- Build output → `web/dist/`. Add a `codebase-rag serve --static`
  flag in the backend that serves dist/ at /; dev mode keeps Vite on
  a separate port with CORS allow-listing 127.0.0.1.

Constraints:
- Mono (JetBrains Mono) for conversation body, sans (Inter) for chrome.
- Phosphor-green palette ported verbatim from designs/browser.html.
  Don't invent new colors.
- All interactive surfaces keyboard-accessible (Tab + arrow keys on
  gate switches and tabs).
- No localStorage / sessionStorage except theme preference.

Acceptance:
- `cd web && npm run dev` opens a working chat UI against a running
  `codebase-rag serve`
- End-to-end: send → retrieval pills → streaming answer → edit
  confirmation card with Approve/Decline that round-trips
- Indexing page shows the same projects as the CLI's `stats`
- Audit page filters and renders real events
- Shield drawer toggles POST back to the server
- `codebase-rag serve --static` serves the built bundle at /

Read first: designs/browser.html (port CSS tokens to Tailwind config),
design/SPEC.md §5, your prompt-4 event shape (or read serve.py).
```

---

## Prompt 6 — Tauri desktop wrapper

```
Wrap the browser app in Tauri. Read design/SPEC.md §6 and
designs/desktop.html (all four scenes) before starting.

Scope:
- New top-level directory `desktop/`. Tauri 2.x scaffold. Frontend =
  the built `web/dist/` from prompt 5.
- On launch, desktop binary spawns `codebase-rag serve --quiet
  --token-file <tmp>` as a sidecar, reads the token, points the
  webview at http://127.0.0.1:8723/?token=…
- Native features (the deltas from designs/desktop.html):
  - Global hotkey ⌥⇧Space → small "ask anywhere" window. Frontmost-
    folder detection sets the project. Use
    tauri-plugin-global-shortcut.
  - Menu-bar tray with the peek panel from scene "tray". Use
    tauri-plugin-tray.
  - Multi-tab sessions via the tab strip — frontend-only; each tab is
    a websocket session.
  - System notifications on: index complete, edit_file pending
    approval > 30s old, shell decline.
  - File watcher: tauri-plugin-fs-watch on the project root, calls
    /api/projects/:slug/reindex?path=… for changed files.
  - First-run permission dialogs map to existing CLI flags (Disk
    access stays available; --allow-shell and --allow-web remain
    opt-in even after permission grants).

Do NOT touch:
- The web frontend's React code beyond a thin `tauri.ts` shim that
  feature-detects window.__TAURI__. The same bundle must still work
  in a plain browser.
- The Python backend.

Acceptance:
- `cd desktop && pnpm tauri dev` launches the app
- App spawns `codebase-rag serve` automatically, kills it on quit
- ⌥⇧Space opens the hotkey window even when the app isn't focused
- Tray peek shows pending approvals + last events
- Editing a tracked file outside the app triggers a re-index event
  within 2s
- `pnpm tauri build` produces a single signed .app / .dmg

Read first: design/SPEC.md §6, designs/desktop.html (all scenes),
tauri.app/v2 docs for global-shortcut + tray, your prompt-5 output.
```

---

## How to use these

1. `git checkout -b tui-scaffold` (one branch per prompt).
2. `cd ~/code/codebase-rag && claude`.
3. Paste prompt 1 verbatim. Let it run. Review the diff before letting it commit.
4. Smoke test (`pip install -e .[tui]`, run the app, run the AST check).
5. Merge or stash, branch again, paste prompt 2.

If a prompt produces too much in one session, ask Claude Code to split: "Stop after scope item 2; I'll come back for the rest." The prompts are sized to fit one session each on Sonnet, but the TUI wiring (prompt 2) is the largest and may need splitting.
