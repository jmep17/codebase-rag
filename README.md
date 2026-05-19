# codebase-rag

Local codebase RAG with Ollama. Index your code, ask questions, get answers grounded in actual files — no cloud, no API keys.

- **Embeddings:** `nomic-embed-text` (via Ollama)
- **Chat:** `mistral-nemo` (via Ollama)
- **Vector store:** ChromaDB (persistent, on-disk)
- **Everything local by default.** Nothing leaves your machine unless you opt in to a network feature.

### Security & privacy at a glance

| Feature | Network surface added | Default |
|---|---|---|
| Default `chat` | Localhost only (Ollama daemon at :11434) | On |
| `--allow-web` | SearXNG (your own host) + `web_fetch` destinations | Off |
| `--provider anthropic` | api.anthropic.com (chat only; embeddings stay local) | Off |
| `--allow-shell` (host) | None | Off; commands run on host |
| `--allow-shell --shell-runner docker:IMAGE` | Only what you allow via `--shell-network` (default `none`) | Off |
| `train` | None; writes local artifacts and can call local Ollama with `--create` | On demand |

Defenses against prompt injection in retrieved content / tool results:

- Retrieved chunks, file contents, web pages, and shell output are wrapped in `<<<UNTRUSTED-...>>>` markers and the system prompt tells the model to treat anything between them as data, not instructions.
- `--read-only` removes create/write/edit/shell tools from the model's schema entirely.
- By default, every model-driven `create_project`, `write_file`, or `edit_file` prompts for approval before it executes. Use `--no-confirm-writes` only if you want writes/edits auto-applied.
- `--web-allow` / `--web-block` cap which hosts `web_fetch` may contact (post-redirect host re-checked).
- Every tool call and slash command is recorded in a per-project `audit.log` (`codebase-rag audit` to view).

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com).

```bash
ollama pull mistral-nemo
ollama pull nomic-embed-text

pip install -e .
```

To make `codebase-rag` available from any directory without activating the
virtual environment, install the global wrapper:

```bash
make install-global
codebase-rag --help
```

This creates `.venv` if needed, installs the package editable into that venv,
then writes a tiny executable wrapper to `~/.local/bin/codebase-rag`. Make sure
`~/.local/bin` is on your `PATH`; the Makefile prints a reminder if it is not.
Install optional extras the same way:

```bash
make install-global EXTRAS=web,serve,tui
```

### Install troubleshooting

If `pip install -e .` fails after activating a virtual environment, first make sure `pip` belongs to that environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip --version
python -m pip install -e .
```

Using `python -m pip` avoids accidentally calling a system `pip` or a `pip` from another environment. The version output should point somewhere under this repo's `.venv/`.

If editable installs are blocked or unsupported, a normal install is fine:

```bash
python -m pip install .
```

If dependency downloads are blocked by your network or package policy, install the runtime dependencies from your approved package source first, then install this package without resolving dependencies:

```bash
python -m pip install ollama chromadb
python -m pip install -e . --no-deps
```

For Python environments that report an "externally managed environment" error, create and activate a virtual environment instead of installing into the system Python. `codebase-rag` requires Python 3.10+.

## Development workflow

Install the development extra to get the lint and format tooling:

```bash
python -m pip install -e .[dev]
```

Enable the committed git hooks in this checkout:

```bash
python scripts/install_hooks.py
```

The hooks enforce:

- Conventional Commit headers, e.g. `feat(tui): add retrieval help overlay`.
- `ruff format --check`.
- `ruff check`.
- The same Python AST smoke check used during local development.

Run checks manually:

```bash
python scripts/check_quality.py        # check only
python scripts/check_quality.py --fix  # format and apply safe lint fixes
```

Commit messages must use:

```text
<type>[optional scope][!]: <description>
```

Allowed types are `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`,
`refactor`, `revert`, `style`, and `test`.

## Usage

Index a codebase:

```bash
codebase-rag index ~/code/my-project
```

This walks the directory, chunks each source file, embeds the chunks, and stores them in `~/.codebase-rag/db` by default. Re-running is safe — chunks are upserted by `path:line-range`, so the index updates incrementally.

If files have been **deleted** or **renamed** in the source tree, `index` leaves stale chunks behind. Run `reindex` instead to wipe and rebuild:

```bash
codebase-rag reindex ~/code/my-project
```

### Excluding files

Nested git repositories (e.g. cloned dependencies sitting inside your project) are detected automatically — any directory below the project root that contains its own `.git/` is treated as a nested repo and its whole subtree is skipped. The indexer prints which nested repos it found at startup.

Skip additional files or patterns via `--exclude` (repeatable, glob syntax). Patterns match against the relative path and the bare filename:

```bash
codebase-rag index ~/code/my-project \
    --exclude '*.test.js' \
    --exclude 'tests/*' \
    --exclude 'secret_*.py'
```

Or drop a `.codebaseragignore` file in the project root — one pattern per line, `#` for comments:

```
# don't index test data or anything generated
fixtures/*
**/*.generated.ts
notes/scratch.md
```

Files already in the index that are now excluded stay there until you `reindex` (cheap; just leaves dead chunks). Common junk (`.git`, `node_modules`, build artifacts, binaries, source maps, minified bundles) is already excluded by default — `--exclude` is for project-specific additions.

### Inspecting the index

List every project you've indexed:

```bash
codebase-rag stats
# Database: /Users/you/.codebase-rag/db  (38.4 MB on disk, 2 project(s))
#
#   /Users/you/code/project-a
#     chunks: 4231, files: 812, collection: cbr_you_project_a_4f8a7b3c1d9e
#   /Users/you/code/project-b
#     chunks: 1106, files: 245, collection: cbr_you_project_b_a2c98e110f3a
```

Detail one project:

```bash
codebase-rag stats --root ~/code/project-a
# Project: /Users/you/code/project-a
# Chunks:  4231
# Files:   812
#
# Largest files by chunk count:
#     47  src/parser.ts
#     33  packages/core/state.ts
#     ...
```

Run a one-shot retrieval query (what chat would see for that question) without burning chat tokens. Full chunk content is printed by default:

```bash
codebase-rag search "where is the database connection set up"

# Filter to a specific file or glob:
codebase-rag search "where is auth checked" --file 'src/middleware/*'

# Just headers, no chunk content (quick scan):
codebase-rag search "auth" --headers-only --top-k 20
```

Print every chunk for a specific file (no query — just dump what's indexed for that path):

```bash
codebase-rag show src/auth.py
codebase-rag show 'src/*.py'       # glob is fine
```

`--top-k` (default 5), `--db <path>` for non-default index locations.

Chat:

```bash
cd ~/code/my-project
codebase-rag chat
> how does the auth middleware work?
> add a docstring to the parse_token function
> :reset                 # clears conversation history
> :q                     # quit
```

Each indexed project lives in its own ChromaDB collection, keyed by the absolute path you indexed. `chat` (and `search`/`show`) defaults the scope to the current working directory — `cd ~/code/project-a && codebase-rag chat` only retrieves chunks from project-a, never from any other project you've indexed. Use `--root <path>` to talk to a different project's index from somewhere else.

## CLI flag reference

Most commands accept either a project root (`--root` / `--for-project`) or database path (`--db`). The default database is `~/.codebase-rag/db`; project-specific metadata such as notes, IDE diagnostics, conversations, web cache, and audit logs lives under `~/.codebase-rag/meta/<project-hash>/`.

### Indexing and retrieval

| Command | Flag | What it does | Example |
|---|---|---|---|
| `index PATH` | `--db PATH` | Store vectors in a non-default ChromaDB directory. | `codebase-rag index ~/code/app --db ~/.cache/cbr/app-db` |
| `index PATH` | `--exclude GLOB`, `-x GLOB` | Skip files matching a glob; repeatable. | `codebase-rag index ~/code/app -x 'fixtures/*' -x '*.generated.ts'` |
| `reindex PATH` | `--db PATH` | Rebuild the project index in a non-default DB. | `codebase-rag reindex ~/code/app --db ~/.cache/cbr/app-db` |
| `reindex PATH` | `--exclude GLOB`, `-x GLOB` | Skip files during rebuild; repeatable. | `codebase-rag reindex ~/code/app --exclude 'dist/*'` |
| `stats` | `--db PATH` | Inspect a non-default DB. | `codebase-rag stats --db ~/.cache/cbr/app-db` |
| `stats` | `--root PATH` | Show details for one indexed project instead of listing all projects. | `codebase-rag stats --root ~/code/app` |
| `search QUERY` | `--db PATH` | Search a non-default DB. | `codebase-rag search 'auth middleware' --db ~/.cache/cbr/app-db` |
| `search QUERY` | `--root PATH` | Search a project other than the current directory. | `codebase-rag search 'database setup' --root ~/code/app` |
| `search QUERY` | `--top-k N`, `-k N` | Return `N` chunks instead of the default 5. | `codebase-rag search 'token validation' --top-k 12` |
| `search QUERY` | `--file GLOB`, `-f GLOB` | Restrict results to matching indexed paths. | `codebase-rag search 'auth' --file 'src/middleware/*'` |
| `search QUERY` | `--headers-only` | Print only file/line headers, not chunk text. | `codebase-rag search 'auth' --headers-only` |
| `show GLOB` | `--db PATH` | Read chunks from a non-default DB. | `codebase-rag show 'src/*.py' --db ~/.cache/cbr/app-db` |
| `show GLOB` | `--root PATH` | Show indexed chunks for a project other than the current directory. | `codebase-rag show 'src/auth.py' --root ~/code/app` |

### IDE diagnostics

`codebase-rag` can read a local cache of IDE/LSP "Problems" through the `get_diagnostics` tool. The cache lives outside the repo at `~/.codebase-rag/meta/<project-hash>/diagnostics.json`; nothing is written into the project.

Publish diagnostics from an editor bridge or script:

```bash
codebase-rag diagnostics --root ~/code/app --set diagnostics.json
# or
codebase-rag diagnostics --root ~/code/app --set - < diagnostics.json
```

The JSON may be a list, or an object with `diagnostics`, `items`, or `problems`:

```json
{
  "diagnostics": [
    {
      "uri": "file:///Users/you/code/app/src/main.ts",
      "range": { "start": [12, 4], "end": [12, 19] },
      "severity": "error",
      "source": "typescript",
      "code": "TS2322",
      "message": "Type 'string' is not assignable to type 'number'."
    }
  ]
}
```

View or clear the cache:

```bash
codebase-rag diagnostics --root ~/code/app
codebase-rag diagnostics --root ~/code/app --severity error --json
codebase-rag diagnostics --root ~/code/app --clear
```

When `codebase-rag serve` is running, an IDE extension can also `PUT` the same JSON to `/api/projects/<slug>/diagnostics` with `Authorization: Bearer <token>`. The VS Code APIs to feed this are `vscode.languages.getDiagnostics()` and `vscode.languages.onDidChangeDiagnostics(...)`.

### Project notes and references

| Command | Flag | What it does | Example |
|---|---|---|---|
| `notes` | `--root PATH` | Read or edit notes for a project other than the current directory. | `codebase-rag notes --root ~/code/app` |
| `notes` | `--edit` | Open the notes file in `$EDITOR` (`vi` fallback). | `codebase-rag notes --edit` |
| `notes` | `--set TEXT` | Replace notes with `TEXT`; use `-` to read from stdin. | `codebase-rag notes --set 'Django app; prefer small patches.'` |
| `notes` | `--append TEXT` | Append `TEXT`; use `-` to read from stdin. | `codebase-rag notes --append 'Avoid legacy/v1 unless asked.'` |
| `notes` | `--clear` | Delete the notes file. | `codebase-rag notes --clear` |
| `train` | `--root PATH` | Build assistant artifacts from a project other than the current directory. | `codebase-rag train --root ~/code/app` |
| `train` | `--profile TEXT` | Add personal preferences to the generated assistant profile; use `-` to read stdin. | `codebase-rag train --profile 'Prefer small, reviewed patches.'` |
| `train` | `--profile-file PATH` | Read personal preferences from a local file. | `codebase-rag train --profile-file ~/assistant-style.md` |
| `train` | `--output PATH` | Write artifacts somewhere other than the default project meta directory. | `codebase-rag train --output ~/.codebase-rag/my-assistant` |
| `train` | `--name NAME` | Name the generated Ollama model. | `codebase-rag train --name my-coder` |
| `train` | `--base-model NAME` | Use a different local Ollama base model in the Modelfile. | `codebase-rag train --base-model qwen2.5-coder:7b` |
| `train` | `--no-conversation` | Skip exporting examples from the last saved conversation. | `codebase-rag train --no-conversation` |
| `train` | `--max-examples N` | Limit exported assistant-response examples. | `codebase-rag train --max-examples 50` |
| `train` | `--create` | Run local `ollama create <name> -f Modelfile` after writing files. | `codebase-rag train --name my-coder --create` |
| `add-reference SOURCE` | `--label NAME` | Name the reference set; defaults to the source directory name. | `codebase-rag add-reference ~/docs/api --label api-spec` |
| `add-reference SOURCE` | `--for-project PATH` | Attach references to a project other than the current directory. | `codebase-rag add-reference ~/docs/api --for-project ~/code/app` |
| `add-reference SOURCE` | `--db PATH` | Store reference chunks in a non-default DB. | `codebase-rag add-reference ~/docs/api --db ~/.cache/cbr/app-db` |
| `add-reference SOURCE` | `--exclude GLOB`, `-x GLOB` | Skip matching files inside the reference source; repeatable. | `codebase-rag add-reference ~/docs/api -x 'archive/*'` |
| `remove-reference LABEL` | `--for-project PATH` | Remove the label from a project other than the current directory. | `codebase-rag remove-reference api-spec --for-project ~/code/app` |
| `remove-reference LABEL` | `--db PATH` | Remove reference chunks from a non-default DB. | `codebase-rag remove-reference api-spec --db ~/.cache/cbr/app-db` |

### Chat

| Flag | What it does | Example |
|---|---|---|
| `--db PATH` | Use a non-default index database. | `codebase-rag chat --db ~/.cache/cbr/app-db` |
| `--root PATH` | Project root the agent may retrieve from and use tools against. | `codebase-rag chat --root ~/code/app` |
| `--model NAME` | Ollama chat model; overrides `CODEBASE_RAG_CHAT_MODEL` and the default `mistral-nemo`. | `codebase-rag chat --model qwen3:8b` |
| `--show-context` | Print retrieved file paths and line ranges for each turn. | `codebase-rag chat --show-context` |
| `--verbose`, `-v` | Print timing/token stats and full tool results. | `codebase-rag chat --verbose` |
| `--resume` | Continue the last saved conversation for this project. | `codebase-rag chat --resume` |
| `--read-only` | Expose only read/grep tools; disables project creation, writes, edits, and shell. | `codebase-rag chat --read-only` |
| `--confirm-writes`, `--no-confirm-writes` | Toggle approval prompts for model-driven project creation, writes, and edits; default is on. | `codebase-rag chat --no-confirm-writes` |
| `--allow-shell` | Enable model-driven `run_shell` and user-driven `:run`; each command prompts before execution. | `codebase-rag chat --allow-shell` |
| `--shell-timeout SECONDS` | Per-command shell timeout; default 30. | `codebase-rag chat --allow-shell --shell-timeout 90` |
| `--shell-runner host\|docker:IMAGE` | Run shell commands on the host or in a transient Docker container. | `codebase-rag chat --allow-shell --shell-runner docker:python:3.13-slim` |
| `--shell-network none\|bridge\|host` | Docker network mode for shell commands; default `none`. | `codebase-rag chat --allow-shell --shell-runner docker:python:3.13-slim --shell-network none` |
| `--allow-web` | Enable SearXNG search and URL fetch tools; requires `SEARXNG_URL`. | `SEARXNG_URL=http://127.0.0.1:8080 codebase-rag chat --allow-web` |
| `--web-allow HOST_GLOB` | Allow `web_fetch` only for matching hosts; repeatable. | `codebase-rag chat --allow-web --web-allow docs.python.org` |
| `--web-block HOST_GLOB` | Block `web_fetch` for matching hosts; repeatable and takes precedence. | `codebase-rag chat --allow-web --web-block '*.social.example'` |
| `--architect-model NAME` | Use a planning model first, then let `--model` execute with tools. | `codebase-rag chat --architect-model qwen3:30b-a3b --model qwen2.5-coder:7b` |
| `--provider ollama\|anthropic` | Select chat provider; Anthropic is opt-in cloud and requires `[cloud]` plus `ANTHROPIC_API_KEY`. | `codebase-rag chat --provider anthropic --model claude-opus-4-7` |
| `--tui` | Launch the Textual interface; requires `pip install -e .[tui]`. | `codebase-rag chat --tui` |

### Audit

| Flag | What it does | Example |
|---|---|---|
| `--root PATH` | Read the audit log for a project other than the current directory. | `codebase-rag audit --root ~/code/app` |
| `--tool NAME` | Show only events for a tool such as `grep` or `edit_file`. | `codebase-rag audit --tool edit_file` |
| `--event NAME` | Show only one event type, such as `tool_call`, `tool_result`, `slash_command`, `session_start`, or `session_end`. | `codebase-rag audit --event tool_call` |
| `--since WHEN` | Filter by time, e.g. `today`, `yesterday`, `15m`, `2h`, `7d`, or an ISO date. | `codebase-rag audit --since 2h` |
| `--limit N` | Limit rows; default 50, `0` means all. | `codebase-rag audit --limit 100` |
| `--pretty` | Pretty-print each audit entry over multiple lines. | `codebase-rag audit --pretty` |

### Serve

`serve` starts the local HTTP/WebSocket backend. It requires `pip install -e '.[serve]'`.
If the editable install already exists and your package index blocks build dependencies, install the runtime pieces directly instead: `pip install starlette uvicorn websockets`.
Without `--static DIR`, `serve` runs the API only; use `--static web/dist` or another built SPA directory for the browser app.

| Flag | What it does | Example |
|---|---|---|
| `--host HOST` | Bind address; default `127.0.0.1` or `CODEBASE_RAG_SERVE_HOST`. Non-loopback hosts print a warning. | `codebase-rag serve --host 127.0.0.1` |
| `--port PORT` | Bind port; default `8723` or `CODEBASE_RAG_SERVE_PORT`. | `codebase-rag serve --port 9000` |
| `--db PATH` | Use a non-default index database. | `codebase-rag serve --db ~/.cache/cbr/app-db` |
| `--root PATH` | Default project root for browser/WebSocket sessions. | `codebase-rag serve --root ~/code/app` |
| `--static DIR` | Serve a built SPA bundle at `/`. | `codebase-rag serve --static web/dist` |
| `--reuse-token` | Keep the existing bearer token instead of rotating on start. | `codebase-rag serve --reuse-token` |
| `--token-file PATH` | Store/read the bearer token at a custom path. | `codebase-rag serve --token-file ~/.cache/cbr/serve.token` |
| `--quiet` | Suppress uvicorn access logs. | `codebase-rag serve --quiet` |
| `--model NAME` | Default chat model for WebSocket sessions. | `codebase-rag serve --model qwen3:8b` |
| `--provider ollama\|anthropic` | Default chat provider for WebSocket sessions. | `codebase-rag serve --provider anthropic --model claude-opus-4-7` |
| `--architect-model NAME` | Optional planning model for WebSocket sessions. | `codebase-rag serve --architect-model qwen3:30b-a3b --model qwen2.5-coder:7b` |
| `--read-only` | Default WebSocket chats to read-only tools. | `codebase-rag serve --read-only` |
| `--confirm-writes`, `--no-confirm-writes` | Toggle approval prompts for model-driven project creation, writes, and edits in WebSocket chats; default is on. | `codebase-rag serve --no-confirm-writes` |
| `--allow-shell` | Enable shell tools for WebSocket chats. | `codebase-rag serve --allow-shell` |
| `--shell-runner host\|docker:IMAGE` | Shell execution backend for WebSocket chats. | `codebase-rag serve --allow-shell --shell-runner docker:python:3.13-slim` |
| `--shell-network none\|bridge\|host` | Docker network mode for WebSocket shell commands. | `codebase-rag serve --allow-shell --shell-runner docker:python:3.13-slim --shell-network none` |
| `--shell-timeout SECONDS` | Per-command shell timeout for WebSocket chats. | `codebase-rag serve --allow-shell --shell-timeout 90` |
| `--allow-web` | Enable web tools for WebSocket chats; requires `SEARXNG_URL`. | `SEARXNG_URL=http://127.0.0.1:8080 codebase-rag serve --allow-web` |
| `--web-allow HOST_GLOB` | Allow matching `web_fetch` hosts; repeatable. | `codebase-rag serve --allow-web --web-allow docs.python.org` |
| `--web-block HOST_GLOB` | Block matching `web_fetch` hosts; repeatable and takes precedence. | `codebase-rag serve --allow-web --web-block '*.social.example'` |

### Project notes & reference docs

Two extra sources of context, both stored **outside the repo** so nothing extra ends up in your project:

**Project notes** — short-form context the model sees in every turn (stack, conventions, constraints, current focus). Auto-injected into the chat system prompt.

```bash
cd ~/code/my-project
codebase-rag notes --edit                            # opens $EDITOR (default vi)
codebase-rag notes --set "React + Vite app, Postgres, follow snake_case in DB."
codebase-rag notes --append "Avoid touching legacy/v1/ unless asked."
codebase-rag notes                                   # print current notes
codebase-rag notes --clear                           # delete
```

Notes live at `~/.codebase-rag/meta/<hash>/notes.md`, keyed by the absolute project path.

**Personal assistant training artifacts** — generate a local Ollama `Modelfile`
plus chat-style JSONL examples from the project's last saved conversation.
This is useful for making a project-specific coding assistant that remembers
your notes and personal preferences without sending anything to a hosted service.

```bash
cd ~/code/my-project
codebase-rag train --profile "Prefer small patches, cite files, keep summaries brief."

# optional: create the local Ollama model immediately
codebase-rag train --name my-project-coder --base-model qwen2.5-coder:7b --create

# then use it normally
codebase-rag chat --model my-project-coder
```

By default the generated files live at
`~/.codebase-rag/meta/<hash>/assistant_training/`, outside your repository.
`ollama create` personalizes the model recipe and system prompt; it does not
fine-tune model weights. The exported `training.jsonl` can be used later with a
separate local fine-tuning tool if you want weight training.

**Reference docs** — external documents indexed alongside the project for retrieval. Each reference set has a label, so the model sees project chunks and reference chunks separately in context.

```bash
cd ~/code/my-project
codebase-rag add-reference ~/Documents/api-docs --label api-spec
codebase-rag add-reference ~/Documents/dnd-kit-docs --label dnd-kit
codebase-rag remove-reference api-spec               # drop a reference set

# inspect what's loaded for this project
codebase-rag stats --root .
```

References are stored in the project's ChromaDB collection but tagged `kind=reference` with a label. During chat, retrieval pulls from both project code and reference docs; the model sees them in separate blocks:

```
## Project code
### src/auth.py:42-67
...

## Reference: api-spec
### oauth-flow.md:1-40
...
```

`codebase-rag reindex <project>` wipes only the project's chunks — your reference sets stay attached.

## Advanced chat flags

Web, shell, cloud provider, TUI, and architect mode are opt-in. Default `codebase-rag chat` keeps web/shell/cloud off and prompts before model-driven project creation, writes, and edits.

### Conversation continuity

```bash
codebase-rag chat --resume          # continue this project's last conversation
:reset                              # clear history, keep pins
:forget                             # clear history + delete saved conversation
```

Conversations auto-save per-project at `~/.codebase-rag/meta/<hash>/last_conversation.json` after every turn.

### Pinned files (`:add` / `:drop`)

Force specific files into every turn's context regardless of retrieval:

```
:add codebase_rag/chat.py
:add 'codebase_rag/*.py'            # globs work
:drop codebase_rag/chat.py
:dropall
:pinned                              # list current pins
```

Pins persist with the conversation.

### Git integration

When the project root is a git repo, slash commands let you review and commit the model's edits:

```
:gitstatus
:diff [path]                        # show pending diff
:commit [message]                   # stage only files the model touched, prompt y/N
:undo                               # find last [codebase-rag] commit, prompt y/N, then revert
```

No auto-commit. The model edits files; you review and commit when you're satisfied.

### Shell tool

```bash
codebase-rag chat --allow-shell                                # host runner (default)
codebase-rag chat --allow-shell --shell-runner docker:python:3.13-slim --shell-network none
codebase-rag chat --allow-shell --shell-timeout 60
```

- **`run_shell` tool** — the model can request a shell command. Always prompts `[y/N/edit]` first.
- **`:run <cmd>`** — you run a command directly; output is added to history for the next turn.
- **Sandbox (host):** `cwd` pinned to root, `shell=False`, `shlex.split` parsing (no `$VAR`/pipes/backticks), 30s timeout, output capped at 50KB, env scrubbed (no `ANTHROPIC_API_KEY`, etc.).
- **Sandbox (docker):** the above + transient container, project root bind-mounted at `/work`, `--network=none` by default, `--read-only` rootfs + 64MB tmpfs, non-root user, 1GB / 1 CPU caps.

### Web search & fetch (SearXNG)

```bash
docker run -d -p 8080:8080 --name searxng searxng/searxng
export SEARXNG_URL=http://localhost:8080
pip install -e .[web]

codebase-rag chat --allow-web --web-allow 'docs.python.org,*.readthedocs.io,github.com'
```

- **`web_search(query)`** via your own SearXNG — queries never leave your machine.
- **`web_fetch(url)`** via httpx + trafilatura; content capped at 50KB, cached per project for 24h.
- `--web-allow HOST_GLOB` / `--web-block HOST_GLOB` are repeatable and restrict what `web_fetch` may contact.
- Post-redirect host is re-checked so a fetch through an allowed host can't silently redirect to an attacker.
- Slash commands `:search <query>` and `:fetch <url>` work too.

### Architect–coder split

```bash
codebase-rag chat --architect-model qwen3:30b-a3b --model qwen2.5-coder:7b
```

The architect (no tools) plans the steps; the coder (with tools) executes. Useful when you have a strong model for thinking and a fast model for editing.

### Anthropic (opt-in cloud)

```bash
pip install -e .[cloud]
export ANTHROPIC_API_KEY=sk-ant-...
codebase-rag chat --provider anthropic --model claude-opus-4-7
```

Chat content goes to api.anthropic.com — a startup banner warns you. Embeddings stay on Ollama either way.

### Read-only & write confirmations

```bash
codebase-rag chat --read-only          # create_project/write_file/edit_file/run_shell removed from schema
codebase-rag chat                      # prompts before every project creation/write/edit by default
codebase-rag chat --no-confirm-writes  # auto-apply model project creation/writes/edits
```

### Verbose logging

```bash
codebase-rag chat -v                   # per-inference timing + token rates; full tool results
```

### Audit log

Every tool call and slash command in every session is recorded at `~/.codebase-rag/meta/<hash>/audit.log` (long string args are redacted to length-only). Inspect:

```bash
codebase-rag audit                      # last 50 events for the current project
codebase-rag audit --tool grep
codebase-rag audit --since '2h'         # also: today, yesterday, 15m, 7d, ISO date
codebase-rag audit --event tool_call --pretty
```

The model has these tools (subset depending on flags): `read_file`, `grep`, `get_diagnostics`, `create_project`, `write_file`, `edit_file`, plus `run_shell` with `--allow-shell` and `web_search` / `web_fetch` with `--allow-web`.

After every successful project creation, write, or edit, the affected files are automatically re-chunked and the index is updated — no manual `reindex` needed for edits the agent makes. If `create_project` is used to start a standalone project, index that new directory and restart chat with `--root` pointing at it.

Show what was retrieved (useful for tuning):

```bash
codebase-rag chat --show-context
```

Sandbox tool calls to a specific directory:

```bash
codebase-rag chat --root ~/code/my-project
```

Use a different index location:

```bash
codebase-rag index ~/code/project-a --db ~/.codebase-rag/project-a
codebase-rag chat --db ~/.codebase-rag/project-a --root ~/code/project-a
```

## How it works

1. **Walk** the directory, skipping common junk (`.git`, `node_modules`, build artifacts, binaries, files over 200KB).
2. **Chunk** each file into 50-line windows with 10-line overlap, keyed by `path:start-end`.
3. **Embed** each chunk via Ollama's `nomic-embed-text` (batched 32 at a time).
4. **Store** the vectors in a local Chroma collection.
5. At query time, embed the question, retrieve the top 8 nearest chunks, and call `mistral-nemo` with tools like `read_file`, `grep`, `get_diagnostics`, `create_project`, `write_file`, and `edit_file` and the chunks as a `Context:` block.
6. If the model emits tool calls, execute them (sandboxed under `--root`), feed each JSON result back as a `role: "tool"` message, and re-call the model. Loop until it stops emitting tool calls. Each successful `create_project`, `write_file`, or `edit_file` re-chunks affected files.

The system prompt forbids the model from claiming a write succeeded until it sees a tool result with `"ok": true`, and forbids placeholders like `"..."` or `"[rest omitted]"` in file content.

### Tools

| Tool         | Args                              | Behavior                                                                | Requires |
| ------------ | --------------------------------- | ----------------------------------------------------------------------- | --- |
| `read_file`  | `path`                            | Returns full file content (wrapped in untrusted markers). Refuses files over 200KB. | always |
| `grep`       | `pattern`, `file_glob?`, `literal?` | Regex-search every source file. Caps at 300 matches. **Use this for "list every / find all" queries.** | always |
| `get_diagnostics` | `path?`, `severity?`, `source?`, `limit?` | Read cached IDE/LSP Problems diagnostics from the per-project metadata dir. Messages are wrapped in untrusted markers. | always |
| `create_project` | `project_path`, `description?`, `files?`, `overwrite?` | Prompts by default, then creates a new directory under `--root` with starter files. Returns commands to index and chat with it as its own project; Python scaffolds also return official docs suggestions without fetching them. | not `--read-only` |
| `write_file` | `path`, `content`                 | Prompts by default, then overwrites the file. Reads it back and reports bytes/lines written. | not `--read-only` |
| `edit_file`  | `path`, `old_string`, `new_string`| Prompts by default, then replaces exactly one occurrence; errors on missing or ambiguous match. | not `--read-only` |
| `run_shell`  | `command`                         | Execute a command. Confirmation prompt fires before each run. shlex.split parsing, no shell expansion. Optional Docker runner. | `--allow-shell` |
| `web_search` | `query`, `top_k?`                 | Search via your self-hosted SearXNG.                                    | `--allow-web` |
| `web_fetch`  | `url`                             | Fetch a URL, extract main text via trafilatura, cache 24h.              | `--allow-web` |

All file paths resolve under `--root`. Anything outside is rejected.

When `create_project` detects a Python project, the tool result includes a small `suggested_documentation` list such as Python, packaging, pytest, FastAPI, or Pydantic docs based on the files and dependencies it just wrote. It does not contact the network. If you want the assistant to read those pages in the current session, restart chat with `--allow-web` and narrow `--web-allow` entries such as `docs.python.org` or `fastapi.tiangolo.com`, then approve the specific fetch.

The system prompt instructs the agent to call `grep` for any enumeration question (e.g. "list every API call this app makes") rather than relying on the retrieved Context block, which is semantic top-K and will silently miss matches.

## Tuning

Most knobs live in `codebase_rag/index.py` and `codebase_rag/chat.py`. The chat model can be set per-invocation via `--model NAME`, the env var `CODEBASE_RAG_CHAT_MODEL`, or this constant:

| Setting             | File         | Default | Purpose                                  |
| ------------------- | ------------ | ------- | ---------------------------------------- |
| `CHAT_MODEL`        | `chat.py`    | `mistral-nemo` | Default model when no `--model` / env var. Try `qwen3:8b`, `qwen2.5-coder:32b`, `llama3.3:70b`. |
| `EMBEDDING_MODEL`   | `index.py`   | `nomic-embed-text` | Try `mxbai-embed-large` for higher quality. |
| `TOP_K`             | `chat.py`    | `5`     | More chunks = richer context, more tokens. |
| `MAX_TURNS`         | `chat.py`    | `20`    | Tool-call rounds per user message before bailing. |
| `CHUNK_LINES`       | `index.py`   | `80`    | Bigger chunks = more context per hit.    |
| `OVERLAP_LINES`     | `index.py`   | `15`    | Reduces boundary-miss issues.            |
| `MAX_CHUNK_CHARS`   | `index.py`   | `3500`  | Cap per-chunk size before embedding.     |
| `MAX_FILE_BYTES`    | `index.py`   | `200000` | Skip enormous files (often generated).  |
| `num_ctx`           | `chat.py`    | `65536` | Ollama context window for chat.          |

If you swap the embedding model, **delete and rebuild the index** — embedding spaces aren't interchangeable.

## Known limits

- Line-based chunking is dumb. It doesn't know functions from comments. For most codebases this is fine; for huge generated files it can be noisy.
- `index` upserts but doesn't delete chunks for files that have been removed or renamed. Use `reindex` for that.
- mistral-nemo (12B) is the chat default for speed. For deep reasoning over retrieved code — or if it narrates edits without actually calling tools — point `CHAT_MODEL` at something larger like `qwen2.5-coder:32b` or `llama3.3:70b`.
- The agent caps at 20 tool-call rounds per user message. Tune `MAX_TURNS` in `chat.py` if you need longer multi-step edits.

## License

MIT
