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

Defenses against prompt injection in retrieved content / tool results:

- Retrieved chunks, file contents, web pages, and shell output are wrapped in `<<<UNTRUSTED-...>>>` markers and the system prompt tells the model to treat anything between them as data, not instructions.
- `--read-only` removes write/edit/shell tools from the model's schema entirely.
- `--confirm-writes` prompts you before every `write_file` / `edit_file` executes.
- `--web-allow` / `--web-block` cap which hosts `web_fetch` may contact (post-redirect host re-checked).
- Every tool call and slash command is recorded in a per-project `audit.log` (`codebase-rag audit` to view).

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com).

```bash
ollama pull mistral-nemo
ollama pull nomic-embed-text

pip install -e .
```

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

Everything below is opt-in. Default `codebase-rag chat` behaves exactly as before.

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

### Read-only & confirm-writes

```bash
codebase-rag chat --read-only          # write_file/edit_file/run_shell removed from schema
codebase-rag chat --confirm-writes     # prompt with diff/preview before every write or edit
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

The model has these tools (subset depending on flags): `read_file`, `grep`, `write_file`, `edit_file`, plus `run_shell` with `--allow-shell` and `web_search` / `web_fetch` with `--allow-web`.

After every successful write or edit, the affected file is automatically re-chunked and the index is updated — no manual `reindex` needed for edits the agent makes.

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
5. At query time, embed the question, retrieve the top 8 nearest chunks, and call `mistral-nemo` with `tools=[read_file, write_file, edit_file]` and the chunks as a `Context:` block.
6. If the model emits tool calls, execute them (sandboxed under `--root`), feed each JSON result back as a `role: "tool"` message, and re-call the model. Loop until it stops emitting tool calls. Each successful `write_file` / `edit_file` re-chunks just that file.

The system prompt forbids the model from claiming a write succeeded until it sees a tool result with `"ok": true`, and forbids placeholders like `"..."` or `"[rest omitted]"` in file content.

### Tools

| Tool         | Args                              | Behavior                                                                | Requires |
| ------------ | --------------------------------- | ----------------------------------------------------------------------- | --- |
| `read_file`  | `path`                            | Returns full file content (wrapped in untrusted markers). Refuses files over 200KB. | always |
| `grep`       | `pattern`, `file_glob?`, `literal?` | Regex-search every source file. Caps at 300 matches. **Use this for "list every / find all" queries.** | always |
| `write_file` | `path`, `content`                 | Overwrites the file. Reads it back and reports bytes/lines written.     | not `--read-only` |
| `edit_file`  | `path`, `old_string`, `new_string`| Replaces exactly one occurrence; errors on missing or ambiguous match.  | not `--read-only` |
| `run_shell`  | `command`                         | Execute a command. Confirmation prompt fires before each run. shlex.split parsing, no shell expansion. Optional Docker runner. | `--allow-shell` |
| `web_search` | `query`, `top_k?`                 | Search via your self-hosted SearXNG.                                    | `--allow-web` |
| `web_fetch`  | `url`                             | Fetch a URL, extract main text via trafilatura, cache 24h.              | `--allow-web` |

All file paths resolve under `--root`. Anything outside is rejected.

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
