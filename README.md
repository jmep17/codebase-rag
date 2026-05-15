# codebase-rag

Local codebase RAG with Ollama. Index your code, ask questions, get answers grounded in actual files — no cloud, no API keys.

- **Embeddings:** `nomic-embed-text` (via Ollama)
- **Chat:** `mistral-nemo` (via Ollama)
- **Vector store:** ChromaDB (persistent, on-disk)
- **Everything local.** Nothing leaves your machine.

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

The model has three tools: `read_file`, `write_file`, `edit_file`. It will use them automatically when you ask for changes ("add a test for…", "rename X to Y", "extract this into a helper"). Edits are confined to `--root` (defaults to the current working directory).

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

| Tool         | Args                              | Behavior                                                                |
| ------------ | --------------------------------- | ----------------------------------------------------------------------- |
| `read_file`  | `path`                            | Returns full file content. Refuses files over 200KB; use retrieval instead. |
| `grep`       | `pattern`, `file_glob?`           | Regex-search every source file. Respects ignore rules and nested-repo skips. Caps at 300 matches. **Use this for "list every / find all" queries** — retrieval alone is top-K and will miss matches. |
| `write_file` | `path`, `content`                 | Overwrites the file. Reads it back and reports bytes/lines written.     |
| `edit_file`  | `path`, `old_string`, `new_string`| Replaces exactly one occurrence. Errors if `old_string` is missing or appears more than once. |

All paths resolve under `--root`. Anything outside is rejected.

The system prompt instructs the agent to call `grep` for any enumeration question (e.g. "list every API call this app makes") rather than relying on the retrieved Context block, which is semantic top-K and will silently miss matches.

## Tuning

Most knobs live in `codebase_rag/index.py` and `codebase_rag/chat.py`:

| Setting             | File         | Default | Purpose                                  |
| ------------------- | ------------ | ------- | ---------------------------------------- |
| `CHAT_MODEL`        | `chat.py`    | `mistral-nemo` | Swap for `llama3.1`, `qwen2.5-coder:32b`, etc. |
| `EMBEDDING_MODEL`   | `index.py`   | `nomic-embed-text` | Try `mxbai-embed-large` for higher quality. |
| `TOP_K`             | `chat.py`    | `8`     | More chunks = richer context, more tokens. |
| `CHUNK_LINES`       | `index.py`   | `50`    | Bigger chunks = more context per hit.    |
| `OVERLAP_LINES`     | `index.py`   | `10`    | Reduces boundary-miss issues.            |
| `MAX_FILE_BYTES`    | `index.py`   | `200000` | Skip enormous files (often generated).  |
| `num_ctx`           | `chat.py`    | `32768` | Ollama context window for chat.          |

If you swap the embedding model, **delete and rebuild the index** — embedding spaces aren't interchangeable.

## Known limits

- Line-based chunking is dumb. It doesn't know functions from comments. For most codebases this is fine; for huge generated files it can be noisy.
- `index` upserts but doesn't delete chunks for files that have been removed or renamed. Use `reindex` for that.
- mistral-nemo (12B) is the chat default for speed. For deep reasoning over retrieved code — or if it narrates edits without actually calling tools — point `CHAT_MODEL` at something larger like `qwen2.5-coder:32b` or `llama3.3:70b`.
- The agent caps at 20 tool-call rounds per user message. Tune `MAX_TURNS` in `chat.py` if you need longer multi-step edits.

## License

MIT
