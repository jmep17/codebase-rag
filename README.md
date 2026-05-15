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

Chat:

```bash
codebase-rag chat
> how does the auth middleware work?
> where is the database connection set up?
> :reset                 # clears conversation history
> :q                     # quit
```

Show what was retrieved (useful for tuning):

```bash
codebase-rag chat --show-context
```

Use a different index location:

```bash
codebase-rag index ~/code/project-a --db ~/.codebase-rag/project-a
codebase-rag chat --db ~/.codebase-rag/project-a
```

## How it works

1. **Walk** the directory, skipping common junk (`.git`, `node_modules`, build artifacts, binaries, files over 200KB).
2. **Chunk** each file into 50-line windows with 10-line overlap, keyed by `path:start-end`.
3. **Embed** each chunk via Ollama's `nomic-embed-text` (batched 32 at a time).
4. **Store** the vectors in a local Chroma collection.
5. At query time, embed the question, retrieve the top 8 nearest chunks, and ask `mistral-nemo` with the chunks as a `Context:` block plus your question.

The system prompt instructs the model to cite file paths and line ranges, and to admit when the retrieved context doesn't cover the question.

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
- mistral-nemo (12B) is the chat default for speed. For deep reasoning over retrieved code, point `CHAT_MODEL` at something larger.

## License

MIT
