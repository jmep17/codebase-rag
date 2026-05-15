"""Walk a codebase, chunk source files, embed with Ollama, store in Chroma."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import chromadb
import ollama
from chromadb.config import Settings

CHROMA_SETTINGS = Settings(anonymized_telemetry=False)

EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "codebase"

EXCLUDE_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", "bower_components", "vendor", "third_party", "Pods",
    "__pycache__", ".venv", "venv", "env", ".tox",
    "dist", "build", "out", ".next", ".nuxt", ".turbo", ".svelte-kit",
    "target", ".gradle", "DerivedData",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache", "coverage",
    ".idea", ".vscode",
}
EXCLUDE_EXTS = {
    ".lock", ".log", ".bin", ".exe", ".so", ".dylib", ".o", ".a",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff",
    ".pdf", ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp4", ".mov", ".mp3", ".wav", ".ogg", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".pyc", ".pyo", ".class", ".jar",
    ".map",  # source maps
}
EXCLUDE_NAME_PATTERNS = (
    ".min.js", ".min.css", ".bundle.js", ".bundle.css",
    "-lock.json", "_pb.go", "_pb.py", ".pb.go",
)
MAX_FILE_BYTES = 200_000
CHUNK_LINES = 80
OVERLAP_LINES = 15
EMBED_BATCH = 32
MAX_CHUNK_CHARS = 3500  # ~900 tokens; safe under 8192-token window, with fallback for edge cases
EMBED_NUM_CTX = 8192
EMBED_TRUNCATE_LADDER = (3500, 2000, 1000, 500)


def iter_source_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in EXCLUDE_EXTS:
            continue
        name_lower = path.name.lower()
        if any(name_lower.endswith(pat) for pat in EXCLUDE_NAME_PATTERNS):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def chunk_file(path: Path, root: Path) -> Iterator[dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    lines = text.splitlines()
    if not lines:
        return
    rel = path.relative_to(root)
    step = max(1, CHUNK_LINES - OVERLAP_LINES)
    for i in range(0, len(lines), step):
        chunk_lines = lines[i : i + CHUNK_LINES]
        if not chunk_lines:
            break
        start_line = i + 1
        end_line = min(i + CHUNK_LINES, len(lines))
        content = "\n".join(chunk_lines)
        if len(content) > MAX_CHUNK_CHARS:
            content = content[:MAX_CHUNK_CHARS]
        yield {
            "id": f"{rel}:{start_line}-{end_line}",
            "path": str(rel),
            "start_line": start_line,
            "end_line": end_line,
            "content": content,
        }
        if end_line >= len(lines):
            break


def _embed_one(text: str) -> list[float]:
    return ollama.embed(
        model=EMBEDDING_MODEL,
        input=text,
        options={"num_ctx": EMBED_NUM_CTX},
    )["embeddings"][0]


def _embed_one_with_fallback(text: str) -> list[float]:
    last_err: Exception | None = None
    for limit in EMBED_TRUNCATE_LADDER:
        try:
            return _embed_one(text[:limit])
        except Exception as e:
            last_err = e
    raise RuntimeError(f"could not embed chunk after truncation fallbacks: {last_err}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    try:
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=texts,
            options={"num_ctx": EMBED_NUM_CTX},
        )
        return response["embeddings"]
    except Exception:
        return [_embed_one_with_fallback(t) for t in texts]


def reset_index(db_path: Path) -> None:
    if not db_path.exists():
        return
    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Wiped collection '{COLLECTION_NAME}' at {db_path}.")
    except Exception:
        pass


def reindex_file(rel_path: str, root: Path, db_path: Path) -> None:
    """Drop existing chunks for `rel_path` and re-chunk/embed the current file."""
    root = root.resolve()
    abs_path = root / rel_path
    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    collection.delete(where={"path": rel_path})
    if not abs_path.exists() or not abs_path.is_file():
        return
    chunks = list(chunk_file(abs_path, root))
    if not chunks:
        return
    try:
        current_mtime = abs_path.stat().st_mtime
    except OSError:
        current_mtime = 0.0
    embeddings = embed_texts([c["content"] for c in chunks])
    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["content"] for c in chunks],
        metadatas=[
            {
                "path": c["path"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "mtime": current_mtime,
            }
            for c in chunks
        ],
    )


def _load_indexed_mtimes(collection) -> dict[str, float]:
    """Return {relative_path: mtime} for files already in the collection."""
    try:
        result = collection.get(include=["metadatas"])
    except Exception:
        return {}
    out: dict[str, float] = {}
    for meta in result.get("metadatas") or []:
        if not meta:
            continue
        path = meta.get("path")
        mtime = meta.get("mtime")
        if path and mtime is not None and path not in out:
            out[path] = float(mtime)
    return out


def build_index(root: Path, db_path: Path) -> None:
    root = root.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    indexed_mtimes = _load_indexed_mtimes(collection)

    chunks: list[dict] = []
    files_to_clear: list[str] = []
    skipped = 0

    for source_path in iter_source_files(root):
        rel = str(source_path.relative_to(root))
        try:
            current_mtime = source_path.stat().st_mtime
        except OSError:
            continue
        if indexed_mtimes.get(rel) == current_mtime:
            skipped += 1
            continue
        if rel in indexed_mtimes:
            files_to_clear.append(rel)
        for chunk in chunk_file(source_path, root):
            chunk["mtime"] = current_mtime
            chunks.append(chunk)

    if not chunks:
        print(f"All {skipped} indexable files unchanged; index is up to date.")
        return

    for rel in files_to_clear:
        collection.delete(where={"path": rel})

    summary = f"Indexing {len(chunks)} chunks from {root}"
    if skipped:
        summary += f" ({skipped} unchanged files skipped)"
    print(summary + "...")

    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        embeddings = embed_texts([c["content"] for c in batch])
        collection.upsert(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=[c["content"] for c in batch],
            metadatas=[
                {
                    "path": c["path"],
                    "start_line": c["start_line"],
                    "end_line": c["end_line"],
                    "mtime": c["mtime"],
                }
                for c in batch
            ],
        )
        done = min(i + EMBED_BATCH, len(chunks))
        print(f"  {done}/{len(chunks)}")

    print("Done.")
