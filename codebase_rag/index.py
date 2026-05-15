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
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", "target", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".tox", "coverage", ".nuxt", ".turbo", ".cache",
}
EXCLUDE_EXTS = {
    ".lock", ".log", ".bin", ".exe", ".so", ".dylib", ".o", ".a",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".pdf", ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp4", ".mp3", ".wav", ".ogg", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".pyc", ".pyo", ".class", ".jar",
}
MAX_FILE_BYTES = 200_000
CHUNK_LINES = 50
OVERLAP_LINES = 10
EMBED_BATCH = 32


def iter_source_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in EXCLUDE_EXTS:
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
        yield {
            "id": f"{rel}:{start_line}-{end_line}",
            "path": str(rel),
            "start_line": start_line,
            "end_line": end_line,
            "content": "\n".join(chunk_lines),
        }
        if end_line >= len(lines):
            break


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    return response["embeddings"]


def reset_index(db_path: Path) -> None:
    if not db_path.exists():
        return
    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Wiped collection '{COLLECTION_NAME}' at {db_path}.")
    except Exception:
        pass


def build_index(root: Path, db_path: Path) -> None:
    root = root.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    chunks: list[dict] = []
    for source_path in iter_source_files(root):
        chunks.extend(chunk_file(source_path, root))

    if not chunks:
        print(f"No indexable files found under {root}.")
        return

    print(f"Indexing {len(chunks)} chunks from {root}...")

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
                }
                for c in batch
            ],
        )
        done = min(i + EMBED_BATCH, len(chunks))
        print(f"  {done}/{len(chunks)}")

    print("Done.")
