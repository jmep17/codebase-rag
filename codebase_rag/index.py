"""Walk a codebase, chunk source files, embed with Ollama, store in Chroma."""

from __future__ import annotations

import fnmatch
import hashlib
import re
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import chromadb
import ollama
from chromadb.config import Settings

from .config import meta_root

CHROMA_SETTINGS = Settings(anonymized_telemetry=False)

EMBEDDING_MODEL = "nomic-embed-text"


def _root_digest(root: Path) -> str:
    return hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:12]


def collection_name_for(root: Path) -> str:
    """Stable per-project collection name derived from the absolute root path."""
    digest = _root_digest(root)
    last_parts = [p for p in root.resolve().parts[-2:] if p and p != "/"]
    slug = "_".join(re.sub(r"[^A-Za-z0-9]+", "_", p) for p in last_parts)
    slug = re.sub(r"_+", "_", slug).strip("_") or "root"
    name = f"cbr_{slug[:48]}_{digest}"
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:512]


def project_meta_dir(root: Path) -> Path:
    """Path to the per-project metadata directory (notes, etc.), outside any repo."""
    return meta_root() / _root_digest(root)


def _open_collection(client, root: Path):
    return client.get_or_create_collection(
        collection_name_for(root),
        metadata={"root": str(root.resolve())},
    )


def read_notes(root: Path) -> str:
    notes_path = project_meta_dir(root) / "notes.md"
    if not notes_path.is_file():
        return ""
    return notes_path.read_text(encoding="utf-8", errors="replace")


def write_notes(root: Path, content: str) -> Path:
    meta_dir = project_meta_dir(root)
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(f'{{"root": "{str(root.resolve())}"}}\n', encoding="utf-8")
    notes_path = meta_dir / "notes.md"
    notes_path.write_text(content, encoding="utf-8")
    return notes_path


def clear_notes(root: Path) -> bool:
    notes_path = project_meta_dir(root) / "notes.md"
    if notes_path.is_file():
        notes_path.unlink()
        return True
    return False


EXCLUDE_DIRS = {
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "bower_components",
    "vendor",
    "third_party",
    "Pods",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".tox",
    "dist",
    "build",
    "out",
    ".next",
    ".nuxt",
    ".turbo",
    ".svelte-kit",
    "target",
    ".gradle",
    "DerivedData",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "coverage",
    ".idea",
    ".vscode",
}
EXCLUDE_EXTS = {
    ".lock",
    ".log",
    ".bin",
    ".exe",
    ".so",
    ".dylib",
    ".o",
    ".a",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".webp",
    ".tiff",
    ".pdf",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    ".mp4",
    ".mov",
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".7z",
    ".rar",
    ".pyc",
    ".pyo",
    ".class",
    ".jar",
    ".map",  # source maps
}
EXCLUDE_NAME_PATTERNS = (
    ".min.js",
    ".min.css",
    ".bundle.js",
    ".bundle.css",
    "-lock.json",
    "_pb.go",
    "_pb.py",
    ".pb.go",
)
IGNORE_FILE_NAME = ".codebaseragignore"
MAX_FILE_BYTES = 200_000
CHUNK_LINES = 80
OVERLAP_LINES = 15
EMBED_BATCH = 32
MAX_CHUNK_CHARS = 3500  # ~900 tokens; safe under 8192-token window, with fallback for edge cases
EMBED_NUM_CTX = 8192
EMBED_TRUNCATE_LADDER = (3500, 2000, 1000, 500)


def _find_nested_repos(root: Path) -> list[str]:
    """Return relative paths of subdirectories that contain their own .git directory."""
    nested: list[str] = []
    for git_dir in root.rglob(".git"):
        if not git_dir.is_dir():
            continue
        if any(part in EXCLUDE_DIRS for part in git_dir.parts[:-1]):
            continue
        try:
            rel = git_dir.parent.relative_to(root)
        except ValueError:
            continue
        rel_str = str(rel).replace("\\", "/")
        if rel_str == ".":
            continue
        nested.append(rel_str)
    return nested


def _load_ignore_file(root: Path) -> list[str]:
    ignore_path = root / IGNORE_FILE_NAME
    if not ignore_path.is_file():
        return []
    patterns = []
    for line in ignore_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def _matches_user_pattern(rel_path: str, name: str, patterns: Sequence[str]) -> bool:
    rel_posix = rel_path.replace("\\", "/")
    for pat in patterns:
        if fnmatch.fnmatch(rel_posix, pat) or fnmatch.fnmatch(name, pat):
            return True
    return False


def iter_source_files(
    root: Path,
    user_excludes: Sequence[str] = (),
    nested_repos: Sequence[str] = (),
) -> Iterator[Path]:
    nested_prefixes = tuple(p + "/" for p in nested_repos)
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
        rel = str(path.relative_to(root)).replace("\\", "/")
        if nested_prefixes and rel.startswith(nested_prefixes):
            continue
        if user_excludes and _matches_user_pattern(rel, path.name, user_excludes):
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


def reset_index(db_path: Path, root: Path) -> None:
    """Wipe only project chunks (kind=project) — references are preserved."""
    if not db_path.exists():
        return
    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)
    name = collection_name_for(root)
    try:
        collection = client.get_collection(name)
    except Exception:
        return
    # Delete project chunks. Also nuke legacy chunks that have no 'kind' metadata
    # (from before this feature) so old/new data don't sit side-by-side.
    try:
        collection.delete(where={"kind": "project"})
    except Exception:
        pass
    try:
        all_result = collection.get(include=["metadatas"])
        ids = result_get_ids(all_result)
        metas = all_result.get("metadatas") or []
        legacy_ids = [cid for cid, m in zip(ids, metas) if cid and not (m and m.get("kind"))]
        if legacy_ids:
            collection.delete(ids=legacy_ids)
    except Exception:
        pass
    print(f"Wiped project chunks for {root.resolve()} (references preserved).")


def remove_reference(db_path: Path, project_root: Path, label: str) -> None:
    if not db_path.exists():
        return
    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)
    try:
        collection = client.get_collection(collection_name_for(project_root))
    except Exception:
        print(f"No index for {project_root.resolve()}.")
        return
    try:
        before = collection.count()
        collection.delete(where={"kind": "reference", "label": label})
        after = collection.count()
        print(f"Removed reference '{label}' ({before - after} chunks).")
    except Exception as e:
        print(f"Could not remove reference '{label}': {e}")


def result_get_ids(result: dict) -> list[str]:
    """ChromaDB get() returns ids by default; collection.get's dict has 'ids'."""
    return list(result.get("ids") or [])


def _collection_summary(collection) -> dict:
    total = collection.count()
    result = collection.get(include=["metadatas"])
    metas = result.get("metadatas") or []
    project_files: dict[str, int] = {}
    references: dict[str, dict[str, int]] = {}  # label -> {path -> chunk_count}
    project_chunks = 0
    legacy_chunks = 0
    for meta in metas:
        if not meta:
            continue
        path = meta.get("path", "?")
        kind = meta.get("kind")
        if kind == "reference":
            label = meta.get("label", "reference")
            references.setdefault(label, {})[path] = (
                references.setdefault(label, {}).get(path, 0) + 1
            )
        elif kind == "project":
            project_chunks += 1
            project_files[path] = project_files.get(path, 0) + 1
        else:
            legacy_chunks += 1
            project_files[path] = project_files.get(path, 0) + 1
    top_project = sorted(project_files.items(), key=lambda kv: -kv[1])[:15]
    return {
        "total": total,
        "project_chunks": project_chunks + legacy_chunks,
        "project_files": len(project_files),
        "legacy_chunks": legacy_chunks,
        "references": references,
        "top_project_files": top_project,
    }


def stats(db_path: Path, root: Path | None = None) -> None:
    """Print a summary of indexed projects.

    With `root`, prints stats for that project's collection only.
    Without `root`, lists every indexed project in the database.
    """
    if not db_path.exists():
        print(f"No index found at {db_path}.")
        return
    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)

    if root is not None:
        target_name = collection_name_for(root)
        try:
            collection = client.get_collection(target_name)
        except Exception:
            print(f"No index for {root.resolve()}. Run `codebase-rag index .` first.")
            return
        summary = _collection_summary(collection)
        notes = read_notes(root)
        print(f"Project: {root.resolve()}")
        print(f"Chunks:  {summary['total']}")
        print(f"  project:  {summary['project_chunks']}  ({summary['project_files']} files)")
        if summary["legacy_chunks"]:
            print(f"    (legacy chunks without kind metadata: {summary['legacy_chunks']})")
        for label, files in summary["references"].items():
            ref_count = sum(files.values())
            print(f"  reference '{label}':  {ref_count}  ({len(files)} files)")
        print(f"Notes:   {'present' if notes else '(none)'}")
        if notes:
            first_line = notes.strip().splitlines()[0] if notes.strip() else ""
            if first_line:
                print(f"  > {first_line[:80]}")
        if summary["top_project_files"]:
            print("\nLargest project files by chunk count:")
            for path, n in summary["top_project_files"]:
                print(f"  {n:5d}  {path}")
        return

    collections = client.list_collections()
    indexed = [c for c in collections if c.name.startswith("cbr_")]
    if not indexed:
        print(f"No indexed projects in {db_path}.")
        return

    try:
        size = sum(p.stat().st_size for p in db_path.rglob("*") if p.is_file())
        size_str = f"{size / 1_000_000:.1f} MB"
    except OSError:
        size_str = "?"

    print(f"Database: {db_path}  ({size_str} on disk, {len(indexed)} project(s))\n")

    for collection in indexed:
        meta_root = (collection.metadata or {}).get("root", "(unknown root)")
        summary = _collection_summary(collection)
        ref_summary = ""
        if summary["references"]:
            ref_summary = "  refs: " + ", ".join(
                f"{label}({sum(files.values())})" for label, files in summary["references"].items()
            )
        notes_marker = ""
        try:
            if (Path(meta_root) / "_").parent and read_notes(Path(meta_root)):
                notes_marker = "  notes: yes"
        except Exception:
            pass
        print(f"  {meta_root}")
        print(
            f"    project chunks: {summary['project_chunks']} ({summary['project_files']} files)"
            f"{ref_summary}{notes_marker}"
        )


def _search_hits(
    db_path: Path,
    query: str,
    root: Path,
    *,
    top_k: int = 5,
    file_pattern: str | None = None,
) -> list[dict]:
    """Run a semantic search and return ranked hits as dicts.

    Each hit: {path, start_line, end_line, content, distance, score, kind, label}.
    Same shape as `chat.retrieve()` so HTTP API and chat retrieval stay aligned.
    Returns [] if the project has no index. Raises nothing for missing-index;
    a missing collection just yields an empty list.
    """
    if not db_path.exists():
        return []
    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)
    name = collection_name_for(root)
    try:
        collection = client.get_collection(name)
    except Exception:
        return []

    embedding = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query,
        options={"num_ctx": EMBED_NUM_CTX},
    )["embeddings"][0]

    fetch_n = top_k * 20 if file_pattern else top_k
    result = collection.query(query_embeddings=[embedding], n_results=fetch_n)

    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = (result.get("distances") or [[]])[0]

    hits: list[dict] = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        if file_pattern and not fnmatch.fnmatch(meta.get("path", ""), file_pattern):
            continue
        dist = distances[i] if i < len(distances) else None
        score = (1.0 - dist) if isinstance(dist, (int, float)) else None
        hits.append(
            {
                "path": meta.get("path", ""),
                "start_line": meta.get("start_line", 0),
                "end_line": meta.get("end_line", 0),
                "content": doc,
                "distance": dist,
                "score": score,
                "kind": meta.get("kind") or "project",
                "label": meta.get("label") or "",
            }
        )
        if len(hits) >= top_k:
            break
    return hits


def search(
    db_path: Path,
    query: str,
    root: Path,
    top_k: int = 5,
    file_pattern: str | None = None,
    headers_only: bool = False,
) -> None:
    """One-shot semantic search scoped to `root`'s collection."""
    if not db_path.exists():
        print(f"No index found at {db_path}.")
        return
    # Reuse the same logic the HTTP API uses; only difference is presentation.
    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)
    name = collection_name_for(root)
    try:
        client.get_collection(name)
    except Exception:
        print(f"No index for {root.resolve()}. Run `codebase-rag index .` first.")
        return

    hits = _search_hits(db_path, query, root, top_k=top_k, file_pattern=file_pattern)
    if not hits:
        print("(no results)")
        return

    for i, hit in enumerate(hits):
        header = f"[{i + 1}] {hit['path']}:{hit['start_line']}-{hit['end_line']}"
        if hit["distance"] is not None:
            header += f"  (distance {hit['distance']:.3f})"
        print(header)
        if not headers_only:
            for line in hit["content"].splitlines():
                print(f"    {line}")
            print()


def show_file(db_path: Path, file_pattern: str, root: Path) -> None:
    """List every chunk for files whose path matches `file_pattern` (glob), within `root`."""
    if not db_path.exists():
        print(f"No index found at {db_path}.")
        return
    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)
    name = collection_name_for(root)
    try:
        collection = client.get_collection(name)
    except Exception:
        print(f"No index for {root.resolve()}. Run `codebase-rag index .` first.")
        return

    result = collection.get(include=["documents", "metadatas"])
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []

    matching: list[tuple[dict, str]] = []
    for meta, doc in zip(metas, docs):
        if not meta or not doc:
            continue
        if fnmatch.fnmatch(meta.get("path", ""), file_pattern):
            matching.append((meta, doc))

    if not matching:
        print(f"No indexed chunks match {file_pattern!r}.")
        return

    matching.sort(key=lambda mc: (mc[0].get("path", ""), mc[0].get("start_line", 0)))
    print(f"{len(matching)} chunk(s) match {file_pattern!r}:\n")
    for meta, doc in matching:
        print(f"=== {meta['path']}:{meta['start_line']}-{meta['end_line']} ===")
        for line in doc.splitlines():
            print(line)
        print()


def _chunk_id(kind: str, label: str, rel: str, start: int, end: int) -> str:
    if kind == "reference":
        return f"ref:{label}::{rel}:{start}-{end}"
    return f"proj::{rel}:{start}-{end}"


def reindex_file(rel_path: str, root: Path, db_path: Path) -> None:
    """Drop existing project chunks for `rel_path` and re-chunk/embed the current file."""
    root = root.resolve()
    abs_path = root / rel_path
    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)
    collection = _open_collection(client, root)
    # Only delete project chunks for this path -- preserves any reference that
    # might happen to share a path string under a different label.
    collection.delete(where={"$and": [{"path": rel_path}, {"kind": "project"}]})
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
        ids=[_chunk_id("project", "", c["path"], c["start_line"], c["end_line"]) for c in chunks],
        embeddings=embeddings,
        documents=[c["content"] for c in chunks],
        metadatas=[
            {
                "path": c["path"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "mtime": current_mtime,
                "kind": "project",
                "label": "",
            }
            for c in chunks
        ],
    )


def _load_indexed_mtimes(collection, kind: str = "project", label: str = "") -> dict[str, float]:
    """Return {relative_path: mtime} for chunks matching the kind/label."""
    where: dict = {"kind": kind}
    if kind == "reference":
        where = {"$and": [{"kind": "reference"}, {"label": label}]}
    try:
        result = collection.get(where=where, include=["metadatas"])
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


def _ingest(
    source: Path,
    project_root: Path,
    db_path: Path,
    *,
    kind: str,
    label: str,
    extra_excludes: Sequence[str] = (),
    on_progress: Callable[[dict], None] | None = None,
) -> None:
    source = source.resolve()
    project_root = project_root.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    def _emit(phase: str, **rest: Any) -> None:
        if on_progress is not None:
            try:
                on_progress({"phase": phase, **rest})
            except Exception:
                pass

    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)
    collection = _open_collection(client, project_root)

    if kind == "reference":
        print(f"Reference '{label}' from {source} -> project {project_root}")
        print(f"  (collection: {collection.name})")
    else:
        print(f"Project: {project_root}  (collection: {collection.name})")
    _emit(
        "start",
        source=str(source),
        project=str(project_root),
        collection=collection.name,
        kind=kind,
        label=label,
    )

    indexed_mtimes = _load_indexed_mtimes(collection, kind=kind, label=label)
    user_excludes = tuple(extra_excludes) + tuple(_load_ignore_file(source))
    nested_repos = _find_nested_repos(source) if kind == "project" else []
    if nested_repos:
        print(f"Skipping {len(nested_repos)} nested git repo(s):")
        for nr in sorted(nested_repos):
            print(f"  - {nr}")

    chunks: list[dict] = []
    files_to_clear: list[str] = []
    skipped = 0

    for source_path in iter_source_files(source, user_excludes, nested_repos):
        rel = str(source_path.relative_to(source))
        try:
            current_mtime = source_path.stat().st_mtime
        except OSError:
            continue
        if indexed_mtimes.get(rel) == current_mtime:
            skipped += 1
            continue
        if rel in indexed_mtimes:
            files_to_clear.append(rel)
        for chunk in chunk_file(source_path, source):
            chunk["mtime"] = current_mtime
            chunks.append(chunk)

    _emit(
        "discover",
        total_chunks=len(chunks),
        files_to_clear=len(files_to_clear),
        unchanged=skipped,
        nested_repos=len(nested_repos),
    )

    if not chunks:
        kind_label = f"reference '{label}'" if kind == "reference" else "indexable files"
        print(f"All {skipped} {kind_label} unchanged; index is up to date.")
        _emit("done", total_chunks=0, unchanged=skipped)
        return

    for rel in files_to_clear:
        if kind == "reference":
            collection.delete(
                where={"$and": [{"path": rel}, {"kind": "reference"}, {"label": label}]}
            )
        else:
            collection.delete(where={"$and": [{"path": rel}, {"kind": "project"}]})

    descr = f"reference '{label}'" if kind == "reference" else "project"
    summary = f"Indexing {len(chunks)} {descr} chunks from {source}"
    if skipped:
        summary += f" ({skipped} unchanged files skipped)"
    print(summary + "...")

    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        embeddings = embed_texts([c["content"] for c in batch])
        collection.upsert(
            ids=[_chunk_id(kind, label, c["path"], c["start_line"], c["end_line"]) for c in batch],
            embeddings=embeddings,
            documents=[c["content"] for c in batch],
            metadatas=[
                {
                    "path": c["path"],
                    "start_line": c["start_line"],
                    "end_line": c["end_line"],
                    "mtime": c["mtime"],
                    "kind": kind,
                    "label": label,
                }
                for c in batch
            ],
        )
        done = min(i + EMBED_BATCH, len(chunks))
        print(f"  {done}/{len(chunks)}")
        _emit("embed", done=done, total=len(chunks), current_file=batch[-1]["path"])

    print("Done.")
    _emit("done", total_chunks=len(chunks), unchanged=skipped)


def build_index(
    root: Path,
    db_path: Path,
    *,
    extra_excludes: Sequence[str] = (),
    on_progress: Callable[[dict], None] | None = None,
) -> None:
    _ingest(
        source=root,
        project_root=root,
        db_path=db_path,
        kind="project",
        label="",
        extra_excludes=extra_excludes,
        on_progress=on_progress,
    )


def add_reference(
    source: Path,
    project_root: Path,
    db_path: Path,
    *,
    label: str,
    extra_excludes: Sequence[str] = (),
    on_progress: Callable[[dict], None] | None = None,
) -> None:
    """Index `source` as reference material attached to `project_root`'s collection."""
    _ingest(
        source=source,
        project_root=project_root,
        db_path=db_path,
        kind="reference",
        label=label,
        extra_excludes=extra_excludes,
        on_progress=on_progress,
    )
