"""Agent loop with mistral-nemo: RAG retrieval + file-editing tool calls."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import ollama

from .index import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL,
    collection_name_for,
    read_notes,
    reindex_file,
)
from .tools import TOOL_SCHEMAS, run_tool

CHAT_MODEL = "mistral-nemo"
TOP_K = 5
MAX_TURNS = 20

CHAT_OPTIONS = {
    "num_ctx": 65536,
    "num_predict": -1,
    "temperature": 0.1,
}

SYSTEM_PROMPT = """You are a coding assistant for the user's local codebase.

You have four tools:
- read_file(path)                            — read a file's full contents
- grep(pattern, file_glob?, literal?)        — search the whole project
- write_file(path, content)                  — create or overwrite a file
- edit_file(path, old, new)                  — replace one occurrence in a file

Every user turn also includes a "Context from codebase" block with retrieved chunks. Retrieval is **semantic top-K**, not a complete listing — for any question that asks you to enumerate ("list every X", "where is Y called", "find all Z"), the Context is a starting point, not the answer. Call grep before responding.

Calling grep correctly:
- `pattern` is the search string only. Do not wrap it in `r"..."`, quotes, or `re.compile(...)`. Just the pattern.
- If you want a plain-text search (no regex features), set `literal=true`. This is the safer default for paths, URLs, identifiers, or anything with `( ) . * + ?` in it.
- If a grep call returns `{"ok": false, "error": "invalid regex: ..."}`, do not give up and do not print Python code. Retry: either pass `literal=true`, or rewrite the pattern with proper escaping (`\\(` for a literal paren, `\\.` for a literal dot).
- If grep returns 23 matches, your answer must cover 23, not 5. If grep returns 0, say "no matches" — never invent.

Strict rules:
- For exhaustive queries, call grep first. Retrieval alone is incomplete.
- Every file path or symbol you cite must come from a tool result, a retrieved chunk, or a clearly user-provided string. Do not invent paths, function names, or routes.
- For any file change, emit a real tool call. Never describe a change you "would make" — either do it or ask a question.
- Before edit_file, call read_file first to copy the exact target text. old_string must appear once and match character-for-character including whitespace.
- write_file content must be complete. Never use placeholders like "...", "[rest omitted]", "// continues", or "// ... existing code ...".
- Never claim a file was written or edited until you have received a tool result with "ok": true. If a tool result has "ok": false, address the error — do not pretend it succeeded.
- Cite file paths and line ranges (e.g. src/auth.py:42-67) when explaining code or proposed changes.
"""


def retrieve(collection, query: str, top_k: int = TOP_K) -> list[dict]:
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    for doc, meta in zip(documents, metadatas):
        chunks.append(
            {
                "path": meta["path"],
                "start_line": meta["start_line"],
                "end_line": meta["end_line"],
                "content": doc,
                "kind": meta.get("kind") or "project",
                "label": meta.get("label") or "",
            }
        )
    return chunks


def format_context(chunks: list[dict]) -> str:
    """Group chunks by kind (project / reference label) and render with section headers."""
    project_chunks = []
    references: dict[str, list[dict]] = {}
    for c in chunks:
        kind = c.get("kind") or "project"
        if kind == "reference":
            references.setdefault(c.get("label") or "reference", []).append(c)
        else:
            project_chunks.append(c)

    sections: list[str] = []
    if project_chunks:
        body = "\n\n".join(
            f"### {c['path']}:{c['start_line']}-{c['end_line']}\n```\n{c['content']}\n```"
            for c in project_chunks
        )
        sections.append(f"## Project code\n\n{body}")
    for label in sorted(references):
        body = "\n\n".join(
            f"### {c['path']}:{c['start_line']}-{c['end_line']}\n```\n{c['content']}\n```"
            for c in references[label]
        )
        sections.append(f"## Reference: {label}\n\n{body}")
    return "\n\n".join(sections)


def _assistant_msg_from_response(msg) -> dict:
    """Coerce an Ollama response message into a plain dict for history."""
    out = {"role": "assistant", "content": msg.get("content", "") or ""}
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        out["tool_calls"] = [
            {
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }
            }
            for tc in tool_calls
        ]
    return out


def _system_prompt_for(root: Path) -> str:
    """SYSTEM_PROMPT plus any project notes from the meta dir."""
    notes = read_notes(root).strip()
    if not notes:
        return SYSTEM_PROMPT
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"## Project-specific notes (set by the user via `codebase-rag notes`)\n\n"
        f"{notes}\n"
    )


def agent_loop(db_path: Path, root: Path, *, show_context: bool = False) -> None:
    root = root.resolve()
    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)
    name = collection_name_for(root)
    try:
        collection = client.get_collection(name)
    except Exception:
        print(
            f"No index for {root}. Run `codebase-rag index .` in this directory first."
        )
        return

    def on_change(rel_path: str) -> None:
        try:
            reindex_file(rel_path, root, db_path)
        except Exception as e:
            print(f"  (reindex failed for {rel_path}: {e})")

    system_prompt = _system_prompt_for(root)
    history: list[dict] = [{"role": "system", "content": system_prompt}]
    notes_marker = "with notes" if read_notes(root).strip() else "no notes"
    print(
        f"Chatting with {CHAT_MODEL}.\n"
        f"Project: {root}  (collection: {name}, {notes_marker})\n"
        f"Type :q or Ctrl-D to exit, :reset to clear history."
    )

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_input:
            continue
        if user_input in (":q", "exit", "quit"):
            return
        if user_input == ":reset":
            history = [{"role": "system", "content": _system_prompt_for(root)}]
            print("(history cleared)")
            continue

        chunks = retrieve(collection, user_input)
        context = format_context(chunks)

        if show_context:
            print("\n--- retrieved ---")
            for c in chunks:
                print(f"  {c['path']}:{c['start_line']}-{c['end_line']}")
            print("-----------------")

        augmented = (
            f"Context from codebase:\n\n{context}\n\n---\n\nQuestion: {user_input}"
        )
        history.append({"role": "user", "content": augmented})

        for turn in range(MAX_TURNS):
            try:
                response = ollama.chat(
                    model=CHAT_MODEL,
                    messages=history,
                    tools=TOOL_SCHEMAS,
                    options=CHAT_OPTIONS,
                )
            except ollama.ResponseError as e:
                msg_text = str(e).lower()
                if "context" in msg_text and "length" in msg_text:
                    print(
                        "\n(prompt exceeded context window — try `:reset` to clear history, "
                        f"lower TOP_K in chat.py, or raise num_ctx above {CHAT_OPTIONS['num_ctx']})"
                    )
                else:
                    print(f"\n(ollama error: {e})")
                history.pop()
                break
            msg = response["message"]
            history.append(_assistant_msg_from_response(msg))

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                text = (msg.get("content") or "").strip()
                print(text if text else "(no response)")
                break

            for call in tool_calls:
                name = call["function"]["name"]
                args = call["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                print(f"  -> {name}({', '.join(args.keys())})")
                result = run_tool(name, args, root, on_change)
                print(f"     {result[:200]}")
                history.append({"role": "tool", "content": result})
        else:
            print(f"(stopped after {MAX_TURNS} tool-call rounds)")
