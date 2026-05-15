"""Agent loop with mistral-nemo: RAG retrieval + file-editing tool calls."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import ollama

from .index import CHROMA_SETTINGS, COLLECTION_NAME, EMBEDDING_MODEL, reindex_file
from .tools import TOOL_SCHEMAS, run_tool

CHAT_MODEL = "mistral-nemo"
TOP_K = 8
MAX_TURNS = 20

CHAT_OPTIONS = {
    "num_ctx": 32768,
    "num_predict": -1,
    "temperature": 0.1,
}

SYSTEM_PROMPT = """You are a coding assistant for the user's local codebase.

You have three tools: read_file, write_file, edit_file. Every user turn also includes a "Context from codebase" block with retrieved chunks.

Strict rules:
- For any file change, emit a real tool call. Never describe a change you "would make" — either do it or ask a question.
- Before edit_file, call read_file first to get the exact text. The old_string must appear once and match character-for-character including whitespace.
- write_file content must be complete. Never write placeholders like "...", "[rest omitted]", "// continues", or "// ... existing code ...".
- Never claim a file was written or edited until you have received a tool result with "ok": true. If a tool result has "ok": false, address the error — do not pretend it succeeded.
- Cite file paths and line ranges (e.g. src/auth.py:42-67) when explaining code or proposed changes.
- If retrieved context is insufficient, say so or call read_file. Do not invent functions, files, or symbols.
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
            }
        )
    return chunks


def format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        header = f"### {c['path']}:{c['start_line']}-{c['end_line']}"
        parts.append(f"{header}\n```\n{c['content']}\n```")
    return "\n\n".join(parts)


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


def agent_loop(db_path: Path, root: Path, *, show_context: bool = False) -> None:
    root = root.resolve()
    client = chromadb.PersistentClient(path=str(db_path), settings=CHROMA_SETTINGS)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(
            f"No '{COLLECTION_NAME}' collection found at {db_path}. "
            f"Run `codebase-rag index <path>` first."
        )
        return

    def on_change(rel_path: str) -> None:
        try:
            reindex_file(rel_path, root, db_path)
        except Exception as e:
            print(f"  (reindex failed for {rel_path}: {e})")

    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(
        f"Chatting with {CHAT_MODEL}. Root: {root}\n"
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
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
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
            response = ollama.chat(
                model=CHAT_MODEL,
                messages=history,
                tools=TOOL_SCHEMAS,
                options=CHAT_OPTIONS,
            )
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
