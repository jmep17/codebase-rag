"""Chat loop with mistral-nemo, augmented with retrieval from a Chroma index."""

from __future__ import annotations

from pathlib import Path

import chromadb
import ollama

from .index import COLLECTION_NAME, EMBEDDING_MODEL

CHAT_MODEL = "mistral-nemo"
TOP_K = 8

CHAT_OPTIONS = {
    "num_ctx": 32768,
    "num_predict": -1,
    "temperature": 0.2,
}

SYSTEM_PROMPT = """You are a code assistant. Answer questions about the user's codebase using the provided context chunks.

Rules:
- Cite file paths and line ranges (e.g. src/auth.py:42-67) when making claims about the code.
- If the context doesn't contain enough information to answer, say so plainly. Do not invent functions, files, or behavior.
- Prefer quoting short, exact snippets over paraphrasing.
- Be concise. Code-aware questions deserve code-aware answers.
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


def chat_loop(db_path: Path, *, show_context: bool = False) -> None:
    client = chromadb.PersistentClient(path=str(db_path))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"No '{COLLECTION_NAME}' collection found at {db_path}. Run `codebase-rag index <path>` first.")
        return

    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"Chatting with {CHAT_MODEL}. Type :q or Ctrl-D to exit, :reset to clear history.")

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
            print("-----------------\n")

        augmented = (
            f"Context from codebase:\n\n{context}\n\n---\n\nQuestion: {user_input}"
        )
        history.append({"role": "user", "content": augmented})

        response_text = ""
        for part in ollama.chat(
            model=CHAT_MODEL,
            messages=history,
            options=CHAT_OPTIONS,
            stream=True,
        ):
            piece = part["message"]["content"]
            print(piece, end="", flush=True)
            response_text += piece
        print()

        history[-1] = {"role": "user", "content": user_input}
        history.append({"role": "assistant", "content": response_text})
