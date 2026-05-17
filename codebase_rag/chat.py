"""Agent loop with a configurable chat model: RAG retrieval + file-editing tool calls."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import chromadb
import ollama

from datetime import datetime, timezone

from .index import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL,
    collection_name_for,
    project_meta_dir,
    read_notes,
    reindex_file,
)
from .tools import TOOL_SCHEMAS, run_tool


def _conversation_path(root: Path) -> Path:
    return project_meta_dir(root) / "last_conversation.json"


def _save_conversation(root: Path, history: list[dict], model: str) -> None:
    """Persist the conversation to the project's meta dir (system message stripped)."""
    path = _conversation_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_msgs = [m for m in history if m.get("role") != "system"]
    payload = {
        "model": model,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "messages": saved_msgs,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_conversation(root: Path) -> dict | None:
    path = _conversation_path(root)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _clear_conversation(root: Path) -> None:
    path = _conversation_path(root)
    if path.is_file():
        path.unlink()

# Default chat model, overridable per call. Resolution order:
#   1. `agent_loop(..., model=...)` argument
#   2. `CODEBASE_RAG_CHAT_MODEL` env var
#   3. This constant
CHAT_MODEL = "mistral-nemo"
TOP_K = 5
MAX_TURNS = 20

CHAT_OPTIONS = {
    "num_ctx": 65536,
    "num_predict": -1,
    "temperature": 0.0,
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
- Choose precise patterns. To find function definitions in Python use `^def\\s+\\w+`; the bare string `def` will also match `default`, `defer`, `define`, etc. and produce false positives. Anchor with `^`, use word boundaries `\\b`, require word characters `\\w+` when you mean identifiers.
- If you want a plain-text search (no regex features), set `literal=true`. This is the safer default for paths, URLs, identifiers, or anything with `( ) . * + ?` in it.
- If a grep call returns `{"ok": false, "error": "invalid regex: ..."}`, do not give up and do not print Python code. Retry: either pass `literal=true`, or rewrite the pattern with proper escaping (`\\(` for a literal paren, `\\.` for a literal dot).
- If grep returns 23 matches, your answer must cover 23, not 5. If grep returns 0, say "no matches" — never invent.

Anti-hallucination rules (read these every time before answering):
- Every file path, function name, identifier, or line you cite must come directly from a tool result, a retrieved chunk, or text the user provided. If you cannot point to where you saw it, do not write it.
- When listing items from a tool result, copy each entry verbatim from the result. Do not paraphrase, summarise, normalise capitalisation, or invent.
- The number of items in your answer must equal the number of items visible in the tool result. If you can only see 14 items in the matches array, list 14 — even if `match_count` says more. Never pad to reach a target number with made-up entries.
- Never abbreviate a list. Do not write "... (remaining items follow)", "...etc", "(and so on)", or any similar shortcut. Either list every entry or explain why you cannot and ask the user how to narrow it down.
- If you find yourself making up a name, path, or identifier to fill out a list or to keep talking, stop and either call another tool or report what you actually have.

Strict rules:
- For exhaustive queries, call grep first. Retrieval alone is incomplete.
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


def _stream_inference(
    model: str,
    messages: list,
    tools: list,
    options: dict,
    *,
    verbose: bool,
) -> tuple[str, list, dict]:
    """Stream a chat call; print content as it arrives. Return (content, tool_calls, stats)."""
    t0 = time.time()
    content = ""
    tool_calls: list = []
    last_chunk = None
    for chunk in ollama.chat(
        model=model,
        messages=messages,
        tools=tools,
        options=options,
        stream=True,
    ):
        msg = chunk.get("message") or {}
        piece = msg.get("content") or ""
        if piece:
            print(piece, end="", flush=True)
            content += piece
        tcs = msg.get("tool_calls") or []
        if tcs:
            tool_calls.extend(tcs)
        last_chunk = chunk
    elapsed = time.time() - t0
    if content and not content.endswith("\n"):
        print()
    stats = {
        "elapsed": elapsed,
        "prompt_tokens": (last_chunk or {}).get("prompt_eval_count") or 0,
        "output_tokens": (last_chunk or {}).get("eval_count") or 0,
        "prompt_eval_duration": ((last_chunk or {}).get("prompt_eval_duration") or 0) / 1e9,
        "eval_duration": ((last_chunk or {}).get("eval_duration") or 0) / 1e9,
        "load_duration": ((last_chunk or {}).get("load_duration") or 0) / 1e9,
    }
    if verbose:
        prompt_rate = stats["prompt_tokens"] / stats["prompt_eval_duration"] if stats["prompt_eval_duration"] else 0
        gen_rate = stats["output_tokens"] / stats["eval_duration"] if stats["eval_duration"] else 0
        load_note = f", load {stats['load_duration']:.1f}s" if stats["load_duration"] > 0.05 else ""
        print(
            f"  [{stats['elapsed']:.1f}s · prompt {stats['prompt_tokens']} tok "
            f"in {stats['prompt_eval_duration']:.2f}s ({prompt_rate:.0f} tok/s) · "
            f"gen {stats['output_tokens']} tok in {stats['eval_duration']:.2f}s "
            f"({gen_rate:.0f} tok/s){load_note}]"
        )
    else:
        print(
            f"  [{stats['elapsed']:.1f}s · {stats['prompt_tokens']} in → "
            f"{stats['output_tokens']} out]"
        )
    return content, tool_calls, stats


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


def _resolve_model(model: str | None) -> str:
    return model or os.environ.get("CODEBASE_RAG_CHAT_MODEL") or CHAT_MODEL


def agent_loop(
    db_path: Path,
    root: Path,
    *,
    show_context: bool = False,
    model: str | None = None,
    verbose: bool = False,
    resume: bool = False,
) -> None:
    root = root.resolve()
    chat_model = _resolve_model(model)
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
    resumed_marker = ""
    if resume:
        saved = _load_conversation(root)
        if saved and saved.get("messages"):
            history.extend(saved["messages"])
            saved_at = saved.get("saved_at", "?")
            saved_model = saved.get("model", "?")
            resumed_marker = (
                f"\nResumed: {len(saved['messages'])} prior messages "
                f"(saved {saved_at}, model {saved_model})"
            )
        else:
            resumed_marker = "\n(--resume requested but no saved conversation found)"
    notes_marker = "with notes" if read_notes(root).strip() else "no notes"
    print(
        f"Chatting with {chat_model}.\n"
        f"Project: {root}  (collection: {name}, {notes_marker})"
        f"{resumed_marker}\n"
        f"Type :q or Ctrl-D to exit, :reset to clear history, :forget to delete the saved conversation."
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
            _save_conversation(root, history, chat_model)
            print("(history cleared)")
            continue
        if user_input == ":forget":
            history = [{"role": "system", "content": _system_prompt_for(root)}]
            _clear_conversation(root)
            print("(history cleared and saved conversation deleted)")
            continue

        retrieve_t0 = time.time()
        chunks = retrieve(collection, user_input)
        retrieve_elapsed = time.time() - retrieve_t0
        context = format_context(chunks)

        if verbose:
            project_count = sum(1 for c in chunks if (c.get("kind") or "project") == "project")
            ref_count = len(chunks) - project_count
            print(
                f"[retrieve: {len(chunks)} chunks ({project_count} project, "
                f"{ref_count} reference) in {retrieve_elapsed:.2f}s]"
            )
        else:
            print(f"[retrieve: {len(chunks)} chunks · {retrieve_elapsed:.2f}s]")

        if show_context:
            print("--- retrieved ---")
            for c in chunks:
                label = f"  [{c.get('label')}] " if c.get("kind") == "reference" else "  "
                print(f"{label}{c['path']}:{c['start_line']}-{c['end_line']}")
            print("-----------------")

        augmented = (
            f"Context from codebase:\n\n{context}\n\n---\n\nQuestion: {user_input}"
        )
        history.append({"role": "user", "content": augmented})

        user_turn_start = time.time()
        total_prompt_tokens = 0
        total_output_tokens = 0
        inferences = 0

        for turn in range(MAX_TURNS):
            inferences += 1
            try:
                content, tool_calls, stats = _stream_inference(
                    chat_model,
                    history,
                    TOOL_SCHEMAS,
                    CHAT_OPTIONS,
                    verbose=verbose,
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

            total_prompt_tokens += stats["prompt_tokens"]
            total_output_tokens += stats["output_tokens"]

            history.append(
                _assistant_msg_from_response(
                    {"content": content, "tool_calls": tool_calls}
                )
            )

            if not tool_calls:
                if not content.strip():
                    print("(no response)")
                break

            for call in tool_calls:
                name = call["function"]["name"]
                args = call["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_t0 = time.time()
                result = run_tool(name, args, root, on_change)
                tool_elapsed = time.time() - tool_t0
                preview = result if verbose else result[:200]
                print(f"  -> {name}({', '.join(args.keys())}) [{tool_elapsed:.2f}s]")
                print(f"     {preview}")
                history.append({"role": "tool", "content": result})
        else:
            print(f"(stopped after {MAX_TURNS} tool-call rounds)")

        turn_elapsed = time.time() - user_turn_start
        summary_parts = [
            f"turn: {turn_elapsed:.1f}s",
            f"{inferences} inference{'s' if inferences != 1 else ''}",
            f"{total_prompt_tokens} in → {total_output_tokens} out",
            f"history: {len(history)} messages",
        ]
        print(f"[{' · '.join(summary_parts)}]")

        try:
            _save_conversation(root, history, chat_model)
        except OSError as e:
            print(f"(could not save conversation: {e})")
