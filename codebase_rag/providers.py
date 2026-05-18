"""Chat provider abstraction: Ollama (local) and Anthropic (cloud, opt-in).

The rest of codebase-rag keeps using Ollama-style message dicts:
    {"role": "system", "content": "..."}
    {"role": "user",   "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [{"function": {...}}]}
    {"role": "tool",   "content": "..."}
and Ollama-style tool schemas:
    {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

Providers translate to/from their native formats. They all return the same
(content, tool_calls, stats) tuple so _stream_inference can stay generic.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable


class ProviderUnavailable(RuntimeError):
    """Raised when an optional provider's SDK isn't installed."""


def make_provider(name: str, *, api_key: str | None = None) -> ChatProvider:
    name = name.lower()
    if name == "ollama":
        return OllamaProvider()
    if name == "anthropic":
        return AnthropicProvider(api_key=api_key)
    raise ValueError(f"unknown provider: {name!r} (use 'ollama' or 'anthropic')")


class ChatProvider:
    """Interface used by chat.py to run one streaming inference."""

    name: str = "base"

    def stream_chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        options: dict,
        *,
        verbose: bool,
    ) -> tuple[str, list, dict]:
        raise NotImplementedError

    def iter_chat_events(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        options: dict,
    ):
        """Yield ('token', piece) for each streamed chunk; finally yield
        ('done', content, tool_calls, stats). No I/O — the consumer decides
        how to render. Used by agent_turn so the same code path drives
        the line UI and the Textual TUI."""
        raise NotImplementedError


class OllamaProvider(ChatProvider):
    name = "ollama"

    def iter_chat_events(self, model, messages, tools, options):
        import ollama

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
                yield ("token", piece)
                content += piece
            tcs = msg.get("tool_calls") or []
            if tcs:
                tool_calls.extend(tcs)
            last_chunk = chunk
        elapsed = time.time() - t0
        stats = {
            "elapsed": elapsed,
            "prompt_tokens": (last_chunk or {}).get("prompt_eval_count") or 0,
            "output_tokens": (last_chunk or {}).get("eval_count") or 0,
            "prompt_eval_duration": ((last_chunk or {}).get("prompt_eval_duration") or 0) / 1e9,
            "eval_duration": ((last_chunk or {}).get("eval_duration") or 0) / 1e9,
            "load_duration": ((last_chunk or {}).get("load_duration") or 0) / 1e9,
        }
        yield ("done", content, tool_calls, stats)

    def stream_chat(self, model, messages, tools, options, *, verbose):
        import ollama

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
        return content, tool_calls, stats


class AnthropicProvider(ChatProvider):
    name = "anthropic"

    def __init__(self, *, api_key: str | None = None):
        try:
            import anthropic
        except ImportError as e:
            raise ProviderUnavailable(
                "anthropic SDK not installed. Run `pip install -e .[cloud]`."
            ) from e
        if not api_key:
            raise ProviderUnavailable(
                "ANTHROPIC_API_KEY not set. Export the key before using --provider anthropic."
            )
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key)

    def iter_chat_events(self, model, messages, tools, options):
        system, anth_messages = _ollama_history_to_anthropic(messages)
        anth_tools = _ollama_tools_to_anthropic(tools)
        max_tokens = options.get("num_predict") if options else None
        if not max_tokens or max_tokens <= 0:
            max_tokens = 4096
        temperature = options.get("temperature") if options else None

        t0 = time.time()
        content = ""
        tool_calls: list = []

        stream_kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system or "",
            "messages": anth_messages,
        }
        if anth_tools:
            stream_kwargs["tools"] = anth_tools
        if temperature is not None:
            stream_kwargs["temperature"] = temperature

        try:
            with self.client.messages.stream(**stream_kwargs) as stream:
                for text in stream.text_stream:
                    if text:
                        yield ("token", text)
                        content += text
                final = stream.get_final_message()
        except self._anthropic.APIStatusError as e:
            elapsed = time.time() - t0
            yield (
                "done",
                f"[anthropic error: {e}]",
                [],
                {
                    "elapsed": elapsed,
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                    "prompt_eval_duration": 0,
                    "eval_duration": elapsed,
                    "load_duration": 0,
                },
            )
            return

        elapsed = time.time() - t0

        for block in final.content:
            if getattr(block, "type", None) == "tool_use":
                tool_calls.append(
                    {
                        "function": {
                            "name": block.name,
                            "arguments": block.input or {},
                        },
                        "_anthropic_id": block.id,
                    }
                )
            elif getattr(block, "type", None) == "text" and not content:
                content += block.text

        usage = getattr(final, "usage", None)
        usage_in = (getattr(usage, "input_tokens", 0) or 0) if usage is not None else 0
        usage_out = (getattr(usage, "output_tokens", 0) or 0) if usage is not None else 0

        stats = {
            "elapsed": elapsed,
            "prompt_tokens": usage_in,
            "output_tokens": usage_out,
            "prompt_eval_duration": 0.0,
            "eval_duration": elapsed,
            "load_duration": 0.0,
        }
        yield ("done", content, tool_calls, stats)

    def stream_chat(self, model, messages, tools, options, *, verbose):
        system, anth_messages = _ollama_history_to_anthropic(messages)
        anth_tools = _ollama_tools_to_anthropic(tools)
        max_tokens = options.get("num_predict") if options else None
        if not max_tokens or max_tokens <= 0:
            max_tokens = 4096
        temperature = options.get("temperature") if options else None

        t0 = time.time()
        content = ""
        tool_calls: list = []
        usage_in = 0
        usage_out = 0

        stream_kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system or "",
            "messages": anth_messages,
        }
        if anth_tools:
            stream_kwargs["tools"] = anth_tools
        if temperature is not None:
            stream_kwargs["temperature"] = temperature

        try:
            with self.client.messages.stream(**stream_kwargs) as stream:
                for text in stream.text_stream:
                    if text:
                        print(text, end="", flush=True)
                        content += text
                final = stream.get_final_message()
        except self._anthropic.APIStatusError as e:
            elapsed = time.time() - t0
            print()
            return (
                f"[anthropic error: {e}]",
                [],
                {
                    "elapsed": elapsed,
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                    "prompt_eval_duration": 0,
                    "eval_duration": elapsed,
                    "load_duration": 0,
                },
            )

        elapsed = time.time() - t0
        if content and not content.endswith("\n"):
            print()

        for block in final.content:
            if getattr(block, "type", None) == "tool_use":
                tool_calls.append(
                    {
                        "function": {
                            "name": block.name,
                            "arguments": block.input or {},
                        },
                        # Stash the Anthropic id so we can pair tool_result correctly
                        # when this history is re-translated next turn.
                        "_anthropic_id": block.id,
                    }
                )
            elif getattr(block, "type", None) == "text" and not content:
                # Some responses arrive without text_stream firing (rare); fall back.
                content += block.text

        usage = getattr(final, "usage", None)
        if usage is not None:
            usage_in = getattr(usage, "input_tokens", 0) or 0
            usage_out = getattr(usage, "output_tokens", 0) or 0

        stats = {
            "elapsed": elapsed,
            "prompt_tokens": usage_in,
            "output_tokens": usage_out,
            "prompt_eval_duration": 0.0,
            "eval_duration": elapsed,
            "load_duration": 0.0,
        }
        return content, tool_calls, stats


# ---------- translation helpers ----------


def _ollama_tools_to_anthropic(tools: Iterable[dict]) -> list[dict]:
    """Map [{'type':'function','function':{...}}] -> [{'name','description','input_schema'}]."""
    out: list[dict] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        out.append(
            {
                "name": name,
                "description": fn.get("description") or "",
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def _ollama_history_to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """Walk Ollama-style history. Return (system_prompt, anthropic_messages)."""
    system_parts: list[str] = []
    out: list[dict] = []
    pending_tool_ids: list[str] = []
    counter = 0

    def _new_id() -> str:
        nonlocal counter
        counter += 1
        return f"toolu_{counter:04d}"

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "") or ""

        if role == "system":
            if content.strip():
                system_parts.append(content)
            continue

        if role == "user":
            if not content.strip():
                out.append({"role": "user", "content": " "})
            else:
                out.append({"role": "user", "content": content})
            pending_tool_ids = []
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            blocks: list[dict] = []
            if content.strip():
                blocks.append({"type": "text", "text": content})
            new_pending: list[str] = []
            for tc in tool_calls:
                # Prefer the stashed id from a prior Anthropic call so result
                # blocks line up. Otherwise generate a fresh one.
                tid = tc.get("_anthropic_id") or _new_id()
                new_pending.append(tid)
                fn = tc.get("function", {}) or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tid,
                        "name": fn.get("name", ""),
                        "input": args,
                    }
                )
            if not blocks:
                blocks = [{"type": "text", "text": " "}]
            out.append({"role": "assistant", "content": blocks})
            pending_tool_ids = new_pending
            continue

        if role == "tool":
            if not pending_tool_ids:
                # Unmatched tool message — skip.
                continue
            tid = pending_tool_ids.pop(0)
            result_block = {
                "type": "tool_result",
                "tool_use_id": tid,
                "content": content,
            }
            # Merge into the most recent user message if it's already a tool-result block.
            if out and out[-1].get("role") == "user" and isinstance(out[-1].get("content"), list):
                out[-1]["content"].append(result_block)
            else:
                out.append({"role": "user", "content": [result_block]})
            continue

    return "\n\n".join(system_parts), out
