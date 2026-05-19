from __future__ import annotations

import json
from pathlib import Path

from codebase_rag import audit, chat, index, url_ingest
from codebase_rag.providers import ChatProvider
from codebase_rag.tools import DEFAULT_SHELL_RUNNER, tool_schemas_for


class WriteToolProvider(ChatProvider):
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def iter_chat_events(self, model, messages, tools, options):
        self.calls += 1
        if self.calls == 1:
            yield (
                "done",
                "",
                [
                    {
                        "function": {
                            "name": "write_file",
                            "arguments": {"path": "owned.txt", "content": "changed\n"},
                        }
                    }
                ],
                _stats(),
            )
            return
        yield ("done", "done", [], _stats())


def test_read_only_agent_turn_rejects_unadvertised_write_tool(monkeypatch, tmp_path: Path) -> None:
    provider = WriteToolProvider()
    monkeypatch.setattr(chat, "retrieve", lambda collection, query: [])
    session = chat.ChatSession(
        root=tmp_path,
        db_path=tmp_path / "db",
        chat_model="fake-model",
        architect_model=None,
        provider=provider,
        provider_name="fake",
        collection=object(),
        meta_dir=tmp_path / "meta",
        session="test",
        tool_schemas=tool_schemas_for(read_only=True),
        web_config=None,
        shell_timeout=30,
        shell_runner=DEFAULT_SHELL_RUNNER,
        shell_network="none",
        check_command="",
        repair_attempts=0,
        confirm_writes=False,
        skills_enabled=False,
        skill_dirs=(),
        skill_library=[],
        active_skills={},
        active_snippets={},
        base_system_prompt=chat.SYSTEM_PROMPT,
        history=[{"role": "system", "content": chat.SYSTEM_PROMPT}],
        pinned_paths=[],
        touched_files=set(),
        read_only=True,
        allow_shell=False,
        allow_web=False,
        web_allow=(),
        web_block=(),
        searxng_url="",
        resumed_marker="",
        notes_marker="no notes",
    )

    events = list(chat.agent_turn(session, "please write a file"))

    tool_results = [event for event in events if event[0] == "tool_result"]
    assert tool_results
    parsed = json.loads(tool_results[0][3])
    assert parsed["ok"] is False
    assert "not enabled" in parsed["error"]
    assert not (tmp_path / "owned.txt").exists()
    assert provider.calls == 2


def test_iter_source_files_ignores_excluded_names_only_under_root(tmp_path: Path) -> None:
    root = tmp_path / "build" / "project"
    root.mkdir(parents=True)
    source = root / "main.py"
    source.write_text("print('ok')\n", encoding="utf-8")

    files = list(index.iter_source_files(root))

    assert files == [source]


def test_url_ingest_policy_rejects_malformed_port() -> None:
    ok, reason = url_ingest._validate_url_policy(
        "https://docs.example.com:bad/path",
        allow_patterns=("docs.example.com",),
        block_patterns=(),
    )

    assert not ok
    assert "Port could not be cast" in reason


def test_audit_redacts_sensitive_keys(tmp_path: Path) -> None:
    audit.log_event(
        tmp_path,
        "session",
        "tool_call",
        args={
            "token": "short-secret",
            "headers": {"Authorization": "Bearer short-secret"},
            "path": "src/app.py",
        },
    )

    row = json.loads((tmp_path / audit.AUDIT_FILE).read_text(encoding="utf-8"))
    assert row["args"]["token"] == {"_redacted": True}
    assert row["args"]["headers"]["Authorization"] == {"_redacted": True}
    assert row["args"]["path"] == "src/app.py"


def _stats() -> dict:
    return {
        "prompt_tokens": 1,
        "output_tokens": 1,
        "elapsed": 0.0,
        "prompt_eval_duration": 0.0,
        "eval_duration": 0.0,
        "load_duration": 0.0,
    }
