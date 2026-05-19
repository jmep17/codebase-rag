from __future__ import annotations

import json
from pathlib import Path

from codebase_rag import chat
from codebase_rag import skills as skills_mod


def test_loads_bundled_python_skill() -> None:
    library = skills_mod.load_skill_library(())

    python = next(skill for skill in library if skill.id == "python")

    assert python.name == "Python"
    assert {snippet.id for snippet in python.snippets} >= {
        "pyproject-basic",
        "argparse-cli",
        "pytest-basic",
        "pathlib-file-io",
        "httpx-client",
        "pydantic-settings",
        "fastapi-route",
    }


def test_detects_skill_and_snippet_from_paths_and_text() -> None:
    library = skills_mod.load_skill_library(())

    matches = skills_mod.detect_matches(
        library,
        text="add a pytest test for this behavior",
        paths=("tests/test_example.py",),
    )

    assert [match.skill.id for match in matches] == ["python"]
    assert "pytest-basic" in {snippet.id for match in matches for snippet in match.snippets}


def test_extra_skill_dir_overrides_bundled_skill(tmp_path: Path) -> None:
    skill_dir = tmp_path / "python"
    skill_dir.mkdir()
    (skill_dir / "manifest.json").write_text(
        json.dumps(
            {
                "id": "python",
                "name": "Custom Python",
                "triggers": ["python"],
                "file_globs": ["*.py"],
                "priority": 200,
            }
        ),
        encoding="utf-8",
    )
    (skill_dir / "SKILL.md").write_text("Custom guidance.\n", encoding="utf-8")

    library = skills_mod.load_skill_library((tmp_path,))

    python = next(skill for skill in library if skill.id == "python")
    assert python.name == "Custom Python"
    assert python.guidance == "Custom guidance.\n"


class FakeProvider:
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
                            "arguments": {
                                "path": "main.py",
                                "content": "print('hello')\n",
                            },
                        }
                    }
                ],
                _stats(),
            )
            return
        yield ("done", "done", [], _stats())


def test_agent_turn_activates_skill_before_write(monkeypatch, tmp_path: Path) -> None:
    provider = FakeProvider()
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
        tool_schemas=[],
        web_config=None,
        shell_timeout=30,
        shell_runner="host",
        shell_network="none",
        check_command="",
        repair_attempts=0,
        confirm_writes=False,
        skills_enabled=True,
        skill_dirs=(),
        skill_library=skills_mod.load_skill_library(()),
        active_skills={},
        active_snippets={},
        base_system_prompt=chat.SYSTEM_PROMPT,
        history=[{"role": "system", "content": chat.SYSTEM_PROMPT}],
        pinned_paths=[],
        touched_files=set(),
        read_only=False,
        allow_shell=False,
        allow_web=False,
        web_allow=(),
        web_block=(),
        searxng_url="",
        resumed_marker="",
        notes_marker="no notes",
    )

    events = list(chat.agent_turn(session, "create a tiny script"))

    skill_events = [event for event in events if event[0] == "skill_activated"]
    assert skill_events
    assert skill_events[0][1]["trigger"] == "pre_write:write_file"
    assert "python" in session.active_skills
    assert not (tmp_path / "main.py").exists()
    assert provider.calls == 2


def _stats() -> dict:
    return {
        "prompt_tokens": 1,
        "output_tokens": 1,
        "elapsed": 0.0,
        "prompt_eval_duration": 0.0,
        "eval_duration": 0.0,
        "load_duration": 0.0,
    }
