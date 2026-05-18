"""Prepare local personalization artifacts for a coding assistant."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import chat as chat_mod
from .index import project_meta_dir

TRAINING_DIR = "assistant_training"


@dataclass
class TrainingArtifacts:
    output_dir: Path
    records_path: Path
    modelfile_path: Path
    readme_path: Path
    metadata_path: Path
    model_name: str
    base_model: str
    examples: int
    created_model: bool = False


def default_model_name(root: Path) -> str:
    """Return an Ollama-friendly model name derived from the project name."""
    name = root.resolve().name or "assistant"
    slug = re.sub(r"[^a-z0-9_.-]+", "-", name.lower()).strip("-._")
    return f"{slug or 'assistant'}-assistant"


def default_output_dir(root: Path) -> Path:
    return project_meta_dir(root) / TRAINING_DIR


def _strip_context_from_user(content: str) -> str:
    marker = "\n\n---\n\nQuestion: "
    if content.startswith("Context from codebase:") and marker in content:
        return content.rsplit(marker, 1)[1].strip()
    return content.strip()


def _plain_chat_message(msg: dict) -> dict | None:
    role = msg.get("role")
    if role not in {"user", "assistant"}:
        return None
    content = (msg.get("content") or "").strip()
    if not content:
        return None
    if role == "user":
        content = _strip_context_from_user(content)
    return {"role": role, "content": content}


def _examples_from_last_conversation(
    root: Path, system_prompt: str, max_examples: int
) -> list[dict]:
    saved = chat_mod._load_conversation(root)
    if not saved:
        return []
    history: list[dict] = []
    examples: list[dict] = []
    for raw in saved.get("messages") or []:
        msg = _plain_chat_message(raw)
        if msg is None:
            continue
        if msg["role"] == "assistant" and any(m["role"] == "user" for m in history):
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        *history,
                        msg,
                    ]
                }
            )
            if len(examples) >= max_examples:
                break
        history.append(msg)
    return examples


def _quote_modelfile(text: str) -> str:
    return '"""\n' + text.replace('"""', '\\"\\"\\"') + '\n"""'


def _personalized_system_prompt(root: Path, profile: str) -> str:
    prompt = chat_mod._system_prompt_for(root)
    profile = profile.strip()
    if not profile:
        return prompt
    return (
        f"{prompt}\n\n"
        "## Personal assistant profile\n\n"
        "These user-provided preferences should shape style, defaults, and coding tradeoffs.\n\n"
        f"{profile}\n"
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_modelfile(path: Path, *, base_model: str, system_prompt: str) -> None:
    body = "\n".join(
        [
            f"FROM {base_model}",
            "PARAMETER temperature 0",
            "PARAMETER num_ctx 65536",
            f"SYSTEM {_quote_modelfile(system_prompt)}",
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")


def _write_readme(
    path: Path,
    *,
    root: Path,
    model_name: str,
    base_model: str,
    examples: int,
) -> None:
    body = f"""# Personal Coding Assistant Artifacts

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}.

This directory is local-only. It contains:

- `Modelfile` - an Ollama model recipe that bakes in the codebase-rag system prompt, project notes, and any personal profile text you supplied.
- `training.jsonl` - chat-style examples exported from the last saved conversation for this project.
- `metadata.json` - generation metadata.

Create the local Ollama assistant:

```bash
ollama create {model_name} -f Modelfile
```

Use it with this project:

```bash
codebase-rag chat --root {root} --model {model_name}
```

Base model: `{base_model}`
Training examples exported: `{examples}`

Ollama `create` personalizes the runtime prompt; it does not fine-tune weights.
Use `training.jsonl` with a separate local fine-tuning tool if you want weight
training later.
"""
    path.write_text(body, encoding="utf-8")


def build_artifacts(
    root: Path,
    *,
    output_dir: Path | None = None,
    model_name: str | None = None,
    base_model: str | None = None,
    profile: str = "",
    include_conversation: bool = True,
    max_examples: int = 200,
    create: bool = False,
) -> TrainingArtifacts:
    root = root.resolve()
    out = output_dir.resolve() if output_dir else default_output_dir(root)
    out.mkdir(parents=True, exist_ok=True)

    model = model_name or default_model_name(root)
    base = chat_mod._resolve_model(base_model)
    profile_parts = []
    if profile.strip():
        profile_parts.append(profile.strip())
    system_prompt = _personalized_system_prompt(root, "\n\n".join(profile_parts))

    examples = (
        _examples_from_last_conversation(root, system_prompt, max_examples)
        if include_conversation and max_examples > 0
        else []
    )

    records_path = out / "training.jsonl"
    modelfile_path = out / "Modelfile"
    readme_path = out / "README.md"
    metadata_path = out / "metadata.json"

    _write_jsonl(records_path, examples)
    _write_modelfile(modelfile_path, base_model=base, system_prompt=system_prompt)
    _write_readme(
        readme_path,
        root=root,
        model_name=model,
        base_model=base,
        examples=len(examples),
    )
    metadata_path.write_text(
        json.dumps(
            {
                "root": str(root),
                "model_name": model,
                "base_model": base,
                "examples": len(examples),
                "records_path": str(records_path),
                "modelfile_path": str(modelfile_path),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    created = False
    if create:
        subprocess.run(["ollama", "create", model, "-f", str(modelfile_path)], check=True)
        created = True

    return TrainingArtifacts(
        output_dir=out,
        records_path=records_path,
        modelfile_path=modelfile_path,
        readme_path=readme_path,
        metadata_path=metadata_path,
        model_name=model,
        base_model=base,
        examples=len(examples),
        created_model=created,
    )
