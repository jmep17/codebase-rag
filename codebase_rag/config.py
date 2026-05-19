"""Runtime path configuration for local codebase-rag state."""

from __future__ import annotations

import os
from pathlib import Path

ENV_HOME = "CODEBASE_RAG_HOME"


def data_home() -> Path:
    """Return the directory used for codebase-rag state outside project repos."""
    configured = os.environ.get(ENV_HOME)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".codebase-rag"


def default_db_path() -> Path:
    return data_home() / "db"


def meta_root() -> Path:
    return data_home() / "meta"


def default_user_skill_dir() -> Path:
    return data_home() / "skills"
