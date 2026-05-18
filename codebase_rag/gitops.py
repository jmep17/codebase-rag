"""Git plumbing for codebase-rag: review-then-commit with deterministic undo.

All operations run via `git` subprocess (no extra deps). Every codebase-rag commit
is tagged with `[codebase-rag]` in its subject so `undo_last` can find the most
recent agent commit deterministically and revert it.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

COMMIT_TAG = "[codebase-rag]"


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def is_git_repo(root: Path) -> bool:
    result = _run_git(root, "rev-parse", "--is-inside-work-tree")
    return result.returncode == 0 and result.stdout.strip() == "true"


def status_short(root: Path) -> str:
    result = _run_git(root, "status", "--short")
    if result.returncode != 0:
        return f"(git status failed: {result.stderr.strip()})"
    return result.stdout


def has_pending_changes(root: Path) -> bool:
    result = _run_git(root, "status", "--porcelain")
    return result.returncode == 0 and bool(result.stdout.strip())


def pending_diff(root: Path, path: str | None = None) -> str:
    args = ["diff", "HEAD"]
    if path:
        args += ["--", path]
    result = _run_git(root, *args)
    if result.returncode != 0:
        return f"(git diff failed: {result.stderr.strip()})"
    return result.stdout


def diff_stat(root: Path, paths: Sequence[str] | None = None) -> str:
    args = ["diff", "--stat", "HEAD"]
    if paths:
        args += ["--", *paths]
    result = _run_git(root, *args)
    return result.stdout if result.returncode == 0 else ""


def commit_pending(root: Path, message: str, *, paths: Sequence[str] | None = None) -> dict:
    """Stage and commit. If `paths` is given, stage exactly those; otherwise stage all."""
    if paths is not None:
        for rel in paths:
            r = _run_git(root, "add", "--", rel)
            if r.returncode != 0:
                return {"ok": False, "error": f"git add {rel} failed: {r.stderr.strip()}"}
    else:
        r = _run_git(root, "add", "-A")
        if r.returncode != 0:
            return {"ok": False, "error": f"git add failed: {r.stderr.strip()}"}
    # Anything actually staged?
    check = _run_git(root, "diff", "--cached", "--quiet")
    if check.returncode == 0:
        return {"ok": False, "error": "nothing staged after `git add` — working tree may be clean"}
    msg = f"{COMMIT_TAG} {message}".strip() if message else COMMIT_TAG
    commit = _run_git(root, "commit", "-m", msg)
    if commit.returncode != 0:
        return {"ok": False, "error": f"git commit failed: {commit.stderr.strip()}"}
    sha = _run_git(root, "rev-parse", "HEAD").stdout.strip()
    files = (
        _run_git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", sha)
        .stdout.strip()
        .splitlines()
    )
    return {"ok": True, "sha": sha, "short": sha[:12], "message": msg, "files": files}


def last_codebase_rag_commit(root: Path) -> dict | None:
    """Find the most recent commit whose subject starts with [codebase-rag]."""
    result = _run_git(root, "log", f"--grep=^{COMMIT_TAG}", "-n", "1", "--format=%H%n%s")
    if result.returncode != 0:
        return None
    lines = result.stdout.strip().splitlines()
    if len(lines) < 2:
        return None
    sha, subject = lines[0], lines[1]
    files = (
        _run_git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", sha)
        .stdout.strip()
        .splitlines()
    )
    return {"sha": sha, "short": sha[:12], "subject": subject, "files": files}


def undo_last(root: Path) -> dict:
    last = last_codebase_rag_commit(root)
    if last is None:
        return {"ok": False, "error": "no [codebase-rag] commits found in history"}
    if has_pending_changes(root):
        return {
            "ok": False,
            "error": "cannot undo while working tree has uncommitted changes — commit or stash first",
        }
    result = _run_git(root, "revert", "--no-edit", last["sha"])
    if result.returncode != 0:
        return {"ok": False, "error": f"git revert failed: {result.stderr.strip()}"}
    new_sha = _run_git(root, "rev-parse", "HEAD").stdout.strip()
    return {
        "ok": True,
        "reverted_sha": last["short"],
        "revert_sha": new_sha[:12],
        "subject": last["subject"],
        "files": last["files"],
    }
