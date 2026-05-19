"""Local skill discovery and rendering for prompt guidance.

Skills are trusted local files. They are never fetched from the network and
never written into the user's indexed repository.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

DEFAULT_USER_SKILL_DIR = Path.home() / ".codebase-rag" / "skills"


@dataclass(frozen=True)
class Snippet:
    id: str
    title: str
    body: str
    triggers: tuple[str, ...]
    file_globs: tuple[str, ...]
    packages: tuple[str, ...]
    max_tokens: int
    path: Path


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    guidance: str
    triggers: tuple[str, ...]
    file_globs: tuple[str, ...]
    snippet_budget: int
    priority: int
    path: Path
    snippets: tuple[Snippet, ...]


@dataclass(frozen=True)
class SkillMatch:
    skill: Skill
    snippets: tuple[Snippet, ...]
    reason: str


def load_skill_library(extra_dirs: tuple[Path, ...] = ()) -> list[Skill]:
    """Load bundled, user, and explicitly configured skills."""
    roots = [Path(__file__).parent, DEFAULT_USER_SKILL_DIR, *extra_dirs]
    skills: dict[str, Skill] = {}
    for root in roots:
        for skill_dir in _candidate_skill_dirs(root):
            skill = _load_skill(skill_dir)
            if skill is None:
                continue
            existing = skills.get(skill.id)
            if existing is None or _source_rank(skill.path) >= _source_rank(existing.path):
                skills[skill.id] = skill
    return sorted(skills.values(), key=lambda s: (-s.priority, s.id))


def detect_matches(
    library: list[Skill],
    *,
    text: str = "",
    paths: tuple[str, ...] = (),
    active_skill_ids: set[str] | None = None,
    active_snippet_ids: set[str] | None = None,
) -> list[SkillMatch]:
    """Return newly relevant skills/snippets for the supplied signals."""
    active_skill_ids = active_skill_ids or set()
    active_snippet_ids = active_snippet_ids or set()
    text_l = text.lower()
    out: list[SkillMatch] = []
    for skill in library:
        skill_hit = _matches(skill.triggers, skill.file_globs, text_l, paths)
        matched_snippets = tuple(
            snip
            for snip in skill.snippets
            if snip.id not in active_snippet_ids and _snippet_matches(snip, text_l, paths)
        )
        if skill.id not in active_skill_ids and (skill_hit or matched_snippets):
            reason = _reason(skill.triggers, skill.file_globs, text_l, paths)
            out.append(SkillMatch(skill=skill, snippets=matched_snippets, reason=reason))
        elif skill.id in active_skill_ids and matched_snippets:
            out.append(
                SkillMatch(
                    skill=skill,
                    snippets=matched_snippets,
                    reason=_reason(
                        (*skill.triggers, *(p for s in matched_snippets for p in s.triggers)),
                        (*skill.file_globs, *(g for s in matched_snippets for g in s.file_globs)),
                        text_l,
                        paths,
                    ),
                )
            )
    return out


def render_active_block(
    active_skills: dict[str, Skill], active_snippets: dict[str, Snippet]
) -> str:
    """Render trusted active skill guidance for the system prompt."""
    if not active_skills:
        return ""
    parts = [
        "## Active local skills",
        "",
        "The following trusted local skill instructions were selected automatically. "
        "Use them as best-practice guidance, but continue to follow the user's request "
        "and the repository's existing patterns.",
    ]
    for skill in sorted(active_skills.values(), key=lambda s: (-s.priority, s.id)):
        parts.extend(["", f"### {skill.name} (`{skill.id}`)", "", skill.guidance.strip()])
        budget_chars = max(0, skill.snippet_budget * 4)
        used = 0
        rendered = []
        for snip in sorted(skill.snippets, key=lambda s: s.id):
            if snip.id not in active_snippets:
                continue
            body = snip.body.strip()
            cap = max(0, snip.max_tokens * 4)
            if cap and len(body) > cap:
                body = body[:cap].rstrip() + "\n..."
            candidate = f"#### Snippet: {snip.title} (`{snip.id}`)\n\n{body}"
            if budget_chars and used + len(candidate) > budget_chars:
                continue
            rendered.append(candidate)
            used += len(candidate)
        if rendered:
            parts.extend(["", "Examples are reference snippets, not mandatory templates.", ""])
            parts.append("\n\n".join(rendered))
    return "\n".join(parts).strip() + "\n"


def matches_to_event(matches: list[SkillMatch]) -> dict[str, Any]:
    skill_ids = []
    snippet_ids = []
    sources = {}
    reasons = {}
    for match in matches:
        if match.skill.id not in skill_ids:
            skill_ids.append(match.skill.id)
            sources[match.skill.id] = str(match.skill.path)
            reasons[match.skill.id] = match.reason
        for snip in match.snippets:
            if snip.id not in snippet_ids:
                snippet_ids.append(snip.id)
                sources[snip.id] = str(snip.path)
    return {"skills": skill_ids, "snippets": snippet_ids, "sources": sources, "reasons": reasons}


def _candidate_skill_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if (root / "manifest.json").is_file() and (root / "SKILL.md").is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and (p / "manifest.json").is_file() and (p / "SKILL.md").is_file()
    )


def _load_skill(skill_dir: Path) -> Skill | None:
    try:
        manifest = json.loads((skill_dir / "manifest.json").read_text(encoding="utf-8"))
        guidance = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        return None
    skill_id = str(manifest.get("id") or skill_dir.name).strip()
    if not skill_id:
        return None
    snippets = []
    snippets_dir = skill_dir / "snippets"
    if snippets_dir.is_dir():
        for path in sorted(snippets_dir.glob("*.md")):
            snip = _load_snippet(path)
            if snip is not None:
                snippets.append(snip)
    return Skill(
        id=skill_id,
        name=str(manifest.get("name") or skill_id),
        guidance=guidance,
        triggers=_string_tuple(manifest.get("triggers")),
        file_globs=_string_tuple(manifest.get("file_globs")),
        snippet_budget=int(manifest.get("snippet_budget") or 1200),
        priority=int(manifest.get("priority") or 0),
        path=skill_dir,
        snippets=tuple(snippets),
    )


def _load_snippet(path: Path) -> Snippet | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    meta, body = _split_frontmatter(raw)
    snippet_id = str(meta.get("id") or path.stem).strip()
    if not snippet_id:
        return None
    return Snippet(
        id=snippet_id,
        title=str(meta.get("title") or snippet_id),
        body=body,
        triggers=_string_tuple(meta.get("triggers")),
        file_globs=_string_tuple(meta.get("file_globs")),
        packages=_string_tuple(meta.get("packages")),
        max_tokens=int(meta.get("max_tokens") or 300),
        path=path,
    )


def _split_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    if not raw.startswith("---\n"):
        return {}, raw
    end = raw.find("\n---\n", 4)
    if end == -1:
        return {}, raw
    meta_text = raw[4:end]
    body = raw[end + 5 :]
    meta: dict[str, Any] = {}
    for line in meta_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                meta[key] = json.loads(value)
            except json.JSONDecodeError:
                meta[key] = []
        elif value.isdigit():
            meta[key] = int(value)
        else:
            meta[key] = value.strip('"')
    return meta, body


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item).strip())
    return ()


def _matches(
    triggers: tuple[str, ...], file_globs: tuple[str, ...], text_l: str, paths: tuple[str, ...]
) -> bool:
    return _trigger_match(triggers, text_l) or _path_match(file_globs, paths)


def _snippet_matches(snip: Snippet, text_l: str, paths: tuple[str, ...]) -> bool:
    if _trigger_match((*snip.triggers, *snip.packages), text_l):
        return True
    specific_globs = tuple(glob for glob in snip.file_globs if glob not in {"*.py", "**/*.py"})
    return _path_match(specific_globs, paths)


def _trigger_match(triggers: tuple[str, ...], text_l: str) -> bool:
    for trigger in triggers:
        t = trigger.lower().strip()
        if not t:
            continue
        if re.search(rf"(^|[^a-z0-9_.-]){re.escape(t)}([^a-z0-9_.-]|$)", text_l):
            return True
    return False


def _path_match(file_globs: tuple[str, ...], paths: tuple[str, ...]) -> bool:
    for path in paths:
        normalized = path.replace("\\", "/")
        name = normalized.rsplit("/", 1)[-1]
        for glob in file_globs:
            if fnmatch(normalized, glob) or fnmatch(name, glob):
                return True
    return False


def _reason(
    triggers: tuple[str, ...], file_globs: tuple[str, ...], text_l: str, paths: tuple[str, ...]
) -> str:
    for trigger in triggers:
        t = trigger.lower().strip()
        if t and re.search(rf"(^|[^a-z0-9_.-]){re.escape(t)}([^a-z0-9_.-]|$)", text_l):
            return f"text:{trigger}"
    for path in paths:
        normalized = path.replace("\\", "/")
        name = normalized.rsplit("/", 1)[-1]
        for glob in file_globs:
            if fnmatch(normalized, glob) or fnmatch(name, glob):
                return f"path:{glob}"
    return "matched"


def _source_rank(path: Path) -> int:
    try:
        path.relative_to(DEFAULT_USER_SKILL_DIR)
        return 2
    except ValueError:
        pass
    try:
        path.relative_to(Path(__file__).parent)
        return 1
    except ValueError:
        return 3
