"""Textual TUI for codebase-rag (Phase 2 — live agent loop).

Wires the chat-scene widgets to a real chat.agent_turn generator:
retrieval, streaming tokens, tool calls, and per-tool confirmations all
render into the same cards mocked in Phase 1.

Architecture:
- run_tui() builds a ChatSession via chat.init_chat_session, then launches
  the App. Same flag surface as chat.agent_loop.
- CodebaseRagApp keeps the three-pane layout (projects · conversation ·
  context) from Phase 1. Conversation is empty initially; each user
  submission appends a Turn widget that grows as agent_turn yields events.
- run_turn() is a Textual @work(thread=True) worker. It drives the
  generator and uses call_from_thread to mutate widgets on the UI thread.
  For ("confirm", ...) events the worker pushes a ModalScreen and blocks
  on a queue.Queue until the modal puts the resolved args (or None) in.
- All audit logging happens inside chat.agent_turn — the TUI never
  touches the audit log, so events stay byte-identical to the line mode.

Entry point: run_tui(). Called by codebase_rag/__main__.py when --tui is set.
"""

from __future__ import annotations

import difflib
import json
import queue
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Input, Markdown, Static

from . import audit as audit_mod
from . import chat as chat_mod
from . import gitops

HINT_TEXT = (
    "^R retrieval inspector   ^L audit overlay   ^K command palette   "
    "/ slash menu   j/k select turns when composer is unfocused   ? help"
)
COMPOSER_PLACEHOLDER = "type a message — Enter to send · :q to quit"


def _clip(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


# --- Modals ---------------------------------------------------------------


class _ConfirmBase(ModalScreen):
    """Shared bones for the two confirm modals: hold a result queue, expose
    approve/reject actions that put a value in the queue and dismiss the
    screen so the worker thread unblocks."""

    BINDINGS = [
        Binding("a", "approve", "Approve", priority=True),
        Binding("y", "approve", "Approve", show=False, priority=True),
        Binding("r", "reject", "Reject", priority=True),
        Binding("n", "reject", "Reject", show=False, priority=True),
        Binding("escape", "reject", "Reject", show=False, priority=True),
    ]

    def __init__(self, args: dict, result_queue: queue.Queue) -> None:
        super().__init__()
        self.args = args
        self.result_queue = result_queue

    def action_approve(self) -> None:
        self.result_queue.put(dict(self.args))
        self.dismiss()

    def action_reject(self) -> None:
        self.result_queue.put(None)
        self.dismiss()


class ConfirmWriteModal(_ConfirmBase):
    """Mirrors designs/cli.html scene 'confirm-write'."""

    BINDINGS = _ConfirmBase.BINDINGS + [
        Binding("up", "preview_up", "Scroll up", show=False, priority=True),
        Binding("down", "preview_down", "Scroll down", show=False, priority=True),
        Binding("pageup", "preview_page_up", "Page up", show=False, priority=True),
        Binding("pagedown", "preview_page_down", "Page down", show=False, priority=True),
        Binding("home", "preview_home", "Top", show=False, priority=True),
        Binding("end", "preview_end", "Bottom", show=False, priority=True),
    ]

    def __init__(self, tname: str, args: dict, result_queue: queue.Queue) -> None:
        super().__init__(args, result_queue)
        self.tname = tname

    def compose(self) -> ComposeResult:
        path = self.args.get("project_path") if self.tname == "create_project" else None
        path = path or self.args.get("path", "?")
        with Vertical(classes="confirm-modal write"):
            yield Static(
                f"[bold]✎ model wants to {self.tname}[/]",
                classes="modal-head write",
            )
            yield Static(path, classes="modal-target")
            with VerticalScroll(id="confirm-preview", classes="modal-diff-scroll"):
                yield Static(self._build_preview(), classes="modal-diff")
            yield Static(
                "[bold]\\[a][/]pprove   [bold]\\[r][/]eject   [bold]↑/↓ PgUp/PgDn[/] scroll",
                classes="modal-prompt",
            )

    def _preview(self) -> VerticalScroll:
        return self.query_one("#confirm-preview", VerticalScroll)

    def action_preview_up(self) -> None:
        self._preview().scroll_up()

    def action_preview_down(self) -> None:
        self._preview().scroll_down()

    def action_preview_page_up(self) -> None:
        self._preview().scroll_page_up()

    def action_preview_page_down(self) -> None:
        self._preview().scroll_page_down()

    def action_preview_home(self) -> None:
        self._preview().scroll_home()

    def action_preview_end(self) -> None:
        self._preview().scroll_end()

    def _build_preview(self) -> Text:
        preview = Text()
        if self.tname == "create_project":
            files = self.args.get("files")
            if isinstance(files, list):
                paths = [item.get("path", "?") for item in files if isinstance(item, dict)]
                file_count = len(files)
            else:
                paths = ["README.md", ".gitignore"]
                file_count = 2
            preview.append(f"{file_count} file(s)\n", style="#5b8b73")
            for path in paths:
                preview.append(f"+ {path}\n", style="#4ade80")
            return preview
        if self.tname == "edit_file":
            old = self.args.get("old_string") or ""
            new = self.args.get("new_string") or ""
            path = self.args.get("path", "?")
            for line in difflib.unified_diff(
                old.splitlines(),
                new.splitlines(),
                lineterm="",
                fromfile=f"{path} (current)",
                tofile=f"{path} (proposed)",
            ):
                if line.startswith("-"):
                    style = "#f87171"
                elif line.startswith("+"):
                    style = "#4ade80"
                else:
                    style = ""
                preview.append(line + "\n", style=style)
            return preview
        content = self.args.get("content", "") or ""
        text_lines = content.splitlines() or [""]
        size = len(content.encode("utf-8"))
        head = f"{len(text_lines)} lines, {size} bytes"
        preview.append(f"{head}\n", style="#5b8b73")
        for i, ln in enumerate(text_lines, 1):
            preview.append(f"{i:4d}: {ln}\n")
        return preview


class ConfirmShellModal(_ConfirmBase):
    """Mirrors designs/cli.html scene 'confirm-shell'."""

    def __init__(
        self,
        args: dict,
        result_queue: queue.Queue,
        *,
        shell_runner: str,
        shell_network: str,
        shell_timeout: float,
    ) -> None:
        super().__init__(args, result_queue)
        self.shell_runner = shell_runner
        self.shell_network = shell_network
        self.shell_timeout = shell_timeout

    def compose(self) -> ComposeResult:
        cmd = (self.args.get("command") or "").strip()
        runner = self.shell_runner
        net = self.shell_network if runner != "host" else "host"
        meta = (
            f"[#5b8b73]runner:[/] {runner}  "
            f"[#5b8b73]· network:[/] {net}  "
            f"[#5b8b73]· timeout:[/] {self.shell_timeout:.0f}s"
        )
        with Vertical(classes="confirm-modal shell"):
            yield Static(
                "[bold]▶ model wants to run a shell command[/]",
                classes="modal-head shell",
            )
            yield Static(meta, classes="modal-meta")
            yield Static(f"[#4ade80]$[/] {cmd}", classes="modal-cmd")
            yield Static(
                "[bold]\\[a][/]pprove (run)   [bold]\\[r][/]eject",
                classes="modal-prompt",
            )


class SlashConfirmModal(ModalScreen):
    """Confirmation prompt for slash commands with side effects."""

    BINDINGS = [
        Binding("a", "approve", "Approve", priority=True),
        Binding("y", "approve", "Approve", show=False, priority=True),
        Binding("r", "reject", "Reject", priority=True),
        Binding("n", "reject", "Reject", show=False, priority=True),
        Binding("escape", "reject", "Reject", show=False, priority=True),
    ]

    def __init__(self, prompt: str, result_queue: queue.Queue) -> None:
        super().__init__()
        self.prompt = prompt
        self.result_queue = result_queue

    def compose(self) -> ComposeResult:
        with Vertical(classes="confirm-modal shell"):
            yield Static("[bold]confirm command[/]", classes="modal-head shell")
            yield Static(self.prompt, classes="modal-cmd")
            yield Static(
                "[bold]\\[a][/]pprove   [bold]\\[r][/]eject",
                classes="modal-prompt",
            )

    def action_approve(self) -> None:
        self.result_queue.put(True)
        self.dismiss()

    def action_reject(self) -> None:
        self.result_queue.put(False)
        self.dismiss()


# --- Palette (slash + command) -------------------------------------------


@dataclass(frozen=True)
class PaletteItem:
    """One row in a palette. `payload` is the value reported back to the
    caller's ``on_accept`` when the user hits Enter (for slash items this
    is the command prefix to insert; for command items this is a callable)."""

    name: str
    desc: str
    requires: str = ""
    payload: object = None


def _palette_score(query: str, item: PaletteItem) -> int:
    """Tiny fuzzy ranker: substring + char-sequence + name-prefix bonus.
    Returns a positive int for a match, or 0 to filter out."""
    q = query.strip().lower()
    if not q:
        return 1
    hay = f"{item.name} {item.desc}".lower()
    if q in item.name.lower():
        return 200 - item.name.lower().find(q)
    if q in hay:
        return 100 - hay.find(q)
    # Char-sequence match
    i = 0
    for ch in hay:
        if ch == q[i]:
            i += 1
            if i == len(q):
                return 10
    return 0


class PaletteModal(ModalScreen):
    """Shared palette UI. Configurations vary by:

    - ``items``: the rows to filter.
    - ``initial``: prefilled query (slash palette starts with "" after the
      consumer strips the leading "/").
    - ``on_accept``: invoked with the chosen item's ``payload`` and the
      remainder of the query string (everything after the matched name).

    Esc dismisses. ↑/↓ navigate. Enter accepts.
    """

    BINDINGS = [
        Binding("escape", "dismiss_palette", "Dismiss", priority=True),
        Binding("up", "move(-1)", "Up", show=False, priority=True),
        Binding("down", "move(1)", "Down", show=False, priority=True),
        Binding("ctrl+p", "move(-1)", "Up", show=False, priority=True),
        Binding("ctrl+n", "move(1)", "Down", show=False, priority=True),
        Binding("enter", "accept", "Accept", show=False, priority=True),
        Binding("tab", "complete", "Complete", show=False, priority=True),
    ]

    def __init__(
        self,
        items: list[PaletteItem],
        *,
        title: str,
        initial: str = "",
        on_accept: Callable[[PaletteItem, str], None],
    ) -> None:
        super().__init__()
        self._items = items
        self._title = title
        self._initial = initial
        self._on_accept = on_accept
        self._sel = 0
        self._filtered: list[PaletteItem] = list(items)

    def compose(self) -> ComposeResult:
        with Vertical(classes="palette-modal"), Vertical(classes="palette"):
            with Horizontal(classes="pHead"):
                yield Static(self._title, classes="gt")
                yield Input(value=self._initial, id="palette-input")
                yield Static("esc", classes="esc")
            yield VerticalScroll(id="palette-list", classes="pList")
            yield Static(
                "[#5b8b73]↑↓ navigate   ↵ run   tab complete   esc dismiss[/]",
                classes="pFoot",
            )

    def on_mount(self) -> None:
        self.query_one("#palette-input", Input).focus()
        self._rebuild_list()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "palette-input":
            return
        self._sel = 0
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        query = self.query_one("#palette-input", Input).value
        scored: list[tuple[int, PaletteItem]] = []
        for it in self._items:
            s = _palette_score(query.strip().lstrip(":"), it)
            if s > 0:
                scored.append((s, it))
        scored.sort(key=lambda t: -t[0])
        self._filtered = [it for _, it in scored]
        if self._sel >= len(self._filtered):
            self._sel = max(0, len(self._filtered) - 1)
        list_node = self.query_one("#palette-list", VerticalScroll)
        list_node.remove_children()
        for i, it in enumerate(self._filtered):
            row = Horizontal(classes=f"pItem{' sel' if i == self._sel else ''}")
            list_node.mount(row)
            row.mount(Static(it.name, classes="pName"))
            row.mount(Static(it.desc, classes="pDesc"))
            if it.requires:
                row.mount(Static(it.requires, classes="pReq"))
            else:
                row.mount(Static("", classes="pReq"))

    def action_move(self, delta: int) -> None:
        if not self._filtered:
            return
        self._sel = (self._sel + delta) % len(self._filtered)
        self._rebuild_list()

    def action_accept(self) -> None:
        if not self._filtered:
            self.dismiss()
            return
        choice = self._filtered[self._sel]
        query = self.query_one("#palette-input", Input).value
        cb = self._on_accept
        self.dismiss()
        cb(choice, query)

    def action_complete(self) -> None:
        if not self._filtered:
            return
        choice = self._filtered[self._sel]
        inp = self.query_one("#palette-input", Input)
        inp.value = choice.name + " "
        inp.cursor_position = len(inp.value)

    def action_dismiss_palette(self) -> None:
        self.dismiss()


# --- Retrieval inspector --------------------------------------------------


class RetrievalInspectorModal(ModalScreen):
    """Full chunk list + top chunk rendered in-file. ``turn`` is the focused
    ``Turn`` widget — we read its captured retrieval data so the inspector
    is always tied to a specific turn."""

    BINDINGS = [
        Binding("escape", "dismiss", "Dismiss", priority=True),
        Binding("q", "dismiss", "Dismiss", show=False, priority=True),
    ]

    def __init__(self, turn: Turn | None, root: Path) -> None:
        super().__init__()
        self.turn = turn
        self.root = root

    def compose(self) -> ComposeResult:
        with Vertical(classes="inspector-modal"):
            yield Static("[bold]⟢ retrieval inspector[/]", classes="inspector-head")
            if self.turn is None or not self.turn.retrieved_chunks:
                yield Static(
                    "[#5b8b73](no turn selected — submit a question first, then ^R)[/]",
                    classes="inspector-empty",
                )
                yield Static("[#5b8b73]esc to close[/]", classes="inspector-foot")
                return

            chunks = self.turn.retrieved_chunks
            elapsed = self.turn.retrieved_elapsed
            pinned = self.turn.retrieved_pinned
            pin_note = f" · {len(pinned)} pinned" if pinned else ""
            yield Static(
                f"[#5b8b73]turn:[/] [bold]{self.turn.user_text[:80]}[/]\n"
                f"[#5b8b73]{len(chunks)} chunks{pin_note} · {elapsed * 1000:.0f}ms[/]",
                classes="inspector-meta",
            )

            yield Static("[bold]chunks[/]", classes="inspector-secH")
            list_box = VerticalScroll(classes="inspector-list")
            yield list_box

            top = max(
                chunks,
                key=lambda c: c.get("score") if isinstance(c.get("score"), (int, float)) else -1,
            )
            yield Static("[bold]top chunk in its file[/]", classes="inspector-secH")
            yield VerticalScroll(
                Static(self._render_top(top), id="inspector-file-body"),
                classes="inspector-file",
            )
            yield Static(
                "[#5b8b73]esc to close · scores are 1.0 − cosine distance[/]",
                classes="inspector-foot",
            )

    def on_mount(self) -> None:
        if self.turn is None or not self.turn.retrieved_chunks:
            return
        list_box = self.query_one(".inspector-list", VerticalScroll)
        for c in self.turn.retrieved_chunks:
            score = c.get("score")
            score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
            kind = (c.get("kind") or "project").lower()
            kind_color = "#22d3ee" if kind == "project" else "#c084fc"
            kind_tag = f"[{kind_color}]{kind}[/]"
            label = c.get("label") or ""
            label_part = f" [#5b8b73]\\[{label}][/]" if label else ""
            row = Static(
                f"  [#4ade80]{score_text}[/]  {kind_tag}{label_part}  "
                f"[#d6f5e3]{c['path']}[/][#5b8b73]:{c['start_line']}-{c['end_line']}[/]",
                classes="inspector-row",
            )
            list_box.mount(row)

    def _render_top(self, chunk: dict) -> str:
        """Render the top chunk in the context of its file, with line numbers
        and a marker on the chunk's range."""
        path = chunk.get("path") or ""
        start = int(chunk.get("start_line") or 1)
        end = int(chunk.get("end_line") or start)
        full = self.root / path
        if not full.is_file():
            # Fall back to the embedded chunk content
            body = chunk.get("content") or ""
            lines = body.splitlines() or [""]
            head = f"[#5b8b73]{path}:{start}-{end} (file not on disk; showing indexed chunk)[/]"
            return (
                head
                + "\n"
                + "\n".join(f"[#5b8b73]{start + i:5d}:[/] {ln}" for i, ln in enumerate(lines))
            )
        try:
            text = full.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"[#f87171](could not read {path}: {e})[/]"
        all_lines = text.splitlines()
        ctx_before, ctx_after = 3, 3
        lo = max(1, start - ctx_before)
        hi = min(len(all_lines), end + ctx_after)
        out: list[str] = [f"[#5b8b73]{path} (lines {lo}–{hi}, chunk lines {start}–{end})[/]"]
        for n in range(lo, hi + 1):
            mark = "▎" if start <= n <= end else " "
            color = "#4ade80" if start <= n <= end else "#5b8b73"
            line = all_lines[n - 1] if 0 < n <= len(all_lines) else ""
            out.append(f"[{color}]{mark}{n:5d}:[/] {line}")
        return "\n".join(out)

    def action_dismiss(self) -> None:
        self.dismiss()


# --- Audit overlay --------------------------------------------------------


_AUDIT_KIND_COLOR: dict[str, str] = {
    "tool_call": "#c084fc",
    "tool_result": "#22d3ee",
    "slash_command": "#4ade80",
    "session_start": "#d6f5e3",
    "session_end": "#d6f5e3",
    "architect_plan": "#fbbf24",
}


class AuditOverlayModal(ModalScreen):
    """Scrollable timeline of the current session's audit events.
    Cycle filters with `t` (tool) and `e` (event-kind)."""

    BINDINGS = [
        Binding("escape", "dismiss", "Dismiss", priority=True),
        Binding("q", "dismiss", "Dismiss", show=False, priority=True),
        Binding("t", "cycle_tool", "Cycle tool filter", priority=True),
        Binding("e", "cycle_event", "Cycle event filter", priority=True),
        Binding("a", "clear_filters", "Clear filters", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
    ]

    def __init__(self, meta_dir: Path, session_id: str) -> None:
        super().__init__()
        self.meta_dir = meta_dir
        self.session_id = session_id
        self._tool_idx = 0
        self._event_idx = 0
        self._tools: list[str | None] = [None]
        self._events: list[str | None] = [None]

    def compose(self) -> ComposeResult:
        with Vertical(classes="audit-modal"):
            with Horizontal(classes="audit-bar"):
                yield Static("filter", classes="audit-bar-label")
                yield Static("[bold]session[/] " + self.session_id[:8], classes="audit-chip on")
                yield Static("tool: any", classes="audit-chip", id="audit-tool-chip")
                yield Static("event: any", classes="audit-chip", id="audit-event-chip")
                yield Static(
                    f"[#5b8b73]audit.log at {self.meta_dir / 'audit.log'}[/]",
                    classes="audit-meta",
                )
            yield VerticalScroll(id="audit-list", classes="audit-list")
            yield Static(
                "[#5b8b73]t cycle tool · e cycle event · a clear · r refresh · esc dismiss[/]",
                classes="audit-foot",
            )

    def on_mount(self) -> None:
        rows = audit_mod.tail_audit(self.meta_dir, limit=0)
        my_rows = [r for r in rows if r.get("session") == self.session_id]
        tools = sorted({r["tool"] for r in my_rows if r.get("tool")})
        events = sorted({r.get("event") for r in my_rows if r.get("event")})
        self._tools = [None] + tools
        self._events = [None] + events
        self._tool_idx = 0
        self._event_idx = 0
        self._repaint()

    def _current_filter(self) -> tuple[str | None, str | None]:
        return (
            self._tools[self._tool_idx] if self._tools else None,
            self._events[self._event_idx] if self._events else None,
        )

    def _repaint(self) -> None:
        tool, event = self._current_filter()
        rows = audit_mod.tail_audit(
            self.meta_dir,
            tool=tool,
            event=event,
            limit=0,
        )
        rows = [r for r in rows if r.get("session") == self.session_id]

        self.query_one("#audit-tool-chip", Static).update(f"tool: [bold]{tool or 'any'}[/]")
        self.query_one("#audit-event-chip", Static).update(f"event: [bold]{event or 'any'}[/]")

        list_node = self.query_one("#audit-list", VerticalScroll)
        list_node.remove_children()
        if not rows:
            list_node.mount(
                Static(
                    "[#5b8b73](no events match — try `a` to clear filters)[/]",
                    classes="audit-empty",
                )
            )
            return
        for r in rows:
            list_node.mount(Static(self._format_row(r), classes="audit-row"))
        list_node.scroll_end(animate=False)

    def _format_row(self, row: dict) -> str:
        ts = row.get("ts", "")
        # Show HH:MM:SS.ms (drop date + timezone)
        try:
            dt = datetime.fromisoformat(ts)
            time_str = dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"
        except (ValueError, TypeError):
            time_str = ts[-12:] if ts else "?"
        kind = row.get("event", "?")
        color = _AUDIT_KIND_COLOR.get(kind, "#98c9af")
        body = self._format_body(row)
        return f"[#5b8b73]{time_str}[/]  [{color}]{kind:<14}[/]  {body}"

    def _format_body(self, row: dict) -> str:
        event = row.get("event")
        if event == "tool_call":
            tname = row.get("tool", "?")
            args = row.get("args") or {}
            if isinstance(args, dict):
                parts = [f"{k}={self._brief(v)}" for k, v in args.items()]
            else:
                parts = [str(args)]
            return f"[bold]{tname}[/]([#22d3ee]{', '.join(parts)}[/])"
        if event == "tool_result":
            tname = row.get("tool", "?")
            result = row.get("result") or {}
            ok = result.get("ok", True) if isinstance(result, dict) else True
            ok_mark = "[#4ade80]ok[/]" if ok else "[#f87171]error[/]"
            extra: list[str] = []
            if isinstance(result, dict):
                for k in ("path", "command", "url", "match_count", "lines", "exit_code"):
                    if k in result:
                        extra.append(f"{k}={self._brief(result[k])}")
            dur = row.get("duration_s")
            if isinstance(dur, (int, float)):
                extra.append(f"{dur:.2f}s")
            tail = (" · " + " · ".join(extra)) if extra else ""
            return f"[bold]{tname}[/] → {ok_mark}{tail}"
        if event == "slash_command":
            cmd = row.get("command", "?")
            arg = row.get("arg") or ""
            return f"[bold]:{cmd}[/] {self._brief(arg)}"
        if event == "session_start":
            keys = ("provider", "model", "read_only", "allow_shell", "allow_web")
            parts = [f"{k}={self._brief(row.get(k))}" for k in keys if k in row]
            return " · ".join(parts)
        if event == "session_end":
            return f"reason={row.get('reason', '?')}"
        if event == "architect_plan":
            return f"model={row.get('model', '?')} · len={row.get('plan_len', '?')}"
        # Fallback: drop bookkeeping keys, show the rest
        clean = {k: v for k, v in row.items() if k not in {"ts", "session", "event"}}
        return self._brief(clean)

    def _brief(self, v: object) -> str:
        if isinstance(v, str):
            return v if len(v) < 80 else v[:77] + "…"
        if isinstance(v, dict):
            inner = json.dumps(v, separators=(",", ":"))
            return inner if len(inner) < 80 else inner[:77] + "…"
        if isinstance(v, list):
            return f"[{len(v)} items]"
        return repr(v) if v is not None else "—"

    def action_cycle_tool(self) -> None:
        if not self._tools:
            return
        self._tool_idx = (self._tool_idx + 1) % len(self._tools)
        self._repaint()

    def action_cycle_event(self) -> None:
        if not self._events:
            return
        self._event_idx = (self._event_idx + 1) % len(self._events)
        self._repaint()

    def action_clear_filters(self) -> None:
        self._tool_idx = 0
        self._event_idx = 0
        self._repaint()

    def action_refresh(self) -> None:
        self.on_mount()

    def action_dismiss(self) -> None:
        self.dismiss()


class HelpModal(ModalScreen):
    """Compact keyboard reference and current-session guardrails."""

    BINDINGS = [
        Binding("escape", "dismiss", "Dismiss", priority=True),
        Binding("q", "dismiss", "Dismiss", show=False, priority=True),
        Binding("question_mark", "dismiss", "Dismiss", show=False, priority=True),
    ]

    def __init__(self, session: chat_mod.ChatSession) -> None:
        super().__init__()
        self.session = session

    def compose(self) -> ComposeResult:
        s = self.session
        shell = "on" if s.allow_shell and not s.read_only else "off"
        web = "on" if s.allow_web else "off"
        writes = "confirm" if s.confirm_writes else "auto"
        with Vertical(classes="help-modal"):
            yield Static("[bold]? help[/]", classes="help-head")
            yield Static(
                f"[#5b8b73]project[/] {s.root}\n"
                f"[#5b8b73]model[/] {s.chat_model} · {s.provider_name}  "
                f"[#5b8b73]mode[/] {'read-only' if s.read_only else 'read+write'}  "
                f"[#5b8b73]shell[/] {shell}  [#5b8b73]web[/] {web}  "
                f"[#5b8b73]writes[/] {writes}",
                classes="help-meta",
            )
            yield Static("[bold]Navigation[/]", classes="help-sec")
            yield Static(
                "j / k        select next / previous turn\n"
                "^P           toggle project rail\n"
                "^R           inspect retrieval for selected turn\n"
                "^L           open audit overlay\n"
                "^K           command palette\n"
                "Esc / q      close overlays",
                classes="help-body",
            )
            yield Static("[bold]Composer[/]", classes="help-sec")
            yield Static(
                "Enter        send message or slash command\n"
                "/            open slash menu when composer is empty\n"
                ":add PATH    pin file context\n"
                ":drop PATH   remove pinned context\n"
                ":reset       clear conversation history\n"
                ":q           quit",
                classes="help-body",
            )
            yield Static("[#5b8b73]esc to close[/]", classes="help-foot")

    def action_dismiss(self) -> None:
        self.dismiss()


# --- Layout widgets -------------------------------------------------------


class ShieldBar(Horizontal):
    """Top status bar derived from the live session. Subscribe to
    ``epoch`` (an opaque counter the App bumps on flag toggles) to repaint
    inside one render frame."""

    epoch: reactive[int] = reactive(0)

    def __init__(self, session: chat_mod.ChatSession, **kwargs) -> None:
        super().__init__(**kwargs)
        self.session = session

    def compose(self) -> ComposeResult:
        yield Static("", id="shield-left")
        yield Static("^K command   ^P projects   ? help", id="shield-right")

    def on_mount(self) -> None:
        self._paint()

    def watch_epoch(self, _old: int, _new: int) -> None:
        self._paint()

    def _paint(self) -> None:
        s = self.session
        if s.provider_name == "anthropic":
            dot_color = "#f87171"
            posture = "anthropic"
        elif s.allow_web:
            dot_color = "#fbbf24"
            posture = "+ web"
        else:
            dot_color = "#4ade80"
            posture = "local"
        flags = ["read-only" if s.read_only else "read+write"]
        flags.append(f"shell:{'on' if s.allow_shell and not s.read_only else 'off'}")
        flags.append(f"web:{'on' if s.allow_web else 'off'}")
        if s.confirm_writes:
            flags.append("confirm")
        try:
            chunk_count = s.collection.count()
        except Exception:
            chunk_count = 0
        left = (
            f"[{dot_color}]●[/] {posture}"
            f" [#3d5e4d]│[/] [bold]{s.chat_model}[/] · {s.provider_name}"
            f" [#3d5e4d]│[/] [#5b8b73]{' · '.join(flags)}[/]"
            f" [#3d5e4d]│[/] {chunk_count} chunks"
        )
        try:
            self.query_one("#shield-left", Static).update(left)
        except Exception:
            pass


class ProjectsPane(VerticalScroll):
    """Left rail — the active project from the session."""

    def __init__(self, session: chat_mod.ChatSession, **kwargs) -> None:
        super().__init__(**kwargs)
        self.session = session

    def compose(self) -> ComposeResult:
        yield Static("PROJECT", classes="colHead")
        try:
            chunk_count = self.session.collection.count()
        except Exception:
            chunk_count = 0
        name = self.session.root.name or str(self.session.root)
        meta = f"{chunk_count} chunks · {self.session.notes_marker}"
        yield Static(
            f"[bold]{name}[/]\n[#5b8b73]{meta}[/]\n[#3d5e4d]{self.session.root}[/]",
            classes="proj active",
        )
        yield Static("SESSION", classes="secH")
        yield Static("▸ live · " + self.session.session, classes="session cur")
        yield Static(
            f"provider  {self.session.provider_name}\nmodel     {_clip(self.session.chat_model, 18)}",
            classes="side-kv",
        )


class ContextPane(VerticalScroll):
    """Right rail — pinned files and git state from the session."""

    def __init__(self, session: chat_mod.ChatSession, **kwargs) -> None:
        super().__init__(**kwargs)
        self.session = session

    def compose(self) -> ComposeResult:
        s = self.session
        yield Static("CONTEXT", classes="colHead")

        yield Static("Mode", classes="secH")
        shell = "on" if s.allow_shell and not s.read_only else "off"
        web = "on" if s.allow_web else "off"
        writes = "confirm" if s.confirm_writes else "auto"
        yield Static(
            f"[#5b8b73]access[/] {'read-only' if s.read_only else 'read+write'}\n"
            f"[#5b8b73]shell[/]  {shell}\n"
            f"[#5b8b73]web[/]    {web}\n"
            f"[#5b8b73]writes[/] {writes}",
            classes="side-kv",
        )

        yield Static(f"Pinned ({len(s.pinned_paths)})", classes="secH")
        if not s.pinned_paths:
            yield Static("[#5b8b73](none — use :add)[/]", classes="ref")
        for rel in s.pinned_paths:
            full = s.root / rel
            try:
                size = full.stat().st_size
                size_str = f"{size} B"
            except OSError:
                size_str = "?"
            with Horizontal(classes="pin"):
                yield Static(rel, classes="pin-path")
                yield Static(size_str, classes="pin-size")

        yield Static("Git", classes="secH")
        if gitops.is_git_repo(s.root):
            status = (gitops.status_short(s.root) or "").strip()
            dirty_lines = [ln for ln in status.splitlines() if ln.strip()]
            yield Static(
                f"[#4ade80]●[/] repo  [#5b8b73]· {len(dirty_lines)} dirty[/]",
                classes="gitline",
            )
            for ln in dirty_lines[:8]:
                yield Static(f"    {ln}", classes="gitline gitfile")
        else:
            yield Static("[#5b8b73](not a git repo)[/]", classes="gitline")

        yield Static("Session", classes="secH")
        yield Static(
            f"[#5b8b73]id[/] {s.session}",
            classes="retrieval-last",
        )


class Composer(Vertical):
    """Bottom input row with an optional pin chip strip."""

    def __init__(self, session: chat_mod.ChatSession, **kwargs) -> None:
        super().__init__(**kwargs)
        self.session = session

    def compose(self) -> ComposeResult:
        with Horizontal(classes="pinned-row"):
            for rel in self.session.pinned_paths[:6]:
                yield Static(f"📎 {rel}", classes="chip")
        with Horizontal(classes="composer-row"):
            yield Static(">", classes="gt")
            yield Input(placeholder=COMPOSER_PLACEHOLDER, id="composer-input")


class HintBar(Static):
    """Bottom hint bar."""


# --- Conversation widgets -------------------------------------------------


class Turn(Vertical):
    """One conversation turn. Constructed empty; agent_turn events append
    children incrementally via the helper methods below."""

    def __init__(self, user_text: str) -> None:
        super().__init__(classes="turn")
        self.user_text = user_text
        self._current_answer: Markdown | None = None
        self._answer_buf = ""
        self._tool_cards: dict[int, Static] = {}
        self._next_tool_key = 0
        self.retrieved_chunks: list[dict] = []
        self.retrieved_pinned: list[dict] = []
        self.retrieved_elapsed: float = 0.0

    def compose(self) -> ComposeResult:
        yield Static(f"> {self.user_text}", classes="user")

    async def add_retrieval(
        self,
        chunks: list[dict],
        pinned: list[dict],
        elapsed: float,
    ) -> None:
        self.retrieved_chunks = list(chunks)
        self.retrieved_pinned = list(pinned)
        self.retrieved_elapsed = elapsed
        pin_note = f" · {len(pinned)} pinned" if pinned else ""
        head = (
            f"[bold]⟢ retrieved[/]  {len(chunks)} chunks{pin_note}  "
            f"[#5b8b73]{elapsed * 1000:.0f}ms · ^R inspect[/]"
        )
        body_lines: list[str] = []
        if not chunks:
            body_lines.append("[#5b8b73](no retrieved chunks)[/]")
        for c in chunks:
            score = c.get("score")
            score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "—"
            tag = ""
            if c.get("kind") == "reference":
                lbl = c.get("label") or "ref"
                tag = f"[#5b8b73]\\[{lbl}][/] "
            body_lines.append(
                f"[#4ade80]{score_text:>4}[/]  "
                f"{tag}[#98c9af]{_clip(c['path'], 58)}"
                f":[#5b8b73]{c['start_line']}-{c['end_line']}[/]"
            )
        card = Static(head + "\n" + "\n".join(body_lines), classes="card retrieval")
        await self.mount(card)

    async def add_token(self, piece: str) -> None:
        if self._current_answer is None:
            self._answer_buf = ""
            new_card = Markdown("", classes="card answer", open_links=False)
            await self.mount(new_card)
            self._current_answer = new_card
        self._answer_buf += piece
        await self._current_answer.update(self._answer_buf)

    async def finalize_inference(self, content: str, stats: dict, provider_name: str) -> None:
        if self._current_answer is None and content.strip():
            self._answer_buf = content
            self._current_answer = Markdown(content, classes="card answer", open_links=False)
            await self.mount(self._current_answer)
        elif self._current_answer is not None and content and content != self._answer_buf:
            self._answer_buf = content
            await self._current_answer.update(content)
        if self._current_answer is not None and not self._answer_buf.strip():
            await self._current_answer.remove()
        self._current_answer = None
        self._answer_buf = ""
        stats_line = (
            f"[#5b8b73][{provider_name}] {stats.get('elapsed', 0):.1f}s · "
            f"{stats.get('prompt_tokens', 0)} in → {stats.get('output_tokens', 0)} out[/]"
        )
        await self.mount(Static(stats_line, classes="turn-meta"))

    async def add_tool_card(self, tname: str, args: dict) -> int:
        key = self._next_tool_key
        self._next_tool_key += 1
        argstr = ", ".join(args.keys()) if args else ""
        card = Static(
            f"[bold]⛁ {tname}[/]({argstr})  [#fbbf24]· pending[/]",
            classes="card tool",
        )
        await self.mount(card)
        self._tool_cards[key] = card
        return key

    async def update_tool_card_result(
        self,
        key: int,
        tname: str,
        args: dict,
        summary: dict,
        elapsed: float,
    ) -> None:
        card = self._tool_cards.get(key)
        if card is None:
            return
        ok = bool(summary.get("ok", True))
        status = "[#4ade80]ok[/]" if ok else "[#f87171]error[/]"
        argstr = ", ".join(args.keys()) if args else ""
        detail = ""
        if isinstance(summary, dict):
            for k in ("path", "command", "match_count", "lines", "exit_code", "url"):
                if k in summary:
                    detail = f" [#5b8b73]· {k}={summary[k]}[/]"
                    break
        card.update(f"[bold]⛁ {tname}[/]({argstr})  {status} [#5b8b73]· {elapsed:.2f}s[/]{detail}")

    async def update_tool_card_declined(
        self,
        key: int,
        tname: str,
        args: dict,
    ) -> None:
        card = self._tool_cards.get(key)
        if card is None:
            return
        argstr = ", ".join(args.keys()) if args else ""
        card.update(f"[bold]⛁ {tname}[/]({argstr})  [#f87171]· declined by user[/]")

    async def add_marker(self, text: str, *, cls: str = "turn-meta") -> None:
        await self.mount(Static(text, classes=cls))

    async def add_turn_summary(self, stats: dict) -> None:
        parts = [
            f"turn: {stats['elapsed']:.1f}s",
            f"{stats['inferences']} inference{'s' if stats['inferences'] != 1 else ''}",
            f"{stats['prompt_tokens']} in → {stats['output_tokens']} out",
            f"history: {stats['history_len']} messages",
        ]
        await self.mount(Static(f"[#5b8b73][{' · '.join(parts)}][/]", classes="turn-meta"))


# --- App ------------------------------------------------------------------


class CodebaseRagApp(App):
    """The chat-pane App, now wired to a real ChatSession."""

    CSS_PATH = "tui.tcss"
    TITLE = "codebase-rag"
    # See Phase 1: must disable Textual's built-in palette to reclaim ^P
    # and ^K for our overlays.
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("ctrl+p", "toggle_projects", "Projects", priority=True),
        Binding("ctrl+k", "open_command_palette", "Command palette", priority=True),
        Binding("ctrl+r", "open_retrieval_inspector", "Retrieval inspector", priority=True),
        Binding("ctrl+l", "open_audit_overlay", "Audit overlay", priority=True),
        Binding("j", "focus_next_turn", "Next turn", show=False),
        Binding("k", "focus_prev_turn", "Previous turn", show=False),
        Binding("question_mark", "help_key", "Help", priority=True),
        # Priority on "slash" so we beat the composer's Input widget to the
        # keystroke when the field is empty.
        Binding("slash", "slash_key", "Slash menu", priority=True),
    ]

    def __init__(self, session: chat_mod.ChatSession) -> None:
        super().__init__()
        self.session = session
        self.sub_title = f"chat — {session.root} — {session.chat_model}"
        self._confirm_queue: queue.Queue = queue.Queue(maxsize=1)
        self._active_turn: Turn | None = None
        self._pending_tool_key: int | None = None
        self._shield: ShieldBar | None = None
        self._turns: list[Turn] = []
        self._focused_turn_idx: int = -1

    def compose(self) -> ComposeResult:
        self._shield = ShieldBar(self.session, id="shield")
        yield self._shield
        with Horizontal(id="body"):
            yield ProjectsPane(self.session, id="projects")
            with Vertical(id="middle"):
                yield VerticalScroll(id="conversation")
                yield Composer(self.session, id="composer")
                yield HintBar(HINT_TEXT, id="hintbar")
            yield ContextPane(self.session, id="context")

    def on_mount(self) -> None:
        self._install_welcome()
        self.query_one("#composer-input", Input).focus()

    def _install_welcome(self) -> None:
        conv = self.query_one("#conversation", VerticalScroll)
        try:
            chunk_count = self.session.collection.count()
        except Exception:
            chunk_count = 0
        shell = "on" if self.session.allow_shell and not self.session.read_only else "off"
        web = "on" if self.session.allow_web else "off"
        conv.mount(
            Static(
                "[bold]codebase-rag[/]\n"
                f"[#98c9af]{self.session.root}[/]\n\n"
                f"[#5b8b73]index[/] {chunk_count} chunks   "
                f"[#5b8b73]mode[/] {'read-only' if self.session.read_only else 'read+write'}   "
                f"[#5b8b73]shell[/] {shell}   [#5b8b73]web[/] {web}\n\n"
                "Ask about the codebase or type [bold]/[/] for commands.",
                id="welcome",
                classes="welcome",
            )
        )

    def action_toggle_projects(self) -> None:
        self.query_one("#projects").toggle_class("hidden")

    def action_help_key(self) -> None:
        try:
            inp = self.query_one("#composer-input", Input)
        except Exception:
            self.action_open_help()
            return
        if inp.has_focus and (inp.value or "") != "":
            inp.insert_text_at_cursor("?")
            return
        self.action_open_help()

    def action_open_help(self) -> None:
        self.push_screen(HelpModal(self.session))

    def action_focus_next_turn(self) -> None:
        self._move_turn_focus(1)

    def action_focus_prev_turn(self) -> None:
        self._move_turn_focus(-1)

    def _move_turn_focus(self, delta: int) -> None:
        if not self._turns:
            return
        if self._focused_turn_idx < 0:
            self._focused_turn_idx = len(self._turns) - 1
        else:
            self._focused_turn_idx = (self._focused_turn_idx + delta) % len(self._turns)
        self._paint_turn_focus()

    def _paint_turn_focus(self) -> None:
        for i, turn in enumerate(self._turns):
            turn.set_class(i == self._focused_turn_idx, "focused")
        if 0 <= self._focused_turn_idx < len(self._turns):
            self._active_turn = self._turns[self._focused_turn_idx]
            self._turns[self._focused_turn_idx].scroll_visible(animate=False)

    # --- Slash palette: opened by typing "/" in the composer -------------

    def action_slash_key(self) -> None:
        """Priority binding for "/". Opens the slash palette by default;
        only swallowed by the composer when it has focus AND already has
        text (so the user can still type paths or regex containing "/")."""
        try:
            inp = self.query_one("#composer-input", Input)
        except Exception:
            self._open_slash_palette()
            return
        if inp.has_focus and (inp.value or "") != "":
            inp.insert_text_at_cursor("/")
            return
        # Empty composer, or focus elsewhere — open the palette.
        self._open_slash_palette()

    def _open_slash_palette(self) -> None:
        items = [
            PaletteItem(spec.name, spec.desc, spec.requires, payload=spec)
            for spec in chat_mod.SLASH_SPECS
        ]
        self.push_screen(
            PaletteModal(
                items,
                title="/",
                initial="",
                on_accept=self._slash_accepted,
            )
        )

    def _slash_accepted(self, item: PaletteItem, query: str) -> None:
        """Drop the matched command (plus a space when it takes an arg) into
        the composer so the user can type the argument and submit."""
        spec = item.payload if isinstance(item.payload, chat_mod.SlashSpec) else None
        suffix = " " if (spec and spec.takes_arg) else ""
        # If the user already typed an arg into the palette filter, carry it.
        extra = ""
        if query and query.strip() and " " in query.strip():
            extra = query.strip().split(" ", 1)[1]
        try:
            inp = self.query_one("#composer-input", Input)
        except Exception:
            return
        if extra:
            inp.value = item.name + " " + extra
        else:
            inp.value = item.name + suffix
        inp.cursor_position = len(inp.value)
        inp.focus()
        # If the command takes no arg, fire it immediately.
        if spec and not spec.takes_arg:
            self._run_slash(inp.value.strip())
            inp.value = ""

    # --- Command palette: ^K ---------------------------------------------

    def action_open_command_palette(self) -> None:
        items: list[PaletteItem] = [
            PaletteItem(
                ":switch-project", "switch to a different indexed project", requires="not impl"
            ),
            PaletteItem(
                ":toggle-read-only",
                "flip read-only on/off (rebuilds tool list)",
                payload="toggle_read_only",
            ),
            PaletteItem(
                ":toggle-confirm-writes",
                "flip create/write/edit confirmation on/off",
                payload="toggle_confirm_writes",
            ),
            PaletteItem(":open-audit", "open the audit-log overlay", payload="open_audit"),
            PaletteItem(
                ":open-retrieval",
                "open the retrieval inspector for last turn",
                payload="open_retrieval",
            ),
        ] + [
            PaletteItem(spec.name, spec.desc, spec.requires, payload=spec)
            for spec in chat_mod.SLASH_SPECS
        ]
        self.push_screen(
            PaletteModal(
                items,
                title="⌘",
                initial="",
                on_accept=self._command_accepted,
            )
        )

    def _command_accepted(self, item: PaletteItem, query: str) -> None:
        payload = item.payload
        if payload == "toggle_read_only":
            self.session.read_only = not self.session.read_only
            self._rebuild_tool_schemas()
            self._refresh_shield()
            return
        if payload == "toggle_confirm_writes":
            self.session.confirm_writes = not self.session.confirm_writes
            self._refresh_shield()
            return
        if payload == "open_audit":
            self.action_open_audit_overlay()
            return
        if payload == "open_retrieval":
            self.action_open_retrieval_inspector()
            return
        # Fall through to slash-style behavior for SlashSpec payloads.
        self._slash_accepted(item, query)

    def _rebuild_tool_schemas(self) -> None:
        from .tools import tool_schemas_for

        s = self.session
        s.tool_schemas = tool_schemas_for(
            read_only=s.read_only,
            allow_shell=s.allow_shell and not s.read_only,
            allow_web=s.allow_web,
        )

    def _refresh_shield(self) -> None:
        if self._shield is not None:
            self._shield.epoch += 1

    # --- Run a slash command via the shared dispatcher ------------------

    @work(thread=True, exclusive=False)
    def _run_slash(self, text: str) -> None:
        if not text:
            return
        confirm_queue: queue.Queue = queue.Queue(maxsize=1)

        def confirm(prompt: str) -> bool:
            self.call_from_thread(self._show_slash_confirm_modal, prompt, confirm_queue)
            return bool(confirm_queue.get())

        lines = chat_mod.dispatch_slash(self.session, text, confirm=confirm)
        self.call_from_thread(self._finish_slash, lines)

    def _show_slash_confirm_modal(
        self,
        prompt: str,
        result_queue: queue.Queue,
    ) -> None:
        self.push_screen(SlashConfirmModal(prompt, result_queue))

    def _finish_slash(self, lines: list[tuple[str, str]] | None) -> None:
        if lines is None:
            return
        for level, msg in lines:
            if level == "exit":
                self.exit()
                return
            if not msg:
                continue
            color = {
                "ok": "#4ade80",
                "warn": "#fbbf24",
                "err": "#f87171",
                "info": "#98c9af",
            }.get(level, "#98c9af")
            self._post_marker(f"[{color}]{msg}[/]")
        # Pinned-row + context-pane may have changed.
        self._refresh_side_panels()
        # recompose() detaches focus; restore it so the next "/" opens the
        # slash palette again.
        try:
            self.query_one("#composer-input", Input).focus()
        except Exception:
            pass

    def _post_marker(self, markup: str) -> None:
        """Append a system-style marker line into the conversation pane."""
        conv = self.query_one("#conversation", VerticalScroll)
        conv.mount(Static(markup, classes="slash-line"))
        conv.scroll_end(animate=False)

    def _refresh_side_panels(self) -> None:
        """Recompose ProjectsPane / ContextPane / Composer after a slash
        command might have mutated session state. `refresh(recompose=True)`
        re-runs the widget's compose() generator in place."""
        for sel in ("#projects", "#context", "#composer"):
            try:
                self.query_one(sel).refresh(recompose=True)
            except Exception:
                pass

    # --- ^R retrieval inspector -----------------------------------------

    def action_open_retrieval_inspector(self) -> None:
        self.push_screen(RetrievalInspectorModal(self._active_turn, self.session.root))

    # --- ^L audit overlay -----------------------------------------------

    def action_open_audit_overlay(self) -> None:
        self.push_screen(AuditOverlayModal(self.session.meta_dir, self.session.session))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "composer-input":
            return
        text = (event.value or "").strip()
        event.input.value = ""
        if not text:
            return
        # Route slash commands through the shared dispatcher so the line
        # loop and the TUI share one implementation.
        if text.startswith(":") or text in ("exit", "quit"):
            if text in ("exit", "quit"):
                text = ":q"
            self._run_slash(text)
            return
        self.run_turn(text)

    @work(thread=True, exclusive=True)
    def run_turn(self, user_input: str) -> None:
        """Worker thread that drives one agent_turn end-to-end."""
        self.call_from_thread(self._start_turn, user_input)
        gen = chat_mod.agent_turn(self.session, user_input)
        try:
            event = next(gen)
        except StopIteration:
            event = None
        while event is not None:
            kind = event[0]
            if kind == "confirm":
                _, tname, args = event
                self.call_from_thread(self._show_confirm_modal, tname, args)
                resolved = self._confirm_queue.get()
                try:
                    event = gen.send(resolved)
                except StopIteration:
                    event = None
            else:
                self.call_from_thread(self._handle_event, event)
                try:
                    event = next(gen)
                except StopIteration:
                    event = None
        self.call_from_thread(self._save_after_turn)

    # --- UI thread helpers (called via call_from_thread) ------------------

    def _start_turn(self, user_input: str) -> None:
        conv = self.query_one("#conversation", VerticalScroll)
        try:
            self.query_one("#welcome").add_class("hidden")
        except Exception:
            pass
        turn = Turn(user_input)
        self._turns.append(turn)
        self._focused_turn_idx = len(self._turns) - 1
        self._active_turn = turn
        self._pending_tool_key = None
        conv.mount(turn)
        self._paint_turn_focus()
        conv.scroll_end(animate=False)

    def _handle_event(self, event: tuple) -> None:
        turn = self._active_turn
        if turn is None:
            return
        kind = event[0]
        if kind == "retrieved":
            _, chunks, pinned, elapsed = event
            self.run_worker(turn.add_retrieval(chunks, pinned, elapsed), exclusive=False)
        elif kind == "token":
            self.run_worker(turn.add_token(event[1]), exclusive=False)
        elif kind == "inference_done":
            _, content, _tool_calls, stats = event
            self.run_worker(
                turn.finalize_inference(content, stats, self.session.provider.name),
                exclusive=False,
            )
        elif kind == "tool_call_request":
            _, tname, args = event

            async def _add_and_remember() -> None:
                key = await turn.add_tool_card(tname, args)
                self._pending_tool_key = key

            self.run_worker(_add_and_remember(), exclusive=False)
        elif kind == "tool_result":
            _, tname, args, _raw, summary, elapsed = event
            key = self._pending_tool_key
            self._pending_tool_key = None
            if key is not None:
                self.run_worker(
                    turn.update_tool_card_result(key, tname, args, summary, elapsed),
                    exclusive=False,
                )
        elif kind == "tool_declined":
            _, tname, args, _decl, _raw = event
            key = self._pending_tool_key
            self._pending_tool_key = None
            if key is not None:
                self.run_worker(
                    turn.update_tool_card_declined(key, tname, args),
                    exclusive=False,
                )
        elif kind == "architect_start":
            self.run_worker(
                turn.add_marker(f"[#c084fc][architect ({event[1]}) thinking…][/]"),
                exclusive=False,
            )
        elif kind == "architect_error":
            self.run_worker(
                turn.add_marker(f"[#f87171](architect error: {event[1]}; falling back)[/]"),
                exclusive=False,
            )
        elif kind == "empty_response":
            self.run_worker(turn.add_marker("[#5b8b73](no response)[/]"), exclusive=False)
        elif kind == "error":
            _, sub, msg = event
            tag = "context-length" if sub == "context_length" else sub
            self.run_worker(
                turn.add_marker(f"[#f87171]({tag}: {msg})[/]"),
                exclusive=False,
            )
        elif kind == "max_turns":
            self.run_worker(
                turn.add_marker(f"[#fbbf24](stopped after {event[1]} tool-call rounds)[/]"),
                exclusive=False,
            )
        elif kind == "turn_done":
            self.run_worker(turn.add_turn_summary(event[1]), exclusive=False)
            conv = self.query_one("#conversation", VerticalScroll)
            conv.scroll_end(animate=False)

    def _show_confirm_modal(self, tname: str, args: dict) -> None:
        if tname == "run_shell":
            modal = ConfirmShellModal(
                args,
                self._confirm_queue,
                shell_runner=self.session.shell_runner,
                shell_network=self.session.shell_network,
                shell_timeout=self.session.shell_timeout,
            )
        else:
            modal = ConfirmWriteModal(tname, args, self._confirm_queue)
        self.push_screen(modal)

    def _save_after_turn(self) -> None:
        try:
            chat_mod._save_conversation(
                self.session.root,
                self.session.history,
                self.session.chat_model,
                pinned=self.session.pinned_paths,
            )
        except OSError:
            pass


def run_tui(
    *,
    db_path: Path,
    root: Path,
    model: str | None,
    provider_name: str = "ollama",
    api_key: str | None = None,
    show_context: bool = False,
    verbose: bool = False,
    resume: bool = False,
    read_only: bool = False,
    allow_shell: bool = False,
    shell_timeout: float = 30,
    shell_runner: str = "host",
    shell_network: str = "none",
    confirm_writes: bool = True,
    allow_web: bool = False,
    web_allow: tuple[str, ...] = (),
    web_block: tuple[str, ...] = (),
    searxng_url: str = "",
    architect_model: str | None = None,
) -> None:
    """Launch the TUI against a real ChatSession."""
    session = chat_mod.init_chat_session(
        db_path,
        root,
        model=model,
        show_context=show_context,
        verbose=verbose,
        resume=resume,
        read_only=read_only,
        allow_shell=allow_shell,
        shell_timeout=shell_timeout,
        shell_runner=shell_runner,
        shell_network=shell_network,
        confirm_writes=confirm_writes,
        allow_web=allow_web,
        web_allow=web_allow,
        web_block=web_block,
        searxng_url=searxng_url,
        architect_model=architect_model,
        provider_name=provider_name,
        api_key=api_key,
    )
    if session is None:
        return
    CodebaseRagApp(session).run()
