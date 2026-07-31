from __future__ import annotations

import curses
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import unicodedata
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from ..core.event import (
    AgentEvent,
    ConversationEvent,
    ContextCompactionEvent,
    ContextStatusEvent,
    ErrorEvent,
    LoopCompleteEvent,
    PermissionChoice,
    PermissionRequestEvent,
    PlanReviewEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    UsageEvent,
)
from .slash_command import handle_slash_command, slash_command_suggestions

DUCK = [
    r"  __",
    r"<(o )___",
    r" ( ._> /",
    r"  `---'",
]

DUCK_COLOR = 1
MUTED_COLOR = 2
USER_COLOR = 3
ERROR_COLOR = 4
TEXT_COLOR = 5
WAIT_FRAMES = ["thinking .  ", "thinking .. ", "thinking ..."]
SHORT_TOOL_RESULT_LINES = 10
SHORT_TOOL_RESULT_CHARS = 1_000
PERMISSION_OPTIONS: tuple[tuple[PermissionChoice, str], ...] = (
    ("allow_once", "允许一次"),
    ("allow_always", "始终允许"),
    ("deny", "拒绝"),
)
PLAN_REVIEW_OPTIONS = (
    ("approve", "批准并执行"),
    ("deny", "拒绝计划"),
    ("feedback", "输入修改意见"),
)
PERMISSION_MODE_OPTIONS = (
    ("full_access", "Full access · 读写文件和 Bash 均放行"),
    ("ask_for_approval", "Ask for approval · 写文件和 Bash 需确认"),
    ("accept_edits", "Accept edits · 文件读写放行，Bash 需确认"),
)
PERMISSION_MODE_NAMES = {
    mode: label.partition(" ·")[0] for mode, label in PERMISSION_MODE_OPTIONS
}


class PipeBackend:
    def __init__(self, process: Any | None = None) -> None:
        self.process = process or subprocess.Popen(
            [sys.executable, "-m", "duckduckcode.main", "--backend"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

    def stream(self, message: str) -> Iterator[AgentEvent]:
        self.process.stdin.write(json.dumps({"message": message}) + "\n")
        self.process.stdin.flush()

        for line in self.process.stdout:
            event = _event_from_json(json.loads(line))
            yield event
            if isinstance(event, LoopCompleteEvent):
                break

    def compact(self) -> Iterator[AgentEvent]:
        self.process.stdin.write(json.dumps({"type": "compact"}) + "\n")
        self.process.stdin.flush()

        for line in self.process.stdout:
            event = _event_from_json(json.loads(line))
            yield event
            if isinstance(event, LoopCompleteEvent):
                break

    def context_status(self) -> Iterator[AgentEvent]:
        self.process.stdin.write(json.dumps({"type": "status"}) + "\n")
        self.process.stdin.flush()

        for line in self.process.stdout:
            event = _event_from_json(json.loads(line))
            yield event
            if isinstance(event, LoopCompleteEvent):
                break

    def respond_permission(self, call_id: str, decision: PermissionChoice) -> None:
        self.process.stdin.write(
            json.dumps(
                {
                    "type": "permission_response",
                    "call_id": call_id,
                    "decision": decision,
                }
            )
            + "\n"
        )
        self.process.stdin.flush()

    def set_mode(self, mode: str) -> None:
        self.process.stdin.write(json.dumps({"type": "set_mode", "mode": mode}) + "\n")
        self.process.stdin.flush()

    def set_permission_mode(self, mode: str) -> None:
        self.process.stdin.write(
            json.dumps({"type": "set_permission_mode", "mode": mode}) + "\n"
        )
        self.process.stdin.flush()

    def respond_plan_review(self, approved: bool, feedback: str = "") -> None:
        self.process.stdin.write(
            json.dumps(
                {
                    "type": "plan_review_response",
                    "approved": approved,
                    "feedback": feedback,
                }
            )
            + "\n"
        )
        self.process.stdin.flush()

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        self.process.terminate()

    def cancel(self) -> None:
        # ponytail: SIGINT is enough for POSIX; use a pipe command if Windows matters.
        self.process.send_signal(signal.SIGINT)


def run_tui(
    model: str,
    cwd: str | None = None,
    backend: PipeBackend | None = None,
    permission_mode: str = "ask_for_approval",
) -> None:
    curses.wrapper(
        lambda screen: _Tui(
            screen,
            model,
            cwd or os.getcwd(),
            backend or PipeBackend(),
            permission_mode,
        ).run()
    )


class _Tui:
    def __init__(
        self,
        screen: Any,
        model: str,
        cwd: str,
        backend: PipeBackend,
        permission_mode: str = "ask_for_approval",
    ) -> None:
        self.screen = screen
        self.model = model
        self.cwd = cwd
        self.backend = backend
        self.messages: list[tuple[str, str]] = []
        self.input = ""
        self.cursor_index = 0
        self.selection_anchor: int | None = None
        self._clipboard = ""
        self._input_width = 1
        self._input_geometry: list[tuple[int, int, int, str]] = []
        self._input_dragging = False
        self._chat_rows: list[str] = []
        self._chat_geometry: dict[int, tuple[int, str]] = {}
        self._chat_selection_anchor: tuple[int, int] | None = None
        self._chat_selection_cursor: tuple[int, int] | None = None
        self._chat_dragging = False
        self._chat_pressed_tool: str | None = None
        self.tool_results: dict[str, ToolResultEvent] = {}
        self._expanded_tool_results: set[str] = set()
        self._chat_tool_rows: dict[int, str] = {}
        self.tokens = 0
        self.scroll_offset = 0
        self._scrollbar_geometry: tuple[int, int, int, int, int, int] | None = None
        self._scroll_drag_offset: int | None = None
        self._events: queue.Queue[AgentEvent | None] | None = None
        self._waiting = False
        self._wait_frame = 0
        self._interrupting = False
        self._compaction_active = False
        self._permission_request: PermissionRequestEvent | None = None
        self._permission_selection = len(PERMISSION_OPTIONS) - 1
        self._plan_review: PlanReviewEvent | None = None
        self._plan_review_selection = 0
        self._permission_mode_open = False
        self._permission_mode_selection = 1
        self.permission_mode = permission_mode
        self._slash_selection = 0
        self.mode = "default"

    def run(self) -> None:
        curses.curs_set(1)
        curses.set_escdelay(25)
        _init_colors()
        self.screen.bkgd(" ", _color(TEXT_COLOR))
        curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
        _set_mouse_tracking(True)
        self.screen.timeout(100)
        self._draw()
        try:
            while True:
                self._consume_events()
                self._animate_wait()
                self._draw()
                try:
                    key = self.screen.get_wch()
                except curses.error:
                    continue
                except KeyboardInterrupt:
                    if self._interrupt_or_exit():
                        break
                    continue
                if self._permission_request is not None and key not in {
                    3,
                    "\x03",
                }:
                    self._handle_permission_key(key)
                    continue
                if self._plan_review is not None and self._handle_plan_review_key(key):
                    continue
                if self._permission_mode_open and key not in {3, "\x03"}:
                    self._handle_permission_mode_key(key)
                    continue
                if self._handle_slash_key(key):
                    continue
                if key in {3, "\x03"}:
                    if self._copy_selection():
                        continue
                    if self._interrupt_or_exit():
                        break
                if key in {27, "\x1b"}:
                    event = self._read_sgr_mouse()
                    if event is not None:
                        self._handle_mouse_event(*event)
                    elif self._interrupt_or_exit():
                        break
                elif key in {10, 13, "\n", "\r"}:
                    self._send()
                elif key in {24, "\x18"}:
                    self._cut_selection()
                elif key in {22, "\x16"}:
                    self._paste()
                elif key in {1, "\x01"}:
                    self.selection_anchor = 0
                    self.cursor_index = len(self.input)
                elif key in {curses.KEY_BACKSPACE, 127, 8, "\b", "\x7f"}:
                    self._backspace()
                elif key == curses.KEY_DC:
                    self._delete_forward()
                elif key == curses.KEY_LEFT:
                    self._move_cursor(-1)
                elif key == curses.KEY_RIGHT:
                    self._move_cursor(1)
                elif key == getattr(curses, "KEY_SLEFT", -1):
                    self._move_cursor(-1, selecting=True)
                elif key == getattr(curses, "KEY_SRIGHT", -1):
                    self._move_cursor(1, selecting=True)
                elif key == getattr(curses, "KEY_SR", -1):
                    self._move_vertical(-1, selecting=True)
                elif key == getattr(curses, "KEY_SF", -1):
                    self._move_vertical(1, selecting=True)
                elif key == curses.KEY_HOME:
                    self._set_cursor(0)
                elif key == curses.KEY_END:
                    self._set_cursor(len(self.input))
                elif key == getattr(curses, "KEY_SHOME", -1):
                    self._set_cursor(0, selecting=True)
                elif key == getattr(curses, "KEY_SEND", -1):
                    self._set_cursor(len(self.input), selecting=True)
                elif key == curses.KEY_UP:
                    self._move_vertical(-1)
                elif key == curses.KEY_DOWN:
                    self._move_vertical(1)
                elif key == curses.KEY_PPAGE:
                    self.scroll_offset += 3
                elif key == curses.KEY_NPAGE:
                    self.scroll_offset = max(0, self.scroll_offset - 3)
                elif key == curses.KEY_MOUSE:
                    self._handle_curses_mouse()
                elif isinstance(key, str) and key >= " ":
                    self._insert_text(key)
        finally:
            _set_mouse_tracking(False)
            self.backend.close()

    def _send(self) -> None:
        if self._plan_review is not None:
            choice = PLAN_REVIEW_OPTIONS[self._plan_review_selection][0]
            feedback = self.input.strip() if choice == "feedback" else ""
            if choice == "feedback" and not feedback:
                return
            if feedback:
                self.messages.append(("you", feedback))
            self.backend.respond_plan_review(choice == "approve", feedback)
            self._plan_review = None
            if choice == "approve":
                self.mode = "default"
            self.messages.append(("duckduckcode", ""))
            self._waiting = True
            self._wait_frame = 0
            self.input = ""
            self.cursor_index = 0
            self.selection_anchor = None
            return
        if self._events is not None:
            return
        prompt = self.input.strip()
        if not prompt:
            return

        self.scroll_offset = 0
        slash_command = handle_slash_command(prompt)
        if slash_command is not None:
            if slash_command[0] == "mode":
                self.mode = "default" if self.mode == "plan" else "plan"
                self.backend.set_mode(self.mode)
            elif slash_command[0] == "permissions":
                self._permission_mode_open = True
                self._permission_mode_selection = next(
                    index
                    for index, (mode, _) in enumerate(PERMISSION_MODE_OPTIONS)
                    if mode == self.permission_mode
                )
            elif slash_command[0] == "compact":
                self.messages.append(("duckduckcode", ""))
                self._events = queue.Queue()
                self._waiting = True
                self._wait_frame = 0
                threading.Thread(
                    target=_read_compact,
                    args=(self.backend, self._events),
                    daemon=True,
                ).start()
            elif slash_command[0] == "status":
                self.messages.append(("duckduckcode", ""))
                self._events = queue.Queue()
                self._waiting = True
                self._wait_frame = 0
                threading.Thread(
                    target=_read_status,
                    args=(self.backend, self._events),
                    daemon=True,
                ).start()
            else:
                self.messages.append(slash_command)
            self.input = ""
            self.cursor_index = 0
            self.selection_anchor = None
            return

        self.messages.append(("you", prompt))
        self.messages.append(("duckduckcode", ""))
        self._events = queue.Queue()
        self._waiting = True
        self._wait_frame = 0
        threading.Thread(
            target=_read_stream,
            args=(self.backend, prompt, self._events),
            daemon=True,
        ).start()
        self.input = ""
        self.cursor_index = 0
        self.selection_anchor = None

    def _consume_events(self) -> None:
        if self._events is None:
            return
        try:
            first = self._events.get_nowait()
        except queue.Empty:
            return

        for event in _ready_events(first, self._events):
            if event is None:
                if self._compaction_active and self.messages:
                    self.messages.pop()
                    self._compaction_active = False
                if self._waiting:
                    self.messages.pop()
                self._events = None
                self._waiting = False
                self._interrupting = False
                self._permission_request = None
                self._plan_review = None
            elif isinstance(event, ConversationEvent):
                if self._waiting and event.delta:
                    self.messages[-1] = ("duckduckcode", "")
                    self._waiting = False
                elif self.messages[-1][0] != "duckduckcode":
                    self.messages.append(("duckduckcode", ""))
                role, text = self.messages[-1]
                self.messages[-1] = (role, text + event.delta)
            elif isinstance(event, ToolCallEvent):
                if self._waiting:
                    self.messages.pop()
                    self._waiting = False
                self.messages.append(
                    (
                        f"tool:{event.tool_call.call_id}",
                        f"→ {event.tool_call.name} running…",
                    )
                )
            elif isinstance(event, ToolResultEvent):
                if self._waiting:
                    self.messages.pop()
                    self._waiting = False
                self.tool_results[event.call_id] = event
                role = f"tool:{event.call_id}"
                if not any(message_role == role for message_role, _ in self.messages):
                    self.messages.append((role, f"→ {event.name} running…"))
            elif isinstance(event, PermissionRequestEvent):
                if self._waiting:
                    self.messages.pop()
                    self._waiting = False
                self._permission_request = event
                self._permission_selection = len(PERMISSION_OPTIONS) - 1
            elif isinstance(event, PlanReviewEvent):
                if self._waiting:
                    self.messages.pop()
                    self._waiting = False
                self._plan_review = event
                self._plan_review_selection = 0
                self.messages.append(("duckduckcode", event.content))
                self.messages.append(("duckduckcode", "Plan completed"))
            elif isinstance(event, UsageEvent):
                self.tokens += event.total_tokens
            elif isinstance(event, ContextCompactionEvent):
                if event.status == "started":
                    self._compaction_active = True
                    if self._waiting:
                        self.messages[-1] = ("duckduckcode", "Compacting context…")
                        self._waiting = False
                    else:
                        self.messages.append(("duckduckcode", "Compacting context…"))
                elif event.status == "completed":
                    self._compaction_active = False
                    label = (
                        "Context compacted automatically"
                        if event.automatic
                        else "Context compacted"
                    )
                    self.messages[-1] = (
                        "duckduckcode",
                        f"{label} ({event.before_tokens:,} → {event.after_tokens:,} tokens).",
                    )
                    self.messages.append(("duckduckcode", ""))
                    self._waiting = True
                elif event.status == "skipped":
                    self._compaction_active = False
                    self.messages[-1] = (
                        "duckduckcode",
                        "Nothing to compact; recent conversation was kept intact.",
                    )
                    self._waiting = False
            elif isinstance(event, ContextStatusEvent):
                status = _context_status_text(event)
                if self._waiting:
                    self.messages[-1] = ("duckduckcode", status)
                    self._waiting = False
                else:
                    self.messages.append(("duckduckcode", status))
            elif isinstance(event, TurnCompleteEvent):
                if not self._waiting:
                    self.messages.append(("duckduckcode", ""))
                    self._waiting = True
                    self._wait_frame = 0
            elif isinstance(event, LoopCompleteEvent):
                if self._waiting:
                    self.messages.pop()
                self._waiting = False
                self._permission_request = None
                self._plan_review = None
            elif isinstance(event, ErrorEvent):
                if self._compaction_active:
                    if self.messages:
                        self.messages.pop()
                    self._compaction_active = False
                elif self._waiting:
                    self.messages.pop()
                self._waiting = False
                self._permission_request = None
                self._plan_review = None
                self.messages.append(("error", event.message))

    def _animate_wait(self) -> None:
        if self._events is None or not self._waiting or self._interrupting:
            return
        self.messages[-1] = (
            "duckduckcode",
            WAIT_FRAMES[self._wait_frame % len(WAIT_FRAMES)],
        )
        self._wait_frame += 1

    def _interrupt_or_exit(self) -> bool:
        if self._events is None:
            return True
        if not self._interrupting:
            self.backend.cancel()
            self._interrupting = True
            if self._waiting:
                self.messages.pop()
                self._waiting = False
        return False

    def _handle_permission_key(self, key: object) -> None:
        request = self._permission_request
        if request is None:
            return
        if key == curses.KEY_UP:
            self._permission_selection = (self._permission_selection - 1) % len(
                PERMISSION_OPTIONS
            )
            return
        if key == curses.KEY_DOWN:
            self._permission_selection = (self._permission_selection + 1) % len(
                PERMISSION_OPTIONS
            )
            return
        if key in {27, "\x1b"}:
            choice: PermissionChoice = "deny"
        elif key in {10, 13, "\n", "\r"}:
            choice = PERMISSION_OPTIONS[self._permission_selection][0]
        else:
            return
        self.backend.respond_permission(request.call_id, choice)
        self._permission_request = None

    def _handle_plan_review_key(self, key: object) -> bool:
        if self._plan_review is None:
            return False
        if key == curses.KEY_UP:
            self._plan_review_selection = (self._plan_review_selection - 1) % len(
                PLAN_REVIEW_OPTIONS
            )
            return True
        if key == curses.KEY_DOWN:
            self._plan_review_selection = (self._plan_review_selection + 1) % len(
                PLAN_REVIEW_OPTIONS
            )
            return True
        if key in {27, "\x1b"}:
            self._plan_review_selection = 1
            self._send()
            return True
        if key in {10, 13, "\n", "\r"}:
            self._send()
            return True
        if isinstance(key, str) and key >= " ":
            self._plan_review_selection = 2
        return False

    def _handle_permission_mode_key(self, key: object) -> None:
        if key == curses.KEY_UP:
            self._permission_mode_selection = (
                self._permission_mode_selection - 1
            ) % len(PERMISSION_MODE_OPTIONS)
            return
        if key == curses.KEY_DOWN:
            self._permission_mode_selection = (
                self._permission_mode_selection + 1
            ) % len(PERMISSION_MODE_OPTIONS)
            return
        if key in {27, "\x1b"}:
            self._permission_mode_open = False
            return
        if key not in {10, 13, "\n", "\r"}:
            return
        self.permission_mode = PERMISSION_MODE_OPTIONS[self._permission_mode_selection][
            0
        ]
        self.backend.set_permission_mode(self.permission_mode)
        self._permission_mode_open = False

    def _handle_slash_key(self, key: object) -> bool:
        if self._plan_review is not None:
            return False
        suggestions = slash_command_suggestions(self.input)
        if suggestions is None:
            return False
        if key == curses.KEY_UP:
            if suggestions:
                self._slash_selection = (self._slash_selection - 1) % len(suggestions)
            return True
        if key == curses.KEY_DOWN:
            if suggestions:
                self._slash_selection = (self._slash_selection + 1) % len(suggestions)
            return True
        if key in {10, 13, "\n", "\r"} and suggestions:
            self._slash_selection %= len(suggestions)
            self.input = suggestions[self._slash_selection][0]
            self.cursor_index = len(self.input)
            self.selection_anchor = None
            self._send()
            return True
        return False

    def _selection(self) -> tuple[int, int] | None:
        if self.selection_anchor is None or self.selection_anchor == self.cursor_index:
            return None
        return tuple(sorted((self.selection_anchor, self.cursor_index)))

    def _set_cursor(self, index: int, selecting: bool = False) -> None:
        old_cursor = self.cursor_index
        self.cursor_index = min(len(self.input), max(0, index))
        if selecting:
            if self.selection_anchor is None:
                self.selection_anchor = old_cursor
        else:
            self.selection_anchor = None

    def _move_cursor(self, delta: int, selecting: bool = False) -> None:
        self._set_cursor(self.cursor_index + delta, selecting)

    def _move_vertical(self, delta: int, selecting: bool = False) -> None:
        rows = _input_rows(self.input, self._input_width)
        row_index, column = _cursor_position(rows, self.cursor_index)
        target = min(len(rows) - 1, max(0, row_index + delta))
        start, end, text = rows[target]
        self._set_cursor(_index_at_column(start, end, text, column), selecting)

    def _replace_selection(self, text: str) -> None:
        selection = self._selection()
        start, end = selection or (self.cursor_index, self.cursor_index)
        self.input = self.input[:start] + text + self.input[end:]
        self.cursor_index = start + len(text)
        self.selection_anchor = None
        self._slash_selection = 0

    def _insert_text(self, text: str) -> None:
        self._replace_selection(text)

    def _backspace(self) -> None:
        if self._selection() is not None:
            self._replace_selection("")
        elif self.cursor_index:
            self.input = (
                self.input[: self.cursor_index - 1] + self.input[self.cursor_index :]
            )
            self.cursor_index -= 1
            self._slash_selection = 0

    def _delete_forward(self) -> None:
        if self._selection() is not None:
            self._replace_selection("")
        elif self.cursor_index < len(self.input):
            self.input = (
                self.input[: self.cursor_index] + self.input[self.cursor_index + 1 :]
            )
            self._slash_selection = 0

    def _copy_selection(self) -> bool:
        selection = self._selection()
        if selection is not None:
            self._clipboard = self.input[slice(*selection)]
        else:
            chat_selection = self._chat_selection_text()
            if chat_selection is None:
                return False
            self._clipboard = chat_selection
        _write_clipboard(self._clipboard)
        return True

    def _cut_selection(self) -> None:
        if self._copy_selection():
            self._replace_selection("")

    def _paste(self) -> None:
        self._insert_text(_read_clipboard() or self._clipboard)

    def _display_messages(self) -> list[tuple[str, str]]:
        displayed = []
        for role, text in self.messages:
            call_id = role.removeprefix("tool:") if role.startswith("tool:") else None
            event = self.tool_results.get(call_id) if call_id is not None else None
            displayed.append(
                (
                    role,
                    (
                        _format_tool_result(
                            event,
                            call_id in self._expanded_tool_results,
                        )
                        if event is not None
                        else text
                    ),
                )
            )
        return displayed

    def _input_index_at(self, x: int, y: int, clamp: bool = False) -> int | None:
        for row, start, end, text in self._input_geometry:
            if y == row:
                return _index_at_column(start, end, text, max(0, x - 2))
        if clamp and self._input_geometry:
            if y < self._input_geometry[0][0]:
                return self._input_geometry[0][1]
            if y > self._input_geometry[-1][0]:
                return self._input_geometry[-1][2]
        return None

    def _chat_position_at(
        self, x: int, y: int, clamp: bool = False
    ) -> tuple[int, int] | None:
        geometry = self._chat_geometry.get(y)
        if geometry is None and clamp and self._chat_geometry:
            row = (
                min(self._chat_geometry)
                if y < min(self._chat_geometry)
                else max(self._chat_geometry)
            )
            geometry = self._chat_geometry[row]
        if geometry is None:
            return None
        row, text = geometry
        return row, _index_at_column(0, len(text), text, max(0, x - 2))

    def _chat_selection(
        self,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        if (
            self._chat_selection_anchor is None
            or self._chat_selection_cursor is None
            or self._chat_selection_anchor == self._chat_selection_cursor
        ):
            return None
        return tuple(sorted((self._chat_selection_anchor, self._chat_selection_cursor)))

    def _chat_selection_text(self) -> str | None:
        selection = self._chat_selection()
        if selection is None:
            return None
        (start_row, start), (end_row, end) = selection
        selected = self._chat_rows[start_row : end_row + 1]
        selected[0] = selected[0][start:]
        selected[-1] = selected[-1][:end]
        return "\n".join(selected)

    def _draw(self) -> None:
        self.screen.erase()
        height, width = self.screen.getmaxyx()
        header_height = len(DUCK) + 2
        self._input_width = max(1, width - 3)
        input_rows = _input_rows(self.input, self._input_width)
        cursor_row, cursor_column = _cursor_position(input_rows, self.cursor_index)
        permission_height = (
            len(PERMISSION_OPTIONS) + 3 if self._permission_request is not None else 0
        )
        plan_review_height = (
            len(PLAN_REVIEW_OPTIONS) + 1 if self._plan_review is not None else 0
        )
        permission_mode_height = (
            len(PERMISSION_MODE_OPTIONS) + 2 if self._permission_mode_open else 0
        )
        slash_suggestions = (
            slash_command_suggestions(self.input)
            if self._permission_request is None
            and self._plan_review is None
            and not self._permission_mode_open
            else None
        )
        if slash_suggestions:
            self._slash_selection = min(
                self._slash_selection, len(slash_suggestions) - 1
            )
        slash_height = (
            max(1, len(slash_suggestions)) + 1 if slash_suggestions is not None else 0
        )
        panel_height = (
            permission_height
            or plan_review_height
            or permission_mode_height
            or slash_height
        )
        max_input_rows = max(
            1,
            min(
                height // 3,
                height - header_height - panel_height - 3,
            ),
        )
        first_input_row = min(
            max(0, cursor_row - max_input_rows + 1),
            max(0, len(input_rows) - max_input_rows),
        )
        visible_input = input_rows[first_input_row : first_input_row + max_input_rows]
        input_y = max(
            header_height,
            height - len(visible_input) - panel_height - 3,
        )

        for index, line in enumerate(DUCK):
            self.screen.addstr(index, 0, _clip(line, width), _color(DUCK_COLOR))
        title = "duckduckcode"
        self.screen.addstr(
            0,
            12,
            _clip(title, width - 12),
            curses.A_BOLD,
        )
        self.screen.addstr(
            0,
            12 + len(title) + 1,
            _clip(app_version(), width - 13 - len(title)),
            _color(MUTED_COLOR) | curses.A_DIM,
        )
        self.screen.addstr(
            1,
            12,
            _clip(f"cwd  {self.cwd}", width - 12),
            _color(MUTED_COLOR) | curses.A_DIM,
        )
        separator = _color(MUTED_COLOR) | curses.A_DIM
        self.screen.hline(header_height - 1, 0, curses.ACS_HLINE, width, separator)

        chat_width = max(1, width - 1)
        rows = _message_rows(self._display_messages(), chat_width)
        self._chat_rows = [text for _, text, _ in rows]
        viewport_height = max(0, input_y - header_height)
        self.scroll_offset = min(
            self.scroll_offset, max(0, len(rows) - viewport_height)
        )
        visible_rows = _visible_rows(rows, viewport_height, self.scroll_offset)
        visible_start = max(0, len(rows) - self.scroll_offset - len(visible_rows))
        self._chat_tool_rows = {}
        self._chat_geometry = {}
        chat_selection = self._chat_selection()
        for row_index, (role, text, source_role) in enumerate(
            visible_rows, start=visible_start
        ):
            row = header_height + row_index - visible_start
            self._chat_geometry[row] = (row_index, text)
            if source_role.startswith("tool:"):
                self._chat_tool_rows[row] = source_role.removeprefix("tool:")
            if role:
                display_role = "tool" if role.startswith("tool:") else role
                color = {
                    "you": USER_COLOR,
                    "duckduckcode": DUCK_COLOR,
                    "tool": MUTED_COLOR,
                    "error": ERROR_COLOR,
                }.get(display_role, MUTED_COLOR)
                self.screen.addstr(row, 0, "●", _color(color))
            text_color = ERROR_COLOR if source_role == "error" else TEXT_COLOR
            self.screen.addstr(
                row,
                2,
                _clip(text, max(1, chat_width - 2)),
                _color(text_color),
            )
            if chat_selection is not None:
                (start_row, start), (end_row, end) = chat_selection
                selected_start = start if row_index == start_row else 0
                selected_end = end if row_index == end_row else len(text)
                if start_row <= row_index <= end_row and selected_start < selected_end:
                    self.screen.addstr(
                        row,
                        2 + _text_width(text[:selected_start]),
                        text[selected_start:selected_end],
                        _color(text_color) | curses.A_REVERSE,
                    )

        thumb_start, thumb_height, max_offset = _scrollbar(
            len(rows), viewport_height, self.scroll_offset
        )
        self._scrollbar_geometry = None
        if max_offset:
            self._scrollbar_geometry = (
                width - 1,
                header_height,
                viewport_height,
                thumb_start,
                thumb_height,
                max_offset,
            )
            for row in range(viewport_height):
                marker = (
                    curses.ACS_CKBOARD
                    if thumb_start <= row < thumb_start + thumb_height
                    else curses.ACS_VLINE
                )
                self.screen.addch(
                    header_height + row,
                    width - 1,
                    marker,
                    (
                        _color(DUCK_COLOR)
                        if thumb_start <= row < thumb_start + thumb_height
                        else separator
                    ),
                )

        self.screen.hline(input_y, 0, curses.ACS_HLINE, width, separator)
        self.screen.addstr(
            input_y,
            2,
            _clip(
                f" permissions: {PERMISSION_MODE_NAMES[self.permission_mode]} ",
                max(1, width - 4),
            ),
            _color(MUTED_COLOR),
        )
        self.screen.addstr(input_y + 1, 0, "›", _color(DUCK_COLOR) | curses.A_BOLD)
        self._input_geometry = []
        selection = self._selection()
        for row, (start, end, line) in enumerate(visible_input, start=input_y + 1):
            self._input_geometry.append((row, start, end, line))
            self.screen.addstr(row, 2, line)
            if selection is not None:
                selected_start = max(start, selection[0])
                selected_end = min(end, selection[1])
                if selected_start < selected_end:
                    self.screen.addstr(
                        row,
                        2 + _text_width(self.input[start:selected_start]),
                        self.input[selected_start:selected_end],
                        curses.A_REVERSE,
                    )
        self.screen.hline(
            input_y + len(visible_input) + 1,
            0,
            curses.ACS_HLINE,
            width,
            separator,
        )
        status_left = f"{self.model} · plan" if self.mode == "plan" else self.model
        status_right = f"{self.tokens:,} tokens"
        self.screen.addstr(
            height - 1, 0, _clip(status_left, width), _color(MUTED_COLOR)
        )
        if _text_width(status_left) + _text_width(status_right) + 2 < width:
            self.screen.addstr(
                height - 1,
                width - _text_width(status_right) - 1,
                status_right,
                _color(MUTED_COLOR) | curses.A_DIM,
            )
        if self._permission_request is not None:
            self._draw_permission_panel(
                input_y + len(visible_input) + 2,
                width,
                separator,
            )
        elif self._plan_review is not None:
            self._draw_plan_review(
                input_y + len(visible_input) + 2,
                width,
                separator,
            )
        elif self._permission_mode_open:
            self._draw_permission_mode_panel(
                input_y + len(visible_input) + 2,
                width,
                separator,
            )
        elif slash_suggestions is not None:
            self._draw_slash_panel(
                input_y + len(visible_input) + 2,
                width,
                separator,
                slash_suggestions,
            )
        self.screen.move(
            input_y + 1 + cursor_row - first_input_row,
            min(width - 1, cursor_column + 2),
        )
        self.screen.refresh()

    def _draw_plan_review(self, top: int, width: int, separator: int) -> None:
        if self._plan_review is None:
            return
        for index, (_, label) in enumerate(PLAN_REVIEW_OPTIONS):
            attributes = _color(TEXT_COLOR)
            if index == self._plan_review_selection:
                attributes |= curses.A_REVERSE
            self.screen.addstr(
                top + index,
                2,
                _clip(
                    ("› " if index == self._plan_review_selection else "  ") + label,
                    max(1, width - 2),
                ),
                attributes,
            )
        self.screen.hline(
            top + len(PLAN_REVIEW_OPTIONS),
            0,
            curses.ACS_HLINE,
            width,
            separator,
        )

    def _draw_permission_panel(
        self,
        top: int,
        width: int,
        separator: int,
    ) -> None:
        request = self._permission_request
        if request is None:
            return
        self.screen.addstr(
            top,
            0,
            _clip(f"? {request.name} 请求权限", width),
            _color(DUCK_COLOR) | curses.A_BOLD,
        )
        self.screen.addstr(
            top + 1,
            2,
            _clip(request.content, max(1, width - 2)),
            _color(MUTED_COLOR),
        )
        for index, (_, label) in enumerate(PERMISSION_OPTIONS):
            attributes = _color(TEXT_COLOR)
            if index == self._permission_selection:
                attributes |= curses.A_REVERSE
            self.screen.addstr(
                top + 2 + index,
                2,
                _clip(
                    ("› " if index == self._permission_selection else "  ") + label,
                    max(1, width - 2),
                ),
                attributes,
            )
        self.screen.hline(
            top + 2 + len(PERMISSION_OPTIONS),
            0,
            curses.ACS_HLINE,
            width,
            separator,
        )

    def _draw_permission_mode_panel(
        self,
        top: int,
        width: int,
        separator: int,
    ) -> None:
        self.screen.addstr(
            top,
            0,
            _clip("? 选择权限模式", width),
            _color(DUCK_COLOR) | curses.A_BOLD,
        )
        for index, (_, label) in enumerate(PERMISSION_MODE_OPTIONS):
            attributes = _color(TEXT_COLOR)
            if index == self._permission_mode_selection:
                attributes |= curses.A_REVERSE
            self.screen.addstr(
                top + 1 + index,
                2,
                _clip(
                    ("› " if index == self._permission_mode_selection else "  ")
                    + label,
                    max(1, width - 2),
                ),
                attributes,
            )
        self.screen.hline(
            top + 1 + len(PERMISSION_MODE_OPTIONS),
            0,
            curses.ACS_HLINE,
            width,
            separator,
        )

    def _draw_slash_panel(
        self,
        top: int,
        width: int,
        separator: int,
        suggestions: list[tuple[str, str]],
    ) -> None:
        if not suggestions:
            self.screen.addstr(
                top,
                2,
                _clip("No matching commands", max(1, width - 2)),
                _color(MUTED_COLOR),
            )
        for index, (name, description) in enumerate(suggestions):
            attributes = _color(TEXT_COLOR)
            if index == self._slash_selection:
                attributes |= curses.A_REVERSE
            self.screen.addstr(
                top + index,
                2,
                _clip(
                    ("› " if index == self._slash_selection else "  ")
                    + f"{name}  {description}",
                    max(1, width - 2),
                ),
                attributes,
            )
        self.screen.hline(
            top + max(1, len(suggestions)),
            0,
            curses.ACS_HLINE,
            width,
            separator,
        )

    def _read_sgr_mouse(self) -> tuple[int, int, int, bool] | None:
        sequence = ""
        while len(sequence) < 32:
            try:
                key = self.screen.get_wch()
            except curses.error:
                break
            if not isinstance(key, str):
                break
            sequence += key
            if key in {"M", "m"}:
                break
        return _parse_sgr_mouse(sequence)

    def _handle_curses_mouse(self) -> None:
        _, x, y, _, state = curses.getmouse()
        shifted = 4 if state & getattr(curses, "BUTTON_SHIFT", 0) else 0
        if state & getattr(curses, "BUTTON4_PRESSED", 0):
            self._handle_mouse_event(64, x, y, False)
        elif state & getattr(curses, "BUTTON5_PRESSED", 0):
            self._handle_mouse_event(65, x, y, False)
        elif state & getattr(curses, "BUTTON1_PRESSED", 0):
            self._handle_mouse_event(shifted, x, y, False)
        elif state & getattr(curses, "BUTTON1_RELEASED", 0):
            self._handle_mouse_event(0, x, y, True)
        elif state & getattr(curses, "REPORT_MOUSE_POSITION", 0):
            self._handle_mouse_event(32, x, y, False)

    def _handle_mouse_event(self, button: int, x: int, y: int, released: bool) -> None:
        shifted = bool(button & 4)
        button &= ~(4 | 8 | 16)
        if button == 64:
            self.scroll_offset += 3
            return
        if button == 65:
            self.scroll_offset = max(0, self.scroll_offset - 3)
            return
        if released:
            if self._input_dragging:
                index = self._input_index_at(x, y, clamp=True)
                if index is not None:
                    self.cursor_index = index
            elif self._chat_dragging:
                position = self._chat_position_at(x, y, clamp=True)
                if position is not None:
                    self._chat_selection_cursor = position
                if self._chat_selection() is None:
                    call_id = self._chat_tool_rows.get(y)
                    if call_id is not None and call_id == self._chat_pressed_tool:
                        if call_id in self._expanded_tool_results:
                            self._expanded_tool_results.remove(call_id)
                        else:
                            self._expanded_tool_results.add(call_id)
                    self._chat_selection_anchor = None
                    self._chat_selection_cursor = None
                else:
                    self._copy_selection()
            self._scroll_drag_offset = None
            self._input_dragging = False
            self._chat_dragging = False
            self._chat_pressed_tool = None
            if self.selection_anchor == self.cursor_index:
                self.selection_anchor = None
            return
        if button == 0:
            index = self._input_index_at(x, y)
            if index is not None:
                old_cursor = self.cursor_index
                self.cursor_index = index
                self.selection_anchor = old_cursor if shifted else self.cursor_index
                self._chat_selection_anchor = None
                self._chat_selection_cursor = None
                self._input_dragging = True
                return
            bar_x = self._scrollbar_geometry[0] if self._scrollbar_geometry else -1
            position = self._chat_position_at(x, y)
            if position is not None and x != bar_x:
                self.selection_anchor = None
                self._chat_selection_anchor = position
                self._chat_selection_cursor = position
                self._chat_dragging = True
                self._chat_pressed_tool = self._chat_tool_rows.get(y)
                return
        if button == 32 and self._input_dragging:
            index = self._input_index_at(x, y, clamp=True)
            if index is not None:
                self.cursor_index = index
            return
        if button == 32 and self._chat_dragging:
            position = self._chat_position_at(x, y, clamp=True)
            if position is not None:
                self._chat_selection_cursor = position
            return
        if self._scrollbar_geometry is None:
            return

        bar_x, bar_y, bar_height, thumb_start, thumb_height, max_offset = (
            self._scrollbar_geometry
        )
        if button == 0 and x == bar_x and bar_y <= y < bar_y + bar_height:
            absolute_thumb = bar_y + thumb_start
            self._scroll_drag_offset = (
                y - absolute_thumb
                if absolute_thumb <= y < absolute_thumb + thumb_height
                else thumb_height // 2
            )
        elif button != 32 or self._scroll_drag_offset is None:
            return

        track = max(1, bar_height - thumb_height)
        position = min(track, max(0, y - bar_y - self._scroll_drag_offset))
        self.scroll_offset = max_offset - round(position * max_offset / track)


def app_version() -> str:
    try:
        return version("duckduckcode")
    except PackageNotFoundError:
        return "0.1.0"


def _init_colors() -> None:
    if not curses.has_colors():
        return
    curses.start_color()
    curses.use_default_colors()
    if curses.COLORS >= 256:
        colors = (179, 245, 73, 167, 230)
        background = 235
    else:
        colors = (
            curses.COLOR_YELLOW,
            curses.COLOR_WHITE,
            curses.COLOR_CYAN,
            curses.COLOR_RED,
            curses.COLOR_WHITE,
        )
        background = -1
    for pair, foreground in enumerate(colors, start=1):
        curses.init_pair(pair, foreground, background)


def _color(pair: int) -> int:
    return curses.color_pair(pair) if curses.has_colors() else 0


def _set_mouse_tracking(enabled: bool) -> None:
    mode = "h" if enabled else "l"
    sys.stdout.write(f"\x1b[?1002{mode}\x1b[?1006{mode}")
    sys.stdout.flush()


def _parse_sgr_mouse(sequence: str) -> tuple[int, int, int, bool] | None:
    match = re.fullmatch(r"\[<(\d+);(\d+);(\d+)([Mm])", sequence)
    if match is None:
        return None
    button, x, y, end = match.groups()
    return int(button), int(x) - 1, int(y) - 1, end == "m"


def _event_from_json(data: dict[str, Any]) -> AgentEvent:
    event_type = data["type"]
    if event_type == "stream_text":
        return ConversationEvent(str(data.get("delta", "")))
    if event_type == "tool_use":
        return ToolCallEvent.create(
            str(data.get("call_id", "")),
            str(data.get("name", "")),
            dict(data.get("arguments", {})),
        )
    if event_type == "tool_result":
        return ToolResultEvent(
            str(data.get("call_id", "")),
            str(data.get("name", "")),
            str(data.get("content", "")),
            bool(data.get("is_error", False)),
        )
    if event_type == "permission_request":
        return PermissionRequestEvent(
            str(data.get("call_id", "")),
            str(data.get("name", "")),
            str(data.get("content", "")),
            str(data.get("message", "")),
        )
    if event_type == "plan_review":
        return PlanReviewEvent(
            str(data.get("plan_file", "")),
            str(data.get("content", "")),
        )
    if event_type == "usage":
        return UsageEvent(int(data.get("total_tokens", 0)))
    if event_type == "context_compaction":
        return ContextCompactionEvent(
            data.get("status", "skipped"),
            bool(data.get("automatic", False)),
            int(data.get("before_tokens", 0)),
            int(data.get("after_tokens", 0)),
        )
    if event_type == "context_status":
        return ContextStatusEvent(
            int(data.get("used_tokens", 0)),
            int(data.get("max_tokens", 0)),
            int(data.get("auto_compact_tokens", 0)),
        )
    if event_type == "turn_complete":
        return TurnCompleteEvent(int(data.get("iteration", 0)))
    if event_type == "loop_complete":
        return LoopCompleteEvent(
            data.get("reason", "error"),
            int(data.get("iterations", 0)),
        )
    if event_type == "error":
        return ErrorEvent(str(data.get("message", "")), data.get("code"))
    raise ValueError(f"Unknown backend event type: {event_type}")


def _read_stream(
    backend: PipeBackend, prompt: str, events: queue.Queue[AgentEvent | None]
) -> None:
    try:
        for event in backend.stream(prompt):
            events.put(event)
    except Exception as exc:
        events.put(ErrorEvent(str(exc)))
    finally:
        events.put(None)


def _read_compact(backend: PipeBackend, events: queue.Queue[AgentEvent | None]) -> None:
    try:
        for event in backend.compact():
            events.put(event)
    except Exception as exc:
        events.put(ErrorEvent(str(exc)))
    finally:
        events.put(None)


def _read_status(backend: PipeBackend, events: queue.Queue[AgentEvent | None]) -> None:
    try:
        for event in backend.context_status():
            events.put(event)
    except Exception as exc:
        events.put(ErrorEvent(str(exc)))
    finally:
        events.put(None)


def _context_status_text(event: ContextStatusEvent) -> str:
    width = 20
    percentage = event.used_tokens / max(1, event.max_tokens) * 100
    filled = min(width, max(0, round(width * percentage / 100)))
    bar = "█" * filled + "░" * (width - filled)
    compact_percentage = event.auto_compact_tokens / max(1, event.max_tokens) * 100
    return (
        f"Context estimate {percentage:.1f}%\n"
        f"[{bar}] {event.used_tokens:,} / {event.max_tokens:,} tokens\n"
        f"Auto compact at {event.auto_compact_tokens:,} "
        f"({compact_percentage:.1f}%)"
    )


def _ready_events(
    first: AgentEvent | None, events: queue.Queue[AgentEvent | None]
) -> list[AgentEvent | None]:
    ready = [first]
    while True:
        try:
            ready.append(events.get_nowait())
        except queue.Empty:
            return ready


def _wrap_messages(
    messages: list[tuple[str, str]], width: int
) -> list[tuple[str, str]]:
    return [(role, text) for role, text, _ in _message_rows(messages, width)]


def _message_rows(
    messages: list[tuple[str, str]], width: int
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    content_width = max(1, width - 2)
    for role, text in messages:
        row_role = role
        for line in (text or " ").split("\n"):
            remaining = line or " "
            while remaining:
                chunk, remaining = _take_width(remaining, content_width)
                if not chunk:
                    chunk, remaining = " ", remaining[1:]
                rows.append((row_role, chunk, role))
                row_role = ""
    return rows


def _wrap_input(text: str, width: int) -> list[str]:
    return [row[2] for row in _input_rows(text, width)]


def _input_rows(text: str, width: int) -> list[tuple[int, int, str]]:
    width = max(1, width)
    rows: list[tuple[int, int, str]] = []
    start = 0
    used = 0
    for index, char in enumerate(text):
        if char == "\n":
            rows.append((start, index, text[start:index]))
            start = index + 1
            used = 0
            continue
        char_width = _char_width(char)
        if index > start and used + char_width > width:
            rows.append((start, index, text[start:index]))
            start = index
            used = 0
        used += char_width
    rows.append((start, len(text), text[start:]))
    return rows


def _cursor_position(
    rows: list[tuple[int, int, str]], cursor_index: int
) -> tuple[int, int]:
    row_index = 0
    column = 0
    for index, (start, end, text) in enumerate(rows):
        if start <= cursor_index <= end:
            row_index = index
            column = _text_width(text[: cursor_index - start])
    return row_index, column


def _index_at_column(start: int, end: int, text: str, column: int) -> int:
    used = 0
    for offset, char in enumerate(text):
        char_width = _char_width(char)
        if column < used + (char_width + 1) // 2:
            return start + offset
        used += char_width
        if column < used:
            return start + offset + 1
    return end


def _format_tool_result(event: ToolResultEvent, expanded: bool) -> str:
    marker = "✗" if event.is_error else "✓"
    status = "failed" if event.is_error else "completed"
    header = f"{marker} {event.name} {status}"
    if expanded:
        return f"{header} · [收起]\n{event.content}"
    result_content = _tool_result_content(event)
    if event.is_error:
        summary = next(
            (line.strip() for line in result_content.splitlines() if line.strip()),
            "unknown error",
        )
        return f"{header} · {summary[:160]} · [查看详情]"
    if (
        len(result_content.splitlines()) <= SHORT_TOOL_RESULT_LINES
        and len(result_content) <= SHORT_TOOL_RESULT_CHARS
    ):
        return f"{header}\n{result_content}" if result_content else header
    return f"{header} · {_tool_result_size(event)} · [查看详情]"


def _tool_result_size(event: ToolResultEvent) -> str:
    content = _tool_result_content(event)
    lines = content.splitlines()
    if event.name == "Grep":
        matches = sum(bool(re.match(r"^.+:\d+:", line)) for line in lines)
        return f"{matches} matches"
    if event.name == "ReadFile":
        size = len(event.content.encode("utf-8"))
        return f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} bytes"
    label = "results" if event.name == "Glob" else "output lines"
    return f"{len(lines)} {label}"


def _tool_result_content(event: ToolResultEvent) -> str:
    if event.name == "Bash":
        try:
            output = json.loads(event.content).get("output", "")
            if isinstance(output, str):
                return output
        except (AttributeError, json.JSONDecodeError):
            pass
    return event.content


def _clipboard_commands(write: bool) -> list[list[str]]:
    if sys.platform == "darwin":
        return [["pbcopy"]] if write else [["pbpaste"]]
    if os.name == "nt":
        return (
            [["clip"]]
            if write
            else [["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"]]
        )
    if write:
        return [["wl-copy"], ["xclip", "-selection", "clipboard"]]
    return [["wl-paste", "--no-newline"], ["xclip", "-selection", "clipboard", "-o"]]


def _write_clipboard(text: str) -> None:
    for command in _clipboard_commands(write=True):
        if shutil.which(command[0]) is None:
            continue
        try:
            subprocess.run(
                command,
                input=text,
                text=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except (OSError, subprocess.CalledProcessError):
            pass


def _read_clipboard() -> str:
    for command in _clipboard_commands(write=False):
        if shutil.which(command[0]) is None:
            continue
        try:
            return subprocess.run(
                command,
                text=True,
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            pass
    return ""


def _visible_rows(rows: list[Any], height: int, scroll_offset: int) -> list[Any]:
    if height <= 0:
        return []
    scroll_offset = min(scroll_offset, max(0, len(rows) - height))
    end = max(0, len(rows) - scroll_offset)
    start = max(0, end - height)
    return rows[start:end]


def _scrollbar(
    total_rows: int, height: int, scroll_offset: int
) -> tuple[int, int, int]:
    max_offset = max(0, total_rows - height)
    if height <= 0 or max_offset == 0:
        return 0, max(0, height), max_offset
    thumb_height = max(1, height * height // total_rows)
    thumb_start = round(
        (max_offset - min(scroll_offset, max_offset))
        * (height - thumb_height)
        / max_offset
    )
    return thumb_start, thumb_height, max_offset


def _clip(text: str, width: int) -> str:
    return _take_width(text, max(0, width - 1))[0]


def _take_width(text: str, width: int) -> tuple[str, str]:
    used = 0
    for index, char in enumerate(text):
        char_width = _char_width(char)
        if used + char_width > width:
            return text[:index], text[index:]
        used += char_width
    return text, ""


def _text_width(text: str) -> int:
    return sum(_char_width(char) for char in text)


def _char_width(char: str) -> int:
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
