from __future__ import annotations

import curses
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import unicodedata
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from ..core.event import ConversationEvent, DoneEvent, ErrorEvent, StreamEvent

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


class PipeBackend:
    def __init__(self, process: Any | None = None) -> None:
        self.process = process or subprocess.Popen(
            [sys.executable, "-m", "duckduckcode.main", "--backend"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def stream(self, message: str) -> Iterator[StreamEvent]:
        self.process.stdin.write(json.dumps({"message": message}) + "\n")
        self.process.stdin.flush()

        for line in self.process.stdout:
            event = _event_from_json(json.loads(line))
            yield event
            if isinstance(event, DoneEvent | ErrorEvent):
                break

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        self.process.terminate()

    def cancel(self) -> None:
        # ponytail: SIGINT is enough for POSIX; use a pipe command if Windows matters.
        self.process.send_signal(signal.SIGINT)


def run_tui(
    model: str, cwd: str | None = None, backend: PipeBackend | None = None
) -> None:
    curses.wrapper(
        lambda screen: _Tui(
            screen, model, cwd or os.getcwd(), backend or PipeBackend()
        ).run()
    )


class _Tui:
    def __init__(self, screen: Any, model: str, cwd: str, backend: PipeBackend) -> None:
        self.screen = screen
        self.model = model
        self.cwd = cwd
        self.backend = backend
        self.messages: list[tuple[str, str]] = []
        self.input = ""
        self.tokens = 0
        self.scroll_offset = 0
        self._scrollbar_geometry: tuple[int, int, int, int, int, int] | None = None
        self._scroll_drag_offset: int | None = None
        self._events: queue.Queue[StreamEvent | None] | None = None
        self._waiting = False
        self._wait_frame = 0
        self._interrupting = False

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
                if key in {3, "\x03"}:
                    break
                if key in {27, "\x1b"}:
                    event = self._read_sgr_mouse()
                    if event is not None:
                        self._handle_mouse_event(*event)
                    elif self._interrupt_or_exit():
                        break
                elif key in {10, 13, "\n", "\r"}:
                    self._send()
                elif key in {curses.KEY_BACKSPACE, 127, 8, "\b", "\x7f"}:
                    self.input = self.input[:-1]
                elif key in {curses.KEY_PPAGE, curses.KEY_UP}:
                    self.scroll_offset += 3
                elif key in {curses.KEY_NPAGE, curses.KEY_DOWN}:
                    self.scroll_offset = max(0, self.scroll_offset - 3)
                elif key == curses.KEY_MOUSE:
                    self._handle_curses_mouse()
                elif isinstance(key, str) and key >= " ":
                    self.input += key
        finally:
            _set_mouse_tracking(False)
            self.backend.close()

    def _send(self) -> None:
        if self._events is not None:
            return
        prompt = self.input.strip()
        self.input = ""
        if not prompt:
            return

        self.scroll_offset = 0
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

    def _consume_events(self) -> None:
        if self._events is None:
            return
        try:
            first = self._events.get_nowait()
        except queue.Empty:
            return

        for event in _ready_events(first, self._events):
            if event is None:
                if self._waiting:
                    self.messages.pop()
                self._events = None
                self._waiting = False
                self._interrupting = False
            elif isinstance(event, ConversationEvent):
                if self._waiting and event.delta:
                    self.messages[-1] = ("duckduckcode", "")
                    self._waiting = False
                role, text = self.messages[-1]
                self.messages[-1] = (role, text + event.delta)
            elif isinstance(event, DoneEvent):
                self.tokens += event.token_usage
                self._waiting = False
            elif isinstance(event, ErrorEvent):
                if self._waiting:
                    self.messages.pop()
                    self._waiting = False
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

    def _draw(self) -> None:
        self.screen.erase()
        height, width = self.screen.getmaxyx()
        header_height = len(DUCK) + 2
        input_y = max(header_height, height - 4)

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
        rows = _wrap_messages(self.messages, chat_width)
        viewport_height = max(0, input_y - header_height)
        self.scroll_offset = min(
            self.scroll_offset, max(0, len(rows) - viewport_height)
        )
        visible_rows = _visible_rows(rows, viewport_height, self.scroll_offset)
        for row, (role, text) in enumerate(visible_rows, start=header_height):
            if role:
                color = {
                    "you": USER_COLOR,
                    "duckduckcode": DUCK_COLOR,
                    "error": ERROR_COLOR,
                }.get(role, MUTED_COLOR)
                self.screen.addstr(row, 0, "●", _color(color))
            text_color = ERROR_COLOR if role == "error" else TEXT_COLOR
            self.screen.addstr(
                row,
                2,
                _clip(text, max(1, chat_width - 2)),
                _color(text_color),
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
        self.screen.addstr(input_y + 1, 0, "›", _color(DUCK_COLOR) | curses.A_BOLD)
        self.screen.addstr(input_y + 1, 2, _clip(self.input, width - 2))
        self.screen.hline(input_y + 2, 0, curses.ACS_HLINE, width, separator)
        status_left = self.model
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
        self.screen.move(input_y + 1, min(width - 1, _text_width(self.input) + 2))
        self.screen.refresh()

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
        if state & getattr(curses, "BUTTON4_PRESSED", 0):
            self._handle_mouse_event(64, x, y, False)
        elif state & getattr(curses, "BUTTON5_PRESSED", 0):
            self._handle_mouse_event(65, x, y, False)
        elif state & getattr(curses, "BUTTON1_PRESSED", 0):
            self._handle_mouse_event(0, x, y, False)
        elif state & getattr(curses, "BUTTON1_RELEASED", 0):
            self._handle_mouse_event(0, x, y, True)
        elif state & getattr(curses, "REPORT_MOUSE_POSITION", 0):
            self._handle_mouse_event(32, x, y, False)

    def _handle_mouse_event(self, button: int, x: int, y: int, released: bool) -> None:
        button &= ~(4 | 8 | 16)
        if button == 64:
            self.scroll_offset += 3
            return
        if button == 65:
            self.scroll_offset = max(0, self.scroll_offset - 3)
            return
        if released:
            self._scroll_drag_offset = None
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


def _event_from_json(data: dict[str, Any]) -> StreamEvent:
    if data["type"] == "delta":
        return ConversationEvent(str(data.get("text", "")))
    if data["type"] == "done":
        return DoneEvent(int(data.get("token_usage", 0)))
    if data["type"] == "error":
        return ErrorEvent(str(data.get("message", "")), data.get("code"))
    return ConversationEvent("")


def _read_stream(
    backend: PipeBackend, prompt: str, events: queue.Queue[StreamEvent | None]
) -> None:
    try:
        for event in backend.stream(prompt):
            events.put(event)
    except Exception as exc:
        events.put(ErrorEvent(str(exc)))
    finally:
        events.put(None)


def _ready_events(
    first: StreamEvent | None, events: queue.Queue[StreamEvent | None]
) -> list[StreamEvent | None]:
    ready = [first]
    while True:
        try:
            ready.append(events.get_nowait())
        except queue.Empty:
            return ready


def _wrap_messages(
    messages: list[tuple[str, str]], width: int
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    content_width = max(1, width - 2)
    for role, text in messages:
        row_role = role
        for line in (text or " ").split("\n"):
            remaining = line or " "
            while remaining:
                chunk, remaining = _take_width(remaining, content_width)
                if not chunk:
                    chunk, remaining = " ", remaining[1:]
                rows.append((row_role, chunk))
                row_role = ""
    return rows


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
