from __future__ import annotations

import curses
import io
import json
import queue
import signal
import threading
import unittest
from unittest.mock import patch

from duckduckcode.event import ConversationEvent, DoneEvent, ErrorEvent
from duckduckcode.tui import (
    PipeBackend,
    _Tui,
    _clip,
    _parse_sgr_mouse,
    _ready_events,
    _read_stream,
    _scrollbar,
    _visible_rows,
    _wrap_messages,
)


class TuiTest(unittest.TestCase):
    def test_pipe_backend_writes_prompt_and_reads_events_until_done(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = io.StringIO()
                self.stdout = io.StringIO(
                    '{"type": "delta", "text": "he"}\n'
                    '{"type": "delta", "text": "llo"}\n'
                    '{"type": "done", "token_usage": 5}\n'
                )

        process = FakeProcess()
        backend = PipeBackend(process)

        self.assertEqual(
            list(backend.stream("hello")),
            [ConversationEvent("he"), ConversationEvent("llo"), DoneEvent(5)],
        )
        process.stdin.seek(0)
        self.assertEqual(json.loads(process.stdin.read()), {"message": "hello"})

    def test_pipe_backend_cancels_current_process_stream(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.signals = []

            def send_signal(self, value):
                self.signals.append(value)

        process = FakeProcess()

        PipeBackend(process).cancel()

        self.assertEqual(process.signals, [signal.SIGINT])

    def test_read_stream_forwards_events_and_sentinel(self) -> None:
        class FakeBackend:
            def stream(self, message):
                yield ConversationEvent(message)
                yield DoneEvent(3)

        events = queue.Queue()

        _read_stream(FakeBackend(), "hello", events)

        self.assertEqual(events.get_nowait(), ConversationEvent("hello"))
        self.assertEqual(events.get_nowait(), DoneEvent(3))
        self.assertIsNone(events.get_nowait())

    def test_ready_events_drains_pending_queue_items(self) -> None:
        events = queue.Queue()
        events.put(ConversationEvent("b"))
        events.put(ConversationEvent("c"))

        self.assertEqual(
            _ready_events(ConversationEvent("a"), events),
            [ConversationEvent("a"), ConversationEvent("b"), ConversationEvent("c")],
        )

    def test_wrap_messages_uses_terminal_width_for_chinese_text(self) -> None:
        self.assertEqual(
            _wrap_messages([("duckduckcode", "你好世界abc")], 16),
            [("duckduckcode", "你好世界abc")],
        )

    def test_wrap_messages_splits_newlines_before_rendering(self) -> None:
        self.assertEqual(
            _wrap_messages([("duckduckcode", "first\nsecond")], 20),
            [("duckduckcode", "first"), ("", "second")],
        )

    def test_clip_uses_terminal_width_for_chinese_text(self) -> None:
        self.assertEqual(_clip("你好abc", 6), "你好a")

    def test_visible_rows_scrolls_back_from_bottom(self) -> None:
        rows = ["one", "two", "three", "four", "five"]

        self.assertEqual(_visible_rows(rows, 3, 0), ["three", "four", "five"])
        self.assertEqual(_visible_rows(rows, 3, 1), ["two", "three", "four"])
        self.assertEqual(_visible_rows(rows, 3, 20), ["one", "two", "three"])

    def test_sgr_mouse_distinguishes_wheel_directions(self) -> None:
        self.assertEqual(_parse_sgr_mouse("[<64;10;4M"), (64, 9, 3, False))
        self.assertEqual(_parse_sgr_mouse("[<65;10;4M"), (65, 9, 3, False))

    def test_scrollbar_tracks_scroll_offset_from_bottom(self) -> None:
        self.assertEqual(_scrollbar(20, 5, 0), (4, 1, 15))
        self.assertEqual(_scrollbar(20, 5, 15), (0, 1, 15))

    def test_scrollbar_can_be_dragged_to_the_top(self) -> None:
        tui = _Tui(object(), "model", "/tmp", object())
        tui._scrollbar_geometry = (79, 6, 10, 8, 2, 15)

        tui._handle_mouse_event(0, 79, 14, False)
        tui._handle_mouse_event(32, 79, 6, False)

        self.assertEqual(tui.scroll_offset, 15)

    def test_send_returns_while_backend_is_generating(self) -> None:
        release = threading.Event()

        class FakeBackend:
            def stream(self, message):
                release.wait()
                yield DoneEvent()

        tui = _Tui(object(), "model", "/tmp", FakeBackend())
        tui.input = "hello"
        tui._draw = lambda: None
        call = threading.Thread(target=tui._send)

        call.start()
        call.join(0.05)
        returned = not call.is_alive()
        release.set()
        call.join(1)

        self.assertTrue(returned)

    def test_escape_interrupts_generation_then_exits_when_idle(self) -> None:
        class FakeBackend:
            def __init__(self) -> None:
                self.cancelled = 0

            def cancel(self):
                self.cancelled += 1

        backend = FakeBackend()
        tui = _Tui(object(), "model", "/tmp", backend)
        tui.messages = [("duckduckcode", "partial")]
        tui._events = queue.Queue()

        self.assertFalse(tui._interrupt_or_exit())
        self.assertEqual(backend.cancelled, 1)

        tui._events.put(ErrorEvent("interrupted", "interrupted"))
        tui._events.put(None)
        tui._consume_events()

        self.assertEqual(
            tui.messages,
            [("duckduckcode", "partial"), ("error", "interrupted")],
        )
        self.assertTrue(tui._interrupt_or_exit())

    def test_run_reduces_standalone_escape_delay(self) -> None:
        class FakeScreen:
            def __init__(self) -> None:
                self.keys = iter(["\x1b"])

            def bkgd(self, *args):
                pass

            def timeout(self, *args):
                pass

            def get_wch(self):
                try:
                    return next(self.keys)
                except StopIteration:
                    raise curses.error

        class FakeBackend:
            def close(self):
                pass

        tui = _Tui(FakeScreen(), "model", "/tmp", FakeBackend())
        tui._draw = lambda: None

        with (
            patch("duckduckcode.tui.curses.curs_set"),
            patch("duckduckcode.tui.curses.set_escdelay") as set_escdelay,
            patch("duckduckcode.tui.curses.mousemask"),
            patch("duckduckcode.tui._init_colors"),
            patch("duckduckcode.tui._color", return_value=0),
            patch("duckduckcode.tui._set_mouse_tracking"),
        ):
            tui.run()

        set_escdelay.assert_called_once_with(25)

    def test_draw_uses_colored_dots_and_warm_duck_separators(self) -> None:
        class FakeScreen:
            def __init__(self) -> None:
                self.lines = []
                self.strings = []

            def erase(self):
                pass

            def getmaxyx(self):
                return 16, 40

            def addstr(self, *args):
                self.strings.append(args)

            def hline(self, y, *args):
                self.lines.append(y)

            def addch(self, *args):
                pass

            def move(self, *args):
                pass

            def refresh(self):
                pass

        screen = FakeScreen()
        tui = _Tui(screen, "model", "/tmp", object())
        tui.messages = [
            ("you", "hello"),
            ("duckduckcode", "hi"),
            ("error", "interrupted"),
        ]

        with (
            patch("duckduckcode.tui._color", side_effect=lambda pair: pair),
            patch("duckduckcode.tui.curses.ACS_HLINE", 0, create=True),
            patch("duckduckcode.tui.curses.ACS_CKBOARD", 0, create=True),
            patch("duckduckcode.tui.curses.ACS_VLINE", 0, create=True),
        ):
            tui._draw()

        dots = [args for args in screen.strings if args[1:3] == (0, "●")]
        self.assertEqual(len(dots), 3)
        self.assertNotEqual(dots[0][3], dots[1][3])
        interrupted = next(args for args in screen.strings if args[2] == "interrupted")
        self.assertEqual(interrupted[3], dots[2][3])
        self.assertEqual(screen.lines, [5, 12, 14])


if __name__ == "__main__":
    unittest.main()
