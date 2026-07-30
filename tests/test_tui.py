from __future__ import annotations

import curses
import io
import json
import queue
import signal
import threading
import unittest
from unittest.mock import patch

from duckduckcode.core.event import (
    ConversationEvent,
    ErrorEvent,
    LoopCompleteEvent,
    PermissionRequestEvent,
    PlanReviewEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    UsageEvent,
)
from duckduckcode.interfaces import slash_command as slash_command_module
from duckduckcode.interfaces import tui as tui_module
from duckduckcode.interfaces.tui import (
    PipeBackend,
    _Tui,
    _clip,
    _input_rows,
    _parse_sgr_mouse,
    _ready_events,
    _read_stream,
    _scrollbar,
    _visible_rows,
    _wrap_messages,
)
from duckduckcode.tools.tool import ToolCall


class TuiTest(unittest.TestCase):
    def test_pipe_backend_starts_in_a_separate_process_session(self) -> None:
        with patch("duckduckcode.interfaces.tui.subprocess.Popen") as popen:
            PipeBackend()

        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_wrap_input_uses_terminal_width_for_ascii_and_chinese(self) -> None:
        wrap_input = getattr(tui_module, "_wrap_input", lambda *_args: None)

        self.assertEqual(wrap_input("", 4), [""])
        self.assertEqual(wrap_input("abcdefgh", 4), ["abcd", "efgh"])
        self.assertEqual(wrap_input("你好abc", 4), ["你好", "abc"])

    def test_input_edits_at_cursor_without_forcing_it_to_the_end(self) -> None:
        tui = _Tui(object(), "model", "/tmp", object())
        tui.input = "helo"
        tui.cursor_index = 3

        tui._insert_text("l")
        tui._move_cursor(-2)
        tui._delete_forward()
        tui._backspace()

        self.assertEqual(tui.input, "hlo")
        self.assertEqual(tui.cursor_index, 1)

    def test_input_home_end_and_selection_replacement(self) -> None:
        tui = _Tui(object(), "model", "/tmp", object())
        tui.input = "hello"
        tui.cursor_index = 5

        tui._set_cursor(0)
        tui._set_cursor(2, selecting=True)
        tui._insert_text("HE")
        tui._set_cursor(len(tui.input))

        self.assertEqual(tui.input, "HEllo")
        self.assertEqual(tui.cursor_index, 5)
        self.assertIsNone(tui.selection_anchor)

    def test_copy_cut_and_paste_use_selection_and_clipboard(self) -> None:
        tui = _Tui(object(), "model", "/tmp", object())
        tui.input = "hello"
        tui.cursor_index = 4
        tui.selection_anchor = 1

        with (
            patch("duckduckcode.interfaces.tui._write_clipboard") as copy,
            patch("duckduckcode.interfaces.tui._read_clipboard", return_value="i"),
        ):
            self.assertTrue(tui._copy_selection())
            tui._cut_selection()
            tui._paste()

        copy.assert_called_with("ell")
        self.assertEqual(tui.input, "hio")
        self.assertEqual(tui.cursor_index, 2)

    def test_input_rows_preserve_indexes_across_wraps_and_newlines(self) -> None:
        self.assertEqual(
            _input_rows("abcd\nef", 3),
            [(0, 3, "abc"), (3, 4, "d"), (5, 7, "ef")],
        )

    def test_pipe_backend_writes_prompt_and_reads_events_until_done(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = io.StringIO()
                self.stdout = io.StringIO(
                    '{"type": "stream_text", "delta": "he"}\n'
                    '{"type": "tool_use", "call_id": "call_1", '
                    '"name": "ReadFile", "arguments": {"path": "README.md"}}\n'
                    '{"type": "tool_result", "call_id": "call_1", '
                    '"name": "ReadFile", "content": "contents", '
                    '"is_error": false}\n'
                    '{"type": "usage", "total_tokens": 5}\n'
                    '{"type": "turn_complete", "iteration": 1}\n'
                    '{"type": "loop_complete", "reason": "completed", '
                    '"iterations": 1}\n'
                )

        process = FakeProcess()
        backend = PipeBackend(process)

        self.assertEqual(
            list(backend.stream("hello")),
            [
                ConversationEvent("he"),
                ToolCallEvent(ToolCall("call_1", "ReadFile", {"path": "README.md"})),
                ToolResultEvent("call_1", "ReadFile", "contents"),
                UsageEvent(5),
                TurnCompleteEvent(1),
                LoopCompleteEvent("completed", 1),
            ],
        )
        process.stdin.seek(0)
        self.assertEqual(json.loads(process.stdin.read()), {"message": "hello"})

    def test_pipe_backend_writes_permission_response(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = io.StringIO()

        process = FakeProcess()

        PipeBackend(process).respond_permission("call_1", "allow_always")

        process.stdin.seek(0)
        self.assertEqual(
            json.loads(process.stdin.read()),
            {
                "type": "permission_response",
                "call_id": "call_1",
                "decision": "allow_always",
            },
        )

    def test_pipe_backend_writes_plan_mode_and_review_responses(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = io.StringIO()

        process = FakeProcess()
        backend = PipeBackend(process)

        backend.set_mode("plan")
        backend.set_mode("default")
        backend.respond_plan_review(False, "Use SQLite instead")

        process.stdin.seek(0)
        self.assertEqual(
            [json.loads(line) for line in process.stdin.read().splitlines()],
            [
                {"type": "set_mode", "mode": "plan"},
                {"type": "set_mode", "mode": "default"},
                {
                    "type": "plan_review_response",
                    "approved": False,
                    "feedback": "Use SQLite instead",
                },
            ],
        )

    def test_pipe_backend_reads_plan_review_event(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdin = io.StringIO()
                self.stdout = io.StringIO(
                    '{"type": "plan_review", '
                    '"plan_file": "/repo/.duckduckcode/plan.md", '
                    '"content": "# Plan"}\n'
                    '{"type": "loop_complete", "reason": "completed", '
                    '"iterations": 1}\n'
                )

        self.assertEqual(
            list(PipeBackend(FakeProcess()).stream("review")),
            [
                PlanReviewEvent("/repo/.duckduckcode/plan.md", "# Plan"),
                LoopCompleteEvent("completed", 1),
            ],
        )

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
                yield UsageEvent(3)
                yield LoopCompleteEvent("completed", 1)

        events = queue.Queue()

        _read_stream(FakeBackend(), "hello", events)

        self.assertEqual(events.get_nowait(), ConversationEvent("hello"))
        self.assertEqual(events.get_nowait(), UsageEvent(3))
        self.assertEqual(events.get_nowait(), LoopCompleteEvent("completed", 1))
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

    def test_mouse_click_and_drag_position_input_cursor_and_selection(self) -> None:
        tui = _Tui(object(), "model", "/tmp", object())
        tui.input = "你好abc"
        tui.cursor_index = len(tui.input)
        tui._input_geometry = [(10, 0, len(tui.input), tui.input)]

        tui._handle_mouse_event(0, 4, 10, False)
        tui._handle_mouse_event(32, 8, 10, False)
        tui._handle_mouse_event(0, 8, 10, True)

        self.assertEqual(tui.selection_anchor, 1)
        self.assertEqual(tui.cursor_index, 4)

    def test_send_returns_while_backend_is_generating(self) -> None:
        release = threading.Event()

        class FakeBackend:
            def stream(self, message):
                release.wait()
                yield LoopCompleteEvent("completed", 1)

        tui = _Tui(object(), "model", "/tmp", FakeBackend())
        tui.input = "hello"
        tui.cursor_index = 3
        tui.selection_anchor = 1
        tui._draw = lambda: None
        call = threading.Thread(target=tui._send)

        call.start()
        call.join(0.05)
        returned = not call.is_alive()
        release.set()
        call.join(1)

        self.assertTrue(returned)
        self.assertEqual(tui.input, "")
        self.assertEqual(tui.cursor_index, 0)
        self.assertIsNone(tui.selection_anchor)

    def test_send_does_not_clear_input_when_no_message_is_sent(self) -> None:
        tui = _Tui(object(), "model", "/tmp", object())
        tui.input = "   "
        tui.cursor_index = 2

        tui._send()

        self.assertEqual(tui.input, "   ")
        self.assertEqual(tui.cursor_index, 2)

    def test_slash_commands_are_handled_without_starting_the_backend(self) -> None:
        cases = [
            (
                "/help",
                (
                    "duckduckcode",
                    "Available slash commands:\n"
                    "/help  Show available commands\n"
                    "/plan  Toggle Plan Mode",
                ),
            ),
            (
                "/missing",
                ("error", "Unknown command '/missing'. Use /help to list commands."),
            ),
        ]
        for prompt, expected_message in cases:
            with self.subTest(prompt=prompt):
                tui = _Tui(object(), "model", "/tmp", object())
                tui.input = prompt
                tui.cursor_index = len(prompt)

                tui._send()

                self.assertEqual(tui.messages, [expected_message])
                self.assertIsNone(tui._events)
                self.assertEqual(tui.input, "")
                self.assertEqual(tui.cursor_index, 0)

    def test_slash_command_suggestions_filter_by_prefix(self) -> None:
        suggestions = getattr(
            slash_command_module,
            "slash_command_suggestions",
            lambda _text: None,
        )

        self.assertEqual(
            suggestions("/h"),
            [("/help", "Show available commands")],
        )
        self.assertEqual(suggestions("/p"), [("/plan", "Toggle Plan Mode")])
        self.assertEqual(suggestions("/missing"), [])
        self.assertIsNone(suggestions("hello"))

    def test_plan_slash_command_switches_backend_without_calling_model(self) -> None:
        class FakeBackend:
            def __init__(self) -> None:
                self.modes = []

            def set_mode(self, mode):
                self.modes.append(mode)

        backend = FakeBackend()
        tui = _Tui(object(), "model", "/tmp", backend)
        tui.input = "/plan"
        tui.cursor_index = len(tui.input)

        tui._send()

        tui.input = "/plan"
        tui.cursor_index = len(tui.input)
        tui._send()

        self.assertEqual(backend.modes, ["plan", "default"])
        self.assertEqual(tui.mode, "default")
        self.assertEqual(tui.messages, [])
        self.assertIsNone(tui._events)

    def test_plan_slash_command_enters_plan_mode(self) -> None:
        class FakeBackend:
            def __init__(self) -> None:
                self.modes = []

            def set_mode(self, mode):
                self.modes.append(mode)

        backend = FakeBackend()
        tui = _Tui(object(), "model", "/tmp", backend)
        tui.input = "/plan"
        tui.cursor_index = len(tui.input)

        tui._send()

        self.assertEqual(backend.modes, ["plan"])
        self.assertEqual(tui.mode, "plan")
        self.assertEqual(tui.messages, [])
        self.assertIsNone(tui._events)

    def test_slash_dropdown_uses_arrows_and_enter(self) -> None:
        tui = _Tui(object(), "model", "/tmp", object())
        tui.input = "/"
        tui.cursor_index = 1
        handler = getattr(tui, "_handle_slash_key", lambda _key: False)
        suggestions = [
            ("/other", "Other command"),
            ("/help", "Show available commands"),
        ]

        with patch(
            "duckduckcode.interfaces.tui.slash_command_suggestions",
            return_value=suggestions,
            create=True,
        ):
            self.assertTrue(handler(curses.KEY_DOWN))
            self.assertEqual(getattr(tui, "_slash_selection", None), 1)
            self.assertTrue(handler("\n"))

        self.assertEqual(
            tui.messages,
            [
                (
                    "duckduckcode",
                    "Available slash commands:\n"
                    "/help  Show available commands\n"
                    "/plan  Toggle Plan Mode",
                )
            ],
        )
        self.assertIsNone(tui._events)

    def test_plan_review_requires_an_explicit_approve_selection(self) -> None:
        class FakeBackend:
            def __init__(self) -> None:
                self.responses = []

            def respond_plan_review(self, approved, feedback=""):
                self.responses.append((approved, feedback))

        backend = FakeBackend()
        tui = _Tui(object(), "model", "/tmp", backend)
        tui.mode = "plan"
        tui._events = queue.Queue()
        tui._plan_review = PlanReviewEvent("/tmp/.duckduckcode/plan.md", "# Plan")

        tui._handle_plan_review_key("\n")

        self.assertEqual(backend.responses, [(True, "")])
        self.assertEqual(tui.mode, "default")
        self.assertIsNone(tui._plan_review)

    def test_plan_review_supports_deny_and_free_form_feedback(self) -> None:
        class FakeBackend:
            def __init__(self) -> None:
                self.responses = []

            def respond_plan_review(self, approved, feedback=""):
                self.responses.append((approved, feedback))

        backend = FakeBackend()
        tui = _Tui(object(), "model", "/tmp", backend)
        tui.mode = "plan"
        tui._events = queue.Queue()
        tui._plan_review = PlanReviewEvent("/tmp/.duckduckcode/plan.md", "# Plan")

        tui._handle_plan_review_key(curses.KEY_DOWN)
        tui._handle_plan_review_key("\n")

        self.assertEqual(backend.responses, [(False, "")])
        self.assertEqual(tui.mode, "plan")

        tui._plan_review = PlanReviewEvent(
            "/tmp/.duckduckcode/plan.md", "# Revised Plan"
        )
        tui._plan_review_selection = 2
        tui.input = "Use SQLite instead"
        tui.cursor_index = len(tui.input)

        tui._handle_plan_review_key("\n")

        self.assertEqual(
            backend.responses, [(False, ""), (False, "Use SQLite instead")]
        )
        self.assertIn(("you", "Use SQLite instead"), tui.messages)

    def test_plan_review_printable_input_selects_feedback(self) -> None:
        tui = _Tui(object(), "model", "/tmp", object())
        tui._plan_review = PlanReviewEvent("/tmp/.duckduckcode/plan.md", "# Plan")

        self.assertFalse(tui._handle_plan_review_key("x"))

        self.assertEqual(tui._plan_review_selection, 2)

    def test_tool_permission_approval_does_not_approve_the_plan(self) -> None:
        class FakeBackend:
            def __init__(self) -> None:
                self.responses = []

            def respond_permission(self, call_id, decision):
                self.responses.append((call_id, decision))

        backend = FakeBackend()
        tui = _Tui(object(), "model", "/tmp", backend)
        tui.mode = "plan"
        tui._permission_request = PermissionRequestEvent(
            "call_1", "Bash", "git status", "approval required"
        )
        tui._permission_selection = 0

        tui._handle_permission_key("\n")

        self.assertEqual(backend.responses, [("call_1", "allow_once")])
        self.assertEqual(tui.mode, "plan")

    def test_plan_review_event_keeps_input_available(self) -> None:
        tui = _Tui(object(), "model", "/tmp", object())
        tui.messages = [("duckduckcode", "waiting")]
        tui._waiting = True
        tui._events = queue.Queue()
        tui._events.put(
            PlanReviewEvent("/tmp/.duckduckcode/plan.md", "# Plan\n\nDo the work.")
        )

        tui._consume_events()
        tui._insert_text("change the database")

        self.assertEqual(
            tui._plan_review,
            PlanReviewEvent(
                "/tmp/.duckduckcode/plan.md",
                "# Plan\n\nDo the work.",
            ),
        )
        self.assertEqual(
            tui.messages,
            [
                ("duckduckcode", "# Plan\n\nDo the work."),
                ("duckduckcode", "Plan completed"),
            ],
        )
        self.assertEqual(tui.input, "change the database")
        self.assertFalse(tui._waiting)

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

    def test_ctrl_c_exits_cleanly_while_idle(self) -> None:
        class FakeScreen:
            def bkgd(self, *_args):
                pass

            def timeout(self, _value):
                pass

            def get_wch(self):
                raise KeyboardInterrupt

        class FakeBackend:
            def __init__(self):
                self.closed = 0

            def close(self):
                self.closed += 1

        backend = FakeBackend()
        tui = _Tui(FakeScreen(), "model", "/tmp", backend)

        with (
            patch("duckduckcode.interfaces.tui.curses.curs_set"),
            patch("duckduckcode.interfaces.tui.curses.set_escdelay"),
            patch("duckduckcode.interfaces.tui.curses.mousemask"),
            patch("duckduckcode.interfaces.tui._init_colors"),
            patch("duckduckcode.interfaces.tui._set_mouse_tracking"),
            patch("duckduckcode.interfaces.tui._color", return_value=0),
            patch.object(tui, "_draw"),
        ):
            tui.run()

        self.assertEqual(backend.closed, 1)

    def test_ctrl_c_cancels_active_stream_once_without_exiting_with_traceback(
        self,
    ) -> None:
        class FakeScreen:
            def __init__(self):
                self.calls = 0
                self.tui = None

            def bkgd(self, *_args):
                pass

            def timeout(self, _value):
                pass

            def get_wch(self):
                self.calls += 1
                if self.calls == 1:
                    self.tui._events.put(LoopCompleteEvent("cancelled", 1))
                    self.tui._events.put(None)
                raise KeyboardInterrupt

        class FakeBackend:
            def __init__(self):
                self.cancelled = 0
                self.closed = 0

            def cancel(self):
                self.cancelled += 1

            def close(self):
                self.closed += 1

        screen = FakeScreen()
        backend = FakeBackend()
        tui = _Tui(screen, "model", "/tmp", backend)
        screen.tui = tui
        tui._events = queue.Queue()

        with (
            patch("duckduckcode.interfaces.tui.curses.curs_set"),
            patch("duckduckcode.interfaces.tui.curses.set_escdelay"),
            patch("duckduckcode.interfaces.tui.curses.mousemask"),
            patch("duckduckcode.interfaces.tui._init_colors"),
            patch("duckduckcode.interfaces.tui._set_mouse_tracking"),
            patch("duckduckcode.interfaces.tui._color", return_value=0),
            patch.object(tui, "_draw"),
        ):
            tui.run()

        self.assertEqual(backend.cancelled, 1)
        self.assertEqual(backend.closed, 1)

    def test_consume_events_displays_tools_without_mixing_model_text(self) -> None:
        tui = _Tui(object(), "model", "/tmp", object())
        tui.messages = [("duckduckcode", "thinking")]
        tui._waiting = True
        tui._events = queue.Queue()
        for event in (
            ToolCallEvent(ToolCall("call_1", "ReadFile", {"path": "README.md"})),
            ToolResultEvent("call_1", "ReadFile", "contents"),
            UsageEvent(4),
            TurnCompleteEvent(1),
            ConversationEvent("done"),
            UsageEvent(2),
            TurnCompleteEvent(2),
            LoopCompleteEvent("completed", 2),
            None,
        ):
            tui._events.put(event)

        tui._consume_events()

        self.assertEqual(
            tui.messages,
            [
                ("tool:call_1", "→ ReadFile running…"),
                ("duckduckcode", "done"),
            ],
        )
        self.assertEqual(tui.tool_results["call_1"].content, "contents")
        self.assertEqual(
            tui._display_messages()[0],
            ("tool:call_1", "✓ ReadFile completed\ncontents"),
        )
        self.assertEqual(tui.tokens, 6)
        self.assertIsNone(tui._events)

    def test_permission_dialog_uses_arrows_enter_and_escape(self) -> None:
        class FakeBackend:
            def __init__(self) -> None:
                self.responses = []

            def respond_permission(self, call_id, decision):
                self.responses.append((call_id, decision))

        backend = FakeBackend()
        tui = _Tui(object(), "model", "/tmp", backend)
        tui.messages = [("tool:call_1", "→ Bash running…")]
        tui._events = queue.Queue()
        tui._events.put(
            PermissionRequestEvent(
                "call_1",
                "Bash",
                "git push origin main",
                "approval required",
            )
        )
        tui._consume_events()

        self.assertEqual(tui._permission_selection, 2)
        tui._handle_permission_key(curses.KEY_UP)
        self.assertEqual(tui._permission_selection, 1)
        tui._handle_permission_key("\n")
        self.assertEqual(backend.responses, [("call_1", "allow_always")])
        self.assertIsNone(tui._permission_request)

        tui._permission_request = PermissionRequestEvent(
            "call_2", "Bash", "git push origin next", "approval required"
        )
        tui._handle_permission_key("\x1b")

        self.assertEqual(
            backend.responses,
            [("call_1", "allow_always"), ("call_2", "deny")],
        )
        self.assertIsNone(tui._permission_request)

    def test_long_tool_result_is_collapsed_but_preserved_and_expandable(self) -> None:
        tui = _Tui(object(), "model", "/tmp", object())
        content = "\n".join(f"src/file.py:{line}:match" for line in range(1, 101))
        event = ToolResultEvent("call_1", "Grep", content)
        tui.messages = [("tool:call_1", "→ Grep running…")]
        tui.tool_results["call_1"] = event

        self.assertEqual(
            tui._display_messages(),
            [("tool:call_1", "✓ Grep completed · 100 matches · [查看详情]")],
        )

        tui._chat_tool_rows = {8: "call_1"}
        tui._handle_mouse_event(0, 4, 8, False)

        self.assertEqual(tui.tool_results["call_1"].content, content)
        self.assertIn(content, tui._display_messages()[0][1])
        self.assertIn("[收起]", tui._display_messages()[0][1])

    def test_failed_tool_result_collapses_full_error(self) -> None:
        tui = _Tui(object(), "model", "/tmp", object())
        content = "permission denied\n" + "trace\n" * 20
        tui.messages = [("tool:call_1", "→ Bash running…")]
        tui.tool_results["call_1"] = ToolResultEvent(
            "call_1", "Bash", content, is_error=True
        )

        displayed = tui._display_messages()[0][1]

        self.assertEqual(
            displayed,
            "✗ Bash failed · permission denied · [查看详情]",
        )
        self.assertNotIn("trace", displayed)

    def test_bash_result_size_uses_output_lines_from_json(self) -> None:
        tui = _Tui(object(), "model", "/tmp", object())
        content = json.dumps(
            {"output": "".join(f"line {line}\n" for line in range(20)), "exit_code": 0}
        )
        tui.messages = [("tool:call_1", "→ Bash running…")]
        tui.tool_results["call_1"] = ToolResultEvent("call_1", "Bash", content)

        self.assertEqual(
            tui._display_messages(),
            [("tool:call_1", "✓ Bash completed · 20 output lines · [查看详情]")],
        )

    def test_short_bash_result_displays_output_instead_of_json(self) -> None:
        tui = _Tui(object(), "model", "/tmp", object())
        content = json.dumps({"output": "hello\n", "exit_code": 0})
        tui.messages = [("tool:call_1", "→ Bash running…")]
        tui.tool_results["call_1"] = ToolResultEvent("call_1", "Bash", content)

        self.assertEqual(
            tui._display_messages(),
            [("tool:call_1", "✓ Bash completed\nhello\n")],
        )

    def test_empty_final_turn_removes_the_waiting_placeholder(self) -> None:
        tui = _Tui(object(), "model", "/tmp", object())
        tui.messages = [("duckduckcode", "thinking")]
        tui._waiting = True
        tui._events = queue.Queue()
        for event in (
            UsageEvent(0),
            TurnCompleteEvent(1),
            LoopCompleteEvent("completed", 1),
            None,
        ):
            tui._events.put(event)

        tui._consume_events()

        self.assertEqual(tui.messages, [])
        self.assertIsNone(tui._events)

    def test_ctrl_c_cancels_active_generation_then_exits_when_idle(self) -> None:
        class FakeScreen:
            def __init__(self) -> None:
                self.keys = iter(["\x03", "\x03"])

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
            def __init__(self) -> None:
                self.cancelled = 0
                self.closed = 0
                self.events = None

            def cancel(self):
                self.cancelled += 1
                self.events.put(ErrorEvent("interrupted", "interrupted"))
                self.events.put(LoopCompleteEvent("cancelled", 1))
                self.events.put(None)

            def close(self):
                self.closed += 1

        backend = FakeBackend()
        tui = _Tui(FakeScreen(), "model", "/tmp", backend)
        tui.messages = [("duckduckcode", "thinking")]
        tui._events = queue.Queue()
        tui._waiting = True
        backend.events = tui._events
        tui._draw = lambda: None

        with (
            patch("duckduckcode.interfaces.tui.curses.curs_set"),
            patch("duckduckcode.interfaces.tui.curses.set_escdelay"),
            patch("duckduckcode.interfaces.tui.curses.mousemask"),
            patch("duckduckcode.interfaces.tui._init_colors"),
            patch("duckduckcode.interfaces.tui._color", return_value=0),
            patch("duckduckcode.interfaces.tui._set_mouse_tracking"),
        ):
            tui.run()

        self.assertEqual(backend.cancelled, 1)
        self.assertEqual(backend.closed, 1)
        self.assertIn(("error", "interrupted"), tui.messages)

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
            patch("duckduckcode.interfaces.tui.curses.curs_set"),
            patch("duckduckcode.interfaces.tui.curses.set_escdelay") as set_escdelay,
            patch("duckduckcode.interfaces.tui.curses.mousemask"),
            patch("duckduckcode.interfaces.tui._init_colors"),
            patch("duckduckcode.interfaces.tui._color", return_value=0),
            patch("duckduckcode.interfaces.tui._set_mouse_tracking"),
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
            patch(
                "duckduckcode.interfaces.tui._color",
                side_effect=lambda pair: pair,
            ),
            patch("duckduckcode.interfaces.tui.curses.ACS_HLINE", 0, create=True),
            patch("duckduckcode.interfaces.tui.curses.ACS_CKBOARD", 0, create=True),
            patch("duckduckcode.interfaces.tui.curses.ACS_VLINE", 0, create=True),
        ):
            tui._draw()

        dots = [args for args in screen.strings if args[1:3] == (0, "●")]
        self.assertEqual(len(dots), 3)
        self.assertNotEqual(dots[0][3], dots[1][3])
        interrupted = next(args for args in screen.strings if args[2] == "interrupted")
        self.assertEqual(interrupted[3], dots[2][3])
        self.assertEqual(screen.lines, [5, 12, 14])

    def test_draw_grows_input_to_one_third_and_keeps_the_end_visible(self) -> None:
        class FakeScreen:
            def __init__(self) -> None:
                self.lines = []
                self.strings = []
                self.cursor = None

            def erase(self):
                pass

            def getmaxyx(self):
                return 18, 12

            def addstr(self, *args):
                self.strings.append(args)

            def hline(self, y, *args):
                self.lines.append(y)

            def addch(self, *args):
                pass

            def move(self, y, x):
                self.cursor = (y, x)

            def refresh(self):
                pass

        screen = FakeScreen()
        tui = _Tui(screen, "model", "/tmp", object())
        tui.input = "".join(str(number) * 9 for number in range(8))
        tui.cursor_index = len(tui.input)

        with (
            patch("duckduckcode.interfaces.tui._color", return_value=0),
            patch("duckduckcode.interfaces.tui.curses.ACS_HLINE", 0, create=True),
            patch("duckduckcode.interfaces.tui.curses.ACS_CKBOARD", 0, create=True),
            patch("duckduckcode.interfaces.tui.curses.ACS_VLINE", 0, create=True),
        ):
            tui._draw()

        input_lines = [
            args[2]
            for args in screen.strings
            if len(args) >= 3 and args[0] in range(10, 16) and args[1] == 2
        ]
        self.assertEqual(
            input_lines,
            [
                "2" * 9,
                "3" * 9,
                "4" * 9,
                "5" * 9,
                "6" * 9,
                "7" * 9,
            ],
        )
        self.assertIn((10, 0, "›", curses.A_BOLD), screen.strings)
        self.assertEqual(screen.lines, [5, 9, 16])
        self.assertEqual(screen.cursor, (15, 11))

        screen.lines.clear()
        screen.strings.clear()
        tui.cursor_index = 20
        with (
            patch("duckduckcode.interfaces.tui._color", return_value=0),
            patch("duckduckcode.interfaces.tui.curses.ACS_HLINE", 0, create=True),
            patch("duckduckcode.interfaces.tui.curses.ACS_CKBOARD", 0, create=True),
            patch("duckduckcode.interfaces.tui.curses.ACS_VLINE", 0, create=True),
        ):
            tui._draw()

        self.assertEqual(screen.cursor, (12, 4))

    def test_permission_panel_is_below_input_without_covering_chat(self) -> None:
        class FakeScreen:
            def __init__(self) -> None:
                self.strings = []

            def erase(self):
                pass

            def getmaxyx(self):
                return 24, 60

            def addstr(self, *args):
                self.strings.append(args)

            def hline(self, *args):
                pass

            def addch(self, *args):
                pass

            def move(self, *args):
                pass

            def refresh(self):
                pass

        screen = FakeScreen()
        tui = _Tui(screen, "model", "/tmp", object())
        tui.messages = [("you", "history")]
        tui.input = "next request"
        tui.cursor_index = len(tui.input)
        tui._permission_request = PermissionRequestEvent(
            "call_1", "Bash", "pwd", "approval required"
        )

        with (
            patch("duckduckcode.interfaces.tui._color", return_value=0),
            patch("duckduckcode.interfaces.tui.curses.ACS_HLINE", 0, create=True),
            patch("duckduckcode.interfaces.tui.curses.ACS_CKBOARD", 0, create=True),
            patch("duckduckcode.interfaces.tui.curses.ACS_VLINE", 0, create=True),
        ):
            tui._draw()

        history_row = next(args[0] for args in screen.strings if args[2] == "history")
        input_row = next(args[0] for args in screen.strings if args[2] == "›")
        panel_row = next(
            args[0]
            for args in screen.strings
            if len(args) >= 3 and "请求权限" in args[2]
        )
        status_row = next(args[0] for args in screen.strings if args[2] == "model")

        self.assertLess(history_row, input_row)
        self.assertLess(input_row, panel_row)
        self.assertLess(panel_row, status_row)

    def test_plan_review_panel_is_below_input_and_shows_plan_status(self) -> None:
        class FakeScreen:
            def __init__(self) -> None:
                self.strings = []

            def erase(self):
                pass

            def getmaxyx(self):
                return 20, 80

            def addstr(self, *args):
                self.strings.append(args)

            def hline(self, *args):
                pass

            def addch(self, *args):
                pass

            def move(self, *args):
                pass

            def refresh(self):
                pass

        screen = FakeScreen()
        tui = _Tui(screen, "model", "/tmp", object())
        tui.mode = "plan"
        tui.messages = [("you", "history")]
        tui._plan_review = PlanReviewEvent("/tmp/.duckduckcode/plan.md", "# Plan")

        with (
            patch("duckduckcode.interfaces.tui._color", return_value=0),
            patch("duckduckcode.interfaces.tui.curses.ACS_HLINE", 0, create=True),
            patch("duckduckcode.interfaces.tui.curses.ACS_CKBOARD", 0, create=True),
            patch("duckduckcode.interfaces.tui.curses.ACS_VLINE", 0, create=True),
        ):
            tui._draw()

        input_row = next(args[0] for args in screen.strings if args[2] == "›")
        panel_row = next(
            args[0]
            for args in screen.strings
            if len(args) >= 3 and "批准并执行" in args[2]
        )
        status_row = next(
            args[0] for args in screen.strings if args[2] == "model · plan"
        )
        labels = [args[2].strip("› ") for args in screen.strings]
        rendered_text = "\n".join(args[2] for args in screen.strings if len(args) >= 3)

        self.assertLess(input_row, panel_row)
        self.assertLess(panel_row, status_row)
        self.assertIn("批准并执行", labels)
        self.assertIn("拒绝计划", labels)
        self.assertIn("输入修改意见", labels)
        self.assertNotIn("plan.md", rendered_text)

    def test_slash_dropdown_is_below_input_without_covering_chat(self) -> None:
        class FakeScreen:
            def __init__(self) -> None:
                self.strings = []

            def erase(self):
                pass

            def getmaxyx(self):
                return 18, 60

            def addstr(self, *args):
                self.strings.append(args)

            def hline(self, *args):
                pass

            def addch(self, *args):
                pass

            def move(self, *args):
                pass

            def refresh(self):
                pass

        screen = FakeScreen()
        tui = _Tui(screen, "model", "/tmp", object())
        tui.messages = [("you", "history")]
        tui.input = "/h"
        tui.cursor_index = len(tui.input)

        with (
            patch("duckduckcode.interfaces.tui._color", return_value=0),
            patch("duckduckcode.interfaces.tui.curses.ACS_HLINE", 0, create=True),
            patch("duckduckcode.interfaces.tui.curses.ACS_CKBOARD", 0, create=True),
            patch("duckduckcode.interfaces.tui.curses.ACS_VLINE", 0, create=True),
        ):
            tui._draw()

        history_row = next(args[0] for args in screen.strings if args[2] == "history")
        input_row = next(args[0] for args in screen.strings if args[2] == "›")
        command_rows = [
            args[0] for args in screen.strings if len(args) >= 3 and "/help" in args[2]
        ]
        status_row = next(args[0] for args in screen.strings if args[2] == "model")

        self.assertEqual(len(command_rows), 1)
        self.assertLess(history_row, input_row)
        self.assertLess(input_row, command_rows[0])
        self.assertLess(command_rows[0], status_row)


if __name__ == "__main__":
    unittest.main()
