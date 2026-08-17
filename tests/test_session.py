from __future__ import annotations

import json
import stat
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from duckduckcode.core.agent import Agent
from duckduckcode.core.context import ContextManager, Message
from duckduckcode.core.event import (
    ConversationEvent,
    DoneEvent,
    ErrorEvent,
    LoopCompleteEvent,
    ToolCallEvent,
)
from duckduckcode.core.prompts import PLAN_MODE_REMINDER
from duckduckcode.memory import SessionManager, SessionPersistenceError
from duckduckcode.memory.session import SYNTHETIC_TOOL_ERROR
from duckduckcode.tools.tool import ToolCall, ToolManager, ToolResult


class SessionManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.now = datetime(2026, 8, 3, 12, 0)

    def manager(self, context: ContextManager | None = None) -> SessionManager:
        return SessionManager(
            self.workspace,
            context or ContextManager(system_prompt="system"),
            clock=lambda: self.now,
        )

    def test_create_uses_collision_suffix_permissions_and_strict_jsonl(self) -> None:
        first = self.manager()
        first_snapshot = first.start()
        first.commit_message("user", "你好 🦆")
        second = self.manager()
        second_snapshot = second.create()

        self.assertEqual(first_snapshot.session_id, "20260803-120000")
        self.assertEqual(second_snapshot.session_id, "20260803-120000-2")
        self.assertEqual(
            stat.S_IMODE(first.directory.stat().st_mode),
            0o700,
        )
        self.assertEqual(stat.S_IMODE(first.current_path.stat().st_mode), 0o600)
        record = json.loads(first.current_path.read_text(encoding="utf-8"))
        self.assertEqual(set(record), {"role", "context", "ts"})
        self.assertEqual(record["context"]["content"], "你好 🦆")
        self.assertEqual(record["ts"], int(self.now.timestamp()))

    def test_start_restores_latest_valid_session_and_keeps_invalid_file(self) -> None:
        manager = self.manager()
        old = manager.start()
        manager.commit_message("user", "old")
        self.now += timedelta(seconds=1)
        newest = manager.create()
        manager.commit_message("assistant", "new", token_usage=7)
        invalid = manager.directory / "broken.jsonl"
        invalid.write_text("not json\n", encoding="utf-8")

        restored_context = ContextManager(system_prompt="system")
        snapshot = self.manager(restored_context).start()

        self.assertNotEqual(old.session_id, newest.session_id)
        self.assertEqual(snapshot.session_id, newest.session_id)
        self.assertTrue(snapshot.restored)
        self.assertEqual(snapshot.invalid, ("broken",))
        self.assertTrue(invalid.exists())
        self.assertEqual(
            restored_context.messages(), [Message("assistant", "new", token_usage=7)]
        )

    def test_replay_preserves_raw_history_and_applies_compaction(self) -> None:
        manager = self.manager()
        snapshot = manager.start()
        manager.commit_message("user", "old")
        manager.commit_message("assistant", "answer", token_usage=3)
        manager.commit_message("user", "current")
        manager.commit_compaction("summary", 2, token_usage=5)

        context = ContextManager(system_prompt="system")
        restored = self.manager(context).resume(snapshot.session_id)

        self.assertEqual(len(restored.records), 4)
        self.assertEqual(restored.token_usage, 8)
        self.assertEqual(context.abstraction, "summary")
        self.assertEqual(context.messages(), [Message("user", "current")])

    def test_replay_preserves_reasoning_needed_by_tool_calls(self) -> None:
        manager = self.manager()
        snapshot = manager.start()
        index = manager.context.start_assistant_stream()
        manager.context.append_assistant_reasoning_delta(index, "private reasoning")
        manager.commit_assistant_stream(index, "completed")
        manager.commit_tool_call(ToolCall("call_1", "ReadFile", {"path": "README.md"}))
        manager.commit_tool_result(ToolCall("call_1", "ReadFile", {}), ToolResult("ok"))

        context = ContextManager(system_prompt="system")
        self.manager(context).resume(snapshot.session_id)

        self.assertEqual(context.messages()[0].reasoning_content, "private reasoning")

    def test_missing_tool_result_is_repaired_without_advancing_activity(self) -> None:
        manager = self.manager()
        snapshot = manager.start()
        call = ToolCall("call_1", "ReadFile", {"path": "README.md"})
        manager.commit_tool_call(call)
        activity = manager.list()[0].last_activity

        context = ContextManager(system_prompt="system")
        restored_manager = self.manager(context)
        restored_manager.resume(snapshot.session_id)

        self.assertEqual(restored_manager.list()[0].last_activity, activity)
        self.assertEqual(
            context.messages()[0],
            Message.tool_call("call_1", "ReadFile", {"path": "README.md"}),
        )
        self.assertEqual(context.messages()[1].kind, "tool_result")
        self.assertIn(SYNTHETIC_TOOL_ERROR, context.messages()[1].content)

    def test_stale_reminder_has_strict_24_hour_boundary(self) -> None:
        manager = self.manager()
        snapshot = manager.start()
        manager.commit_message("user", "hello")

        self.now += timedelta(hours=24)
        exact_context = ContextManager(system_prompt="system")
        self.manager(exact_context).resume(snapshot.session_id)
        self.assertEqual(exact_context.reminder, "")

        self.now += timedelta(seconds=1)
        stale_context = ContextManager(system_prompt="system")
        self.manager(stale_context).resume(snapshot.session_id)
        self.assertIn("More than 24 hours", stale_context.reminder)
        self.assertEqual(stale_context.model_messages()[1].role, "system")

    def test_summary_and_stale_reminder_precede_mode_and_history(self) -> None:
        context = ContextManager(system_prompt="system", abstraction="summary")
        context.set_reminder("stale")
        context.set_mode("plan")
        context.add_user("current")

        messages = context.model_messages()

        self.assertEqual(messages[0], Message("system", "system"))
        self.assertIn("summary", messages[1].content)
        self.assertEqual(messages[2], Message("system", "stale"))
        self.assertEqual(messages[3], Message("system", PLAN_MODE_REMINDER))
        self.assertEqual(messages[4], Message("user", "current"))

    def test_start_deletes_only_valid_sessions_strictly_older_than_30_days(
        self,
    ) -> None:
        manager = self.manager()
        old = manager.start()
        manager.commit_message("user", "old")
        invalid = manager.directory / "invalid.jsonl"
        invalid.write_text("broken\n", encoding="utf-8")

        self.now += timedelta(days=30)
        exact = self.manager().start()
        self.assertEqual(exact.cleaned, 0)

        self.now += timedelta(seconds=1)
        cleaned = self.manager().start()
        self.assertEqual(cleaned.cleaned, 1)
        self.assertFalse((manager.directory / f"{old.session_id}.jsonl").exists())
        self.assertTrue(invalid.exists())

    def test_rejects_traversal_and_symlink_sessions(self) -> None:
        manager = self.manager()
        manager.start()
        outside = self.workspace / "outside.jsonl"
        outside.write_text("", encoding="utf-8")
        link = manager.directory / "link.jsonl"
        link.symlink_to(outside)

        with self.assertRaises(ValueError):
            manager.resume("../outside")
        with self.assertRaises(ValueError):
            manager.resume("link")
        self.assertEqual(
            next(info.status for info in manager.list() if info.id == "link"),
            "invalid",
        )

    def test_failed_fsync_does_not_update_context(self) -> None:
        context = ContextManager(system_prompt="system")
        manager = self.manager(context)
        manager.start()

        with patch(
            "duckduckcode.memory.session.os.fsync", side_effect=OSError("disk full")
        ):
            with self.assertRaises(SessionPersistenceError):
                manager.commit_message("user", "lost")

        self.assertEqual(context.messages(), [])

    def test_delete_current_creates_empty_replacement(self) -> None:
        manager = self.manager()
        original = manager.start()
        manager.commit_message("user", "hello")

        replacement = manager.delete()

        self.assertNotEqual(replacement.session_id, original.session_id)
        self.assertEqual(replacement.records, ())
        self.assertFalse((manager.directory / f"{original.session_id}.jsonl").exists())

    def test_agent_restores_mcp_state_and_schemas_on_each_session_switch(self) -> None:
        context = ContextManager(system_prompt="system")
        session_manager = self.manager(context)
        tools = ToolManager()

        class MCP:
            def __init__(self):
                self.restored = []

            def permission_request(self):
                return None

            def initialize(self, choice):
                return []

            def catalog_block(self):
                return ""

            def restore_session(self, records):
                self.restored.append(list(records))
                return ["restore warning"]

        class Client:
            def stream(self, messages, tools=None, reasoning=None):
                yield DoneEvent()

        mcp = MCP()
        agent = Agent(
            Client(),
            context,
            tools,
            session_manager=session_manager,
            mcp_manager=mcp,
        )
        list(agent.initialize())
        session_manager.commit_message("user", "old session")
        old_session = session_manager.current_session_id
        mcp.restored.clear()

        switches = [
            ("after-new", agent.new_session),
            ("after-resume", lambda: agent.resume_session(old_session)),
            ("after-delete", agent.delete_session),
        ]
        for tool_name, switch in switches:
            tools.register(
                tool_name,
                tool_name,
                {"type": "object"},
                lambda: "ok",
            )

            events = list(switch())

            self.assertEqual(
                mcp.restored[-1],
                [record.as_dict() for record in session_manager.snapshot().records],
            )
            self.assertEqual(
                [event for event in events if isinstance(event, ErrorEvent)],
                [ErrorEvent("restore warning", "mcp")],
            )
            self.assertEqual(context.tool_schemas(), tools.schemas())

        self.assertEqual(len(mcp.restored), 3)

    def test_mcp_restore_receives_load_history_before_compaction(self) -> None:
        original_context = ContextManager(system_prompt="system")
        original = self.manager(original_context)
        snapshot = original.start()
        call = ToolCall(
            "load",
            "LoadTools",
            {"names": ["mcp__docs__search"]},
        )
        original.commit_tool_call(call)
        original.commit_tool_result(call, ToolResult("loaded"))
        original.commit_compaction("summary", 2)

        restored_context = ContextManager(system_prompt="system")
        restored_sessions = self.manager(restored_context)

        class MCP:
            def __init__(self):
                self.restored = []

            def permission_request(self):
                return None

            def initialize(self, choice):
                return []

            def catalog_block(self):
                return ""

            def restore_session(self, records):
                self.restored = list(records)
                return []

        mcp = MCP()
        agent = Agent(
            object(),
            restored_context,
            ToolManager(),
            session_manager=restored_sessions,
            mcp_manager=mcp,
        )

        list(agent.resume_session(snapshot.session_id))
        list(agent.initialize())

        self.assertEqual(
            [record["context"]["type"] for record in mcp.restored],
            ["tool_call", "tool_result", "compaction"],
        )
        self.assertEqual(
            mcp.restored[0]["context"],
            {
                "type": "tool_call",
                "call_id": "load",
                "name": "LoadTools",
                "arguments": {"names": ["mcp__docs__search"]},
            },
        )
        self.assertFalse(mcp.restored[1]["context"]["is_error"])

    def test_agent_persists_calls_before_tools_run_and_replays_complete_chain(
        self,
    ) -> None:
        context = ContextManager(system_prompt="system")
        manager = self.manager(context)
        tools = ToolManager()
        call = ToolCall("call_1", "Check", {"text": "你好"})

        def check(text: str) -> ToolResult:
            records = [
                json.loads(line)
                for line in manager.current_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(records[-1]["context"]["type"], "tool_call")
            return ToolResult(text)

        tools.register(
            "Check",
            "check persistence",
            {"type": "object", "properties": {"text": {"type": "string"}}},
            check,
        )

        class Client:
            calls = 0

            def stream(self, messages, tools=None, reasoning=None):
                self.calls += 1
                if self.calls == 1:
                    yield ConversationEvent("working")
                    yield ToolCallEvent(call)
                    yield DoneEvent(4)
                else:
                    yield ConversationEvent("done")
                    yield DoneEvent(5)

        agent = Agent(Client(), context, tools, session_manager=manager)
        events = list(agent.stream("run"))
        records = manager.snapshot().records

        self.assertEqual(
            [record.context["type"] for record in records],
            ["message", "message", "tool_call", "tool_result", "message"],
        )
        self.assertEqual(manager.snapshot().token_usage, 9)
        self.assertIn(LoopCompleteEvent("completed", 2), events)

    def test_agent_persists_partial_reply_as_error_on_interrupt(self) -> None:
        context = ContextManager(system_prompt="system")
        manager = self.manager(context)

        class Client:
            def stream(self, messages, tools=None, reasoning=None):
                yield ConversationEvent("partial")
                raise KeyboardInterrupt

        events = list(Agent(Client(), context, session_manager=manager).stream("run"))

        self.assertIn(ErrorEvent("interrupted", "interrupted"), events)
        self.assertEqual(manager.snapshot().records[-1].context["status"], "error")
        self.assertEqual(manager.snapshot().records[-1].context["content"], "partial")


if __name__ == "__main__":
    unittest.main()
