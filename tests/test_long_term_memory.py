from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from duckduckcode.core.agent import Agent
from duckduckcode.core.context import ContextManager, Message, ReasoningConfig
from duckduckcode.core.event import (
    ConversationEvent,
    DoneEvent,
    ErrorEvent,
    LoopCompleteEvent,
    ToolCallEvent,
)
from duckduckcode.memory import MemoryError, MemoryManager, SessionManager
from duckduckcode.memory.consolidate import (
    CONSOLIDATE_AFTER_SECONDS,
    _consolidation_tools,
    count_active_sessions,
    maybe_consolidate,
)
from duckduckcode.memory.long_term import (
    MEMORY_MAX_BYTES,
    MEMORY_MAX_LINES,
    MemoryRecord,
    MemoryStore,
    build_memory_block,
    read_memory_file,
)
from duckduckcode.memory.worker import (
    ACTION_TOOL,
    build_extraction_input,
    read_session_slice,
    request_actions,
)
from duckduckcode.tools.tool import ToolCall


def action(
    operation: str = "create",
    memory_id: str | None = None,
    category: str = "project",
    scope: str = "project",
    summary: str = "Uses Python 3.12",
    tags: list[str] | None = None,
    body: str = "The project requires Python 3.12.",
) -> dict[str, object]:
    return {
        "operation": operation,
        "id": memory_id,
        "category": category,
        "scope": scope,
        "summary": summary,
        "tags": tags or ["python"],
        "body": body,
    }


class LongTermStorageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_tmp = tempfile.TemporaryDirectory()
        self.home_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace_tmp.cleanup)
        self.addCleanup(self.home_tmp.cleanup)
        self.workspace = Path(self.workspace_tmp.name)
        self.manager = MemoryManager(self.workspace, home=self.home_tmp.name)
        self.now = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)

    def test_crud_permissions_unicode_and_atomic_index(self) -> None:
        self.manager.apply_actions(
            [
                action(),
                action(
                    category="preference",
                    scope="user",
                    summary="回复可以简洁 🦆",
                    body="用户偏好简洁的中文回复。",
                ),
            ],
            "session-1",
            now=self.now,
        )
        inventory = self.manager.inventory()
        project = next(item for item in inventory if item["scope"] == "project")
        user = next(item for item in inventory if item["scope"] == "user")

        self.assertEqual(stat.S_IMODE(self.manager.user.root.stat().st_mode), 0o700)
        path = self.manager.user.root / "preference" / f"{user['id']}.md"
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertIn("回复可以简洁 🦆", self.manager.user.index_path.read_text())
        record = read_memory_file(path, "user")
        self.assertEqual(record.source_session, "session-1")
        self.assertEqual(record.created_at, "2026-08-03T12:00:00Z")

        self.manager.apply_actions(
            [
                action(
                    "update",
                    project["id"],
                    summary="Uses Python 3.12 and uv",
                    body="The project requires Python 3.12 and uses uv.",
                ),
                action(
                    "delete",
                    user["id"],
                    category="preference",
                    scope="user",
                    summary="",
                    tags=[],
                    body="",
                ),
            ],
            "session-2",
            now=self.now,
        )

        updated = self.manager.project.load()[0][project["id"]]
        self.assertEqual(updated.created_at, "2026-08-03T12:00:00Z")
        self.assertEqual(updated.source_session, "session-2")
        self.assertEqual(self.manager.user.load()[0], {})

    def test_rejects_invalid_actions_secrets_duplicates_and_symlinks(self) -> None:
        cases = [
            action(category="feedback", scope="project"),
            action(body="api_key = abcdefghijklmnop"),
            {**action(), "extra": True},
        ]
        for invalid in cases:
            with self.subTest(invalid=invalid), self.assertRaises(MemoryError):
                self.manager.apply_actions([invalid], "session", now=self.now)
        self.assertEqual(self.manager.inventory(), [])

        outside = self.workspace / "outside.md"
        outside.write_text("outside", encoding="utf-8")
        link = self.manager.project.root / "project"
        link.symlink_to(self.workspace)
        self.manager.project.index_path.write_text(
            "# DuckDuckCode Memory\n\n"
            "- [project/aaaaaaaa.md](project/aaaaaaaa.md): outside\n",
            encoding="utf-8",
        )
        with self.assertRaises(MemoryError):
            self.manager.project.load()

    def test_invalid_refresh_keeps_old_snapshot_and_state_warns_once(self) -> None:
        self.manager.apply_actions([action()], "session", now=self.now)
        snapshot, warning = self.manager.refresh()
        self.assertIsNone(warning)
        self.manager.project.index_path.write_text("broken\n", encoding="utf-8")

        stale, warning = self.manager.refresh()

        self.assertEqual(stale, snapshot)
        self.assertIn("refresh failed", warning or "")
        self.manager.project.index_path.write_text(
            "# DuckDuckCode Memory\n", encoding="utf-8"
        )
        for path in (self.manager.project.root / "project").glob("*.md"):
            path.unlink()
        self.manager.write_state("model returned no tool call")
        _, first = self.manager.refresh()
        _, second = self.manager.refresh()
        self.assertIn("background task failed", first or "")
        self.assertIsNone(second)


class LongTermInjectionTest(unittest.TestCase):
    def test_project_first_budget_with_user_first_render_and_utf8_limits(self) -> None:
        user = "# DuckDuckCode Memory\n\n" + "\n".join(
            f"user-{number}-你好" for number in range(180)
        )
        project = "# DuckDuckCode Memory\n\n" + "\n".join(
            f"project-{number}-🦆" for number in range(180)
        )

        block, truncated = build_memory_block(user, project)

        self.assertTrue(truncated)
        self.assertLessEqual(len(block.splitlines()), MEMORY_MAX_LINES)
        self.assertLessEqual(len(block.encode("utf-8")), MEMORY_MAX_BYTES)
        self.assertIn("project-179-🦆", block)
        self.assertNotIn("user-179-你好", block)
        self.assertLess(block.index("<user_memory>"), block.index("<project_memory>"))
        self.assertIn("WARNING", block)
        block.encode("utf-8").decode("utf-8")

    def test_memory_follows_abstraction_and_survives_restore_and_compaction(
        self,
    ) -> None:
        context = ContextManager(
            system_prompt="system",
            abstraction="summary",
            long_term_memory="memory",
        )
        context.add_user("old " + "x" * 100_000)
        context.add_assistant("old answer")
        context.add_user("current")

        transcript, cutoff = context.compaction_input() or ("", 0)
        self.assertNotIn("memory", transcript)
        context.apply_compaction("new summary", cutoff)
        context.restore(context.messages(), context.abstraction)

        messages = context.model_messages()
        self.assertEqual(messages[0], Message("system", "system"))
        self.assertIn("new summary", messages[1].content)
        self.assertEqual(messages[2], Message("system", "memory"))


class MemoryWorkerTest(unittest.TestCase):
    def test_reads_only_a_strict_in_workspace_session_slice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            context = ContextManager(system_prompt="system")
            sessions = SessionManager(workspace, context)
            sessions.start()
            sessions.commit_message("user", "first")
            sessions.commit_message("assistant", "second")

            records = read_session_slice(workspace, sessions.current_path, 1, 2)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["context"]["content"], "second")
            outside = workspace / "outside.jsonl"
            outside.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(MemoryError):
                read_session_slice(workspace, outside, 0, 1)

    def test_strict_tool_protocol_accepts_one_call_and_rejects_other_shapes(
        self,
    ) -> None:
        expected = [action()]

        class Client:
            def stream(self, messages, tools=None, reasoning=None):
                self.messages = messages
                self.tools = tools
                yield ToolCallEvent(
                    ToolCall("call", ACTION_TOOL["name"], {"actions": expected})
                )
                yield DoneEvent()

        client = Client()
        self.assertEqual(request_actions(client, "turn", ReasoningConfig()), expected)
        self.assertEqual(client.tools, [ACTION_TOOL])

        class InvalidClient:
            def stream(self, messages, tools=None, reasoning=None):
                yield ConversationEvent("prose")
                yield DoneEvent()

        with self.assertRaises(MemoryError):
            request_actions(InvalidClient(), "turn", ReasoningConfig())

    def test_extraction_input_prunes_old_results_before_final_reply(self) -> None:
        with (
            tempfile.TemporaryDirectory() as workspace,
            tempfile.TemporaryDirectory() as home,
        ):
            manager = MemoryManager(workspace, home=home)
            records = [
                {
                    "role": "user",
                    "context": {
                        "type": "message",
                        "content": "question",
                        "status": "completed",
                        "token_usage": 0,
                        "visible": True,
                    },
                    "ts": 1,
                },
                {
                    "role": "tool",
                    "context": {
                        "type": "tool_result",
                        "call_id": "call",
                        "name": "ReadFile",
                        "content": "x" * 200_000,
                        "is_error": False,
                    },
                    "ts": 2,
                },
                {
                    "role": "assistant",
                    "context": {
                        "type": "message",
                        "content": "final answer",
                        "status": "completed",
                        "token_usage": 0,
                        "visible": True,
                    },
                    "ts": 3,
                },
            ]

            payload = build_extraction_input(manager, records)

        self.assertLessEqual(len(payload.encode("utf-8")), 128 * 1024)
        self.assertIn("final answer", payload)
        self.assertLess(payload.count("x"), 2000)

    def test_agent_launches_once_only_after_completed_final_reply(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            context = ContextManager(system_prompt="system")
            sessions = SessionManager(workspace, context)
            memory = Mock()
            memory.refresh.return_value = ("memory", None)

            class Client:
                calls = 0

                def stream(self, messages, tools=None, reasoning=None):
                    self.calls += 1
                    if self.calls == 1:
                        yield ToolCallEvent(ToolCall("call", "missing", {}))
                    else:
                        yield ConversationEvent("done")
                    yield DoneEvent()

            events = list(
                Agent(
                    Client(),
                    context,
                    session_manager=sessions,
                    memory_manager=memory,
                ).stream("hello")
            )

        self.assertIn(LoopCompleteEvent("completed", 2), events)
        memory.spawn_worker.assert_called_once()
        arguments = memory.spawn_worker.call_args.args
        self.assertEqual(arguments[2:], (0, 5))

    def test_agent_does_not_launch_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = ContextManager(system_prompt="system")
            sessions = SessionManager(directory, context)
            memory = Mock()
            memory.refresh.return_value = ("", None)

            class Client:
                def stream(self, messages, tools=None, reasoning=None):
                    yield ErrorEvent("nope")

            events = list(
                Agent(
                    Client(),
                    context,
                    session_manager=sessions,
                    memory_manager=memory,
                ).stream("hello")
            )

        self.assertIn(LoopCompleteEvent("error", 1), events)
        memory.spawn_worker.assert_not_called()


class ConsolidationTest(unittest.TestCase):
    def test_activity_requires_five_distinct_valid_sessions_after_boundary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            sessions = workspace / ".duckduckcode" / "sessions"
            sessions.mkdir(parents=True)
            since = 100
            for number in range(5):
                record = {
                    "role": "user",
                    "context": {
                        "type": "message",
                        "content": "activity",
                        "status": "completed",
                        "token_usage": 0,
                        "visible": True,
                    },
                    "ts": since + 1,
                }
                sessions.joinpath(f"session-{number}.jsonl").write_text(
                    json.dumps(record) + "\n", encoding="utf-8"
                )
            sessions.joinpath("invalid.jsonl").write_text("broken\n")

            self.assertEqual(count_active_sessions(workspace, since), 5)
            self.assertEqual(count_active_sessions(workspace, since + 1), 0)

    def test_seven_day_boundary_and_failure_restore_lock_mtime(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as home,
        ):
            workspace = Path(directory)
            manager = MemoryManager(workspace, home=home)
            lock = manager.project.root / ".consolidate-lock"
            self.assertFalse(maybe_consolidate(Mock(), manager, now=1_000_000))
            exact = 2_000_000 - CONSOLIDATE_AFTER_SECONDS
            os.utime(lock, (exact, exact))
            with (
                patch(
                    "duckduckcode.memory.consolidate.count_active_sessions",
                    return_value=5,
                ),
                patch(
                    "duckduckcode.memory.consolidate._run_staged_consolidation"
                ) as run,
            ):
                self.assertFalse(maybe_consolidate(Mock(), manager, now=2_000_000))
                self.assertFalse(run.called)

            old = exact - 1
            os.utime(lock, (old, old))
            with (
                patch(
                    "duckduckcode.memory.consolidate.count_active_sessions",
                    return_value=5,
                ),
                patch(
                    "duckduckcode.memory.consolidate._run_staged_consolidation",
                    side_effect=RuntimeError("failed"),
                ),
                self.assertRaises(RuntimeError),
            ):
                maybe_consolidate(Mock(), manager, now=2_000_000)
            self.assertEqual(lock.stat().st_mtime_ns, int(old * 1_000_000_000))

    def test_restricted_bash_rejects_shell_network_and_outside_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            user = MemoryStore(root / "stage" / "user", "user")
            project = MemoryStore(root / "stage" / "project", "project")
            sessions = root / "sessions"
            user.ensure()
            project.ensure()
            sessions.mkdir()
            tools = _consolidation_tools(user, project, sessions)

            for command in (
                "cat /etc/passwd",
                f"cat {user.index_path} | grep x",
                "curl https://example.com",
                f"cat {user.index_path} > /tmp/out",
            ):
                with self.subTest(command=command):
                    result = tools.execute(
                        ToolCall("call", "Bash", {"command": command})
                    )
                    self.assertTrue(result.is_error)


if __name__ == "__main__":
    unittest.main()
