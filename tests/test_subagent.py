from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import time
import unittest

from duckduckcode.core.context import ContextManager
from duckduckcode.core.agent import Agent
from duckduckcode.core.event import (
    ConversationEvent,
    DoneEvent,
    LoopCompleteEvent,
    SubagentEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from duckduckcode.permissions import PermissionChecker, PermissionDecision
from duckduckcode.core.subagent import (
    DefinitionManager,
    SubagentManager,
    _worker_request,
)
from duckduckcode.tools.tool import (
    QuerySource,
    ToolCall,
    ToolManager,
    ToolResult,
    create_agent_tool,
)


class DefinitionTest(unittest.TestCase):
    def test_discovers_builtins_and_project_overrides_user(self) -> None:
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            user = Path(home) / ".duckduckcode" / "agents"
            project = workspace / ".duckduckcode" / "agents"
            user.mkdir(parents=True)
            project.mkdir(parents=True)
            user.joinpath("custom.md").write_text(
                "---\ntype: custom\nwhenToUse: user\ndisallowedTools: []\nmaxTurns: 4\n---\nUser body\n",
                encoding="utf-8",
            )
            project.joinpath("custom.md").write_text(
                "---\ntype: custom\nwhenToUse: project\ndisallowedTools: [Grep]\nmaxTurns: 7\n---\nProject body\n",
                encoding="utf-8",
            )

            definitions, warning = DefinitionManager(
                workspace,
                home=home,
                known_tools={"ReadFile", "Glob", "Grep"},
            ).refresh()

            by_type = {definition.type: definition for definition in definitions}
            self.assertIsNone(warning)
            self.assertEqual({"explore", "plan", "custom"}, set(by_type))
            self.assertEqual(by_type["custom"].scope, "project")
            self.assertEqual(by_type["custom"].when_to_use, "project")
            self.assertEqual(by_type["custom"].max_turns, 7)
            self.assertEqual(by_type["custom"].disallowed_tools, ("Grep",))
            self.assertEqual(by_type["custom"].body, "Project body")

    def test_skips_duplicate_and_unsafe_or_invalid_definitions(self) -> None:
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as wd:
            root = Path(wd) / ".duckduckcode" / "agents"
            root.mkdir(parents=True)
            valid = "---\ntype: same\nwhenToUse: x\ndisallowedTools: []\nmaxTurns: 1\n---\nBody\n"
            root.joinpath("one.md").write_text(valid, encoding="utf-8")
            root.joinpath("two.md").write_text(valid, encoding="utf-8")
            root.joinpath("bad-name.md").write_text(
                "---\ntype: Bad\nwhenToUse: x\ndisallowedTools: []\nmaxTurns: 1\n---\nBody\n",
                encoding="utf-8",
            )
            root.joinpath("unknown.md").write_text(
                "---\ntype: unknown\nwhenToUse: x\ndisallowedTools: [Bash]\nmaxTurns: 1\n---\nBody\n",
                encoding="utf-8",
            )
            target = root / "target.txt"
            target.write_text(valid, encoding="utf-8")
            root.joinpath("linked.md").symlink_to(target)
            root.joinpath("large.md").write_bytes(b"x" * (256 * 1024 + 1))

            definitions, warning = DefinitionManager(
                Path(wd), home=home, known_tools={"ReadFile", "Glob", "Grep"}
            ).refresh()

            self.assertEqual([item.type for item in definitions], ["explore", "plan"])
            self.assertIn("duplicate definition type 'same'", warning or "")
            self.assertIn("lowercase kebab-case", warning or "")
            self.assertIn("unknown tool 'Bash'", warning or "")
            self.assertIn("must not be a symlink", warning or "")
            self.assertIn("exceeds", warning or "")


class AgentToolTest(unittest.TestCase):
    def test_schema_is_strict_nullable_and_has_dynamic_definition_enum(self) -> None:
        captured = []
        tool = create_agent_tool(
            ["explore", "plan", "custom"],
            lambda **arguments: captured.append(arguments),
        )

        schema = tool.schema()
        properties = schema["parameters"]["properties"]
        self.assertEqual(schema["name"], "Agent")
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["parameters"]["additionalProperties"])
        self.assertEqual(set(schema["parameters"]["required"]), set(properties))
        self.assertEqual(
            properties["subagent_type"],
            {"type": ["string", "null"], "enum": ["explore", "plan", "custom", None]},
        )
        self.assertEqual(properties["model"]["type"], ["string", "null"])
        self.assertEqual(properties["name"]["type"], ["string", "null"])

    def test_validator_rejects_unknown_type_and_empty_values_and_backgrounds_fork(
        self,
    ) -> None:
        captured = []
        tool = create_agent_tool(
            ["explore"], lambda **arguments: captured.append(arguments) or "ok"
        )
        base = {
            "prompt": "do it",
            "description": "work",
            "subagent_type": "explore",
            "model": None,
            "run_in_background": False,
            "name": None,
            "isolation": True,
        }
        manager = ToolManager()
        manager.register(tool)

        self.assertFalse(manager.execute(ToolCall("1", "Agent", base)).is_error)
        for key, value in (
            ("prompt", " "),
            ("description", ""),
            ("subagent_type", "missing"),
            ("model", " "),
            ("name", " "),
        ):
            invalid = dict(base)
            invalid[key] = value
            self.assertTrue(
                manager.execute(ToolCall(key, "Agent", invalid)).is_error, key
            )
        fork = dict(base, subagent_type=None, run_in_background=False)
        result = manager.execute(ToolCall("fork", "Agent", fork))
        self.assertFalse(result.is_error)
        self.assertTrue(captured[-1]["run_in_background"])

    def test_subagent_source_hard_rejects_recursive_agent_execution(self) -> None:
        manager = ToolManager(source=QuerySource.SUBAGENT)
        manager.register(create_agent_tool(["explore"], lambda **_: "ran"))
        result = manager.execute(
            ToolCall(
                "1",
                "Agent",
                {
                    "prompt": "do it",
                    "description": "work",
                    "subagent_type": "explore",
                    "model": None,
                    "run_in_background": True,
                    "name": None,
                    "isolation": True,
                },
            )
        )

        self.assertEqual(
            result,
            ToolResult("Subagents cannot invoke the Agent tool.", is_error=True),
        )


class SubagentManagerTest(unittest.TestCase):
    def _script(self, directory: Path, source: str) -> list[str]:
        path = directory / "worker.py"
        path.write_text(source, encoding="utf-8")
        return [sys.executable, str(path)]

    def _arguments(self, **overrides):
        arguments = {
            "prompt": "inspect",
            "description": "inspection",
            "subagent_type": "explore",
            "model": None,
            "run_in_background": False,
            "name": "reader",
            "isolation": False,
        }
        arguments.update(overrides)
        return arguments

    def test_foreground_forwards_nested_events_and_returns_final_result(self) -> None:
        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            command = self._script(
                workspace,
                "import json, sys\n"
                "request = json.loads(sys.stdin.readline())\n"
                "print(json.dumps({'type':'tool_use','call_id':'inner','name':'ReadFile','arguments':{'path':'x'}}), flush=True)\n"
                "print(json.dumps({'type':'usage','total_tokens':9}), flush=True)\n"
                "print(json.dumps({'type':'worker_result','status':'completed','result':'evidence'}), flush=True)\n",
            )
            manager = SubagentManager(workspace, worker_command=command)
            self.addCleanup(manager.close)

            stream = manager.run(
                "outer",
                self._arguments(),
                session_key="session-a",
                context=ContextManager(system_prompt="system"),
                permission_mode="ask_for_approval",
            )
            events = []
            try:
                while True:
                    events.append(next(stream))
            except StopIteration as done:
                result = done.value

            task_id = next(
                event.task_id
                for event in events
                if isinstance(event, SubagentEvent) and event.status == "started"
            )
            self.assertIn(SubagentEvent(task_id, "reader", "started", False), events)
            self.assertIn(
                ToolCallEvent(ToolCall("outer/inner", "ReadFile", {"path": "x"})),
                events,
            )
            self.assertIn(UsageEvent(9), events)
            self.assertIn(SubagentEvent(task_id, "reader", "completed", False), events)
            payload = json.loads(result.content)
            self.assertFalse(result.is_error)
            self.assertEqual(payload["result"], "evidence")
            self.assertEqual(payload["status"], "completed")
            self.assertFalse(payload["background"])

    def test_background_result_waits_for_its_session_inbox(self) -> None:
        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            command = self._script(
                workspace,
                "import json, sys, time\n"
                "json.loads(sys.stdin.readline())\n"
                "time.sleep(.05)\n"
                "print(json.dumps({'type':'usage','total_tokens':4}), flush=True)\n"
                "print(json.dumps({'type':'worker_result','status':'completed','result':'later'}), flush=True)\n",
            )
            manager = SubagentManager(workspace, worker_command=command)
            self.addCleanup(manager.close)
            stream = manager.run(
                "outer",
                self._arguments(run_in_background=True),
                session_key="session-a",
                context=ContextManager(system_prompt="system"),
                permission_mode="ask_for_approval",
            )
            events = []
            try:
                while True:
                    events.append(next(stream))
            except StopIteration as done:
                result = done.value

            payload = json.loads(result.content)
            self.assertEqual(payload["status"], "running")
            self.assertTrue(payload["background"])
            self.assertIn(
                SubagentEvent(payload["task_id"], "reader", "backgrounded", True),
                events,
            )
            deadline = time.monotonic() + 2
            while manager.running_count and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(manager.drain("session-b"), ([], []))
            inbox_events, messages = manager.drain("session-a")
            self.assertIn(UsageEvent(4), inbox_events)
            self.assertIn(
                SubagentEvent(payload["task_id"], "reader", "completed", True),
                inbox_events,
            )
            self.assertEqual(len(messages), 1)
            self.assertIn("later", messages[0])
            self.assertIn("untrusted", messages[0].lower())

    def test_isolation_discards_changes_and_cleans_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            workspace.joinpath("source.txt").write_text("original", encoding="utf-8")
            command = self._script(
                workspace,
                "import json, pathlib, sys\n"
                "request = json.loads(sys.stdin.readline())\n"
                "pathlib.Path('source.txt').write_text('changed')\n"
                "print(json.dumps({'type':'worker_result','status':'completed','result':str(pathlib.Path.cwd())}), flush=True)\n",
            )
            manager = SubagentManager(workspace, worker_command=command)
            self.addCleanup(manager.close)
            stream = manager.run(
                "outer",
                self._arguments(isolation=True),
                session_key="session-a",
                context=ContextManager(system_prompt="system"),
                permission_mode="full_access",
            )
            try:
                while True:
                    next(stream)
            except StopIteration as done:
                result = done.value

            snapshot = Path(json.loads(result.content)["result"])
            self.assertEqual(workspace.joinpath("source.txt").read_text(), "original")
            self.assertFalse(snapshot.exists())

    def test_nonisolated_fork_holds_single_writer_lease(self) -> None:
        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            command = self._script(
                workspace,
                "import json, sys, time\njson.loads(sys.stdin.readline())\ntime.sleep(2)\n",
            )
            manager = SubagentManager(workspace, worker_command=command)
            self.addCleanup(manager.close)
            first = manager.run(
                "one",
                self._arguments(
                    subagent_type=None, run_in_background=True, isolation=False
                ),
                session_key="s",
                context=ContextManager(system_prompt="system"),
                permission_mode="ask_for_approval",
            )
            try:
                while True:
                    next(first)
            except StopIteration:
                pass
            second = manager.run(
                "two",
                self._arguments(
                    subagent_type=None, run_in_background=True, isolation=False
                ),
                session_key="s",
                context=ContextManager(system_prompt="system"),
                permission_mode="ask_for_approval",
            )
            try:
                while True:
                    next(second)
            except StopIteration as done:
                result = done.value
            self.assertTrue(result.is_error)
            self.assertIn("write lease", result.content)
            self.assertTrue(manager.workspace_busy)

    def test_timeout_releases_lease_and_delivers_structured_failure(self) -> None:
        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            command = self._script(
                workspace,
                "import json, sys, time\njson.loads(sys.stdin.readline())\ntime.sleep(2)\n",
            )
            manager = SubagentManager(workspace, worker_command=command, timeout=0.05)
            self.addCleanup(manager.close)
            stream = manager.run(
                "one",
                self._arguments(
                    subagent_type=None, run_in_background=True, isolation=False
                ),
                session_key="s",
                context=ContextManager(system_prompt="system"),
                permission_mode="ask_for_approval",
            )
            try:
                while True:
                    next(stream)
            except StopIteration:
                pass
            deadline = time.monotonic() + 2
            while manager.running_count and time.monotonic() < deadline:
                time.sleep(0.01)

            events, messages = manager.drain("s")

            self.assertFalse(manager.workspace_busy)
            self.assertTrue(
                any(
                    isinstance(event, SubagentEvent) and event.status == "timed_out"
                    for event in events
                )
            )
            self.assertIn("timed out", messages[0])

    def test_concurrency_limit_rejects_an_extra_worker(self) -> None:
        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            command = self._script(
                workspace,
                "import json, sys, time\njson.loads(sys.stdin.readline())\ntime.sleep(2)\n",
            )
            manager = SubagentManager(workspace, worker_command=command, max_tasks=1)
            self.addCleanup(manager.close)
            first = manager.run(
                "one",
                self._arguments(run_in_background=True, isolation=True),
                session_key="s",
                context=ContextManager(system_prompt="system"),
                permission_mode="ask_for_approval",
            )
            try:
                while True:
                    next(first)
            except StopIteration:
                pass
            second = manager.run(
                "two",
                self._arguments(run_in_background=True, isolation=True),
                session_key="s",
                context=ContextManager(system_prompt="system"),
                permission_mode="ask_for_approval",
            )
            try:
                while True:
                    next(second)
            except StopIteration as done:
                result = done.value

            self.assertTrue(result.is_error)
            self.assertIn("at most 1", result.content)

    def test_foreground_task_can_detach_without_restarting_worker(self) -> None:
        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            command = self._script(
                workspace,
                "import json, sys, time\n"
                "json.loads(sys.stdin.readline())\n"
                "time.sleep(.1)\n"
                "print(json.dumps({'type':'worker_result','status':'completed','result':'same process'}), flush=True)\n",
            )
            manager = SubagentManager(workspace, worker_command=command)
            self.addCleanup(manager.close)
            stream = manager.run(
                "one",
                self._arguments(run_in_background=False),
                session_key="s",
                context=ContextManager(system_prompt="system"),
                permission_mode="ask_for_approval",
            )
            started = next(stream)
            self.assertEqual(started.status, "started")
            self.assertTrue(manager.detach_foreground())
            backgrounded = next(stream)
            self.assertEqual(backgrounded.status, "backgrounded")
            with self.assertRaises(StopIteration) as stopped:
                next(stream)
            self.assertEqual(
                json.loads(stopped.exception.value.content)["status"], "running"
            )
            deadline = time.monotonic() + 2
            while manager.running_count and time.monotonic() < deadline:
                time.sleep(0.01)
            events, messages = manager.drain("s")
            self.assertTrue(any(event.status == "completed" for event in events))
            self.assertIn("same process", messages[0])

    def test_definition_request_does_not_copy_parent_conversation(self) -> None:
        context = ContextManager(system_prompt="parent secret")
        context.add_user("private history")
        with tempfile.TemporaryDirectory() as wd:
            manager = DefinitionManager(Path(wd))
            definitions, _ = manager.refresh()
            definition = next(item for item in definitions if item.type == "explore")
            request = _worker_request(
                self._arguments(),
                context,
                "ask_for_approval",
                definition,
                Path(wd),
            )

        self.assertEqual(request["messages"], [])
        self.assertEqual(request["abstraction"], "")
        self.assertEqual(request["long_term_memory"], "")
        self.assertEqual(request["system_prompt"], "")


class AgentIntegrationTest(unittest.TestCase):
    def test_agent_tool_routes_through_subagent_manager(self) -> None:
        class Client:
            def __init__(self):
                self.calls = 0

            def stream(self, messages, tools=None, reasoning=None):
                self.calls += 1
                if self.calls == 1:
                    yield ToolCallEvent(
                        ToolCall(
                            "parent",
                            "Agent",
                            {
                                "prompt": "inspect",
                                "description": "inspection",
                                "subagent_type": "explore",
                                "model": None,
                                "run_in_background": False,
                                "name": None,
                                "isolation": True,
                            },
                        )
                    )
                    yield DoneEvent(2)
                    return
                yield ConversationEvent("parent answer")
                yield DoneEvent(1)

        class Manager:
            workspace_busy = False

            def __init__(self):
                self.arguments = None

            def run(self, call_id, arguments, **context):
                self.arguments = arguments
                yield SubagentEvent("task", "inspection", "started", False)
                return ToolResult("evidence")

            def drain(self, session_key):
                return [], []

            def close(self):
                pass

        definitions = DefinitionManager(Path.cwd())
        definitions.refresh()
        manager = Manager()
        tools = ToolManager()
        tools.register(create_agent_tool(["explore", "plan"], lambda **_: "unused"))
        agent = Agent(
            Client(),
            ContextManager(system_prompt="system"),
            tools,
            definition_manager=definitions,
            subagent_manager=manager,
        )

        events = list(agent.stream("hello"))

        self.assertIn(SubagentEvent("task", "inspection", "started", False), events)
        self.assertIn(ToolResultEvent("parent", "Agent", "evidence"), events)
        self.assertEqual(manager.arguments["prompt"], "inspect")
        self.assertEqual(events[-1], LoopCompleteEvent("completed", 2))

    def test_parent_write_tools_are_busy_while_fork_holds_lease(self) -> None:
        class Client:
            def stream(self, messages, tools=None, reasoning=None):
                yield ToolCallEvent(ToolCall("write", "WriteFile", {"path": "x"}))
                yield DoneEvent()

        class Manager:
            workspace_busy = True

            def drain(self, session_key):
                return [], []

            def close(self):
                pass

        tools = ToolManager()
        tools.register(
            "WriteFile",
            "write",
            {"type": "object", "properties": {}},
            lambda path: "written",
        )
        events = list(
            Agent(
                Client(),
                ContextManager(system_prompt="system"),
                tools,
                subagent_manager=Manager(),
            ).stream("hello")
        )

        result = next(event for event in events if isinstance(event, ToolResultEvent))
        self.assertTrue(result.is_error)
        self.assertIn("workspace is busy", result.content.lower())

    def test_subagent_permission_prompts_are_denied_without_an_event(self) -> None:
        class Client:
            def __init__(self):
                self.calls = 0

            def stream(self, messages, tools=None, reasoning=None):
                self.calls += 1
                if self.calls == 1:
                    yield ToolCallEvent(ToolCall("write", "WriteFile", {"path": "x"}))
                yield DoneEvent()

        class Policy:
            permission_mode = "ask_for_approval"

            def check(self, tool_call):
                return PermissionDecision("ask", "approval required", tool_call.name)

            def set_permission_mode(self, mode):
                self.permission_mode = mode

            def remember_allow(self, tool_call):
                raise AssertionError("must not remember subagent approval")

        tools = ToolManager(source=QuerySource.SUBAGENT)
        tools.register(
            "WriteFile",
            "write",
            {"type": "object", "properties": {}},
            lambda path: "written",
        )
        events = list(
            Agent(
                Client(),
                ContextManager(system_prompt="system"),
                tools,
                permission_checker=PermissionChecker(policy=Policy()),
                query_source=QuerySource.SUBAGENT,
            ).stream("hello")
        )

        self.assertFalse(
            any(
                event.__class__.__name__ == "PermissionRequestEvent" for event in events
            )
        )
        result = next(event for event in events if isinstance(event, ToolResultEvent))
        self.assertTrue(result.is_error)
        self.assertIn("cannot request user approval", result.content)


if __name__ == "__main__":
    unittest.main()
