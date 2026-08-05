from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from duckduckcode.config import Config, ModelConfig
from duckduckcode.core.context import Message
from duckduckcode.core.event import (
    ConversationEvent,
    DoneEvent,
    ErrorEvent,
    ToolCallEvent,
    ToolResultEvent,
    LoopCompleteEvent,
    UsageEvent,
)
from duckduckcode.main import build_agent, main, run_subagent_worker
from duckduckcode.tools.tool import QuerySource, ToolCall


class MainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)
        home_patch = patch(
            "duckduckcode.permissions.rule_policy.Path.home",
            return_value=Path(self.home.name),
        )
        home_patch.start()
        self.addCleanup(home_patch.stop)
        memory_home_patch = patch(
            "duckduckcode.memory.long_term.Path.home",
            return_value=Path(self.home.name),
        )
        memory_home_patch.start()
        self.addCleanup(memory_home_patch.stop)
        skill_home_patch = patch(
            "duckduckcode.core.skill.Path.home", return_value=Path(self.home.name)
        )
        skill_home_patch.start()
        self.addCleanup(skill_home_patch.stop)
        worker_patch = patch("duckduckcode.memory.long_term.MemoryManager.spawn_worker")
        self.worker = worker_patch.start()
        self.addCleanup(worker_patch.stop)

    def test_build_agent_registers_core_file_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with patch("duckduckcode.main.create_client", return_value=object()):
                agent = build_agent(Config("test-key"), workspace)
            self.addCleanup(agent.close)

            self.assertEqual(
                [schema["name"] for schema in agent.tools.schemas()],
                [
                    "ReadFile",
                    "WriteFile",
                    "EditFile",
                    "Glob",
                    "Grep",
                    "Bash",
                    "ExitPlanMode",
                    "LoadSkill",
                    "Agent",
                ],
            )
            self.assertIsNotNone(agent.session_manager)
            self.assertTrue(agent.session_manager.current_path.is_file())
            agent.enter_plan_mode()
            self.assertEqual(agent.context.mode, "plan")
            self.assertEqual(
                agent.plan_file,
                workspace.resolve() / ".duckduckcode" / "plan.md",
            )

    def test_subagent_worker_filters_definition_tools_and_emits_jsonl(self) -> None:
        calls = []

        class Context:
            system_prompt = "project startup"
            long_term_memory = ""

            def messages(self):
                return [Message("assistant", "worker answer")]

        class WorkerAgent:
            def __init__(self):
                self.context = Context()

            def set_permission_mode(self, mode):
                calls.append(("permission", mode))

            def stream(self, prompt):
                calls.append(("prompt", prompt))
                yield UsageEvent(3)
                yield LoopCompleteEvent("completed", 1)

            def close(self):
                calls.append(("closed", True))

        request = {
            "prompt": "inspect",
            "model": None,
            "mode": "definition",
            "definition": {
                "type": "explore",
                "body": "Read carefully.",
                "max_turns": 6,
                "disallowed_tools": ["Grep"],
            },
            "workspace": "/tmp",
            "permission_mode": "ask_for_approval",
            "isolation": True,
            "boilerplate": "No questions.",
        }
        output = io.StringIO()
        with patch(
            "duckduckcode.main.build_agent", return_value=WorkerAgent()
        ) as build:
            with patch(
                "duckduckcode.main.sys.stdin", io.StringIO(json.dumps(request) + "\n")
            ):
                with patch("duckduckcode.main.sys.stdout", output):
                    run_subagent_worker(Config("test-key"))

        self.assertEqual(build.call_args.kwargs["allowed_tools"], {"ReadFile", "Glob"})
        self.assertEqual(build.call_args.kwargs["max_iterations"], 6)
        self.assertEqual(build.call_args.kwargs["query_source"], QuerySource.SUBAGENT)
        self.assertEqual(build.call_args.kwargs["model_role"], "subagent")
        self.assertIsNone(build.call_args.kwargs["model_override"])
        self.assertFalse(build.call_args.kwargs["enable_sessions"])
        self.assertIn(("permission", "ask_for_approval"), calls)
        self.assertIn(("prompt", "inspect"), calls)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(events[0], {"type": "usage", "total_tokens": 3})
        self.assertEqual(events[-1]["status"], "completed")
        self.assertEqual(events[-1]["result"], "worker answer")

    def test_filtered_worker_accepts_permission_rules_for_unregistered_tools(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            permissions = Path(self.home.name) / ".duckduckcode" / "permissions.yaml"
            permissions.parent.mkdir(parents=True, exist_ok=True)
            permissions.write_text("Bash: []\n", encoding="utf-8")

            with patch("duckduckcode.main.create_client", return_value=object()):
                agent = build_agent(
                    Config("test-key"),
                    workspace,
                    enable_sessions=False,
                    enable_memory=False,
                    enable_skills=False,
                    enable_exit_plan_mode=False,
                    enable_subagents=False,
                    allowed_tools={"ReadFile", "Glob"},
                    query_source=QuerySource.SUBAGENT,
                )
            self.addCleanup(agent.close)

            self.assertEqual(
                [schema["name"] for schema in agent.tools.schemas()],
                ["ReadFile", "Glob"],
            )

    def test_fork_factory_builds_an_isolated_non_recursive_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with patch(
                "duckduckcode.main.create_client",
                side_effect=lambda *_args, **_kwargs: object(),
            ):
                parent = build_agent(Config("test-key"), workspace)
                child = parent._fork_agent_factory()
            self.addCleanup(parent.close)
            self.addCleanup(child.close)

            self.assertIsNot(parent.client, child.client)
            self.assertIsNone(child.session_manager)
            self.assertIsNone(child.memory_manager)
            self.assertIsNone(child.skill_manager)
            self.assertNotIn(
                "ExitPlanMode", [schema["name"] for schema in child.tools.schemas()]
            )
            self.assertNotIn(
                "LoadSkill", [schema["name"] for schema in child.tools.schemas()]
            )

    def test_build_agent_injects_workspace_into_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with patch("duckduckcode.main.create_client", return_value=object()):
                agent = build_agent(Config("test-key"), workspace)
            self.addCleanup(agent.close)

            temporary_line = next(
                line
                for line in agent.context.system_prompt.splitlines()
                if line.startswith("- Temporary directory: ")
            )
            temporary_directory = Path(
                temporary_line.removeprefix("- Temporary directory: ")
            )

            self.assertIn(
                f"Working directory: {workspace.resolve()}", agent.context.system_prompt
            )
            self.assertIn("Model: o4-mini", agent.context.system_prompt)
            self.assertTrue(temporary_directory.is_dir())
            self.assertEqual(
                temporary_directory.stat().st_mode & 0o777,
                0o700,
            )

            agent.close()

            self.assertFalse(temporary_directory.exists())

    def test_build_agent_uses_the_configured_model_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config = Config(
                "openai-key",
                deepseek_api_key="deepseek-key",
                agent=ModelConfig("deepseek", "deepseek-model"),
            )
            with patch(
                "duckduckcode.main.create_client", return_value=object()
            ) as create:
                agent = build_agent(config, workspace)
            self.addCleanup(agent.close)

            create.assert_called_once_with(config, config.agent, model=None)
            self.assertIn("Model: deepseek-model", agent.context.system_prompt)

    def test_build_agent_loads_instruction_layers_and_static_tool_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            user = Path(self.home.name) / ".duckduckcode" / "DDCODE.md"
            nested = workspace / ".duckduckcode" / "DDCODE.md"
            user.parent.mkdir()
            nested.parent.mkdir()
            user.write_text("USER_LAYER_MARKER", encoding="utf-8")
            workspace.joinpath("DDCODE.md").write_text(
                "PROJECT_LAYER_MARKER", encoding="utf-8"
            )
            nested.write_text("NESTED_LAYER_MARKER", encoding="utf-8")
            workspace.joinpath("DDCODE.local.md").write_text(
                "LOCAL_LAYER_MARKER", encoding="utf-8"
            )

            with patch("duckduckcode.main.create_client", return_value=object()):
                agent = build_agent(Config("test-key"), workspace)
            self.addCleanup(agent.close)

            prompt = agent.context.system_prompt
            positions = [
                prompt.index(text)
                for text in (
                    "USER_LAYER_MARKER",
                    "PROJECT_LAYER_MARKER",
                    "NESTED_LAYER_MARKER",
                    "LOCAL_LAYER_MARKER",
                )
            ]
            self.assertEqual(positions, sorted(positions))
            self.assertLess(
                prompt.index("Environment:"),
                prompt.index("User and project instructions:"),
            )
            self.assertLess(
                prompt.index("User and project instructions:"),
                prompt.index("Mode instructions:"),
            )
            self.assertEqual(agent.context.tool_schemas(), agent.tools.schemas())

    def test_build_agent_can_exclude_user_instructions_for_evaluations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            user = Path(self.home.name) / ".duckduckcode" / "DDCODE.md"
            user.parent.mkdir()
            user.write_text("machine-specific", encoding="utf-8")
            workspace.joinpath("DDCODE.md").write_text(
                "fixture-specific", encoding="utf-8"
            )

            with patch("duckduckcode.main.create_client", return_value=object()):
                agent = build_agent(
                    Config("test-key"),
                    workspace,
                    include_user_instructions=False,
                )
            self.addCleanup(agent.close)

            self.assertNotIn("machine-specific", agent.context.system_prompt)
            self.assertIn("fixture-specific", agent.context.system_prompt)

    def test_build_agent_can_disable_all_memory_io_for_evaluations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with (
                patch("duckduckcode.main.MemoryManager") as memory,
                patch("duckduckcode.main.create_client", return_value=object()),
            ):
                agent = build_agent(
                    Config("test-key"),
                    workspace,
                    enable_sessions=False,
                    enable_memory=False,
                )
            self.addCleanup(agent.close)

            memory.assert_not_called()
            self.assertIsNone(agent.memory_manager)
            self.assertFalse((workspace / ".duckduckcode" / "memory").exists())

    def test_build_agent_rejects_static_context_at_compaction_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            workspace.joinpath("DDCODE.md").write_text("x" * 6_000, encoding="utf-8")

            with (
                patch("duckduckcode.main.create_client", return_value=object()),
                self.assertRaisesRegex(
                    RuntimeError,
                    r"estimated at \d+ tokens.*trigger of 5000 tokens",
                ),
            ):
                build_agent(
                    Config("test-key"),
                    workspace,
                    context_window_tokens=20_000,
                    compaction_trigger_tokens=5_000,
                    compaction_target_tokens=1_000,
                )

    def test_full_access_disables_the_os_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            sandbox = Mock()
            with (
                patch("duckduckcode.main.create_client", return_value=object()),
                patch("duckduckcode.main.OSSandbox", return_value=sandbox) as factory,
            ):
                agent = build_agent(Config("test-key"), workspace)
            self.addCleanup(agent.close)
            enabled = factory.call_args.args[2]

            self.assertTrue(enabled())
            agent.set_permission_mode("full_access")
            self.assertFalse(enabled())

    def test_build_agent_injects_one_workspace_into_all_file_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "source.txt"
            source.write_text("old", encoding="utf-8")
            with (
                patch("duckduckcode.main.create_client", return_value=object()),
                patch("duckduckcode.main.OSSandbox", return_value=None),
            ):
                agent = build_agent(Config("test-key"), workspace)
            self.addCleanup(agent.close)

            read = agent.tools.execute(
                ToolCall(
                    "read",
                    "ReadFile",
                    {"path": "source.txt", "offset": 1, "limit": 10},
                )
            )
            edit = agent.tools.execute(
                ToolCall(
                    "edit",
                    "EditFile",
                    {
                        "path": "source.txt",
                        "old_string": "old",
                        "new_string": "edited",
                    },
                )
            )
            write = agent.tools.execute(
                ToolCall(
                    "write",
                    "WriteFile",
                    {"path": "new.txt", "content": "new"},
                )
            )
            found = agent.tools.execute(
                ToolCall("glob", "Glob", {"pattern": "*.txt", "path": None})
            )
            searched = agent.tools.execute(
                ToolCall(
                    "grep",
                    "Grep",
                    {
                        "pattern": "edited",
                        "path": None,
                        "glob": "*.txt",
                        "context": 0,
                    },
                )
            )
            shell = agent.tools.execute(ToolCall("bash", "Bash", {"command": "pwd"}))

            self.assertFalse(read.is_error)
            self.assertFalse(edit.is_error)
            self.assertFalse(write.is_error)
            self.assertFalse(found.is_error)
            self.assertFalse(searched.is_error)
            self.assertFalse(shell.is_error)
            self.assertEqual(source.read_text(encoding="utf-8"), "edited")
            self.assertEqual((workspace / "new.txt").read_text(encoding="utf-8"), "new")
            self.assertEqual(set(found.content.splitlines()), {"new.txt", "source.txt"})
            self.assertEqual(searched.content, "source.txt:1:edited")
            self.assertEqual(
                json.loads(shell.content),
                {"output": f"{workspace.resolve()}\n", "exit_code": 0},
            )

    def test_build_agent_allows_searching_private_temporary_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with patch("duckduckcode.main.create_client", return_value=object()):
                agent = build_agent(Config("test-key"), workspace)
            self.addCleanup(agent.close)
            temporary_line = next(
                line
                for line in agent.context.system_prompt.splitlines()
                if line.startswith("- Temporary directory: ")
            )
            temporary_directory = Path(
                temporary_line.removeprefix("- Temporary directory: ")
            )
            source = temporary_directory / "cache" / "result.txt"
            source.parent.mkdir()
            source.write_text("temporary result\n", encoding="utf-8")

            glob_call = ToolCall(
                "glob",
                "Glob",
                {"pattern": "*.txt", "path": str(source.parent)},
            )
            grep_call = ToolCall(
                "grep",
                "Grep",
                {
                    "pattern": "temporary",
                    "path": str(source.parent),
                    "glob": "*.txt",
                    "context": 0,
                },
            )

            self.assertEqual(agent.permission_checker.check(glob_call).action, "ask")
            self.assertEqual(agent.permission_checker.check(grep_call).action, "ask")
            self.assertEqual(agent.tools.execute(glob_call).content, str(source))
            self.assertEqual(
                agent.tools.execute(grep_call).content,
                f"{source}:1:temporary result",
            )

    def test_build_agent_rejects_blacklisted_bash_commands(self) -> None:
        call = ToolCall("call_1", "Bash", {"command": "rm -rf build"})

        class FakeClient:
            calls = 0

            def stream(self, messages, tools=None, reasoning=None):
                self.calls += 1
                if self.calls == 1:
                    yield ToolCallEvent(call)
                else:
                    yield ConversationEvent("adjusted")
                yield DoneEvent()

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            marker = workspace / "build" / "marker.txt"
            marker.parent.mkdir()
            marker.write_text("keep", encoding="utf-8")
            with patch("duckduckcode.main.create_client", return_value=FakeClient()):
                agent = build_agent(Config("test-key"), workspace)
                events = list(agent.stream("remove build"))
                agent.close()

            self.assertTrue(marker.exists())
            self.assertIn(
                ToolResultEvent(
                    "call_1",
                    "Bash",
                    "Permission denied: Bash command matches blocked rule 'rm -rf'.",
                    is_error=True,
                ),
                events,
            )

    def test_build_agent_loads_project_permission_rules(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as home,
        ):
            workspace = Path(directory)
            permissions = workspace / ".duckduckcode" / "permissions.yaml"
            permissions.parent.mkdir()
            permissions.write_text(
                "Bash:\n" '  - content: "git push*"\n' "    action: ask\n",
                encoding="utf-8",
            )
            with (
                patch("duckduckcode.main.create_client", return_value=object()),
                patch(
                    "duckduckcode.permissions.rule_policy.Path.home",
                    return_value=Path(home),
                ),
            ):
                agent = build_agent(Config("test-key"), workspace)
            self.addCleanup(agent.close)

            decision = agent.permission_checker.check(
                ToolCall("push", "Bash", {"command": "git push origin main"})
            )

            self.assertEqual(decision.action, "ask")

    def test_main_launches_tui_without_building_an_unused_agent(self) -> None:
        config = Config("test-key")
        with (
            patch("sys.argv", ["duckduckcode"]),
            patch("duckduckcode.main.Config.from_env", return_value=config),
            patch("duckduckcode.main.Path.cwd", return_value=Path("/project")),
            patch(
                "duckduckcode.main.RulePolicy.read_permission_mode",
                return_value="accept_edits",
            ),
            patch("duckduckcode.main.build_agent") as build,
            patch("duckduckcode.main.run_tui") as run,
        ):
            main()

        build.assert_not_called()
        run.assert_called_once_with(
            "o4-mini",
            "/project",
            permission_mode="accept_edits",
        )

    def test_main_exits_cleanly_if_tui_receives_keyboard_interrupt(self) -> None:
        config = Config("test-key")
        with (
            patch("sys.argv", ["duckduckcode"]),
            patch("duckduckcode.main.Config.from_env", return_value=config),
            patch("duckduckcode.main.Path.cwd", return_value=Path("/project")),
            patch("duckduckcode.main.run_tui", side_effect=KeyboardInterrupt),
        ):
            main()

    def test_internal_backend_builds_and_closes_agent(self) -> None:
        config = Config("test-key")
        agent = Mock()
        with (
            patch("sys.argv", ["duckduckcode", "--backend"]),
            patch("duckduckcode.main.Config.from_env", return_value=config),
            patch("duckduckcode.main.Path.cwd", return_value=Path("/project")),
            patch("duckduckcode.main.build_agent", return_value=agent),
            patch("duckduckcode.main.run_backend") as run,
        ):
            main()

        run.assert_called_once_with(agent)
        agent.close.assert_called_once_with()

    def test_prompt_argument_runs_one_headless_turn(self) -> None:
        config = Config("test-key")
        agent = Mock()
        agent.stream.return_value = [ConversationEvent("hello back")]
        output = io.StringIO()
        with (
            patch("sys.argv", ["duckduckcode", "hello"]),
            patch("sys.stdout", output),
            patch("duckduckcode.main.Config.from_env", return_value=config),
            patch("duckduckcode.main.Path.cwd", return_value=Path("/project")),
            patch("duckduckcode.main.build_agent", return_value=agent),
            patch("duckduckcode.main.run_tui") as run,
        ):
            main()

        self.assertEqual(output.getvalue(), "hello back\n")
        agent.stream.assert_called_once_with("hello")
        agent.close.assert_called_once_with()
        run.assert_not_called()

    def test_prompt_argument_does_not_fail_on_memory_warning(self) -> None:
        config = Config("test-key")
        agent = Mock()
        agent.stream.return_value = [
            ErrorEvent("old memory snapshot retained", "memory"),
            ConversationEvent("hello back"),
        ]
        output = io.StringIO()
        with (
            patch("sys.argv", ["duckduckcode", "hello"]),
            patch("sys.stdout", output),
            patch("duckduckcode.main.Config.from_env", return_value=config),
            patch("duckduckcode.main.Path.cwd", return_value=Path("/project")),
            patch("duckduckcode.main.build_agent", return_value=agent),
        ):
            main()

        self.assertEqual(output.getvalue(), "hello back\n")


if __name__ == "__main__":
    unittest.main()
