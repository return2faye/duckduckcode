from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from duckduckcode.config import Config
from duckduckcode.core.event import (
    ConversationEvent,
    DoneEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from duckduckcode.main import build_agent, main
from duckduckcode.tools.tool import ToolCall


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

    def test_build_agent_registers_core_file_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with patch("duckduckcode.main.OpenAIClient", return_value=object()):
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
                ],
            )
            agent.enter_plan_mode()
            self.assertEqual(agent.context.mode, "plan")
            self.assertEqual(
                agent.plan_file,
                workspace.resolve() / ".duckduckcode" / "plan.md",
            )

    def test_build_agent_injects_workspace_into_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with patch("duckduckcode.main.OpenAIClient", return_value=object()):
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

            with patch("duckduckcode.main.OpenAIClient", return_value=object()):
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

            with patch("duckduckcode.main.OpenAIClient", return_value=object()):
                agent = build_agent(
                    Config("test-key"),
                    workspace,
                    include_user_instructions=False,
                )
            self.addCleanup(agent.close)

            self.assertNotIn("machine-specific", agent.context.system_prompt)
            self.assertIn("fixture-specific", agent.context.system_prompt)

    def test_build_agent_rejects_static_context_at_compaction_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            workspace.joinpath("DDCODE.md").write_text("x" * 6_000, encoding="utf-8")

            with (
                patch("duckduckcode.main.OpenAIClient", return_value=object()),
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
                patch("duckduckcode.main.OpenAIClient", return_value=object()),
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
                patch("duckduckcode.main.OpenAIClient", return_value=object()),
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
            with patch("duckduckcode.main.OpenAIClient", return_value=object()):
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
            with patch("duckduckcode.main.OpenAIClient", return_value=FakeClient()):
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
                patch("duckduckcode.main.OpenAIClient", return_value=object()),
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


if __name__ == "__main__":
    unittest.main()
