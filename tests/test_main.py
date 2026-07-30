from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from duckduckcode.config import Config
from duckduckcode.core.event import (
    ConversationEvent,
    DoneEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from duckduckcode.main import build_agent, run_repl
from duckduckcode.tools.tool import ToolCall


class MainTest(unittest.TestCase):
    def test_build_agent_registers_core_file_tools(self) -> None:
        with patch("duckduckcode.main.OpenAIClient", return_value=object()):
            agent = build_agent(Config("test-key"), Path.cwd())

        self.assertEqual(
            [schema["name"] for schema in agent.tools.schemas()],
            ["ReadFile", "WriteFile", "EditFile", "Glob", "Grep", "Bash"],
        )

    def test_build_agent_injects_workspace_into_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with patch("duckduckcode.main.OpenAIClient", return_value=object()):
                agent = build_agent(Config("test-key"), workspace)

        self.assertIn(
            f"Working directory: {workspace.resolve()}", agent.context.system_prompt
        )
        self.assertIn("Model: o4-mini", agent.context.system_prompt)

    def test_build_agent_injects_one_workspace_into_all_file_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = workspace / "source.txt"
            source.write_text("old", encoding="utf-8")
            with patch("duckduckcode.main.OpenAIClient", return_value=object()):
                agent = build_agent(Config("test-key"), workspace)

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
                events = list(
                    build_agent(Config("test-key"), workspace).stream("remove build")
                )

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

    def test_repl_streams_agent_responses_for_multiple_turns(self) -> None:
        calls = []

        class FakeAgent:
            def stream(self, message):
                calls.append(message)
                yield ConversationEvent("answer ")
                yield ConversationEvent(str(len(calls)))
                yield DoneEvent()

        output = io.StringIO()

        run_repl(
            FakeAgent(),
            input_stream=io.StringIO("hello\ncontinue\nquit\n"),
            output_stream=output,
        )

        self.assertEqual(calls, ["hello", "continue"])
        self.assertEqual(
            output.getvalue(),
            "duckduckcode: 你好，我是 DuckDuckCode。输入 exit 或 quit 结束。\n"
            "you: duckduckcode: answer 1\n"
            "you: duckduckcode: answer 2\n"
            "you: ",
        )


if __name__ == "__main__":
    unittest.main()
