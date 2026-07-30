from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from duckduckcode.config import Config
from duckduckcode.core.event import ConversationEvent, DoneEvent
from duckduckcode.main import build_agent, run_repl


class MainTest(unittest.TestCase):
    def test_build_agent_registers_core_file_tools(self) -> None:
        with patch("duckduckcode.main.OpenAIClient", return_value=object()):
            agent = build_agent(Config("test-key"))

        self.assertEqual(
            [schema["name"] for schema in agent.tools.schemas()],
            ["ReadFile", "WriteFile", "EditFile"],
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
