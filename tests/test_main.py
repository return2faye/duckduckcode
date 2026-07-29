from __future__ import annotations

import io
import unittest

from duckduckcode.event import ConversationEvent, DoneEvent
from duckduckcode.main import run_repl


class MainTest(unittest.TestCase):
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
