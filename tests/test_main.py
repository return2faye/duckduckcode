from __future__ import annotations

import io
import unittest

from duckduckcode.main import run_repl


class MainTest(unittest.TestCase):
    def test_repl_reuses_agent_for_multiple_turns(self) -> None:
        calls = []

        class FakeAgent:
            def ask(self, message):
                calls.append(message)
                return f"answer {len(calls)}"

        output = io.StringIO()

        run_repl(FakeAgent(), input_stream=io.StringIO("hello\ncontinue\nquit\n"), output_stream=output)

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
