from __future__ import annotations

import io
import json
import unittest

from duckduckcode.backend import run_backend
from duckduckcode.event import ConversationEvent, DoneEvent


class BackendTest(unittest.TestCase):
    def test_backend_streams_json_lines_for_each_prompt(self) -> None:
        calls = []

        class FakeAgent:
            def stream(self, message):
                calls.append(message)
                yield ConversationEvent("hi")
                yield DoneEvent(token_usage=7)

        output = io.StringIO()

        run_backend(
            FakeAgent(),
            input_stream=io.StringIO('{"message": "hello"}\n'),
            output_stream=output,
        )

        self.assertEqual(calls, ["hello"])
        self.assertEqual(
            [json.loads(line) for line in output.getvalue().splitlines()],
            [{"type": "delta", "text": "hi"}, {"type": "done", "token_usage": 7}],
        )

    def test_backend_survives_interrupted_stream(self) -> None:
        class FakeAgent:
            def stream(self, message):
                if message == "stop":
                    yield ConversationEvent("partial")
                    raise KeyboardInterrupt
                yield ConversationEvent("next")
                yield DoneEvent()

        output = io.StringIO()

        run_backend(
            FakeAgent(),
            input_stream=io.StringIO('{"message": "stop"}\n{"message": "continue"}\n'),
            output_stream=output,
        )

        self.assertEqual(
            [json.loads(line) for line in output.getvalue().splitlines()],
            [
                {"type": "delta", "text": "partial"},
                {"type": "error", "message": "interrupted", "code": "interrupted"},
                {"type": "delta", "text": "next"},
                {"type": "done", "token_usage": 0},
            ],
        )


if __name__ == "__main__":
    unittest.main()
