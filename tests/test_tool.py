from __future__ import annotations

import threading
import unittest

from duckduckcode.tools.tool import (
    ToolCall,
    ToolManager,
    ToolResult,
    create_exit_plan_mode_tool,
    create_tool,
)


def raise_error(message: str) -> None:
    raise ValueError(message)


class ToolTest(unittest.TestCase):
    def test_exit_plan_mode_tool_has_no_arguments_and_cannot_execute_directly(
        self,
    ) -> None:
        tool = create_exit_plan_mode_tool()

        self.assertEqual(tool.name, "ExitPlanMode")
        self.assertEqual(tool.params["properties"], {})
        self.assertTrue(tool.handler().is_error)

    def test_factory_exposes_metadata_and_model_schema(self) -> None:
        params = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        validator = lambda arguments: arguments
        handler = lambda path: path

        tool = create_tool(
            "read_file",
            "Read a file",
            params,
            handler,
            validator,
            is_read_only=True,
            is_dangerous=False,
            is_concurrency_safe=True,
            category="filesystem",
        )

        self.assertEqual(tool.params, params)
        self.assertIs(tool.validator, validator)
        self.assertTrue(tool.is_read_only)
        self.assertFalse(tool.is_dangerous)
        self.assertTrue(tool.is_concurrency_safe)
        self.assertEqual(tool.category, "filesystem")
        self.assertEqual(
            tool.schema(),
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a file",
                "parameters": params,
                "strict": True,
            },
        )

    def test_registry_replaces_duplicate_names_and_preserves_schema_order(self) -> None:
        manager = ToolManager()
        make = lambda name, description: create_tool(
            name,
            description,
            {"type": "object", "properties": {}},
            lambda: description,
            lambda arguments: arguments,
        )

        manager.register(make("first", "old"))
        manager.register(make("second", "second"))
        manager.register(make("first", "new"))

        self.assertEqual(
            [(schema["name"], schema["description"]) for schema in manager.schemas()],
            [("first", "new"), ("second", "second")],
        )

    def test_registry_preserves_keyword_registration(self) -> None:
        manager = ToolManager()

        manager.register(
            name="echo",
            description="Echo text",
            parameters={"type": "object", "properties": {}},
            handler=lambda text: text,
        )

        self.assertEqual(
            manager.execute(ToolCall("call_1", "echo", {"text": "hello"})),
            ToolResult("hello"),
        )

    def test_registry_rejects_overrides_for_complete_tools(self) -> None:
        manager = ToolManager()
        tool = create_tool(
            "echo",
            "Echo text",
            {"type": "object", "properties": {}},
            lambda: "hello",
            lambda arguments: arguments,
        )

        with self.assertRaisesRegex(TypeError, "registration arguments"):
            manager.register(tool, is_dangerous=True)

    def test_registry_uses_a_falsey_validator(self) -> None:
        class FalseyValidator:
            def __bool__(self) -> bool:
                return False

            def __call__(self, arguments):
                return {"text": arguments["text"].strip()}

        manager = ToolManager()
        manager.register(
            name="echo",
            description="Echo text",
            parameters={"type": "object", "properties": {}},
            handler=lambda text: text,
            validator=FalseyValidator(),
        )

        self.assertEqual(
            manager.execute(ToolCall("call_1", "echo", {"text": " hello "})),
            ToolResult("hello"),
        )

    def test_execute_validates_arguments_and_returns_structured_result(self) -> None:
        manager = ToolManager()
        manager.register(
            create_tool(
                "echo",
                "Echo text",
                {"type": "object", "properties": {}},
                lambda text: text,
                lambda arguments: {"text": arguments["text"].strip()},
            )
        )

        result = manager.execute(ToolCall("call_1", "echo", {"text": " normalized "}))

        self.assertEqual(result, ToolResult("normalized"))

    def test_execute_serializes_values_and_preserves_tool_results(self) -> None:
        manager = ToolManager()
        manager.register(
            create_tool(
                "json",
                "Return JSON",
                {"type": "object", "properties": {}},
                lambda: {"answer": 42},
                lambda arguments: arguments,
            )
        )
        manager.register(
            create_tool(
                "result",
                "Return a result",
                {"type": "object", "properties": {}},
                lambda: ToolResult("declined", is_error=True),
                lambda arguments: arguments,
            )
        )

        self.assertEqual(
            manager.execute(ToolCall("call_1", "json", {})),
            ToolResult('{"answer": 42}'),
        )
        self.assertEqual(
            manager.execute(ToolCall("call_2", "result", {})),
            ToolResult("declined", is_error=True),
        )

    def test_execute_converts_failures_to_error_results(self) -> None:
        manager = ToolManager()
        manager.register(
            create_tool(
                "invalid",
                "Reject input",
                {"type": "object", "properties": {}},
                lambda: "unreachable",
                lambda arguments: raise_error("invalid arguments"),
            )
        )
        manager.register(
            create_tool(
                "broken",
                "Fail execution",
                {"type": "object", "properties": {}},
                lambda: raise_error("execution failed"),
                lambda arguments: arguments,
            )
        )
        manager.register(
            create_tool(
                "unserializable",
                "Return an unsupported value",
                {"type": "object", "properties": {}},
                object,
                lambda arguments: arguments,
            )
        )
        manager.register(
            create_tool(
                "bad_validator",
                "Return invalid normalized arguments",
                {"type": "object", "properties": {}},
                lambda: "unreachable",
                lambda arguments: "not a dictionary",
            )
        )

        cases = [
            ("missing", "Unknown tool: missing"),
            ("invalid", "invalid arguments"),
            ("broken", "execution failed"),
            ("unserializable", "not JSON serializable"),
            ("bad_validator", "must return a dictionary"),
        ]
        for name, message in cases:
            with self.subTest(name=name):
                result = manager.execute(ToolCall("call_1", name, {}))
                self.assertTrue(result.is_error)
                self.assertIn(message, result.content)

    def test_execute_does_not_catch_process_interrupts(self) -> None:
        manager = ToolManager()
        manager.register(
            create_tool(
                "interrupt",
                "Interrupt execution",
                {"type": "object", "properties": {}},
                lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
                lambda arguments: arguments,
            )
        )

        with self.assertRaises(KeyboardInterrupt):
            manager.execute(ToolCall("call_1", "interrupt", {}))

    def test_execute_many_runs_safe_tools_concurrently_in_completion_order(
        self,
    ) -> None:
        manager = ToolManager()
        started = {name: threading.Event() for name in ("first", "second")}
        release = {name: threading.Event() for name in ("first", "second")}

        def handler(name):
            started[name].set()
            release[name].wait(5)
            return name

        for name in started:
            manager.register(
                name,
                name,
                {"type": "object", "properties": {}},
                lambda name=name: handler(name),
                is_concurrency_safe=True,
            )

        calls = [
            ToolCall("call_1", "first", {}),
            ToolCall("call_2", "second", {}),
        ]
        results = []
        result_ready = threading.Event()

        def consume():
            for result in manager.execute_many(calls):
                results.append(result)
                result_ready.set()

        worker = threading.Thread(target=consume)
        worker.start()
        try:
            self.assertTrue(started["first"].wait(1))
            self.assertTrue(started["second"].wait(1))
            release["second"].set()
            self.assertTrue(result_ready.wait(1))
            self.assertEqual(results[0], (calls[1], ToolResult("second")))
        finally:
            release["first"].set()
            release["second"].set()
            worker.join(1)

        self.assertFalse(worker.is_alive())
        self.assertEqual(
            results,
            [
                (calls[1], ToolResult("second")),
                (calls[0], ToolResult("first")),
            ],
        )

    def test_execute_many_treats_unsafe_tools_as_exclusive_barriers(self) -> None:
        manager = ToolManager()
        safe_started = threading.Event()
        release_safe = threading.Event()
        unsafe_started = threading.Event()
        release_unsafe = threading.Event()
        safe_after_started = threading.Event()

        def safe_before():
            safe_started.set()
            release_safe.wait(5)
            return "before"

        def unsafe():
            unsafe_started.set()
            release_unsafe.wait(5)
            return "unsafe"

        def safe_after():
            safe_after_started.set()
            return "after"

        manager.register(
            "safe_before",
            "safe",
            {"type": "object", "properties": {}},
            safe_before,
            is_concurrency_safe=True,
        )
        manager.register(
            "unsafe",
            "unsafe",
            {"type": "object", "properties": {}},
            unsafe,
        )
        manager.register(
            "safe_after",
            "safe",
            {"type": "object", "properties": {}},
            safe_after,
            is_concurrency_safe=True,
        )
        calls = [
            ToolCall("call_1", "safe_before", {}),
            ToolCall("call_2", "unsafe", {}),
            ToolCall("call_3", "safe_after", {}),
        ]
        results = []
        worker = threading.Thread(
            target=lambda: results.extend(manager.execute_many(calls))
        )
        worker.start()
        try:
            self.assertTrue(safe_started.wait(1))
            self.assertFalse(unsafe_started.is_set())
            release_safe.set()
            self.assertTrue(unsafe_started.wait(1))
            self.assertFalse(safe_after_started.is_set())
            release_unsafe.set()
        finally:
            release_safe.set()
            release_unsafe.set()
            worker.join(1)

        self.assertFalse(worker.is_alive())
        self.assertTrue(safe_after_started.is_set())
        self.assertEqual(
            [call.call_id for call, _ in results],
            [
                "call_1",
                "call_2",
                "call_3",
            ],
        )

    def test_result_serializes_for_the_model(self) -> None:
        self.assertEqual(
            ToolResult("failed", is_error=True).to_model_output(),
            '{"content": "failed", "isError": true}',
        )


if __name__ == "__main__":
    unittest.main()
