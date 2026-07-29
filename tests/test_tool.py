from __future__ import annotations

import unittest

from duckduckcode.tools.tool import ToolCall, ToolManager, ToolResult, create_tool


def raise_error(message: str) -> None:
    raise ValueError(message)


class ToolTest(unittest.TestCase):
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

    def test_result_serializes_for_the_model(self) -> None:
        self.assertEqual(
            ToolResult("failed", is_error=True).to_model_output(),
            '{"content": "failed", "isError": true}',
        )


if __name__ == "__main__":
    unittest.main()
