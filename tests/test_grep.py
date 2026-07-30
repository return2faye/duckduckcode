from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from duckduckcode.tools.grep import create_grep_tool
from duckduckcode.tools.tool import ToolCall, ToolManager, ToolResult


class GrepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.tool = create_grep_tool(self.root)
        self.manager = ToolManager()
        self.manager.register(self.tool)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def execute(self, **arguments) -> ToolResult:
        return self.manager.execute(ToolCall("call_1", "Grep", arguments))

    def test_exposes_read_only_search_metadata_and_strict_schema(self) -> None:
        self.assertEqual(self.tool.name, "Grep")
        self.assertEqual(self.tool.category, "search")
        self.assertTrue(self.tool.is_read_only)
        self.assertFalse(self.tool.is_dangerous)
        self.assertTrue(self.tool.is_concurrency_safe)
        self.assertEqual(
            self.tool.params,
            {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression to search for.",
                    },
                    "path": {
                        "type": ["string", "null"],
                        "description": "Absolute search root inside the working directory. Use null for the working directory; relative paths are accepted only for compatibility.",
                    },
                    "glob": {
                        "type": ["string", "null"],
                        "description": "Optional glob applied to relative file paths.",
                    },
                    "context": {
                        "type": ["integer", "null"],
                        "minimum": 0,
                        "description": "Context lines before and after each match. Use null for 0.",
                    },
                },
                "required": ["pattern", "path", "glob", "context"],
                "additionalProperties": False,
            },
        )

    def test_searches_utf8_text_and_filters_relative_paths(self) -> None:
        source = self.root / "src" / "pkg" / "app.py"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"prefix \xff\ndef target():\n")
        (self.root / "src" / "pkg" / "app.txt").write_text(
            "def target():\n", encoding="utf-8"
        )
        binary = self.root / "src" / "pkg" / "binary.py"
        binary.write_bytes(b"def target():\x00\n")

        for directory in (
            ".git",
            "node_modules",
            "vendor",
            ".idea",
            "__pycache__",
        ):
            ignored = self.root / directory / "ignored.py"
            ignored.parent.mkdir()
            ignored.write_text("def target():\n", encoding="utf-8")

        result = self.execute(
            pattern=r"def\s+target",
            path=None,
            glob="**/*.py",
            context=0,
        )

        self.assertEqual(result, ToolResult("src/pkg/app.py:2:def target():"))

    def test_glob_uses_standard_recursive_path_semantics(self) -> None:
        for relative_path in ("app.py", "src/app.py", "src/pkg/app.py"):
            destination = self.root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("hit\n", encoding="utf-8")

        cases = [
            ("*.py", "app.py:1:hit"),
            (
                "**/*.py",
                "app.py:1:hit\nsrc/app.py:1:hit\nsrc/pkg/app.py:1:hit",
            ),
            ("src/*.py", "src/app.py:1:hit"),
            (
                "src/**/*.py",
                "src/app.py:1:hit\nsrc/pkg/app.py:1:hit",
            ),
        ]

        for file_glob, expected in cases:
            with self.subTest(glob=file_glob):
                result = self.execute(
                    pattern="hit",
                    path=None,
                    glob=file_glob,
                    context=0,
                )
                self.assertEqual(result, ToolResult(expected))

    def test_merges_overlapping_context_and_marks_matching_lines(self) -> None:
        source = self.root / "example.txt"
        source.write_text(
            "before\nhit one\nbetween\nhit two\nafter\n",
            encoding="utf-8",
        )

        result = self.execute(
            pattern="hit",
            path=None,
            glob=None,
            context=1,
        )

        self.assertEqual(
            result,
            ToolResult(
                "example.txt-1-before\n"
                "example.txt:2:hit one\n"
                "example.txt-3-between\n"
                "example.txt:4:hit two\n"
                "example.txt-5-after"
            ),
        )

    def test_stops_after_100_matching_lines_and_reports_truncation(self) -> None:
        source = self.root / "matches.txt"
        source.write_text(
            "".join(f"match {number}\n" for number in range(101)),
            encoding="utf-8",
        )

        result = self.execute(
            pattern="match",
            path=None,
            glob=None,
            context=0,
        )

        lines = result.content.splitlines()
        self.assertEqual(len(lines), 101)
        self.assertEqual(lines[0], "matches.txt:1:match 0")
        self.assertEqual(lines[99], "matches.txt:100:match 99")
        self.assertEqual(lines[100], "[Results truncated after 100 matches.]")

    def test_restricts_search_roots_to_working_directory(self) -> None:
        file_path = self.root / "file.txt"
        file_path.write_text("", encoding="utf-8")
        with tempfile.TemporaryDirectory() as outside:
            cases = [
                ("missing", "does not exist"),
                ("file.txt", "is not a directory"),
                (outside, "must be inside the working directory"),
            ]

            for path, message in cases:
                with self.subTest(path=path):
                    result = self.execute(
                        pattern="text",
                        path=path,
                        glob=None,
                        context=0,
                    )
                    self.assertTrue(result.is_error)
                    self.assertIn(message, result.content)

    def test_rejects_invalid_arguments_and_regular_expressions(self) -> None:
        cases = [
            ({}, "'pattern' is required"),
            ({"pattern": 1}, "'pattern' must be a string"),
            ({"pattern": ""}, "'pattern' cannot be empty"),
            ({"pattern": "[", "context": 0}, "invalid regular expression"),
            ({"pattern": "x", "path": ""}, "'path' cannot be empty"),
            ({"pattern": "x", "glob": ""}, "'glob' cannot be empty"),
            ({"pattern": "x", "glob": "/tmp/*.py"}, "'glob' must be relative"),
            ({"pattern": "x", "context": -1}, "'context' must be"),
            ({"pattern": "x", "context": True}, "'context' must be"),
            ({"pattern": "x", "extra": True}, "unsupported parameter"),
        ]

        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                result = self.execute(**arguments)
                self.assertTrue(result.is_error)
                self.assertIn(message, result.content)

    def test_returns_an_explicit_empty_result(self) -> None:
        self.assertEqual(
            self.execute(
                pattern="missing",
                path=None,
                glob=None,
                context=None,
            ),
            ToolResult("(no matches)"),
        )


if __name__ == "__main__":
    unittest.main()
