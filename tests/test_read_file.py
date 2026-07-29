from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from duckduckcode.tools.read_file import create_read_file_tool
from duckduckcode.tools.tool import ToolCall, ToolManager, ToolResult


class ReadFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.tool = create_read_file_tool(self.root)
        self.manager = ToolManager()
        self.manager.register(self.tool)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def execute(self, **arguments) -> ToolResult:
        return self.manager.execute(ToolCall("call_1", "ReadFile", arguments))

    def test_exposes_read_only_file_metadata_and_strict_schema(self) -> None:
        self.assertEqual(self.tool.name, "ReadFile")
        self.assertEqual(self.tool.category, "file")
        self.assertTrue(self.tool.is_read_only)
        self.assertFalse(self.tool.is_dangerous)
        self.assertTrue(self.tool.is_concurrency_safe)
        self.assertEqual(
            self.tool.params,
            {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path or path relative to the working directory.",
                    },
                    "offset": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": "1-based first line. Use null for 1.",
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "maximum": 2000,
                        "description": "Number of lines to read. Use null for 2000.",
                    },
                },
                "required": ["path", "offset", "limit"],
                "additionalProperties": False,
            },
        )

    def test_reads_relative_slice_with_line_numbers_and_next_offset(self) -> None:
        (self.root / "story.txt").write_text(
            "one\ntwo\nthree\nfour\n", encoding="utf-8"
        )

        result = self.execute(path="story.txt", offset=2, limit=2)

        self.assertEqual(
            result,
            ToolResult(
                "2: two\n"
                "3: three\n"
                "[More lines available. Continue with offset=4, limit=2.]"
            ),
        )

    def test_reads_absolute_path_and_applies_null_defaults(self) -> None:
        path = self.root / "notes.txt"
        path.write_text("first\n\nthird", encoding="utf-8")

        result = self.execute(path=str(path.resolve()), offset=None, limit=None)

        self.assertEqual(result, ToolResult("1: first\n2: \n3: third"))

    def test_default_limit_is_2000_lines(self) -> None:
        path = self.root / "large.txt"
        path.write_text(
            "".join(f"line {number}\n" for number in range(1, 2002)),
            encoding="utf-8",
        )

        result = self.execute(path="large.txt")

        self.assertFalse(result.is_error)
        self.assertEqual(len(result.content.splitlines()), 2001)
        self.assertTrue(
            result.content.endswith(
                "[More lines available. Continue with offset=2001, limit=2000.]"
            )
        )

    def test_empty_file_returns_an_explicit_result(self) -> None:
        (self.root / "empty.txt").touch()

        self.assertEqual(
            self.execute(path="empty.txt", offset=1, limit=10),
            ToolResult("(empty file)"),
        )

    def test_rejects_invalid_arguments_with_adjustment_hints(self) -> None:
        cases = [
            ({}, "'path' is required", "Provide"),
            ({"path": " "}, "'path' cannot be empty", "Provide"),
            ({"path": "x", "offset": 0}, "'offset' must be", "positive integer"),
            (
                {"path": "x", "limit": 2001},
                "'limit' must be between 1 and 2000",
                "multiple calls",
            ),
            (
                {"path": "x", "extra": True},
                "unsupported parameter",
                "Remove",
            ),
        ]

        for arguments, reason, adjustment in cases:
            with self.subTest(arguments=arguments):
                result = self.execute(**arguments)
                self.assertTrue(result.is_error)
                self.assertIn(reason, result.content)
                self.assertIn(adjustment, result.content)

    def test_reports_distinct_file_errors_with_adjustment_hints(self) -> None:
        (self.root / "folder").mkdir()
        (self.root / "binary.dat").write_bytes(b"text\x00binary")
        (self.root / "invalid.txt").write_bytes(b"\xfftext")
        (self.root / "short.txt").write_text("one\ntwo\n", encoding="utf-8")

        cases = [
            (
                {"path": "missing.txt"},
                "does not exist",
                "Check the path",
            ),
            (
                {"path": "folder"},
                "is a directory",
                "Provide a file path",
            ),
            (
                {"path": "binary.dat"},
                "binary",
                "xxd",
            ),
            (
                {"path": "invalid.txt"},
                "valid UTF-8",
                "iconv",
            ),
            (
                {"path": "short.txt", "offset": 3},
                "last line is 2",
                "offset between 1 and 2",
            ),
        ]

        for arguments, reason, adjustment in cases:
            with self.subTest(arguments=arguments):
                result = self.execute(**arguments)
                self.assertTrue(result.is_error)
                self.assertIn(reason, result.content)
                self.assertIn(adjustment, result.content)

    def test_reports_permission_and_other_operating_system_errors(self) -> None:
        with patch(
            "duckduckcode.tools.read_file.os.open",
            side_effect=PermissionError("permission denied"),
        ):
            denied = self.execute(path="secret.txt")
        with patch(
            "duckduckcode.tools.read_file.os.open",
            side_effect=OSError("device unavailable"),
        ):
            unavailable = self.execute(path="device.txt")

        self.assertTrue(denied.is_error)
        self.assertIn("Permission denied", denied.content)
        self.assertIn("permissions", denied.content)
        self.assertTrue(unavailable.is_error)
        self.assertIn("device unavailable", unavailable.content)
        self.assertIn("Try again", unavailable.content)

    @unittest.skipUnless(Path("/dev/null").exists(), "requires a character device")
    def test_rejects_non_regular_files(self) -> None:
        result = self.execute(path="/dev/null", offset=1, limit=10)

        self.assertTrue(result.is_error)
        self.assertIn("not a regular file", result.content)
        self.assertIn("Bash", result.content)

    def test_rejects_oversized_lines_and_output(self) -> None:
        (self.root / "long-line.txt").write_text("x" * 100_001, encoding="utf-8")
        (self.root / "large-output.txt").write_text(
            ("x" * 70_000 + "\n") * 3,
            encoding="utf-8",
        )

        long_line = self.execute(path="long-line.txt", offset=1, limit=1)
        large_output = self.execute(path="large-output.txt", offset=1, limit=3)

        self.assertTrue(long_line.is_error)
        self.assertIn("line 1 exceeds 100000 characters", long_line.content)
        self.assertIn("Bash", long_line.content)
        self.assertTrue(large_output.is_error)
        self.assertIn("output exceeds 200000 characters", large_output.content)
        self.assertIn("smaller limit", large_output.content)


if __name__ == "__main__":
    unittest.main()
