from __future__ import annotations

from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from duckduckcode.tools.edit_file import create_edit_file_tool
from duckduckcode.tools.tool import ToolCall, ToolManager, ToolResult


class EditFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.tool = create_edit_file_tool(self.root)
        self.manager = ToolManager()
        self.manager.register(self.tool)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def execute(self, **arguments) -> ToolResult:
        return self.manager.execute(ToolCall("call_1", "EditFile", arguments))

    def test_exposes_file_metadata_and_strict_schema(self) -> None:
        self.assertEqual(self.tool.name, "EditFile")
        self.assertEqual(self.tool.category, "file")
        self.assertFalse(self.tool.is_read_only)
        self.assertFalse(self.tool.is_dangerous)
        self.assertFalse(self.tool.is_concurrency_safe)
        self.assertEqual(
            self.tool.params,
            {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path or path relative to the working directory.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to replace. It must occur exactly once in the file.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
                "additionalProperties": False,
            },
        )

    def test_replaces_unique_text_and_returns_numbered_context(self) -> None:
        destination = self.root / "example.txt"
        destination.write_text(
            "".join(f"line {number}\n" for number in range(1, 10)),
            encoding="utf-8",
        )
        destination.chmod(0o600)

        result = self.execute(
            path="example.txt",
            old_string="line 5\nline 6",
            new_string="changed 5\nchanged 6",
        )

        self.assertFalse(result.is_error)
        self.assertEqual(
            destination.read_text(encoding="utf-8"),
            "line 1\nline 2\nline 3\nline 4\nchanged 5\nchanged 6\n"
            "line 7\nline 8\nline 9\n",
        )
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
        self.assertIn("2: line 2", result.content)
        self.assertIn("5: changed 5", result.content)
        self.assertIn("6: changed 6", result.content)
        self.assertIn("9: line 9", result.content)
        self.assertNotIn("1: line 1", result.content)

    def test_rejects_zero_or_multiple_matches_without_editing(self) -> None:
        destination = self.root / "example.txt"
        original = "same\nother\nsame\n"
        destination.write_text(original, encoding="utf-8")

        missing = self.execute(
            path="example.txt", old_string="missing", new_string="new"
        )
        repeated = self.execute(path="example.txt", old_string="same", new_string="new")

        self.assertTrue(missing.is_error)
        self.assertIn("not found", missing.content)
        self.assertTrue(repeated.is_error)
        self.assertIn("appears 2 times", repeated.content)
        self.assertIn("more context", repeated.content)
        self.assertEqual(destination.read_text(encoding="utf-8"), original)

    def test_rejects_invalid_arguments(self) -> None:
        cases = [
            ({}, "'path' is required"),
            ({"path": " "}, "'path' cannot be empty"),
            ({"path": "x", "old_string": ""}, "'old_string' cannot be empty"),
            (
                {"path": "x", "old_string": "a", "new_string": 1},
                "'new_string' must be a string",
            ),
            (
                {
                    "path": "x",
                    "old_string": "a",
                    "new_string": "b",
                    "extra": True,
                },
                "unsupported parameter",
            ),
        ]

        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                result = self.execute(**arguments)
                self.assertTrue(result.is_error)
                self.assertIn(message, result.content)

    def test_rejects_invalid_files_without_replacing_them(self) -> None:
        (self.root / "folder").mkdir()
        (self.root / "invalid.txt").write_bytes(b"\xffold")
        target = self.root / "target.txt"
        target.write_text("old", encoding="utf-8")
        link = self.root / "link.txt"
        link.symlink_to(target)

        cases = [
            ("missing.txt", "does not exist"),
            ("folder", "is a directory"),
            ("invalid.txt", "valid UTF-8"),
            ("link.txt", "symbolic link"),
        ]

        for path, message in cases:
            with self.subTest(path=path):
                result = self.execute(path=path, old_string="old", new_string="new")
                self.assertTrue(result.is_error)
                self.assertIn(message, result.content)
        self.assertEqual(target.read_text(encoding="utf-8"), "old")

    def test_failed_atomic_replace_preserves_original_and_cleans_up(self) -> None:
        destination = self.root / "example.txt"
        destination.write_text("old", encoding="utf-8")

        with patch(
            "duckduckcode.tools.edit_file.os.replace",
            side_effect=OSError("disk unavailable"),
        ):
            result = self.execute(
                path="example.txt", old_string="old", new_string="new"
            )

        self.assertTrue(result.is_error)
        self.assertIn("disk unavailable", result.content)
        self.assertEqual(destination.read_text(encoding="utf-8"), "old")
        self.assertEqual(list(self.root.iterdir()), [destination])


if __name__ == "__main__":
    unittest.main()
