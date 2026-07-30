from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from duckduckcode.tools.glob import create_glob_tool
from duckduckcode.tools.tool import ToolCall, ToolManager, ToolResult


class GlobTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.tool = create_glob_tool(self.root)
        self.manager = ToolManager()
        self.manager.register(self.tool)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def execute(self, **arguments) -> ToolResult:
        return self.manager.execute(ToolCall("call_1", "Glob", arguments))

    def test_exposes_read_only_search_metadata_and_strict_schema(self) -> None:
        self.assertEqual(self.tool.name, "Glob")
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
                        "description": "Glob pattern such as **/*.py.",
                    },
                    "path": {
                        "type": ["string", "null"],
                        "description": "Search root. Use null for the working directory.",
                    },
                },
                "required": ["pattern", "path"],
                "additionalProperties": False,
            },
        )

    def test_recurses_excludes_noisy_directories_and_sorts_newest_first(
        self,
    ) -> None:
        old = self.root / "src" / "old.py"
        new = self.root / "src" / "pkg" / "new.py"
        new.parent.mkdir(parents=True)
        old.write_text("", encoding="utf-8")
        new.write_text("", encoding="utf-8")
        os.utime(old, (100, 100))
        os.utime(new, (200, 200))

        for directory in (
            ".git",
            "node_modules",
            "vendor",
            ".idea",
            "__pycache__",
        ):
            ignored = self.root / directory / "ignored.py"
            ignored.parent.mkdir()
            ignored.write_text("", encoding="utf-8")
            os.utime(ignored, (300, 300))

        result = self.execute(pattern="**/*.py")

        self.assertEqual(result, ToolResult("src/pkg/new.py\nsrc/old.py"))

    def test_optional_search_root_keeps_paths_usable_from_working_directory(
        self,
    ) -> None:
        destination = self.root / "packages" / "app.py"
        destination.parent.mkdir()
        destination.write_text("", encoding="utf-8")

        result = self.execute(pattern="*.py", path="packages")

        self.assertEqual(result, ToolResult("packages/app.py"))

    def test_limits_results_to_200(self) -> None:
        for number in range(205):
            destination = self.root / f"{number:03}.txt"
            destination.write_text("", encoding="utf-8")
            os.utime(destination, (number, number))

        result = self.execute(pattern="*.txt", path=None)
        paths = result.content.splitlines()

        self.assertEqual(len(paths), 200)
        self.assertEqual(paths[0], "204.txt")
        self.assertEqual(paths[-1], "005.txt")

    def test_returns_an_explicit_empty_result(self) -> None:
        self.assertEqual(
            self.execute(pattern="**/*.py", path=None),
            ToolResult("(no matches)"),
        )

    def test_rejects_invalid_arguments_and_search_roots(self) -> None:
        file_path = self.root / "file.txt"
        file_path.write_text("", encoding="utf-8")
        cases = [
            ({}, "'pattern' is required"),
            ({"pattern": ""}, "'pattern' cannot be empty"),
            ({"pattern": 1}, "'pattern' must be a string"),
            ({"pattern": "*.py", "path": ""}, "'path' cannot be empty"),
            (
                {"pattern": "*.py", "path": None, "extra": True},
                "unsupported parameter",
            ),
            ({"pattern": "*.py", "path": "missing"}, "does not exist"),
            ({"pattern": "*.py", "path": "file.txt"}, "is not a directory"),
        ]

        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                result = self.execute(**arguments)
                self.assertTrue(result.is_error)
                self.assertIn(message, result.content)


if __name__ == "__main__":
    unittest.main()
