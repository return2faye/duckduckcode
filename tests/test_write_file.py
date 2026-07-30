from __future__ import annotations

from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from duckduckcode.tools.write_file import create_write_file_tool
from duckduckcode.tools.tool import ToolCall, ToolManager, ToolResult


class WriteFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.tool = create_write_file_tool(self.root)
        self.manager = ToolManager()
        self.manager.register(self.tool)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def execute(self, **arguments) -> ToolResult:
        return self.manager.execute(ToolCall("call_1", "WriteFile", arguments))

    def test_exposes_file_metadata_and_strict_schema(self) -> None:
        self.assertEqual(self.tool.name, "WriteFile")
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
                        "description": "Absolute file path. Relative paths are accepted only for compatibility.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete UTF-8 file content to write.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        )

    def test_creates_parent_directories_and_reports_utf8_bytes(self) -> None:
        result = self.execute(path="one/two/note.txt", content="duck\n你好")
        destination = self.root / "one" / "two" / "note.txt"

        self.assertEqual(
            result,
            ToolResult(f"Successfully wrote 11 bytes to '{destination.resolve()}'."),
        )
        self.assertEqual(destination.read_text(encoding="utf-8"), "duck\n你好")
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o644)
        self.assertEqual(stat.S_IMODE(destination.parent.stat().st_mode), 0o755)
        self.assertEqual(stat.S_IMODE(destination.parent.parent.stat().st_mode), 0o755)

    def test_replaces_the_complete_file_and_sets_mode(self) -> None:
        destination = self.root / "existing.txt"
        destination.write_text("old content that must disappear", encoding="utf-8")
        destination.chmod(0o600)

        result = self.execute(path="existing.txt", content="new")

        self.assertFalse(result.is_error)
        self.assertEqual(destination.read_text(encoding="utf-8"), "new")
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o644)

    def test_rejects_a_symbolic_link_without_changing_its_target(self) -> None:
        target = self.root / "target.txt"
        target.write_text("old", encoding="utf-8")
        link = self.root / "link.txt"
        link.symlink_to(target)

        result = self.execute(path="link.txt", content="new")

        self.assertTrue(result.is_error)
        self.assertIn("symbolic link", result.content)
        self.assertIn("target path", result.content)
        self.assertTrue(link.is_symlink())
        self.assertEqual(target.read_text(encoding="utf-8"), "old")

    def test_does_not_chmod_a_parent_created_by_another_process(self) -> None:
        raced_parent = (self.root / "raced").resolve()
        original_mkdir = Path.mkdir

        def race(directory: Path, *args, **kwargs) -> None:
            if directory == raced_parent:
                original_mkdir(directory, mode=0o700)
                raise FileExistsError("created concurrently")
            original_mkdir(directory, *args, **kwargs)

        with patch("duckduckcode.tools.write_file.Path.mkdir", new=race):
            result = self.execute(path="raced/note.txt", content="new")

        self.assertFalse(result.is_error)
        self.assertEqual(stat.S_IMODE(raced_parent.stat().st_mode), 0o700)

    def test_failed_replace_preserves_existing_file_and_removes_temporary_file(
        self,
    ) -> None:
        destination = self.root / "existing.txt"
        destination.write_text("old", encoding="utf-8")

        with patch(
            "duckduckcode.tools.write_file.os.replace",
            side_effect=OSError("disk unavailable"),
        ):
            result = self.execute(path="existing.txt", content="new")

        self.assertTrue(result.is_error)
        self.assertIn("disk unavailable", result.content)
        self.assertIn("Try again", result.content)
        self.assertEqual(destination.read_text(encoding="utf-8"), "old")
        self.assertEqual(list(self.root.iterdir()), [destination])

    def test_cleanup_failure_does_not_hide_the_write_error(self) -> None:
        with (
            patch(
                "duckduckcode.tools.write_file.os.replace",
                side_effect=OSError("disk unavailable"),
            ),
            patch(
                "duckduckcode.tools.write_file.Path.unlink",
                side_effect=PermissionError("cleanup denied"),
            ),
        ):
            result = self.execute(path="note.txt", content="new")

        self.assertTrue(result.is_error)
        self.assertIn("disk unavailable", result.content)
        self.assertIn("Try again", result.content)

    def test_rejects_content_that_cannot_be_encoded_as_utf8(self) -> None:
        result = self.execute(path="note.txt", content="\ud800")

        self.assertTrue(result.is_error)
        self.assertIn("valid UTF-8", result.content)
        self.assertIn("Unicode", result.content)

    def test_rejects_invalid_arguments_with_adjustment_hints(self) -> None:
        cases = [
            ({}, "'path' is required", "Provide"),
            ({"path": " "}, "'path' cannot be empty", "Provide"),
            ({"path": 1, "content": "x"}, "'path' must be a string", "filesystem"),
            ({"path": "x"}, "'content' is required", "complete file content"),
            (
                {"path": "x", "content": b"bytes"},
                "'content' must be a string",
                "UTF-8 text",
            ),
            (
                {"path": "x", "content": "", "append": True},
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

    def test_reports_directory_and_permission_errors(self) -> None:
        (self.root / "folder").mkdir()

        directory = self.execute(path="folder", content="content")
        with patch(
            "duckduckcode.tools.write_file.tempfile.mkstemp",
            side_effect=PermissionError("permission denied"),
        ):
            denied = self.execute(path="denied.txt", content="content")

        self.assertTrue(directory.is_error)
        self.assertIn("is a directory", directory.content)
        self.assertIn("file path", directory.content)
        self.assertTrue(denied.is_error)
        self.assertIn("Permission denied", denied.content)
        self.assertIn("permissions", denied.content)


if __name__ == "__main__":
    unittest.main()
