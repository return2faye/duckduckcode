from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from duckduckcode.memory import load_instructions


class InstructionMemoryTest(unittest.TestCase):
    def test_loads_nonempty_layers_from_low_to_high_priority(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as home_directory,
        ):
            workspace = Path(directory)
            home = Path(home_directory)
            user = home / ".duckduckcode" / "DDCODE.md"
            project = workspace / "DDCODE.md"
            nested = workspace / ".duckduckcode" / "DDCODE.md"
            local = workspace / "DDCODE.local.md"
            user.parent.mkdir()
            nested.parent.mkdir()
            user.write_text("user", encoding="utf-8")
            project.write_text("project", encoding="utf-8")
            nested.write_text("nested", encoding="utf-8")
            local.write_text("local", encoding="utf-8")

            loaded = load_instructions(workspace, home)

        self.assertEqual(
            loaded,
            "user\n\n---\n\nproject\n\n---\n\nnested\n\n---\n\nlocal",
        )

    def test_skips_missing_and_empty_top_level_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            workspace.joinpath("DDCODE.md").write_text("", encoding="utf-8")
            workspace.joinpath("DDCODE.local.md").write_text("local", encoding="utf-8")

            loaded = load_instructions(workspace, include_user=False)

        self.assertEqual(loaded, "local")

    def test_expands_relative_references_in_place_and_only_once_per_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            parts = workspace / "parts"
            parts.mkdir()
            workspace.joinpath("shared.txt").write_text("shared", encoding="utf-8")
            parts.joinpath("rules.anything").write_text(
                "part\n@../shared.txt", encoding="utf-8"
            )
            workspace.joinpath("DDCODE.md").write_text(
                "before\n@parts/rules.anything\n@parts/rules.anything\nafter",
                encoding="utf-8",
            )

            loaded = load_instructions(workspace, include_user=False)

        self.assertEqual(loaded.count("part"), 1)
        self.assertEqual(loaded.count("shared"), 1)
        self.assertEqual(
            loaded,
            "before\n---\npart\n---\nshared\n---\n---\nafter",
        )

    def test_rejects_invalid_files_and_references_with_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "DDCODE.md"
            cases = (
                ("@missing.md", "does not exist"),
                ("@/absolute.md", "unsupported"),
                ("@https://example.com/rules", "unsupported"),
                ("@rules.md#section", "unsupported"),
                ("@rules/*.md", "unsupported"),
                ("@../outside.md", "escapes"),
            )
            for content, message in cases:
                root.write_text(f"first\n{content}", encoding="utf-8")
                with self.subTest(content=content):
                    with self.assertRaisesRegex(RuntimeError, message) as caught:
                        load_instructions(workspace, include_user=False)
                    error = str(caught.exception)
                    self.assertIn(f"source: {root.resolve()}", error)
                    self.assertIn("line: 2", error)
                    self.assertIn("chain:", error)

            root.write_bytes(b"\xff")
            with self.assertRaisesRegex(RuntimeError, "valid UTF-8"):
                load_instructions(workspace, include_user=False)

            root.write_text("unreadable", encoding="utf-8")
            with (
                patch.object(Path, "read_text", side_effect=PermissionError("denied")),
                self.assertRaisesRegex(RuntimeError, "cannot be read"),
            ):
                load_instructions(workspace, include_user=False)

            root.unlink()
            root.mkdir()
            with self.assertRaisesRegex(RuntimeError, "not a regular file"):
                load_instructions(workspace, include_user=False)

    def test_rejects_outside_symlink_cycle_and_sixth_nested_level(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as outside_directory,
        ):
            workspace = Path(directory)
            outside = Path(outside_directory) / "outside.md"
            outside.write_text("outside", encoding="utf-8")
            root = workspace / "DDCODE.md"
            root.symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "escapes"):
                load_instructions(workspace, include_user=False)

            root.unlink()
            workspace.joinpath("linked.md").symlink_to(outside)
            root.write_text("@linked.md", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "escapes"):
                load_instructions(workspace, include_user=False)

            root.write_text("@a.md", encoding="utf-8")
            workspace.joinpath("a.md").write_text("@b.md", encoding="utf-8")
            workspace.joinpath("b.md").write_text("@a.md", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "cycle") as caught:
                load_instructions(workspace, include_user=False)
            self.assertIn("a.md", str(caught.exception))
            self.assertIn("b.md", str(caught.exception))

            root.write_text("@level1", encoding="utf-8")
            for level in range(1, 7):
                workspace.joinpath(f"level{level}").write_text(
                    f"@level{level + 1}" if level < 6 else "too deep",
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(RuntimeError, "exceed 5 nested levels"):
                load_instructions(workspace, include_user=False)


if __name__ == "__main__":
    unittest.main()
