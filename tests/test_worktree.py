from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from duckduckcode.core.worktree import WorktreeManager


class WorktreeManagerTest(unittest.TestCase):
    def _git(self, workspace: Path, *arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(workspace), *arguments],
            capture_output=True,
            check=True,
            text=True,
        ).stdout

    def _repository(self, root: Path) -> Path:
        workspace = root / "repository"
        workspace.mkdir()
        self._git(workspace, "init", "-q")
        self._git(workspace, "config", "user.name", "Test")
        self._git(workspace, "config", "user.email", "test@example.com")
        workspace.joinpath(".gitignore").write_text(
            ".env\ncache/\n.duckduckcode/worktrees.json\n", encoding="utf-8"
        )
        workspace.joinpath("source.txt").write_text("base\n", encoding="utf-8")
        self._git(workspace, "add", ".")
        self._git(workspace, "commit", "-qm", "initial")
        return workspace

    def _configure_dependencies(self, home: Path) -> None:
        config = home / ".duckduckcode" / "worktree.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("copy:\n  - .env\nsymlinks:\n  - cache\n", encoding="utf-8")

    def test_reuses_stable_worktree_and_returns_cumulative_patch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._repository(root)
            home = root / "home"
            workspace.joinpath(".env").write_text("TOKEN=one\n", encoding="utf-8")
            workspace.joinpath("cache").mkdir()
            self._configure_dependencies(home)
            manager = WorktreeManager(workspace, home=home, clock=lambda: 100)

            first = manager.enter("chat-a", "fix-parser", "task-1")
            first_workspace = manager.workspace_path(first)
            self.assertEqual(first.worktree.id, "fix-parser-6b6e7f77")
            self.assertEqual(first.worktree.branch, "worktree-fix-parser-6b6e7f77")
            self.assertFalse(first_workspace.joinpath(".env").is_symlink())
            self.assertEqual(
                first_workspace.joinpath(".env").read_text(), "TOKEN=one\n"
            )
            self.assertTrue(first_workspace.joinpath("cache").is_symlink())
            with self.assertRaisesRegex(RuntimeError, "already active"):
                manager.enter("chat-a", "fix-parser", "task-2")
            with self.assertRaisesRegex(RuntimeError, "active"):
                manager.preflight_remove(first.worktree.id)
            with self.assertRaisesRegex(RuntimeError, "active"):
                manager.remove(first.worktree.id)

            first_workspace.joinpath("source.txt").write_text(
                "first\n", encoding="utf-8"
            )
            first_changes = manager.leave(first, partial=False)["changes"]
            self.assertIn("first", first_changes["patch"])
            self.assertNotIn(".env", first_changes["patch"])

            second = manager.enter("chat-a", "fix-parser", "task-2")
            self.assertIs(second, first)
            second_workspace = manager.workspace_path(second)
            second_workspace.joinpath("new.txt").write_text(
                "second\n", encoding="utf-8"
            )
            second_changes = manager.leave(second, partial=False)["changes"]

            self.assertEqual(second_changes["worktree_id"], first.worktree.id)
            self.assertIn("first", second_changes["patch"])
            self.assertIn("new.txt", second_changes["patch"])
            self.assertTrue(first.worktree.path.is_dir())
            self.assertFalse(manager.list()[0]["active"])

    def test_recovers_active_state_as_idle_without_resetting_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._repository(root)
            home = root / "home"
            manager = WorktreeManager(workspace, home=home)
            entered = manager.enter("chat", "recover", "old-task")
            manager.workspace_path(entered).joinpath("source.txt").write_text(
                "recovered\n", encoding="utf-8"
            )

            recovered = WorktreeManager(workspace, home=home)
            listing = recovered.list()[0]

            self.assertFalse(listing["active"])
            self.assertTrue(listing["recovered"])
            reused = recovered.enter("chat", "recover", "new-task")
            delivered = recovered.leave(reused, partial=False)
            self.assertIn("recovered", delivered["changes"]["patch"])

            state = json.loads(recovered.state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["worktrees"][0]["status"], "idle")
            self.assertIsNone(state["worktrees"][0]["active_task_id"])

    def test_remove_returns_final_patch_and_removes_checkout_branch_and_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._repository(root)
            manager = WorktreeManager(workspace, home=root / "home")
            entered = manager.enter("chat", "remove", "task")
            manager.workspace_path(entered).joinpath("new.bin").write_bytes(
                bytes(range(256))
            )
            manager.leave(entered, partial=False)

            preflight = manager.preflight_remove(entered.worktree.id)
            self.assertTrue(preflight["dirty"])
            changes = manager.remove(entered.worktree.id)

            self.assertIn("GIT binary patch", changes["patch"])
            self.assertFalse(entered.worktree.path.exists())
            self.assertNotIn(
                entered.worktree.branch, self._git(workspace, "branch", "--list")
            )
            self.assertEqual(manager.list(), ())
            self.assertEqual(
                json.loads(manager.state_path.read_text(encoding="utf-8"))["worktrees"],
                [],
            )

    def test_refreshes_copy_and_preserves_changed_injected_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._repository(root)
            home = root / "home"
            source = workspace / ".env"
            source.write_text("one\n", encoding="utf-8")
            workspace.joinpath("cache").mkdir()
            self._configure_dependencies(home)
            manager = WorktreeManager(workspace, home=home)
            entered = manager.enter("chat", "deps", "one")
            destination = manager.workspace_path(entered) / ".env"
            manager.leave(entered, partial=False)

            source.write_text("two\n", encoding="utf-8")
            refreshed = manager.enter("chat", "deps", "two")
            self.assertEqual(destination.read_text(), "two\n")
            manager.leave(refreshed, partial=False)
            destination.chmod(0o600)
            destination.write_text("local\n", encoding="utf-8")
            source.write_text("three\n", encoding="utf-8")

            conflicted = manager.enter("chat", "deps", "three")
            self.assertEqual(destination.read_text(), "local\n")
            self.assertTrue(
                any("not overwritten" in warning for warning in conflicted.warnings)
            )
            manager.leave(conflicted, partial=False)

    def test_sets_absolute_worktree_hooks_path_and_reports_parent_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._repository(root)
            hooks = workspace / "hooks"
            hooks.mkdir()
            self._git(workspace, "config", "core.hooksPath", "hooks")
            manager = WorktreeManager(workspace, home=root / "home")
            entered = manager.enter("chat", "hooks", "task")

            configured = self._git(
                entered.worktree.path,
                "config",
                "--worktree",
                "--path",
                "--get",
                "core.hooksPath",
            ).strip()
            self.assertEqual(configured, str(hooks.resolve()))
            manager.leave(entered, partial=False)
            workspace.joinpath("source.txt").write_text("parent\n", encoding="utf-8")
            self._git(workspace, "add", "source.txt")
            self._git(workspace, "commit", "-qm", "parent")
            self.assertTrue(manager.list()[0]["parent_changed"])

    def test_refuses_to_remove_a_branch_with_external_commits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._repository(root)
            manager = WorktreeManager(workspace, home=root / "home")
            entered = manager.enter("chat", "external", "task")
            manager.leave(entered, partial=False)
            checkout = entered.worktree.path
            self._git(checkout, "config", "user.name", "External")
            self._git(checkout, "config", "user.email", "external@example.com")
            checkout.joinpath("source.txt").write_text("commit\n", encoding="utf-8")
            self._git(checkout, "add", "source.txt")
            self._git(checkout, "commit", "-qm", "external")

            with self.assertRaisesRegex(RuntimeError, "external commits"):
                manager.preflight_remove(entered.worktree.id)

            self.assertTrue(checkout.is_dir())
            self.assertIn(
                entered.worktree.branch, self._git(workspace, "branch", "--list")
            )

    def test_does_not_adopt_unmanaged_branch_or_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._repository(root)
            manager = WorktreeManager(workspace, home=root / "home")
            digest = hashlib.sha256(b"chat").hexdigest()[:8]
            branch = f"worktree-collision-{digest}"
            self._git(workspace, "branch", branch, "HEAD")

            with self.assertRaisesRegex(RuntimeError, "unmanaged branch"):
                manager.enter("chat", "collision", "task")

            path_id = f"occupied-{digest}"
            occupied = manager.root / path_id
            occupied.mkdir(parents=True)
            marker = occupied / "keep.txt"
            marker.write_text("external\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unmanaged worktree path"):
                manager.enter("chat", "occupied", "task")
            self.assertEqual(marker.read_text(), "external\n")

    def test_does_not_follow_replaced_injection_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._repository(root)
            home = root / "home"
            workspace.joinpath("local").mkdir()
            workspace.joinpath("local", ".env").write_text("source\n", encoding="utf-8")
            with workspace.joinpath(".gitignore").open("a", encoding="utf-8") as stream:
                stream.write("local/\n")
            self._git(workspace, "add", ".gitignore")
            self._git(workspace, "commit", "-qm", "ignore local")
            config = home / ".duckduckcode" / "worktree.yaml"
            config.parent.mkdir(parents=True)
            config.write_text("copy:\n  - local/.env\n", encoding="utf-8")
            manager = WorktreeManager(workspace, home=home)
            entered = manager.enter("chat", "parents", "one")
            checkout = manager.workspace_path(entered)
            manager.leave(entered, partial=False)

            injected = checkout / "local" / ".env"
            injected.unlink()
            injected.parent.rmdir()
            outside = root / "outside"
            outside.mkdir()
            outside.joinpath(".env").write_text("outside\n", encoding="utf-8")
            os.symlink(outside, checkout / "local")

            reused = manager.enter("chat", "parents", "two")

            self.assertEqual(outside.joinpath(".env").read_text(), "outside\n")
            self.assertTrue(any("unsafe parent" in item for item in reused.warnings))
            manager.leave(reused, partial=False)

    def test_corrupt_or_symlinked_state_is_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._repository(root)
            state = workspace / ".duckduckcode" / "worktrees.json"
            state.parent.mkdir()
            state.write_text('{"version":1,"version":1}', encoding="utf-8")

            corrupt = WorktreeManager(workspace, home=root / "home")
            self.assertTrue(
                any("could not be recovered" in item for item in corrupt.warnings)
            )
            with self.assertRaisesRegex(RuntimeError, "could not be recovered"):
                corrupt.enter("chat", "safe", "task")

            state.unlink()
            target = root / "state.json"
            target.write_text("{}", encoding="utf-8")
            state.symlink_to(target)
            linked = WorktreeManager(workspace, home=root / "other-home")
            self.assertTrue(any("symbolic links" in item for item in linked.warnings))


if __name__ == "__main__":
    unittest.main()
