from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from duckduckcode.permissions import RulePolicy
from duckduckcode.tools.tool import ToolCall

TOOLS = {"Bash", "ReadFile", "WriteFile", "EditFile", "Glob", "Grep"}


class RulePolicyTest(unittest.TestCase):
    def test_initializes_all_permission_files_without_overwriting_them(self) -> None:
        with (
            tempfile.TemporaryDirectory() as home_directory,
            tempfile.TemporaryDirectory() as workspace_directory,
            tempfile.TemporaryDirectory() as temporary_directory,
        ):
            home = Path(home_directory)
            workspace = Path(workspace_directory)
            expected = "deny: []\nask: []\nallow: []\n"

            RulePolicy.load(
                workspace,
                Path(temporary_directory),
                TOOLS,
                home=home,
            )

            paths = [
                home / ".duckduckcode" / "permissions.yaml",
                workspace / ".duckduckcode" / "permissions.yaml",
                workspace / ".duckduckcode" / "permissions.local.yaml",
            ]
            self.assertEqual(
                [path.read_text(encoding="utf-8") for path in paths],
                [expected, expected, expected],
            )

            paths[1].write_text("allow:\n  - Bash(git status)\n", encoding="utf-8")
            RulePolicy.load(
                workspace,
                Path(temporary_directory),
                TOOLS,
                home=home,
            )

            self.assertEqual(
                paths[1].read_text(encoding="utf-8"),
                "allow:\n  - Bash(git status)\n",
            )

    def test_defaults_unmatched_known_tools_to_ask(self) -> None:
        with (
            tempfile.TemporaryDirectory() as home_directory,
            tempfile.TemporaryDirectory() as workspace_directory,
            tempfile.TemporaryDirectory() as temporary_directory,
        ):
            policy = RulePolicy.load(
                Path(workspace_directory),
                Path(temporary_directory),
                TOOLS,
                home=Path(home_directory),
            )

            decision = policy.check(ToolCall("call", "Bash", {"command": "git status"}))

            self.assertEqual(decision.action, "ask")
            self.assertEqual(decision.content, "git status")

    def test_merges_rules_normalizes_paths_and_applies_decision_order(self) -> None:
        with (
            tempfile.TemporaryDirectory() as home_directory,
            tempfile.TemporaryDirectory() as workspace_directory,
            tempfile.TemporaryDirectory() as temporary_directory,
        ):
            home = Path(home_directory)
            workspace = Path(workspace_directory)
            private_temp = Path(temporary_directory)
            (home / ".duckduckcode").mkdir()
            (workspace / ".duckduckcode").mkdir()
            (workspace / "src" / "pkg").mkdir(parents=True)
            (workspace / "linked").symlink_to(
                workspace / "src", target_is_directory=True
            )
            (home / ".duckduckcode" / "permissions.yaml").write_text(
                """
deny:
  - "Bash(git push --force*)"
  - "ReadFile(**/.env)"
allow:
  - "Bash(git *)"
""",
                encoding="utf-8",
            )
            (workspace / ".duckduckcode" / "permissions.yaml").write_text(
                """
ask:
  - "Bash(git push *)"
  - "ReadFile(./src/*.py)"
allow:
  - "ReadFile(./src/**)"
  - "WriteFile(${TEMP}/cache.json)"
""",
                encoding="utf-8",
            )
            (workspace / ".duckduckcode" / "permissions.local.yaml").write_text(
                """
allow:
  - "Bash(git push origin main)"
""",
                encoding="utf-8",
            )

            policy = RulePolicy.load(workspace, private_temp, TOOLS, home=home)

            cases = [
                (
                    ToolCall(
                        "force",
                        "Bash",
                        {"command": "git push --force origin main"},
                    ),
                    "deny",
                ),
                (
                    ToolCall(
                        "approved",
                        "Bash",
                        {"command": "git push origin main"},
                    ),
                    "allow",
                ),
                (
                    ToolCall(
                        "ask",
                        "Bash",
                        {"command": "git push origin feature"},
                    ),
                    "ask",
                ),
                (
                    ToolCall("allowed", "Bash", {"command": "git status"}),
                    "allow",
                ),
                (
                    ToolCall(
                        "top-level",
                        "ReadFile",
                        {"path": str(workspace / "src" / "app.py")},
                    ),
                    "ask",
                ),
                (
                    ToolCall(
                        "recursive",
                        "ReadFile",
                        {"path": str(workspace / "src" / "pkg" / "app.py")},
                    ),
                    "allow",
                ),
                (
                    ToolCall(
                        "symlink",
                        "ReadFile",
                        {"path": str(workspace / "linked" / "pkg" / "app.py")},
                    ),
                    "allow",
                ),
                (
                    ToolCall(
                        "secret",
                        "ReadFile",
                        {"path": str(workspace / ".env")},
                    ),
                    "deny",
                ),
                (
                    ToolCall(
                        "temporary",
                        "WriteFile",
                        {"path": str(private_temp / "cache.json")},
                    ),
                    "allow",
                ),
                (
                    ToolCall("unset", "Bash", {"command": "uv run tests"}),
                    "ask",
                ),
            ]
            for call, action in cases:
                with self.subTest(call=call):
                    self.assertEqual(policy.check(call).action, action)

    def test_local_wildcard_allow_does_not_override_ask(self) -> None:
        with (
            tempfile.TemporaryDirectory() as home_directory,
            tempfile.TemporaryDirectory() as workspace_directory,
            tempfile.TemporaryDirectory() as temporary_directory,
        ):
            home = Path(home_directory)
            workspace = Path(workspace_directory)
            (workspace / ".duckduckcode").mkdir()
            (workspace / ".duckduckcode" / "permissions.yaml").write_text(
                'ask:\n  - "Bash(git push *)"\n',
                encoding="utf-8",
            )
            (workspace / ".duckduckcode" / "permissions.local.yaml").write_text(
                'allow:\n  - "Bash(git push origin *)"\n',
                encoding="utf-8",
            )

            policy = RulePolicy.load(
                workspace,
                Path(temporary_directory),
                TOOLS,
                home=home,
            )

            self.assertEqual(
                policy.check(
                    ToolCall(
                        "call",
                        "Bash",
                        {"command": "git push origin main"},
                    )
                ).action,
                "ask",
            )

    def test_fails_closed_for_invalid_permission_files(self) -> None:
        cases = [
            ("deny: [", "YAML"),
            ("unknown:\n  - Bash(*)\n", "unknown field"),
            ("allow: Bash(*)\n", "must be a list"),
            ("allow:\n  - broken\n", "invalid rule"),
            ("allow:\n  - Missing(*)\n", "unknown tool"),
        ]
        for content, message in cases:
            with self.subTest(content=content):
                with (
                    tempfile.TemporaryDirectory() as home_directory,
                    tempfile.TemporaryDirectory() as workspace_directory,
                    tempfile.TemporaryDirectory() as temporary_directory,
                ):
                    workspace = Path(workspace_directory)
                    permission_directory = workspace / ".duckduckcode"
                    permission_directory.mkdir()
                    path = permission_directory / "permissions.yaml"
                    path.write_text(content, encoding="utf-8")

                    with self.assertRaisesRegex(RuntimeError, f"{path}.*{message}"):
                        RulePolicy.load(
                            workspace,
                            Path(temporary_directory),
                            TOOLS,
                            home=Path(home_directory),
                        )

    def test_remember_allow_writes_exact_local_rules_and_reloads_them(self) -> None:
        with (
            tempfile.TemporaryDirectory() as home_directory,
            tempfile.TemporaryDirectory() as workspace_directory,
            tempfile.TemporaryDirectory() as temporary_directory,
        ):
            home = Path(home_directory)
            workspace = Path(workspace_directory)
            private_temp = Path(temporary_directory)
            permission_directory = workspace / ".duckduckcode"
            permission_directory.mkdir()
            (permission_directory / "permissions.yaml").write_text(
                'ask:\n  - "Bash(git push *)"\n',
                encoding="utf-8",
            )
            policy = RulePolicy.load(workspace, private_temp, TOOLS, home=home)
            calls = [
                ToolCall(
                    "bash",
                    "Bash",
                    {"command": "git push origin feature[x]"},
                ),
                ToolCall(
                    "read",
                    "ReadFile",
                    {"path": str(workspace / "src" / "app.py")},
                ),
                ToolCall(
                    "temp",
                    "WriteFile",
                    {"path": str(private_temp / "cache.json")},
                ),
            ]

            for call in calls:
                policy.remember_allow(call)
                policy.remember_allow(call)

            content = (permission_directory / "permissions.local.yaml").read_text(
                encoding="utf-8"
            )
            self.assertEqual(content.count("Bash("), 1)
            self.assertEqual(content.count("ReadFile("), 1)
            self.assertEqual(content.count("WriteFile("), 1)
            self.assertIn("./src/app.py", content)
            self.assertIn("${TEMP}/cache.json", content)

            reloaded = RulePolicy.load(workspace, private_temp, TOOLS, home=home)
            for call in calls:
                with self.subTest(call=call):
                    self.assertEqual(reloaded.check(call).action, "allow")

    def test_failed_local_write_can_be_retried(self) -> None:
        with (
            tempfile.TemporaryDirectory() as home_directory,
            tempfile.TemporaryDirectory() as workspace_directory,
            tempfile.TemporaryDirectory() as temporary_directory,
        ):
            workspace = Path(workspace_directory)
            policy = RulePolicy.load(
                workspace,
                Path(temporary_directory),
                TOOLS,
                home=Path(home_directory),
            )
            call = ToolCall("call", "Bash", {"command": "git status"})

            with (
                patch(
                    "duckduckcode.permissions.rule_policy.os.replace",
                    side_effect=OSError("disk full"),
                ),
                self.assertRaises(OSError),
            ):
                policy.remember_allow(call)

            policy.remember_allow(call)

            local_file = workspace / ".duckduckcode" / "permissions.local.yaml"
            self.assertIn("Bash(git status)", local_file.read_text(encoding="utf-8"))

    def test_refuses_to_initialize_through_a_symlinked_permission_directory(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as home_directory,
            tempfile.TemporaryDirectory() as workspace_directory,
            tempfile.TemporaryDirectory() as outside_directory,
            tempfile.TemporaryDirectory() as temporary_directory,
        ):
            workspace = Path(workspace_directory)
            outside = Path(outside_directory)
            (workspace / ".duckduckcode").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "outside the workspace"):
                RulePolicy.load(
                    workspace,
                    Path(temporary_directory),
                    TOOLS,
                    home=Path(home_directory),
                )

            self.assertFalse((outside / "permissions.local.yaml").exists())


if __name__ == "__main__":
    unittest.main()
