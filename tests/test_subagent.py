from __future__ import annotations

import json
import sys
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

from duckduckcode.core.context import ContextManager
from duckduckcode.core.agent import Agent
from duckduckcode.core.event import (
    ConversationEvent,
    DoneEvent,
    LoopCompleteEvent,
    SubagentEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from duckduckcode.permissions import PermissionChecker, PermissionDecision
from duckduckcode.core.subagent import (
    DefinitionManager,
    SubagentManager,
    load_worktree_configuration,
    _worker_request,
)
from duckduckcode.tools.tool import (
    QuerySource,
    ToolCall,
    ToolManager,
    ToolResult,
    create_agent_tool,
)


class DefinitionTest(unittest.TestCase):
    def test_discovers_builtins_and_project_overrides_user(self) -> None:
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            user = Path(home) / ".duckduckcode" / "agents"
            project = workspace / ".duckduckcode" / "agents"
            user.mkdir(parents=True)
            project.mkdir(parents=True)
            user.joinpath("custom.md").write_text(
                "---\ntype: custom\nwhenToUse: user\ndisallowedTools: []\nmaxTurns: 4\n---\nUser body\n",
                encoding="utf-8",
            )
            project.joinpath("custom.md").write_text(
                "---\ntype: custom\nwhenToUse: project\ndisallowedTools: [Grep]\nmaxTurns: 7\n---\nProject body\n",
                encoding="utf-8",
            )

            definitions, warning = DefinitionManager(
                workspace,
                home=home,
                known_tools={"ReadFile", "Glob", "Grep"},
            ).refresh()

            by_type = {definition.type: definition for definition in definitions}
            self.assertIsNone(warning)
            self.assertEqual({"explore", "plan", "custom"}, set(by_type))
            self.assertEqual(by_type["custom"].scope, "project")
            self.assertEqual(by_type["custom"].when_to_use, "project")
            self.assertEqual(by_type["custom"].max_turns, 7)
            self.assertEqual(by_type["custom"].disallowed_tools, ("Grep",))
            self.assertEqual(by_type["custom"].body, "Project body")

    def test_skips_duplicate_and_unsafe_or_invalid_definitions(self) -> None:
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as wd:
            root = Path(wd) / ".duckduckcode" / "agents"
            root.mkdir(parents=True)
            valid = "---\ntype: same\nwhenToUse: x\ndisallowedTools: []\nmaxTurns: 1\n---\nBody\n"
            root.joinpath("one.md").write_text(valid, encoding="utf-8")
            root.joinpath("two.md").write_text(valid, encoding="utf-8")
            root.joinpath("bad-name.md").write_text(
                "---\ntype: Bad\nwhenToUse: x\ndisallowedTools: []\nmaxTurns: 1\n---\nBody\n",
                encoding="utf-8",
            )
            root.joinpath("unknown.md").write_text(
                "---\ntype: unknown\nwhenToUse: x\ndisallowedTools: [Bash]\nmaxTurns: 1\n---\nBody\n",
                encoding="utf-8",
            )
            target = root / "target.txt"
            target.write_text(valid, encoding="utf-8")
            root.joinpath("linked.md").symlink_to(target)
            root.joinpath("large.md").write_bytes(b"x" * (256 * 1024 + 1))

            definitions, warning = DefinitionManager(
                Path(wd), home=home, known_tools={"ReadFile", "Glob", "Grep"}
            ).refresh()

            self.assertEqual([item.type for item in definitions], ["explore", "plan"])
            self.assertIn("duplicate definition type 'same'", warning or "")
            self.assertIn("lowercase kebab-case", warning or "")
            self.assertIn("unknown tool 'Bash'", warning or "")
            self.assertIn("must not be a symlink", warning or "")
            self.assertIn("exceeds", warning or "")


class AgentToolTest(unittest.TestCase):
    def test_schema_is_strict_nullable_and_has_dynamic_definition_enum(self) -> None:
        captured = []
        tool = create_agent_tool(
            ["explore", "plan", "custom"],
            lambda **arguments: captured.append(arguments),
        )

        schema = tool.schema()
        properties = schema["parameters"]["properties"]
        self.assertEqual(schema["name"], "Agent")
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["parameters"]["additionalProperties"])
        self.assertEqual(set(schema["parameters"]["required"]), set(properties))
        self.assertEqual(
            properties["subagent_type"],
            {"type": ["string", "null"], "enum": ["explore", "plan", "custom", None]},
        )
        self.assertEqual(properties["model"]["type"], ["string", "null"])
        self.assertEqual(properties["name"]["type"], ["string", "null"])

    def test_validator_rejects_unknown_type_and_empty_values_and_backgrounds_fork(
        self,
    ) -> None:
        captured = []
        tool = create_agent_tool(
            ["explore"], lambda **arguments: captured.append(arguments) or "ok"
        )
        base = {
            "prompt": "do it",
            "description": "work",
            "subagent_type": "explore",
            "model": None,
            "run_in_background": False,
            "name": None,
            "isolation": True,
        }
        manager = ToolManager()
        manager.register(tool)

        self.assertFalse(manager.execute(ToolCall("1", "Agent", base)).is_error)
        for key, value in (
            ("prompt", " "),
            ("description", ""),
            ("subagent_type", "missing"),
            ("model", " "),
            ("name", " "),
        ):
            invalid = dict(base)
            invalid[key] = value
            self.assertTrue(
                manager.execute(ToolCall(key, "Agent", invalid)).is_error, key
            )
        fork = dict(base, subagent_type=None, run_in_background=False)
        result = manager.execute(ToolCall("fork", "Agent", fork))
        self.assertFalse(result.is_error)
        self.assertTrue(captured[-1]["run_in_background"])

    def test_validator_enforces_safe_worktree_slugs(self) -> None:
        tool = create_agent_tool(["explore"], lambda **arguments: arguments)
        manager = ToolManager()
        manager.register(tool)
        base = {
            "prompt": "do it",
            "description": "work",
            "subagent_type": None,
            "model": None,
            "run_in_background": True,
            "name": "Fix.Parser_2-a",
            "isolation": True,
        }

        self.assertFalse(manager.execute(ToolCall("ok", "Agent", base)).is_error)
        for name in (
            ".hidden",
            "trailing-",
            "two--parts",
            "mixed._parts",
            "../escape",
            "-option",
            "中文",
            "x" * 49,
        ):
            with self.subTest(name=name):
                arguments = dict(base, name=name)
                self.assertTrue(
                    manager.execute(ToolCall(name, "Agent", arguments)).is_error
                )

    def test_subagent_source_hard_rejects_recursive_agent_execution(self) -> None:
        manager = ToolManager(source=QuerySource.SUBAGENT)
        manager.register(create_agent_tool(["explore"], lambda **_: "ran"))
        result = manager.execute(
            ToolCall(
                "1",
                "Agent",
                {
                    "prompt": "do it",
                    "description": "work",
                    "subagent_type": "explore",
                    "model": None,
                    "run_in_background": True,
                    "name": None,
                    "isolation": True,
                },
            )
        )

        self.assertEqual(
            result,
            ToolResult("Subagents cannot invoke the Agent tool.", is_error=True),
        )


class WorktreeConfigurationTest(unittest.TestCase):
    def _write(self, home: Path, content: str) -> Path:
        path = home / ".duckduckcode" / "worktree.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_safe_relative_symlink_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self._write(
                home,
                "symlinks:\n  - .env\n  - .duckduckcode/permissions.local.yaml\n",
            )

            configuration = load_worktree_configuration(home=home)

            self.assertEqual(
                configuration.symlinks,
                (".env", ".duckduckcode/permissions.local.yaml"),
            )
            self.assertEqual(configuration.warnings, ())

    def test_invalid_config_is_skipped_safely(self) -> None:
        cases = (
            ("duplicate", "symlinks: []\nsymlinks: []\n", "duplicate"),
            ("unknown", "other: []\n", "unknown"),
            ("absolute", "symlinks: [/tmp/secret]\n", "relative"),
            ("parent", "symlinks: [../secret]\n", "relative"),
            ("empty-segment", "symlinks: [a//b]\n", "relative"),
            ("dot-segment", "symlinks: [a/./b]\n", "relative"),
            ("backslash", "symlinks: ['a\\\\b']\n", "backslash"),
            ("many", "symlinks:\n" + "  - x\n" * 65, "64"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content, expected in cases:
                with self.subTest(name=name):
                    home = root / name
                    self._write(home, content)
                    configuration = load_worktree_configuration(home=home)
                    self.assertEqual(configuration.symlinks, ())
                    self.assertIn(expected, configuration.warnings[0])

    def test_rejects_symlink_and_oversized_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = self._write(root / "target", "symlinks: []\n")
            linked = root / "linked" / ".duckduckcode" / "worktree.yaml"
            linked.parent.mkdir(parents=True)
            linked.symlink_to(target)
            large = self._write(root / "large", "x" * (256 * 1024 + 1))

            linked_config = load_worktree_configuration(home=root / "linked")
            large_config = load_worktree_configuration(home=root / "large")

            self.assertIn("symbolic", linked_config.warnings[0])
            self.assertIn("256 KiB", large_config.warnings[0])


class SubagentManagerTest(unittest.TestCase):
    def _script(self, directory: Path, source: str) -> list[str]:
        path = directory / "worker.py"
        path.write_text(source, encoding="utf-8")
        return [sys.executable, str(path)]

    def _arguments(self, **overrides):
        arguments = {
            "prompt": "inspect",
            "description": "inspection",
            "subagent_type": "explore",
            "model": None,
            "run_in_background": False,
            "name": "reader",
            "isolation": False,
        }
        arguments.update(overrides)
        return arguments

    def _git(self, workspace: Path, *arguments: str, input: str | None = None):
        return subprocess.run(
            ["git", "-C", str(workspace), *arguments],
            input=input,
            text=True,
            capture_output=True,
            check=True,
        )

    def _repository(self, root: Path) -> Path:
        workspace = root / "repository"
        workspace.mkdir()
        self._git(workspace, "init", "-q")
        self._git(workspace, "config", "user.name", "Test")
        self._git(workspace, "config", "user.email", "test@example.com")
        workspace.joinpath(".gitignore").write_text(
            ".env\n.duckduckcode/permissions.local.yaml\n", encoding="utf-8"
        )
        workspace.joinpath("source.txt").write_text("original\n", encoding="utf-8")
        workspace.joinpath("deleted.txt").write_text("delete me\n", encoding="utf-8")
        self._git(workspace, "add", ".gitignore", "source.txt", "deleted.txt")
        self._git(workspace, "commit", "-qm", "initial")
        return workspace

    def _finish_background(self, manager: SubagentManager, stream, session="session-a"):
        list(stream)
        deadline = time.monotonic() + 5
        while manager.running_count and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(manager.running_count, 0)
        events, messages = manager.drain(session)
        self.assertEqual(len(messages), 1)
        return events, json.loads(messages[0].splitlines()[-1])

    def test_foreground_forwards_nested_events_and_returns_final_result(self) -> None:
        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            command = self._script(
                workspace,
                "import json, sys\n"
                "request = json.loads(sys.stdin.readline())\n"
                "print(json.dumps({'type':'tool_use','call_id':'inner','name':'ReadFile','arguments':{'path':'x'}}), flush=True)\n"
                "print(json.dumps({'type':'usage','total_tokens':9}), flush=True)\n"
                "print(json.dumps({'type':'worker_result','status':'completed','result':'evidence'}), flush=True)\n",
            )
            manager = SubagentManager(workspace, worker_command=command)
            self.addCleanup(manager.close)

            stream = manager.run(
                "outer",
                self._arguments(),
                session_key="session-a",
                context=ContextManager(system_prompt="system"),
                permission_mode="ask_for_approval",
            )
            events = []
            try:
                while True:
                    events.append(next(stream))
            except StopIteration as done:
                result = done.value

            task_id = next(
                event.task_id
                for event in events
                if isinstance(event, SubagentEvent) and event.status == "started"
            )
            self.assertIn(SubagentEvent(task_id, "reader", "started", False), events)
            self.assertIn(
                ToolCallEvent(ToolCall("outer/inner", "ReadFile", {"path": "x"})),
                events,
            )
            self.assertIn(UsageEvent(9), events)
            self.assertIn(SubagentEvent(task_id, "reader", "completed", False), events)
            payload = json.loads(result.content)
            self.assertFalse(result.is_error)
            self.assertEqual(payload["result"], "evidence")
            self.assertEqual(payload["status"], "completed")
            self.assertFalse(payload["background"])

    def test_background_result_waits_for_its_session_inbox(self) -> None:
        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            command = self._script(
                workspace,
                "import json, sys, time\n"
                "json.loads(sys.stdin.readline())\n"
                "time.sleep(.05)\n"
                "print(json.dumps({'type':'usage','total_tokens':4}), flush=True)\n"
                "print(json.dumps({'type':'worker_result','status':'completed','result':'later'}), flush=True)\n",
            )
            manager = SubagentManager(workspace, worker_command=command)
            self.addCleanup(manager.close)
            stream = manager.run(
                "outer",
                self._arguments(run_in_background=True),
                session_key="session-a",
                context=ContextManager(system_prompt="system"),
                permission_mode="ask_for_approval",
            )
            events = []
            try:
                while True:
                    events.append(next(stream))
            except StopIteration as done:
                result = done.value

            payload = json.loads(result.content)
            self.assertEqual(payload["status"], "running")
            self.assertTrue(payload["background"])
            self.assertIn(
                SubagentEvent(payload["task_id"], "reader", "backgrounded", True),
                events,
            )
            deadline = time.monotonic() + 2
            while manager.running_count and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(manager.drain("session-b"), ([], []))
            inbox_events, messages = manager.drain("session-a")
            self.assertIn(UsageEvent(4), inbox_events)
            self.assertIn(
                SubagentEvent(payload["task_id"], "reader", "completed", True),
                inbox_events,
            )
            self.assertEqual(len(messages), 1)
            self.assertIn("later", messages[0])
            self.assertIn("untrusted", messages[0].lower())

    def test_isolation_discards_changes_and_cleans_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            workspace.joinpath("source.txt").write_text("original", encoding="utf-8")
            command = self._script(
                workspace,
                "import json, pathlib, sys\n"
                "request = json.loads(sys.stdin.readline())\n"
                "pathlib.Path('source.txt').write_text('changed')\n"
                "print(json.dumps({'type':'worker_result','status':'completed','result':str(pathlib.Path.cwd())}), flush=True)\n",
            )
            manager = SubagentManager(workspace, worker_command=command)
            self.addCleanup(manager.close)
            stream = manager.run(
                "outer",
                self._arguments(isolation=True),
                session_key="session-a",
                context=ContextManager(system_prompt="system"),
                permission_mode="full_access",
            )
            try:
                while True:
                    next(stream)
            except StopIteration as done:
                result = done.value

            snapshot = Path(json.loads(result.content)["result"])
            self.assertEqual(workspace.joinpath("source.txt").read_text(), "original")
            self.assertFalse(snapshot.exists())

    def test_isolated_fork_returns_patch_and_retains_worktree_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._repository(root)
            workspace.joinpath(".env").write_text("TOKEN=secret\n", encoding="utf-8")
            workspace.joinpath("ignored-dir").mkdir()
            workspace.joinpath("linked-secret").symlink_to(".env")
            with workspace.joinpath(".gitignore").open("a", encoding="utf-8") as stream:
                stream.write("ignored-dir/\nlinked-secret\n")
            self._git(workspace, "add", ".gitignore")
            self._git(workspace, "commit", "-qm", "ignore local files")
            home = root / "home"
            config = home / ".duckduckcode" / "worktree.yaml"
            config.parent.mkdir(parents=True)
            config.write_text(
                "symlinks:\n"
                "  - .env\n"
                "  - missing.local\n"
                "  - source.txt\n"
                "  - ignored-dir\n"
                "  - linked-secret\n",
                encoding="utf-8",
            )
            command = self._script(
                root,
                "import json, pathlib, sys\n"
                "request = json.loads(sys.stdin.readline())\n"
                "assert request['worktree'] is True\n"
                "assert {pathlib.Path(path).name for path in request['worktree_read_files']} == {'.env', 'ignored-dir'}\n"
                "cwd = pathlib.Path.cwd()\n"
                "assert cwd.joinpath('.env').is_symlink()\n"
                "assert cwd.joinpath('.env').read_text() == 'TOKEN=secret\\n'\n"
                "cwd.joinpath('source.txt').write_text('changed\\n')\n"
                "cwd.joinpath('new.txt').write_text('new\\n')\n"
                "cwd.joinpath('deleted.txt').unlink()\n"
                "cwd.joinpath('binary.bin').write_bytes(bytes(range(256)) * 4)\n"
                "cwd.joinpath('new-link').symlink_to('source.txt')\n"
                "print(json.dumps({'type':'worker_result','status':'completed','result':str(cwd)}), flush=True)\n",
            )
            manager = SubagentManager(workspace, worker_command=command, home=home)
            self.addCleanup(manager.close)

            stream = manager.run(
                "outer",
                self._arguments(
                    subagent_type=None,
                    run_in_background=True,
                    isolation=True,
                    name="Fix.Parser_2-a",
                ),
                session_key="session-a",
                context=ContextManager(system_prompt="system"),
                permission_mode="full_access",
            )
            events, payload = self._finish_background(manager, stream)

            changes = payload["changes"]
            worktree_path = Path(payload["result"])
            self.assertEqual(payload["worktree_id"], changes["worktree_id"])
            self.assertTrue(changes["branch"].startswith("worktree-Fix.Parser_2-a-"))
            self.assertEqual(len(changes["branch"].rsplit("-", 1)[1]), 8)
            self.assertFalse(changes["partial"])
            self.assertFalse(changes["parent_changed"])
            self.assertIn("diff --git a/source.txt b/source.txt", changes["patch"])
            self.assertIn("diff --git a/new.txt b/new.txt", changes["patch"])
            self.assertIn("GIT binary patch", changes["patch"])
            self._git(workspace, "apply", "--check", "-", input=changes["patch"])
            self.assertEqual(
                {item["path"] for item in changes["files"]},
                {"binary.bin", "deleted.txt", "new-link", "new.txt", "source.txt"},
            )
            self.assertTrue(
                any("missing.local" in item for item in payload["warnings"])
            )
            self.assertTrue(any("tracked" in item for item in payload["warnings"]))
            self.assertTrue(any("symlink" in item for item in payload["warnings"]))
            self.assertEqual(workspace.joinpath("source.txt").read_text(), "original\n")
            self.assertEqual(workspace.joinpath(".env").read_text(), "TOKEN=secret\n")
            self.assertTrue(worktree_path.exists())
            self.assertIn(
                changes["branch"], self._git(workspace, "branch", "--list").stdout
            )
            self.assertTrue(any(event.status == "completed" for event in events))

    def test_isolated_fork_requires_clean_git_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._repository(root)
            workspace.joinpath("untracked.txt").write_text("dirty", encoding="utf-8")
            manager = SubagentManager(
                workspace,
                worker_command=self._script(root, "raise AssertionError('not run')\n"),
                home=root / "home",
            )
            self.addCleanup(manager.close)

            stream = manager.run(
                "outer",
                self._arguments(
                    subagent_type=None, run_in_background=True, isolation=True
                ),
                session_key="session-a",
                context=ContextManager(system_prompt="system"),
                permission_mode="full_access",
            )
            with self.assertRaises(StopIteration) as stopped:
                next(stream)

            self.assertTrue(stopped.exception.value.is_error)
            self.assertIn("clean", stopped.exception.value.content.lower())
            self.assertEqual(manager.running_count, 0)

    def test_timed_out_fork_returns_partial_patch_and_retains_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._repository(root)
            command = self._script(
                root,
                "import json, pathlib, sys, time\n"
                "json.loads(sys.stdin.readline())\n"
                "pathlib.Path('source.txt').write_text('partial\\n')\n"
                "time.sleep(10)\n",
            )
            manager = SubagentManager(
                workspace, worker_command=command, timeout=0.05, home=root / "home"
            )
            self.addCleanup(manager.close)

            _, payload = self._finish_background(
                manager,
                manager.run(
                    "outer",
                    self._arguments(
                        subagent_type=None,
                        run_in_background=True,
                        isolation=True,
                    ),
                    session_key="session-a",
                    context=ContextManager(system_prompt="system"),
                    permission_mode="full_access",
                ),
            )

            self.assertTrue(payload["changes"]["partial"])
            self.assertIn("partial", payload["changes"]["patch"])
            self.assertEqual(
                self._git(workspace, "worktree", "list", "--porcelain").stdout.count(
                    "worktree "
                ),
                2,
            )
            self.assertIn(
                payload["changes"]["branch"],
                self._git(workspace, "branch", "--list").stdout,
            )

    def test_parent_change_is_reported_without_overwriting_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._repository(root)
            command = self._script(
                root,
                "import json, pathlib, sys, time\n"
                "json.loads(sys.stdin.readline())\n"
                "pathlib.Path('source.txt').write_text('child\\n')\n"
                "time.sleep(.1)\n"
                "print(json.dumps({'type':'worker_result','status':'completed','result':'done'}), flush=True)\n",
            )
            manager = SubagentManager(workspace, worker_command=command, home=root)
            self.addCleanup(manager.close)
            stream = manager.run(
                "outer",
                self._arguments(
                    subagent_type=None, run_in_background=True, isolation=True
                ),
                session_key="session-a",
                context=ContextManager(system_prompt="system"),
                permission_mode="full_access",
            )
            list(stream)
            workspace.joinpath("source.txt").write_text("parent\n", encoding="utf-8")
            _, payload = self._finish_background(manager, iter(()))

            self.assertTrue(payload["changes"]["parent_changed"])
            self.assertEqual(workspace.joinpath("source.txt").read_text(), "parent\n")

    def test_invalid_worker_output_returns_partial_patch_and_retains_worktree(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._repository(root)
            command = self._script(
                root,
                "import json, pathlib, sys\n"
                "json.loads(sys.stdin.readline())\n"
                "pathlib.Path('source.txt').write_text('partial\\n')\n"
                "print('not-json', flush=True)\n",
            )
            manager = SubagentManager(workspace, worker_command=command, home=root)
            self.addCleanup(manager.close)

            _, payload = self._finish_background(
                manager,
                manager.run(
                    "outer",
                    self._arguments(
                        subagent_type=None,
                        run_in_background=True,
                        isolation=True,
                    ),
                    session_key="session-a",
                    context=ContextManager(system_prompt="system"),
                    permission_mode="full_access",
                ),
            )

            self.assertTrue(payload["changes"]["partial"])
            self.assertIn("partial", payload["changes"]["patch"])
            self.assertEqual(
                self._git(workspace, "worktree", "list", "--porcelain").stdout.count(
                    "worktree "
                ),
                2,
            )

    def test_worker_start_failure_retains_created_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._repository(root)
            manager = SubagentManager(
                workspace,
                worker_command=[str(root / "missing-worker")],
                home=root,
            )
            self.addCleanup(manager.close)

            stream = manager.run(
                "outer",
                self._arguments(
                    subagent_type=None,
                    run_in_background=True,
                    isolation=True,
                ),
                session_key="session-a",
                context=ContextManager(system_prompt="system"),
                permission_mode="full_access",
            )
            with self.assertRaises(StopIteration) as stopped:
                next(stream)

            self.assertTrue(stopped.exception.value.is_error)
            self.assertEqual(
                self._git(workspace, "worktree", "list", "--porcelain").stdout.count(
                    "worktree "
                ),
                2,
            )
            self.assertIn("worktree-", self._git(workspace, "branch", "--list").stdout)

    def test_session_termination_retains_worktree_without_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._repository(root)
            command = self._script(
                root,
                "import json, pathlib, sys, time\n"
                "json.loads(sys.stdin.readline())\n"
                "pathlib.Path('source.txt').write_text('partial\\n')\n"
                "time.sleep(10)\n",
            )
            manager = SubagentManager(workspace, worker_command=command, home=root)
            self.addCleanup(manager.close)
            stream = manager.run(
                "outer",
                self._arguments(
                    subagent_type=None, run_in_background=True, isolation=True
                ),
                session_key="deleted",
                context=ContextManager(system_prompt="system"),
                permission_mode="full_access",
            )
            list(stream)

            manager.terminate_session("deleted")
            deadline = time.monotonic() + 5
            while manager.running_count and time.monotonic() < deadline:
                time.sleep(0.01)

            self.assertEqual(manager.drain("deleted"), ([], []))
            self.assertEqual(
                self._git(workspace, "worktree", "list", "--porcelain").stdout.count(
                    "worktree "
                ),
                2,
            )
            self.assertIn("worktree-", self._git(workspace, "branch", "--list").stdout)

    def test_nonisolated_fork_holds_single_writer_lease(self) -> None:
        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            command = self._script(
                workspace,
                "import json, sys, time\njson.loads(sys.stdin.readline())\ntime.sleep(2)\n",
            )
            manager = SubagentManager(workspace, worker_command=command)
            self.addCleanup(manager.close)
            first = manager.run(
                "one",
                self._arguments(
                    subagent_type=None, run_in_background=True, isolation=False
                ),
                session_key="s",
                context=ContextManager(system_prompt="system"),
                permission_mode="ask_for_approval",
            )
            try:
                while True:
                    next(first)
            except StopIteration:
                pass
            second = manager.run(
                "two",
                self._arguments(
                    subagent_type=None, run_in_background=True, isolation=False
                ),
                session_key="s",
                context=ContextManager(system_prompt="system"),
                permission_mode="ask_for_approval",
            )
            try:
                while True:
                    next(second)
            except StopIteration as done:
                result = done.value
            self.assertTrue(result.is_error)
            self.assertIn("write lease", result.content)
            self.assertTrue(manager.workspace_busy)

    def test_isolated_forks_run_concurrently_without_writer_lease(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self._repository(root)
            command = self._script(
                root,
                "import json, sys, time\n"
                "json.loads(sys.stdin.readline())\n"
                "time.sleep(10)\n",
            )
            manager = SubagentManager(workspace, worker_command=command, home=root)
            for name in ("one", "two"):
                list(
                    manager.run(
                        name,
                        self._arguments(
                            subagent_type=None,
                            run_in_background=True,
                            isolation=True,
                            name=name,
                        ),
                        session_key="session-a",
                        context=ContextManager(system_prompt="system"),
                        permission_mode="full_access",
                    )
                )

            self.assertEqual(manager.running_count, 2)
            self.assertFalse(manager.workspace_busy)
            self.assertEqual(
                self._git(workspace, "worktree", "list", "--porcelain").stdout.count(
                    "worktree "
                ),
                3,
            )

            manager.close()

            self.assertEqual(
                self._git(workspace, "worktree", "list", "--porcelain").stdout.count(
                    "worktree "
                ),
                3,
            )
            self.assertIn("worktree-", self._git(workspace, "branch", "--list").stdout)

    def test_timeout_releases_lease_and_delivers_structured_failure(self) -> None:
        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            command = self._script(
                workspace,
                "import json, sys, time\njson.loads(sys.stdin.readline())\ntime.sleep(2)\n",
            )
            manager = SubagentManager(workspace, worker_command=command, timeout=0.05)
            self.addCleanup(manager.close)
            stream = manager.run(
                "one",
                self._arguments(
                    subagent_type=None, run_in_background=True, isolation=False
                ),
                session_key="s",
                context=ContextManager(system_prompt="system"),
                permission_mode="ask_for_approval",
            )
            try:
                while True:
                    next(stream)
            except StopIteration:
                pass
            deadline = time.monotonic() + 2
            while manager.running_count and time.monotonic() < deadline:
                time.sleep(0.01)

            events, messages = manager.drain("s")

            self.assertFalse(manager.workspace_busy)
            self.assertTrue(
                any(
                    isinstance(event, SubagentEvent) and event.status == "timed_out"
                    for event in events
                )
            )
            self.assertIn("timed out", messages[0])

    def test_concurrency_limit_rejects_an_extra_worker(self) -> None:
        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            command = self._script(
                workspace,
                "import json, sys, time\njson.loads(sys.stdin.readline())\ntime.sleep(2)\n",
            )
            manager = SubagentManager(workspace, worker_command=command, max_tasks=1)
            self.addCleanup(manager.close)
            first = manager.run(
                "one",
                self._arguments(run_in_background=True, isolation=True),
                session_key="s",
                context=ContextManager(system_prompt="system"),
                permission_mode="ask_for_approval",
            )
            try:
                while True:
                    next(first)
            except StopIteration:
                pass
            second = manager.run(
                "two",
                self._arguments(run_in_background=True, isolation=True),
                session_key="s",
                context=ContextManager(system_prompt="system"),
                permission_mode="ask_for_approval",
            )
            try:
                while True:
                    next(second)
            except StopIteration as done:
                result = done.value

            self.assertTrue(result.is_error)
            self.assertIn("at most 1", result.content)

    def test_foreground_task_can_detach_without_restarting_worker(self) -> None:
        with tempfile.TemporaryDirectory() as wd:
            workspace = Path(wd)
            command = self._script(
                workspace,
                "import json, sys, time\n"
                "json.loads(sys.stdin.readline())\n"
                "time.sleep(.1)\n"
                "print(json.dumps({'type':'worker_result','status':'completed','result':'same process'}), flush=True)\n",
            )
            manager = SubagentManager(workspace, worker_command=command)
            self.addCleanup(manager.close)
            stream = manager.run(
                "one",
                self._arguments(run_in_background=False),
                session_key="s",
                context=ContextManager(system_prompt="system"),
                permission_mode="ask_for_approval",
            )
            started = next(stream)
            self.assertEqual(started.status, "started")
            self.assertTrue(manager.detach_foreground())
            backgrounded = next(stream)
            self.assertEqual(backgrounded.status, "backgrounded")
            with self.assertRaises(StopIteration) as stopped:
                next(stream)
            self.assertEqual(
                json.loads(stopped.exception.value.content)["status"], "running"
            )
            deadline = time.monotonic() + 2
            while manager.running_count and time.monotonic() < deadline:
                time.sleep(0.01)
            events, messages = manager.drain("s")
            self.assertTrue(any(event.status == "completed" for event in events))
            self.assertIn("same process", messages[0])

    def test_definition_request_does_not_copy_parent_conversation(self) -> None:
        context = ContextManager(system_prompt="parent secret")
        context.add_user("private history")
        with tempfile.TemporaryDirectory() as wd:
            manager = DefinitionManager(Path(wd))
            definitions, _ = manager.refresh()
            definition = next(item for item in definitions if item.type == "explore")
            request = _worker_request(
                self._arguments(),
                context,
                "ask_for_approval",
                definition,
                Path(wd),
            )

        self.assertEqual(request["messages"], [])
        self.assertEqual(request["abstraction"], "")
        self.assertEqual(request["long_term_memory"], "")
        self.assertEqual(request["system_prompt"], "")


class AgentIntegrationTest(unittest.TestCase):
    def test_large_background_result_uses_context_externalization(self) -> None:
        class Manager:
            workspace_busy = False

            def drain(self, session_key):
                return [], ["Untrusted subagent output\n" + "x" * (81 * 1024)]

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            context = ContextManager(
                system_prompt="system", tool_result_directory=Path(directory)
            )
            agent = Agent(object(), context, subagent_manager=Manager())

            self.assertEqual(list(agent._drain_subagents()), [])

            message = context.messages()[-1]
            self.assertEqual(message.role, "user")
            self.assertIn("Tool result stored on disk", message.content)
            stored = next(Path(directory).rglob("*.txt"))
            self.assertIn("Untrusted subagent output", stored.read_text())

    def test_agent_tool_routes_through_subagent_manager(self) -> None:
        class Client:
            def __init__(self):
                self.calls = 0

            def stream(self, messages, tools=None, reasoning=None):
                self.calls += 1
                if self.calls == 1:
                    yield ToolCallEvent(
                        ToolCall(
                            "parent",
                            "Agent",
                            {
                                "prompt": "inspect",
                                "description": "inspection",
                                "subagent_type": "explore",
                                "model": None,
                                "run_in_background": False,
                                "name": None,
                                "isolation": True,
                            },
                        )
                    )
                    yield DoneEvent(2)
                    return
                yield ConversationEvent("parent answer")
                yield DoneEvent(1)

        class Manager:
            workspace_busy = False

            def __init__(self):
                self.arguments = None

            def run(self, call_id, arguments, **context):
                self.arguments = arguments
                yield SubagentEvent("task", "inspection", "started", False)
                return ToolResult("evidence")

            def drain(self, session_key):
                return [], []

            def close(self):
                pass

        manager = Manager()
        tools = ToolManager()
        builtin = create_agent_tool(["explore", "plan"], lambda **_: "unused")

        class ProtocolTool:
            name = builtin.name
            description = builtin.description
            params = builtin.params
            is_read_only = builtin.is_read_only
            is_dangerous = builtin.is_dangerous
            is_concurrency_safe = builtin.is_concurrency_safe
            category = builtin.category
            strict = builtin.strict

            def schema(self):
                return builtin.schema()

            def execute(self, arguments):
                return builtin.execute(arguments)

            def permission_content(self, arguments):
                return None

        tools.register(ProtocolTool())
        agent = Agent(
            Client(),
            ContextManager(system_prompt="system"),
            tools,
            subagent_manager=manager,
        )

        events = list(agent.stream("hello"))

        self.assertIn(SubagentEvent("task", "inspection", "started", False), events)
        self.assertIn(ToolResultEvent("parent", "Agent", "evidence"), events)
        self.assertEqual(manager.arguments["prompt"], "inspect")
        self.assertEqual(events[-1], LoopCompleteEvent("completed", 2))

    def test_parent_write_tools_are_busy_while_fork_holds_lease(self) -> None:
        class Client:
            def stream(self, messages, tools=None, reasoning=None):
                yield ToolCallEvent(ToolCall("write", "WriteFile", {"path": "x"}))
                yield DoneEvent()

        class Manager:
            workspace_busy = True

            def drain(self, session_key):
                return [], []

            def close(self):
                pass

        tools = ToolManager()
        tools.register(
            "WriteFile",
            "write",
            {"type": "object", "properties": {}},
            lambda path: "written",
        )
        events = list(
            Agent(
                Client(),
                ContextManager(system_prompt="system"),
                tools,
                subagent_manager=Manager(),
            ).stream("hello")
        )

        result = next(event for event in events if isinstance(event, ToolResultEvent))
        self.assertTrue(result.is_error)
        self.assertIn("workspace is busy", result.content.lower())

    def test_subagent_permission_prompts_are_denied_without_an_event(self) -> None:
        class Client:
            def __init__(self):
                self.calls = 0

            def stream(self, messages, tools=None, reasoning=None):
                self.calls += 1
                if self.calls == 1:
                    yield ToolCallEvent(ToolCall("write", "WriteFile", {"path": "x"}))
                yield DoneEvent()

        class Policy:
            permission_mode = "ask_for_approval"

            def check(self, tool_call, *, tool=None):
                return PermissionDecision("ask", "approval required", tool_call.name)

            def set_permission_mode(self, mode):
                self.permission_mode = mode

            def remember_allow(self, tool_call, *, tool=None):
                raise AssertionError("must not remember subagent approval")

        tools = ToolManager(source=QuerySource.SUBAGENT)
        tools.register(
            "WriteFile",
            "write",
            {"type": "object", "properties": {}},
            lambda path: "written",
        )
        events = list(
            Agent(
                Client(),
                ContextManager(system_prompt="system"),
                tools,
                permission_checker=PermissionChecker(policy=Policy()),
                query_source=QuerySource.SUBAGENT,
            ).stream("hello")
        )

        self.assertFalse(
            any(
                event.__class__.__name__ == "PermissionRequestEvent" for event in events
            )
        )
        result = next(event for event in events if isinstance(event, ToolResultEvent))
        self.assertTrue(result.is_error)
        self.assertIn("cannot request user approval", result.content)


if __name__ == "__main__":
    unittest.main()
