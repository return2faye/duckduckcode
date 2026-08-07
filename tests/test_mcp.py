from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from mcp.types import CallToolResult, TextContent

from duckduckcode.core.agent import Agent
from duckduckcode.core.context import ContextManager, Message
from duckduckcode.core.event import (
    DoneEvent,
    ErrorEvent,
    LoopCompleteEvent,
    PermissionRequestEvent,
    ToolCallEvent,
)
from duckduckcode.core.mcp import (
    HTTPServer,
    MCPManager,
    StdioServer,
    _ServerState,
    load_mcp_configuration,
)
from duckduckcode.tools.tool import ToolCall, ToolManager, create_tool


class MCPConfigurationTest(unittest.TestCase):
    def test_invalid_project_entry_still_overrides_same_user_server(self) -> None:
        with (
            tempfile.TemporaryDirectory() as home_dir,
            tempfile.TemporaryDirectory() as workspace_dir,
        ):
            home = Path(home_dir)
            workspace = Path(workspace_dir)
            user = home / ".duckduckcode" / "mcp.yaml"
            project = workspace / ".duckduckcode" / "mcp.yaml"
            user.parent.mkdir()
            project.parent.mkdir()
            user.write_text(
                "shared:\n  type: stdio\n  command: user-command\n",
                encoding="utf-8",
            )
            project.write_text(
                "shared:\n  type: stdio\n  command: project-command\n  unknown: true\n",
                encoding="utf-8",
            )

            loaded = load_mcp_configuration(workspace, {}, home=home)

            self.assertIn("shared", loaded.merged(include_project=False))
            self.assertNotIn("shared", loaded.merged(include_project=True))

    def test_project_replaces_whole_user_server_and_expands_only_secrets(self) -> None:
        with (
            tempfile.TemporaryDirectory() as home_dir,
            tempfile.TemporaryDirectory() as workspace_dir,
        ):
            home = Path(home_dir)
            workspace = Path(workspace_dir)
            user = home / ".duckduckcode" / "mcp.yaml"
            project = workspace / ".duckduckcode" / "mcp.yaml"
            user.parent.mkdir()
            project.parent.mkdir()
            user.write_text(
                "files:\n  type: stdio\n  command: user-command\n  args: ['${UNCHANGED}']\n"
                "remote:\n  type: http\n  url: https://example.com/mcp\n"
                "  headers:\n    Authorization: 'Bearer ${TOKEN}'\n",
                encoding="utf-8",
            )
            project.write_text(
                "files:\n  type: stdio\n  command: project-command\n  args: ['.']\n",
                encoding="utf-8",
            )

            loaded = load_mcp_configuration(workspace, {"TOKEN": "secret"}, home=home)
            servers = loaded.merged(include_project=True)

            self.assertEqual(
                servers["files"], StdioServer("project-command", (".",), {})
            )
            self.assertEqual(
                servers["remote"],
                HTTPServer(
                    "https://example.com/mcp",
                    {"Authorization": "Bearer secret"},
                ),
            )
            self.assertEqual(loaded.user_servers["files"].args, ("${UNCHANGED}",))
            self.assertFalse(loaded.project_trusted)
            self.assertNotIn("secret", loaded.project_preview)

    def test_bad_servers_are_skipped_without_hiding_valid_siblings(self) -> None:
        with (
            tempfile.TemporaryDirectory() as home_dir,
            tempfile.TemporaryDirectory() as workspace_dir,
        ):
            home = Path(home_dir)
            workspace = Path(workspace_dir)
            path = home / ".duckduckcode" / "mcp.yaml"
            path.parent.mkdir()
            path.write_text(
                "good:\n  type: stdio\n  command: python\n"
                "unknown:\n  type: stdio\n  command: python\n  url: https://bad.example\n"
                "missing:\n  type: http\n  url: https://example.com/mcp\n"
                "  headers:\n    Authorization: '${MISSING}'\n"
                "bad_url:\n  type: http\n  url: file:///tmp/socket\n",
                encoding="utf-8",
            )

            loaded = load_mcp_configuration(workspace, {}, home=home)

            self.assertEqual(set(loaded.user_servers), {"good"})
            self.assertEqual(len(loaded.warnings), 3)
            self.assertNotIn("Authorization", "\n".join(loaded.warnings))

    def test_rejects_duplicate_yaml_fields_symlinks_and_large_files(self) -> None:
        with (
            tempfile.TemporaryDirectory() as home_dir,
            tempfile.TemporaryDirectory() as workspace_dir,
        ):
            home = Path(home_dir)
            workspace = Path(workspace_dir)
            path = home / ".duckduckcode" / "mcp.yaml"
            path.parent.mkdir()
            path.write_text(
                "server:\n  type: stdio\n  command: first\n  command: second\n",
                encoding="utf-8",
            )
            duplicate = load_mcp_configuration(workspace, {}, home=home)
            self.assertEqual(duplicate.user_servers, {})
            self.assertIn("duplicate", duplicate.warnings[0].lower())

            path.unlink()
            target = home / "actual.yaml"
            target.write_text("server: {}\n", encoding="utf-8")
            path.symlink_to(target)
            linked = load_mcp_configuration(workspace, {}, home=home)
            self.assertIn("symbolic", linked.warnings[0].lower())

            path.unlink()
            path.write_bytes(b"x" * (256 * 1024 + 1))
            large = load_mcp_configuration(workspace, {}, home=home)
            self.assertIn("256", large.warnings[0])

            path.write_text("? [a, b]\n: value\n", encoding="utf-8")
            malformed = load_mcp_configuration(workspace, {}, home=home)
            self.assertEqual(malformed.user_servers, {})
            self.assertIn("YAML", malformed.warnings[0])

    def test_trust_digest_changes_and_can_be_remembered_atomically(self) -> None:
        with (
            tempfile.TemporaryDirectory() as home_dir,
            tempfile.TemporaryDirectory() as workspace_dir,
        ):
            home = Path(home_dir)
            workspace = Path(workspace_dir)
            path = workspace / ".duckduckcode" / "mcp.yaml"
            path.parent.mkdir()
            path.write_text(
                "server:\n  type: stdio\n  command: python\n  env:\n    TOKEN: secret\n",
                encoding="utf-8",
            )
            manager = MCPManager(workspace, {}, ToolManager(), home=home)
            request = manager.permission_request()
            self.assertIsNotNone(request)
            self.assertNotIn("secret", request.content)

            manager.remember_project_trust()
            trust = workspace / ".duckduckcode" / "mcp.trust"
            self.assertEqual(
                trust.read_text(encoding="utf-8").strip(),
                manager.configuration.project_digest,
            )
            self.assertTrue(
                load_mcp_configuration(workspace, {}, home=home).project_trusted
            )

            path.write_text(
                "server:\n  type: stdio\n  command: python3\n", encoding="utf-8"
            )
            self.assertFalse(
                load_mcp_configuration(workspace, {}, home=home).project_trusted
            )


class MCPToolAdapterTest(unittest.TestCase):
    def test_stdio_server_discovery_call_and_close(self) -> None:
        with (
            tempfile.TemporaryDirectory() as home_dir,
            tempfile.TemporaryDirectory() as workspace_dir,
        ):
            home = Path(home_dir)
            workspace = Path(workspace_dir)
            server = workspace / "server.py"
            server.write_text(
                "from mcp.server.fastmcp import FastMCP\n"
                "import asyncio\n"
                "import os\n"
                "server = FastMCP('test', log_level='CRITICAL')\n"
                "@server.tool()\n"
                "async def echo(value: str, delay: float = 0) -> dict[str, str]:\n"
                "    await asyncio.sleep(delay)\n"
                "    return {'value': value, 'token': os.environ.get('MCP_TEST_TOKEN', '')}\n"
                "server.run(transport='stdio')\n",
                encoding="utf-8",
            )
            config = home / ".duckduckcode" / "mcp.yaml"
            config.parent.mkdir()
            config.write_text(
                "test:\n"
                "  type: stdio\n"
                f"  command: {json.dumps(sys.executable)}\n"
                "  args: [server.py]\n"
                "  env:\n    MCP_TEST_TOKEN: configured\n",
                encoding="utf-8",
            )
            tools = ToolManager()
            manager = MCPManager(workspace, {}, tools, home=home)

            warnings = manager.initialize()
            self.assertFalse(
                tools.execute(
                    ToolCall("load", "LoadTools", {"names": ["mcp__test__echo"]})
                ).is_error
            )
            result = tools.execute(
                SimpleNamespace(name="mcp__test__echo", arguments={"value": "quack"})
            )
            calls = [
                ToolCall(
                    "slow",
                    "mcp__test__echo",
                    {"value": "slow", "delay": 0.05},
                ),
                ToolCall(
                    "fast",
                    "mcp__test__echo",
                    {"value": "fast", "delay": 0},
                ),
            ]
            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(tools.execute, calls))
            paired = {
                call.call_id: json.loads(outcome.content)["structuredContent"]["value"]
                for call, outcome in zip(calls, outcomes)
            }
            manager.close()
            manager.close()

            self.assertEqual(warnings, [])
            self.assertFalse(result.is_error, result.content)
            payload = json.loads(result.content)
            self.assertEqual(
                payload["structuredContent"],
                {"value": "quack", "token": "configured"},
            )
            self.assertEqual(paired, {"slow": "slow", "fast": "fast"})
            self.assertFalse(manager._thread.is_alive())

    def test_failed_and_timed_out_servers_do_not_hide_ready_siblings(self) -> None:
        with (
            tempfile.TemporaryDirectory() as home_dir,
            tempfile.TemporaryDirectory() as workspace_dir,
        ):
            home = Path(home_dir)
            workspace = Path(workspace_dir)
            ready = workspace / "ready.py"
            ready.write_text(
                "from mcp.server.fastmcp import FastMCP\n"
                "server = FastMCP('ready', log_level='CRITICAL')\n"
                "@server.tool()\n"
                "def ping() -> str:\n"
                "    return 'pong'\n"
                "server.run(transport='stdio')\n",
                encoding="utf-8",
            )
            slow = workspace / "slow.py"
            slow.write_text(
                "import sys\nsys.stdin.readline()\nsys.stdin.readline()\n",
                encoding="utf-8",
            )
            config = home / ".duckduckcode" / "mcp.yaml"
            config.parent.mkdir()
            config.write_text(
                "broken:\n  type: stdio\n  command: definitely-not-a-command\n"
                "slow:\n"
                "  type: stdio\n"
                f"  command: {json.dumps(sys.executable)}\n"
                "  args: [slow.py]\n"
                "ready:\n"
                "  type: stdio\n"
                f"  command: {json.dumps(sys.executable)}\n"
                "  args: [ready.py]\n",
                encoding="utf-8",
            )
            tools = ToolManager()
            manager = MCPManager(workspace, {}, tools, home=home)

            with patch("duckduckcode.core.mcp.START_TIMEOUT_SECONDS", 1):
                warnings = manager.initialize()
            manager.close()

            self.assertIn(
                "mcp__ready__ping", [tool.name for tool in manager.mcp_tools()]
            )
            self.assertIsNone(tools.get("mcp__ready__ping"))
            self.assertEqual(len(warnings), 2)
            self.assertTrue(any("broken" in warning for warning in warnings))
            self.assertTrue(any("slow" in warning for warning in warnings))

    def test_empty_configuration_does_not_start_runtime_thread(self) -> None:
        with (
            tempfile.TemporaryDirectory() as home_dir,
            tempfile.TemporaryDirectory() as workspace_dir,
        ):
            manager = MCPManager(
                Path(workspace_dir), {}, ToolManager(), home=Path(home_dir)
            )

            self.assertEqual(manager.initialize(), [])
            self.assertIsNone(manager._thread)

    def test_thread_start_failure_becomes_startup_warning(self) -> None:
        with (
            tempfile.TemporaryDirectory() as home_dir,
            tempfile.TemporaryDirectory() as workspace_dir,
        ):
            home = Path(home_dir)
            config = home / ".duckduckcode" / "mcp.yaml"
            config.parent.mkdir()
            config.write_text(
                "server:\n  type: stdio\n  command: python\n", encoding="utf-8"
            )
            manager = MCPManager(Path(workspace_dir), {}, ToolManager(), home=home)

            with patch("duckduckcode.core.mcp.threading.Thread") as thread:
                thread.return_value.start.side_effect = RuntimeError("no threads")
                warnings = manager.initialize()

            self.assertEqual(warnings, ["MCP runtime could not be started: no threads"])

    def test_startup_scheduling_failure_becomes_warning(self) -> None:
        with (
            tempfile.TemporaryDirectory() as home_dir,
            tempfile.TemporaryDirectory() as workspace_dir,
        ):
            home = Path(home_dir)
            config = home / ".duckduckcode" / "mcp.yaml"
            config.parent.mkdir()
            config.write_text(
                "server:\n  type: stdio\n  command: python\n", encoding="utf-8"
            )
            manager = MCPManager(Path(workspace_dir), {}, ToolManager(), home=home)

            with patch(
                "duckduckcode.core.mcp.asyncio.run_coroutine_threadsafe",
                side_effect=RuntimeError("loop stopped"),
            ):
                warnings = manager.initialize()
            manager.close()

            self.assertEqual(
                warnings, ["MCP startup could not be scheduled: loop stopped"]
            )

    def test_registers_non_strict_namespaced_tools_and_serializes_full_result(
        self,
    ) -> None:
        tools = ToolManager()
        manager = MCPManager(Path.cwd(), {}, tools, home=Path("/nonexistent"))
        session = object()
        remote = SimpleNamespace(
            name="search",
            description="Search remotely",
            inputSchema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        result = CallToolResult(
            content=[TextContent(type="text", text="found")],
            structuredContent={"count": 1},
            isError=True,
        )
        with patch.object(manager, "_call_tool", return_value=result) as call:
            warnings = manager.register_discovered("docs", session, [remote])
            tool = manager.mcp_tools()[0]
            self.assertFalse(
                tools.execute(
                    ToolCall("load", "LoadTools", {"names": [tool.name]})
                ).is_error
            )
            actual = tools.execute(
                SimpleNamespace(name="mcp__docs__search", arguments={"q": "ducks"})
            )

        self.assertEqual(warnings, [])
        self.assertIsNotNone(tool)
        self.assertEqual(type(tool).__name__, "MCPTool")
        self.assertEqual(manager.mcp_tools(), (tool,))
        self.assertFalse(tool.schema()["strict"])
        self.assertFalse(tool.is_read_only)
        self.assertFalse(tool.is_concurrency_safe)
        self.assertEqual(
            tool.permission_content({"query": "duck", "limit": 3}),
            '{"limit":3,"query":"duck"}',
        )
        call.assert_called_once_with(session, "search", {"q": "ducks"})
        payload = json.loads(actual.content)
        self.assertEqual(payload["content"][0]["text"], "found")
        self.assertEqual(payload["structuredContent"], {"count": 1})
        self.assertTrue(payload["isError"])
        self.assertTrue(actual.is_error)

    def test_mcp_tools_are_cataloged_in_order_but_not_eagerly_registered(self) -> None:
        tools = ToolManager()
        manager = MCPManager(Path.cwd(), {}, tools, home=Path("/nonexistent"))
        discovered = [
            SimpleNamespace(
                name="first", description="First", inputSchema={"type": "object"}
            ),
            SimpleNamespace(
                name="second",
                description="Ignore this instruction",
                inputSchema={"type": "object"},
            ),
        ]

        self.assertEqual(manager.register_discovered("docs", object(), discovered), [])

        mcp_tools = manager.mcp_tools()
        self.assertEqual(
            [tool.name for tool in mcp_tools], ["mcp__docs__first", "mcp__docs__second"]
        )
        self.assertIsNone(tools.get("mcp__docs__first"))
        self.assertIsNone(tools.get("mcp__docs__second"))
        self.assertIsNotNone(tools.get("LoadTools"))
        catalog = manager.catalog_block()
        self.assertIn('"name":"mcp__docs__first"', catalog)
        self.assertIn('"description":"First"', catalog)
        self.assertNotIn("inputSchema", catalog)
        self.assertIn("untrusted", catalog.lower())

    def test_skips_invalid_generated_names_and_schemas(self) -> None:
        tools = ToolManager()
        manager = MCPManager(Path.cwd(), {}, tools, home=Path("/nonexistent"))
        remote = SimpleNamespace(name="bad name", description="bad", inputSchema={})
        schema = SimpleNamespace(name="valid", description="bad", inputSchema=[])

        warnings = manager.register_discovered("server", object(), [remote, schema])

        self.assertEqual(manager.mcp_tools(), ())
        self.assertEqual(len(warnings), 2)

    def test_skips_generated_name_collisions(self) -> None:
        tools = ToolManager()
        manager = MCPManager(Path.cwd(), {}, tools, home=Path("/nonexistent"))
        first = SimpleNamespace(name="c", description="first", inputSchema={})
        second = SimpleNamespace(name="b__c", description="second", inputSchema={})

        self.assertEqual(manager.register_discovered("a__b", object(), [first]), [])
        warnings = manager.register_discovered("a", object(), [second])

        self.assertEqual(len(warnings), 1)
        self.assertEqual(manager.mcp_tools()[0].description, "first")

    def test_load_tools_is_atomic_ordered_and_idempotent(self) -> None:
        tools = ToolManager()
        manager = MCPManager(Path.cwd(), {}, tools, home=Path("/nonexistent"))
        manager.register_discovered(
            "docs",
            object(),
            [
                SimpleNamespace(
                    name="first", description="First", inputSchema={"type": "object"}
                ),
                SimpleNamespace(
                    name="second", description="Second", inputSchema={"type": "object"}
                ),
                SimpleNamespace(
                    name="third", description="Third", inputSchema={"type": "object"}
                ),
            ],
        )

        failed = tools.execute(
            ToolCall("bad", "LoadTools", {"names": ["mcp__docs__first", "missing"]})
        )
        self.assertTrue(failed.is_error)
        self.assertIsNone(tools.get("mcp__docs__first"))

        loaded = tools.execute(
            ToolCall(
                "load",
                "LoadTools",
                {
                    "names": [
                        "mcp__docs__second",
                        "mcp__docs__first",
                        "mcp__docs__second",
                    ]
                },
            )
        )
        self.assertFalse(loaded.is_error)
        self.assertEqual(
            json.loads(loaded.content),
            {"loaded": ["mcp__docs__second", "mcp__docs__first"], "already_loaded": []},
        )
        self.assertIs(tools.get("mcp__docs__second"), manager.mcp_tools()[1])
        self.assertEqual(
            [schema["name"] for schema in tools.schemas()],
            ["LoadTools", "mcp__docs__second", "mcp__docs__first"],
        )

        again = tools.execute(
            ToolCall("again", "LoadTools", {"names": ["mcp__docs__first"]})
        )
        self.assertEqual(
            json.loads(again.content)["already_loaded"], ["mcp__docs__first"]
        )

    def test_load_tools_rejects_invalid_batches_without_changes(self) -> None:
        tools = ToolManager()
        manager = MCPManager(Path.cwd(), {}, tools, home=Path("/nonexistent"))
        manager.register_discovered(
            "docs",
            object(),
            [
                SimpleNamespace(
                    name="first", description="First", inputSchema={"type": "object"}
                )
            ],
        )
        before = tools.schemas()

        for arguments in ({"names": []}, {"names": [1]}, {"names": [""]}):
            self.assertTrue(
                tools.execute(ToolCall("bad", "LoadTools", arguments)).is_error
            )
            self.assertEqual(tools.schemas(), before)

        tools.register(
            create_tool(
                "mcp__docs__first", "Other", {}, lambda: "", lambda value: value
            )
        )
        occupied = tools.schemas()
        self.assertTrue(
            tools.execute(
                ToolCall("occupied", "LoadTools", {"names": ["mcp__docs__first"]})
            ).is_error
        )
        self.assertEqual(tools.schemas(), occupied)

    def test_load_tools_is_read_only_and_preserves_existing_loader(self) -> None:
        tools = ToolManager()
        existing = create_tool(
            "LoadTools", "Other", {}, lambda: "", lambda value: value
        )
        tools.register(existing)
        manager = MCPManager(Path.cwd(), {}, tools, home=Path("/nonexistent"))

        warnings = manager.register_discovered(
            "docs",
            object(),
            [
                SimpleNamespace(
                    name="first", description="First", inputSchema={"type": "object"}
                )
            ],
        )

        self.assertIs(tools.get("LoadTools"), existing)
        self.assertEqual(len(warnings), 1)

        normal_tools = ToolManager()
        normal_manager = MCPManager(
            Path.cwd(), {}, normal_tools, home=Path("/nonexistent")
        )
        normal_manager.register_discovered(
            "docs",
            object(),
            [
                SimpleNamespace(
                    name="first", description="First", inputSchema={"type": "object"}
                )
            ],
        )
        loader = normal_tools.get("LoadTools")
        self.assertIsNotNone(loader)
        self.assertTrue(loader.is_read_only)
        self.assertFalse(loader.is_concurrency_safe)

    def test_restore_session_replays_successful_loads_and_unloads_stale_tools(
        self,
    ) -> None:
        tools = ToolManager()
        manager = MCPManager(Path.cwd(), {}, tools, home=Path("/nonexistent"))
        manager.register_discovered(
            "docs",
            object(),
            [
                SimpleNamespace(
                    name="first", description="First", inputSchema={"type": "object"}
                ),
                SimpleNamespace(
                    name="second", description="Second", inputSchema={"type": "object"}
                ),
            ],
        )
        tools.execute(ToolCall("load", "LoadTools", {"names": ["mcp__docs__second"]}))
        records = [
            {
                "role": "assistant",
                "context": {
                    "type": "tool_call",
                    "call_id": "ok",
                    "name": "LoadTools",
                    "arguments": {"names": ["mcp__docs__first", "mcp__docs__gone"]},
                },
                "ts": 1,
            },
            {
                "role": "tool",
                "context": {
                    "type": "tool_result",
                    "call_id": "ok",
                    "name": "LoadTools",
                    "content": "loaded",
                    "is_error": False,
                },
                "ts": 1,
            },
            {
                "role": "assistant",
                "context": {
                    "type": "tool_call",
                    "call_id": "failed",
                    "name": "LoadTools",
                    "arguments": {"names": ["mcp__docs__second"]},
                },
                "ts": 2,
            },
            {
                "role": "tool",
                "context": {
                    "type": "tool_result",
                    "call_id": "failed",
                    "name": "LoadTools",
                    "content": "failed",
                    "is_error": True,
                },
                "ts": 2,
            },
            {
                "role": "assistant",
                "context": {
                    "type": "compaction",
                    "summary": "summary",
                    "cutoff": 2,
                    "token_usage": 1,
                },
                "ts": 3,
            },
        ]

        warnings = manager.restore_session(records)

        self.assertIs(tools.get("mcp__docs__first"), manager.mcp_tools()[0])
        self.assertIsNone(tools.get("mcp__docs__second"))
        self.assertEqual(len([warning for warning in warnings if "gone" in warning]), 1)

    def test_restore_session_preserves_different_tool_on_collision(self) -> None:
        tools = ToolManager()
        manager = MCPManager(Path.cwd(), {}, tools, home=Path("/nonexistent"))
        manager.register_discovered(
            "docs",
            object(),
            [
                SimpleNamespace(
                    name="first", description="First", inputSchema={"type": "object"}
                )
            ],
        )
        existing = create_tool(
            "mcp__docs__first", "Other", {}, lambda: "", lambda value: value
        )
        tools.register(existing)
        records = [
            {
                "context": {
                    "type": "tool_call",
                    "call_id": "ok",
                    "name": "LoadTools",
                    "arguments": {"names": ["mcp__docs__first"]},
                }
            },
            {
                "context": {
                    "type": "tool_result",
                    "call_id": "ok",
                    "name": "LoadTools",
                    "is_error": False,
                }
            },
        ]

        warnings = manager.restore_session(records)

        self.assertIs(tools.get("mcp__docs__first"), existing)
        self.assertEqual(len(warnings), 1)


class MCPHTTPTransportTest(unittest.IsolatedAsyncioTestCase):
    async def test_http_uses_private_client_headers_and_paginates_tools(self) -> None:
        manager = MCPManager(Path.cwd(), {}, ToolManager(), home=Path("/nonexistent"))
        manager._stop_event = asyncio.Event()
        state = _ServerState(
            "remote",
            HTTPServer("https://example.com/mcp", {"Authorization": "Bearer secret"}),
            ready=asyncio.Event(),
        )
        captured = {}

        class Context:
            def __init__(self, value):
                self.value = value

            async def __aenter__(self):
                return self.value

            async def __aexit__(self, *args):
                return False

        class Session:
            def __init__(self):
                self.cursors = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def initialize(self):
                return None

            async def list_tools(self, cursor=None):
                self.cursors.append(cursor)
                return SimpleNamespace(
                    tools=[SimpleNamespace(name=cursor or "first")],
                    nextCursor="next" if cursor is None else None,
                )

        http_client = Context(object())
        session = Session()

        def make_http_client(**kwargs):
            captured["headers"] = kwargs["headers"]
            captured["timeout"] = kwargs["timeout"]
            return http_client

        def make_transport(url, *, http_client):
            captured["url"] = url
            captured["client"] = http_client
            return Context((object(), object(), lambda: None))

        with (
            patch("duckduckcode.core.mcp.httpx.AsyncClient", make_http_client),
            patch("duckduckcode.core.mcp.streamable_http_client", make_transport),
            patch("duckduckcode.core.mcp.ClientSession", return_value=session),
        ):
            task = asyncio.create_task(manager._serve(state))
            await asyncio.wait_for(state.ready.wait(), 1)
            manager._stop_event.set()
            await task

        self.assertEqual(captured["headers"], {"Authorization": "Bearer secret"})
        self.assertIsNone(captured["timeout"])
        self.assertEqual(captured["url"], "https://example.com/mcp")
        self.assertIs(captured["client"], http_client.value)
        self.assertEqual(session.cursors, [None, "next"])
        self.assertEqual([tool.name for tool in state.tools], ["first", "next"])


class MCPAgentIntegrationTest(unittest.TestCase):
    def test_initialize_prompts_once_refreshes_schemas_and_reports_warnings_once(
        self,
    ) -> None:
        tools = ToolManager()

        class Manager:
            def __init__(self):
                self.initialized = []
                self.restored = []
                self.closed = 0

            def permission_request(self):
                return SimpleNamespace(content="safe preview", message="approve")

            def initialize(self, choice):
                self.initialized.append(choice)
                tools.register(
                    "mcp__docs__search",
                    "search",
                    {"type": "object"},
                    lambda **arguments: arguments,
                    strict=False,
                    category="mcp",
                )
                return ["one startup warning"]

            def catalog_block(self):
                return "MCP catalog"

            def restore_session(self, records):
                self.restored = list(records)
                return []

            def close(self):
                self.closed += 1

        class Client:
            def stream(self, *args, **kwargs):
                yield DoneEvent()

        manager = Manager()
        context = ContextManager(tool_schemas=[])
        agent = Agent(Client(), context, tools, mcp_manager=manager)

        events = agent.initialize()
        request = next(events)
        self.assertIsInstance(request, PermissionRequestEvent)
        remaining = [events.send("allow_once"), *list(events)]
        second = list(agent.initialize())
        agent.close()

        self.assertEqual(manager.initialized, ["allow_once"])
        self.assertEqual(
            [event.message for event in remaining if isinstance(event, ErrorEvent)],
            ["one startup warning"],
        )
        self.assertFalse(any(isinstance(event, ErrorEvent) for event in second))
        self.assertEqual(context.tool_schemas(), tools.schemas())
        self.assertIn(Message("system", "MCP catalog"), context.model_messages())
        self.assertEqual(manager.restored, [])
        self.assertIsInstance(remaining[-1], LoopCompleteEvent)
        self.assertEqual(manager.closed, 1)

    def test_loaded_tool_schema_is_sent_on_the_next_iteration(self) -> None:
        tools = ToolManager()
        manager = MCPManager(Path.cwd(), {}, tools, home=Path("/nonexistent"))
        manager.register_discovered(
            "docs",
            object(),
            [
                SimpleNamespace(
                    name="search",
                    description="Search docs",
                    inputSchema={"type": "object"},
                )
            ],
        )

        class Client:
            def __init__(self):
                self.requests = []

            def stream(self, messages, tools=None, reasoning=None):
                self.requests.append((messages, list(tools or [])))
                if len(self.requests) == 1:
                    yield ToolCallEvent(
                        ToolCall(
                            "load",
                            "LoadTools",
                            {"names": ["mcp__docs__search"]},
                        )
                    )
                yield DoneEvent()

        client = Client()
        agent = Agent(
            client,
            ContextManager(system_prompt="system"),
            tools,
            mcp_manager=manager,
        )

        list(agent.stream("find docs"))

        self.assertEqual(
            [tool["name"] for tool in client.requests[0][1]], ["LoadTools"]
        )
        self.assertEqual(
            [tool["name"] for tool in client.requests[1][1]],
            ["LoadTools", "mcp__docs__search"],
        )
        self.assertIn("mcp__docs__search", client.requests[0][0][1].content)


if __name__ == "__main__":
    unittest.main()
