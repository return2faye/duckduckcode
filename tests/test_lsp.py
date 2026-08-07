from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from duckduckcode.core.lsp import (
    MAX_MESSAGE_BYTES,
    LSPManager,
    _end_position,
    _lsp_position,
    _read_frame,
    load_lsp_configuration,
)
from duckduckcode.tools.lsp import create_lsp_tool
from duckduckcode.tools.tool import ToolCall, ToolManager, ToolResult
from duckduckcode.permissions import PermissionChecker


class LSPConfigurationTest(unittest.TestCase):
    def _write(self, home: Path, content: str) -> Path:
        path = home / ".duckduckcode" / "lsp.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_multiple_servers_and_expands_only_configured_environment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self._write(
                home,
                "pyright:\n"
                "  command: pyright-langserver\n"
                "  args: [--stdio]\n"
                "  extensions: {'.py': python, '.pyi': python}\n"
                "  env: {PYTHONPATH: '${PROJECT_PATH}'}\n"
                "  initialization_options: {python: {analysis: {indexing: true}}}\n"
                "  settings: {python: {analysis: {typeCheckingMode: strict}}}\n"
                "rust:\n"
                "  command: rust-analyzer\n"
                "  extensions: {'.rs': rust}\n",
            )

            configuration = load_lsp_configuration(
                {"PROJECT_PATH": "/workspace/src", "UNUSED": "secret"}, home=home
            )

            self.assertEqual(list(configuration.servers), ["pyright", "rust"])
            pyright = configuration.servers["pyright"]
            self.assertEqual(pyright.command, "pyright-langserver")
            self.assertEqual(pyright.args, ("--stdio",))
            self.assertEqual(
                dict(pyright.extensions), {".py": "python", ".pyi": "python"}
            )
            self.assertEqual(dict(pyright.env), {"PYTHONPATH": "/workspace/src"})
            self.assertEqual(configuration.warnings, ())

    def test_empty_config_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self._write(home, "")

            configuration = load_lsp_configuration({}, home=home)

            self.assertEqual(configuration.servers, {})
            self.assertEqual(configuration.warnings, ())

    def test_bad_server_is_skipped_without_hiding_valid_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self._write(
                home,
                "bad:\n"
                "  command: server\n"
                "  extensions: {'.x': x}\n"
                "  unknown: true\n"
                "good:\n"
                "  command: server\n"
                "  extensions: {'.ok': ok}\n",
            )

            configuration = load_lsp_configuration({}, home=home)

            self.assertEqual(list(configuration.servers), ["good"])
            self.assertEqual(len(configuration.warnings), 1)
            self.assertIn("unknown field(s): unknown", configuration.warnings[0])

    def test_missing_environment_skips_server_without_exposing_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self._write(
                home,
                "pyright:\n"
                "  command: server\n"
                "  extensions: {'.py': python}\n"
                "  env: {TOKEN: 'prefix-${MISSING}-secret'}\n",
            )

            configuration = load_lsp_configuration({}, home=home)

            self.assertEqual(configuration.servers, {})
            self.assertIn("MISSING", configuration.warnings[0])
            self.assertNotIn("secret", configuration.warnings[0])

    def test_rejects_duplicate_yaml_keys_symlinks_and_oversize_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, prepare, expected in (
                (
                    "duplicate",
                    lambda home: self._write(
                        home,
                        "server:\n  command: one\n  command: two\n  extensions: {'.x': x}\n",
                    ),
                    "duplicate YAML key",
                ),
                (
                    "symlink",
                    lambda home: (home / ".duckduckcode" / "lsp.yaml").symlink_to(
                        self._write(root / "target", "{}\n")
                    ),
                    "symbolic links",
                ),
                (
                    "large",
                    lambda home: self._write(home, "x" * (256 * 1024 + 1)),
                    "256 KiB",
                ),
            ):
                with self.subTest(name=name):
                    home = root / name
                    (home / ".duckduckcode").mkdir(parents=True)
                    prepare(home)
                    configuration = load_lsp_configuration({}, home=home)
                    self.assertEqual(configuration.servers, {})
                    self.assertIn(expected, configuration.warnings[0])

    def test_duplicate_field_skips_only_its_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self._write(
                home,
                "bad:\n"
                "  command: one\n"
                "  command: two\n"
                "  extensions: {'.bad': bad}\n"
                "good:\n"
                "  command: server\n"
                "  extensions: {'.ok': ok}\n",
            )

            configuration = load_lsp_configuration({}, home=home)

            self.assertEqual(list(configuration.servers), ["good"])
            self.assertEqual(len(configuration.warnings), 1)
            self.assertIn("bad", configuration.warnings[0])
            self.assertIn("duplicate YAML key", configuration.warnings[0])

    def test_rejects_non_json_options_and_invalid_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self._write(
                home,
                "bad_options:\n"
                "  command: server\n"
                "  extensions: {'.x': x}\n"
                "  settings: {.nan}\n"
                "bad_extensions:\n"
                "  command: server\n"
                "  extensions: {py: python}\n",
            )

            configuration = load_lsp_configuration({}, home=home)

            self.assertEqual(configuration.servers, {})
            self.assertEqual(len(configuration.warnings), 2)
            self.assertTrue(
                any("JSON-compatible" in item for item in configuration.warnings)
            )
            self.assertTrue(
                any("start with '.'" in item for item in configuration.warnings)
            )

    def test_rejects_non_string_json_object_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self._write(
                home,
                "bad:\n"
                "  command: server\n"
                "  extensions: {'.x': x}\n"
                "  settings: {1: value}\n",
            )

            configuration = load_lsp_configuration({}, home=home)

            self.assertEqual(configuration.servers, {})
            self.assertIn("JSON-compatible", configuration.warnings[0])


class LSPToolTest(unittest.TestCase):
    def _manager(self, workspace: Path, home: Path) -> LSPManager:
        path = home / ".duckduckcode" / "lsp.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(
            "pyright:\n"
            "  command: unused\n"
            "  extensions: {'.py': python, '.pyi': python}\n"
            "rust:\n"
            "  command: unused\n"
            "  extensions: {'.rs': rust}\n",
            encoding="utf-8",
        )
        return LSPManager(workspace, {}, home=home)

    def test_schema_is_strict_read_only_and_lists_servers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = self._manager(root / "workspace", root / "home")
            tool = create_lsp_tool(manager)

            self.assertEqual(tool.name, "LSP")
            self.assertTrue(tool.strict)
            self.assertTrue(tool.is_read_only)
            self.assertTrue(tool.is_concurrency_safe)
            self.assertEqual(
                tool.params["properties"]["server"]["enum"], ["pyright", "rust"]
            )
            self.assertIn(".py", tool.description)
            self.assertIn(".rs", tool.description)

    def test_validator_enforces_operation_argument_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = self._manager(root / "workspace", root / "home")
            tools = ToolManager()
            tools.register(create_lsp_tool(manager))
            valid = {
                "operation": "hover",
                "server": "pyright",
                "path": "sample.py",
                "line": 1,
                "column": 1,
                "query": None,
                "include_declaration": None,
            }

            with self.subTest("missing required key"):
                result = tools.execute(ToolCall("1", "LSP", {**valid, "line": None}))
                self.assertTrue(result.is_error)
                self.assertIn("line", result.content)
            with self.subTest("workspace query"):
                result = tools.execute(
                    ToolCall(
                        "2",
                        "LSP",
                        {
                            **valid,
                            "operation": "workspace_symbols",
                            "path": None,
                            "line": None,
                            "column": None,
                            "query": "",
                        },
                    )
                )
                self.assertTrue(result.is_error)
                self.assertIn("query", result.content)
            with self.subTest("unknown server"):
                result = tools.execute(
                    ToolCall("3", "LSP", {**valid, "server": "missing"})
                )
                self.assertTrue(result.is_error)
                self.assertIn("server", result.content)
            with self.subTest("irrelevant arguments"):
                result = tools.execute(
                    ToolCall(
                        "4",
                        "LSP",
                        {
                            **valid,
                            "operation": "document_symbols",
                            "line": 1,
                            "column": None,
                        },
                    )
                )
                self.assertTrue(result.is_error)
                self.assertIn("only requires path", result.content)

    def test_path_requests_reject_escape_directory_and_non_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            manager = self._manager(workspace, root / "home")
            tools = ToolManager()
            tools.register(create_lsp_tool(manager))
            arguments = {
                "operation": "document_symbols",
                "server": "pyright",
                "path": "../outside.py",
                "line": None,
                "column": None,
                "query": None,
                "include_declaration": None,
            }

            outside = root / "outside.py"
            outside.write_text("x = 1", encoding="utf-8")
            escaped = tools.execute(ToolCall("1", "LSP", arguments))
            self.assertTrue(escaped.is_error)
            self.assertIn("workspace", escaped.content)

            binary = workspace / "binary.py"
            binary.write_bytes(b"\xff")
            invalid = tools.execute(
                ToolCall("2", "LSP", {**arguments, "path": "binary.py"})
            )
            self.assertTrue(invalid.is_error)
            self.assertIn("UTF-8", invalid.content)

    def test_tool_returns_raw_json_result(self) -> None:
        class Manager:
            server_names = ("test",)
            server_descriptions = ("test: .py → python",)

            def execute(self, **arguments):
                self.arguments = arguments
                return {"uri": "file:///tmp/a.py", "range": {"start": {"line": 0}}}

        manager = Manager()
        tool = create_lsp_tool(manager)
        arguments = {
            "operation": "definition",
            "server": "test",
            "path": "/tmp/a.py",
            "line": 1,
            "column": 1,
            "query": None,
            "include_declaration": None,
        }

        result = tool.execute(arguments)

        self.assertEqual(json.loads(result.content)["uri"], "file:///tmp/a.py")
        self.assertEqual(manager.arguments, arguments)
        self.assertEqual(result, ToolResult(result.content))

    def test_read_only_tool_is_allowed_in_plan_mode(self) -> None:
        class Manager:
            server_names = ("test",)
            server_descriptions = ("test: .py → python",)

            def execute(self, **arguments):
                return None

        tool = create_lsp_tool(Manager())
        call = ToolCall(
            "lsp",
            "LSP",
            {
                "operation": "workspace_symbols",
                "server": "test",
                "path": None,
                "line": None,
                "column": None,
                "query": "symbol",
                "include_declaration": None,
            },
        )

        decision = PermissionChecker().check(call, tool=tool, plan_file=Path("plan.md"))

        self.assertEqual(decision.action, "allow")


FAKE_SERVER = r"""
import json
import os
from pathlib import Path
import sys
import threading

log_path = Path(sys.argv[1])
write_lock = threading.Lock()
pending_symbol = None

def log(value):
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, separators=(",", ":")) + "\n")

def read():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            raise EOFError
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode("ascii").split(":", 1)
        headers[key.lower()] = value.strip()
    return json.loads(sys.stdin.buffer.read(int(headers["content-length"])))

def send(value):
    data = json.dumps(value, separators=(",", ":")).encode()
    with write_lock:
        sys.stdout.buffer.write(b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data)
        sys.stdout.buffer.flush()

log({"serverPid": os.getpid(), "cwd": os.getcwd(), "environment": dict(os.environ)})
if os.environ.get("LSP_FAKE_MODE") == "stubborn":
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
try:
    while True:
        message = read()
        log(message)
        method = message.get("method")
        request_id = message.get("id")
        if method == "initialize":
            if os.environ.get("LSP_FAKE_MODE") == "exit":
                raise SystemExit(2)
            if os.environ.get("LSP_FAKE_MODE") == "hang":
                continue
            send({"jsonrpc":"2.0","id":request_id,"result":{"capabilities":{"positionEncoding":os.environ.get("LSP_ENCODING", "utf-16"),"textDocumentSync":{"openClose":True,"change":2}}}})
            send({"jsonrpc":"2.0","id":"server-config","method":"workspace/configuration","params":{"items":[{"section":"python.analysis"}]}})
            send({"jsonrpc":"2.0","id":"server-folders","method":"workspace/workspaceFolders","params":None})
            send({"jsonrpc":"2.0","id":"server-edit","method":"workspace/applyEdit","params":{"edit":{"changes":{}}}})
            send({"jsonrpc":"2.0","id":"server-register","method":"client/registerCapability","params":{"registrations":[]}})
            send({"jsonrpc":"2.0","id":"server-progress","method":"window/workDoneProgress/create","params":{"token":"work"}})
            send({"jsonrpc":"2.0","id":"server-unknown","method":"custom/unknown","params":{}})
        elif method in ("initialized", "workspace/didChangeConfiguration", "textDocument/didOpen", "textDocument/didChange", "textDocument/didClose", "$/cancelRequest", "exit"):
            if method == "exit":
                if os.environ.get("LSP_FAKE_MODE") != "stubborn":
                    break
        elif method == "shutdown":
            send({"jsonrpc":"2.0","id":request_id,"result":None})
        elif method == "workspace/symbol" and message["params"]["query"] == "timeout":
            continue
        elif method == "workspace/symbol" and message["params"]["query"] == "error":
            send({"jsonrpc":"2.0","id":request_id,"error":{"code":-32001,"message":"fake error"}})
        elif method == "workspace/symbol" and message["params"]["query"] in ("first", "second"):
            if pending_symbol is None:
                pending_symbol = message
            else:
                send({"jsonrpc":"2.0","id":request_id,"result":{"query":message["params"]["query"]}})
                send({"jsonrpc":"2.0","id":pending_symbol["id"],"result":{"query":pending_symbol["params"]["query"]}})
                pending_symbol = None
        elif request_id is not None and method is not None:
            send({"jsonrpc":"2.0","id":request_id,"result":{"method":method,"params":message.get("params")}})
except EOFError:
    pass
"""


class LSPProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.home = self.root / "home"
        self.workspace.mkdir()
        self.server_script = self.root / "fake_lsp.py"
        self.server_script.write_text(FAKE_SERVER, encoding="utf-8")
        self.log = self.root / "lsp.log"

    def _manager(
        self, *, env: dict[str, str] | None = None, extra: str = ""
    ) -> LSPManager:
        config = self.home / ".duckduckcode" / "lsp.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            "test:\n"
            f"  command: {json.dumps(sys.executable)}\n"
            f"  args: [{json.dumps(str(self.server_script))}, {json.dumps(str(self.log))}]\n"
            "  extensions: {'.py': python}\n"
            "  env:\n"
            f"    LSP_FAKE_MODE: {json.dumps((env or {}).get('LSP_FAKE_MODE', 'normal'))}\n"
            f"    LSP_ENCODING: {json.dumps((env or {}).get('LSP_ENCODING', 'utf-16'))}\n"
            "  settings: {python: {analysis: {typeCheckingMode: strict}}}\n" + extra,
            encoding="utf-8",
        )
        return LSPManager(
            self.workspace,
            {"PATH": os.environ.get("PATH", ""), "UNSAFE_SECRET": "hidden"},
            home=self.home,
        )

    def _messages(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_lazy_handshake_position_sync_five_operations_and_close(self) -> None:
        source = self.workspace / "sample.py"
        source.write_text("😀thing\n", encoding="utf-8")
        manager = self._manager()
        self.addCleanup(manager.close)

        self.assertEqual(manager.initialize(), [])
        self.assertIsNone(manager._thread)

        hover = manager.execute(
            operation="hover",
            server="test",
            path="sample.py",
            line=1,
            column=2,
            query=None,
            include_declaration=None,
        )
        self.assertEqual(hover["method"], "textDocument/hover")
        self.assertEqual(hover["params"]["position"], {"line": 0, "character": 2})

        source.write_text("😀changed\n", encoding="utf-8")
        symbols = manager.execute(
            operation="document_symbols",
            server="test",
            path=str(source),
            line=None,
            column=None,
            query=None,
            include_declaration=None,
        )
        self.assertEqual(symbols["method"], "textDocument/documentSymbol")

        for operation, method in (
            ("definition", "textDocument/definition"),
            ("references", "textDocument/references"),
        ):
            result = manager.execute(
                operation=operation,
                server="test",
                path="sample.py",
                line=1,
                column=1,
                query=None,
                include_declaration=False,
            )
            self.assertEqual(result["method"], method)
        workspace = manager.execute(
            operation="workspace_symbols",
            server="test",
            path=None,
            line=None,
            column=None,
            query="Thing",
            include_declaration=None,
        )
        self.assertEqual(workspace["method"], "workspace/symbol")

        manager.close()
        messages = self._messages()
        startup = next(item for item in messages if "serverPid" in item)
        self.assertEqual(startup["cwd"], str(self.workspace.resolve()))
        self.assertIn("PATH", startup["environment"])
        self.assertNotIn("UNSAFE_SECRET", startup["environment"])
        self.assertEqual(startup["environment"]["LSP_FAKE_MODE"], "normal")
        initialize = next(
            item for item in messages if item.get("method") == "initialize"
        )
        self.assertEqual(
            initialize["params"]["rootUri"], self.workspace.resolve().as_uri()
        )
        self.assertEqual(
            initialize["params"]["capabilities"]["general"]["positionEncodings"],
            ["utf-8", "utf-16", "utf-32"],
        )
        self.assertTrue(
            any(item.get("method") == "textDocument/didOpen" for item in messages)
        )
        change = next(
            item for item in messages if item.get("method") == "textDocument/didChange"
        )
        self.assertIn("range", change["params"]["contentChanges"][0])
        config_response = next(
            item
            for item in messages
            if item.get("id") == "server-config" and "method" not in item
        )
        self.assertEqual(config_response["result"], [{"typeCheckingMode": "strict"}])
        folders = next(
            item
            for item in messages
            if item.get("id") == "server-folders" and "method" not in item
        )
        self.assertEqual(folders["result"][0]["uri"], self.workspace.resolve().as_uri())
        edit = next(
            item
            for item in messages
            if item.get("id") == "server-edit" and "method" not in item
        )
        self.assertFalse(edit["result"]["applied"])
        self.assertTrue(
            any(
                item.get("id") == "server-register"
                and item.get("result", "missing") is None
                for item in messages
            )
        )
        self.assertTrue(
            any(
                item.get("id") == "server-progress"
                and item.get("result", "missing") is None
                for item in messages
            )
        )
        unknown = next(
            item
            for item in messages
            if item.get("id") == "server-unknown" and "method" not in item
        )
        self.assertEqual(unknown["error"]["code"], -32601)
        self.assertTrue(
            any(item.get("method") == "textDocument/didClose" for item in messages)
        )
        self.assertTrue(any(item.get("method") == "shutdown" for item in messages))
        self.assertTrue(any(item.get("method") == "exit" for item in messages))

    def test_utf8_utf16_and_utf32_convert_non_bmp_columns(self) -> None:
        (self.workspace / "sample.py").write_text("😀x", encoding="utf-8")
        for encoding, character in (("utf-8", 4), ("utf-16", 2), ("utf-32", 1)):
            with self.subTest(encoding=encoding):
                log = self.root / f"{encoding}.log"
                self.log = log
                manager = self._manager(env={"LSP_ENCODING": encoding})
                try:
                    result = manager.execute(
                        operation="definition",
                        server="test",
                        path="sample.py",
                        line=1,
                        column=2,
                        query=None,
                        include_declaration=None,
                    )
                    self.assertEqual(
                        result["params"]["position"]["character"], character
                    )
                finally:
                    manager.close()

    def test_only_cr_and_lf_are_lsp_line_breaks(self) -> None:
        text = "a\u2028b"

        self.assertEqual(
            _lsp_position(text, 1, 4, "utf-16"), {"line": 0, "character": 3}
        )
        self.assertEqual(_end_position(text, "utf-16"), {"line": 0, "character": 3})

    def test_concurrent_requests_are_paired_by_id(self) -> None:
        manager = self._manager()
        self.addCleanup(manager.close)
        tools = ToolManager()
        tools.register(create_lsp_tool(manager))

        calls = [
            ToolCall(
                name,
                "LSP",
                {
                    "operation": "workspace_symbols",
                    "server": "test",
                    "path": None,
                    "line": None,
                    "column": None,
                    "query": name,
                    "include_declaration": None,
                },
            )
            for name in ("first", "second")
        ]
        results = dict(
            (call.call_id, json.loads(result.content))
            for call, result in tools.execute_many(calls)
        )

        self.assertEqual(results["first"], {"query": "first"})
        self.assertEqual(results["second"], {"query": "second"})

    def test_timeout_sends_cancel_and_failed_server_does_not_poison_another(
        self,
    ) -> None:
        manager = self._manager()
        self.addCleanup(manager.close)
        with patch("duckduckcode.core.lsp.REQUEST_TIMEOUT_SECONDS", 0.05):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                manager.execute(
                    operation="workspace_symbols",
                    server="test",
                    path=None,
                    line=None,
                    column=None,
                    query="timeout",
                    include_declaration=None,
                )
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not any(
            item.get("method") == "$/cancelRequest" for item in self._messages()
        ):
            time.sleep(0.01)
        self.assertTrue(
            any(item.get("method") == "$/cancelRequest" for item in self._messages())
        )

        # The timed-out server remains usable for later independent requests.
        result = manager.execute(
            operation="workspace_symbols",
            server="test",
            path=None,
            line=None,
            column=None,
            query="healthy",
            include_declaration=None,
        )
        self.assertEqual(result["params"]["query"], "healthy")

    def test_startup_failure_is_sticky_and_initialize_warnings_report_once(
        self,
    ) -> None:
        manager = self._manager(env={"LSP_FAKE_MODE": "exit"})
        self.addCleanup(manager.close)
        self.assertEqual(len(manager.initialize()), 0)
        self.assertEqual(manager.initialize(), [])

        arguments = dict(
            operation="workspace_symbols",
            server="test",
            path=None,
            line=None,
            column=None,
            query="x",
            include_declaration=None,
        )
        with self.assertRaises(RuntimeError) as first:
            manager.execute(**arguments)
        with self.assertRaises(RuntimeError) as second:
            manager.execute(**arguments)
        self.assertEqual(str(first.exception), str(second.exception))

    def test_json_rpc_error_and_startup_timeout_are_reported(self) -> None:
        manager = self._manager()
        self.addCleanup(manager.close)
        with self.assertRaisesRegex(RuntimeError, "LSP error -32001: fake error"):
            manager.execute(
                operation="workspace_symbols",
                server="test",
                path=None,
                line=None,
                column=None,
                query="error",
                include_declaration=None,
            )

        self.log = self.root / "hang.log"
        hanging = self._manager(env={"LSP_FAKE_MODE": "hang"})
        self.addCleanup(hanging.close)
        with patch("duckduckcode.core.lsp.START_TIMEOUT_SECONDS", 0.05):
            with self.assertRaisesRegex(RuntimeError, "startup.*timed out"):
                hanging.execute(
                    operation="workspace_symbols",
                    server="test",
                    path=None,
                    line=None,
                    column=None,
                    query="x",
                    include_declaration=None,
                )

    def test_failed_server_is_isolated_from_healthy_server(self) -> None:
        bad_log = self.root / "bad.log"
        extra = (
            "bad:\n"
            f"  command: {json.dumps(sys.executable)}\n"
            f"  args: [{json.dumps(str(self.server_script))}, {json.dumps(str(bad_log))}]\n"
            "  extensions: {'.py': python}\n"
            "  env: {LSP_FAKE_MODE: exit, LSP_ENCODING: utf-16}\n"
        )
        manager = self._manager(extra=extra)
        self.addCleanup(manager.close)
        arguments = dict(
            operation="workspace_symbols",
            path=None,
            line=None,
            column=None,
            query="healthy",
            include_declaration=None,
        )

        with self.assertRaisesRegex(RuntimeError, "server 'bad'.*failed"):
            manager.execute(server="bad", **arguments)
        result = manager.execute(server="test", **arguments)

        self.assertEqual(result["params"]["query"], "healthy")

    def test_close_is_idempotent_without_starting_empty_manager(self) -> None:
        empty = LSPManager(self.workspace, {}, home=self.home)
        self.assertEqual(empty.server_names, ())
        empty.close()
        empty.close()
        self.assertIsNone(empty._thread)

    def test_close_during_startup_terminates_the_process(self) -> None:
        manager = self._manager(env={"LSP_FAKE_MODE": "hang"})
        failures = []

        def call() -> None:
            try:
                manager.execute(
                    operation="workspace_symbols",
                    server="test",
                    path=None,
                    line=None,
                    column=None,
                    query="x",
                    include_declaration=None,
                )
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=call)
        thread.start()
        deadline = time.monotonic() + 2
        server_pid = None
        while time.monotonic() < deadline:
            server_pid = next(
                (item["serverPid"] for item in self._messages() if "serverPid" in item),
                None,
            )
            if server_pid is not None:
                break
            time.sleep(0.01)
        self.assertIsNotNone(server_pid)
        self.addCleanup(
            lambda: _terminate_process(server_pid) if server_pid is not None else None
        )

        manager.close()
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        with self.assertRaises(ProcessLookupError):
            os.kill(server_pid, 0)

    def test_close_deadline_still_kills_stubborn_process(self) -> None:
        manager = self._manager(env={"LSP_FAKE_MODE": "stubborn"})
        manager.execute(
            operation="workspace_symbols",
            server="test",
            path=None,
            line=None,
            column=None,
            query="healthy",
            include_declaration=None,
        )
        server_pid = next(
            item["serverPid"] for item in self._messages() if "serverPid" in item
        )
        self.addCleanup(lambda: _terminate_process(server_pid))

        started = time.monotonic()
        with patch("duckduckcode.core.lsp.CLOSE_TIMEOUT_SECONDS", 0.1):
            manager.close()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.16)
        with self.assertRaises(ProcessLookupError):
            os.kill(server_pid, 0)


class LSPFramingTest(unittest.IsolatedAsyncioTestCase):
    async def test_reads_chunked_and_consecutive_frames(self) -> None:
        import asyncio

        reader = asyncio.StreamReader()
        first = b'{"jsonrpc":"2.0","id":1,"result":{}}'
        second = b'{"jsonrpc":"2.0","id":2,"result":null}'
        data = (
            b"Content-Length: "
            + str(len(first)).encode()
            + b"\r\n\r\n"
            + first
            + b"Content-Length: "
            + str(len(second)).encode()
            + b"\r\n\r\n"
            + second
        )
        reader.feed_data(data[:17])
        reader.feed_data(data[17:])
        reader.feed_eof()

        self.assertEqual((await _read_frame(reader))["id"], 1)
        self.assertEqual((await _read_frame(reader))["id"], 2)

    async def test_rejects_messages_over_limit(self) -> None:
        import asyncio

        reader = asyncio.StreamReader()
        reader.feed_data(
            b"Content-Length: " + str(MAX_MESSAGE_BYTES + 1).encode() + b"\r\n\r\n"
        )
        reader.feed_eof()

        with self.assertRaisesRegex(RuntimeError, "16 MiB"):
            await _read_frame(reader)

    async def test_rejects_non_finite_json_numbers(self) -> None:
        import asyncio

        for constant in (b"NaN", b"Infinity", b"-Infinity"):
            with self.subTest(constant=constant):
                reader = asyncio.StreamReader()
                body = b'{"jsonrpc":"2.0","result":' + constant + b"}"
                reader.feed_data(
                    b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
                )
                reader.feed_eof()

                with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
                    await _read_frame(reader)

    async def test_startup_deadline_covers_process_creation(self) -> None:
        import asyncio

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "home"
            workspace = root / "workspace"
            workspace.mkdir()
            path = home / ".duckduckcode" / "lsp.yaml"
            path.parent.mkdir(parents=True)
            path.write_text(
                "test:\n  command: blocked\n  extensions: {'.py': python}\n",
                encoding="utf-8",
            )
            manager = LSPManager(workspace, {}, home=home)

            async def blocked(*args, **kwargs):
                await asyncio.Event().wait()

            with (
                patch("duckduckcode.core.lsp.asyncio.create_subprocess_exec", blocked),
                patch("duckduckcode.core.lsp.START_TIMEOUT_SECONDS", 0.02),
            ):
                with self.assertRaisesRegex(RuntimeError, "startup.*timed out"):
                    await asyncio.wait_for(manager._start_connection("test"), 0.2)

    async def test_request_deadline_covers_document_sync(self) -> None:
        import asyncio

        class BlockedConnection:
            position_encoding = "utf-16"

            async def sync_document(self, path, language_id, content):
                await asyncio.Event().wait()

        manager = object.__new__(LSPManager)

        async def connection(name):
            return BlockedConnection()

        manager._connection = connection
        document = (Path("sample.py"), "x", "python")
        arguments = {"operation": "document_symbols"}
        with patch("duckduckcode.core.lsp.REQUEST_TIMEOUT_SECONDS", 0.02):
            with self.assertRaisesRegex(RuntimeError, "request timed out"):
                await asyncio.wait_for(
                    manager._execute_async("test", arguments, document), 0.2
                )


def _terminate_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


if __name__ == "__main__":
    unittest.main()
