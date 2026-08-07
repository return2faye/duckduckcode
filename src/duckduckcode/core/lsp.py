from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
from types import MappingProxyType
from typing import Any

import yaml

MAX_CONFIG_BYTES = 256 * 1024
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
START_TIMEOUT_SECONDS = 10
REQUEST_TIMEOUT_SECONDS = 30
CLOSE_TIMEOUT_SECONDS = 5
_SAFE_ENVIRONMENT = ("HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER")
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class LSPServer:
    command: str
    extensions: Mapping[str, str]
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    initialization_options: Any = field(default_factory=dict)
    settings: Any = field(default_factory=dict)


@dataclass(frozen=True)
class LSPConfiguration:
    servers: dict[str, LSPServer]
    warnings: tuple[str, ...] = ()


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


class _DuplicateServerField(ValueError):
    pass


@dataclass(frozen=True)
class _InvalidServer:
    detail: str


def _construct_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    depth = getattr(loader, "_mapping_depth", 0)
    loader._mapping_depth = depth + 1
    mapping: dict[Any, Any] = {}
    try:
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                if depth:
                    raise _DuplicateServerField(f"duplicate YAML key {key!r}")
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            try:
                mapping[key] = loader.construct_object(value_node, deep=deep)
            except _DuplicateServerField as exc:
                if depth:
                    raise
                mapping[key] = _InvalidServer(str(exc))
        return mapping
    finally:
        loader._mapping_depth = depth


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def load_lsp_configuration(
    environment: Mapping[str, str], *, home: Path | None = None
) -> LSPConfiguration:
    home = (home or Path.home()).resolve()
    path = home / ".duckduckcode" / "lsp.yaml"
    try:
        text = _read_secure_file(path, home)
    except (OSError, RuntimeError, UnicodeError) as exc:
        return LSPConfiguration({}, (f"LSP config '{path}' was skipped: {exc}",))
    if text is None:
        return LSPConfiguration({})
    try:
        loaded = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        detail = "duplicate YAML key" if "duplicate key" in str(exc) else "invalid YAML"
        return LSPConfiguration({}, (f"LSP config '{path}' was skipped: {detail}.",))
    if loaded is None:
        return LSPConfiguration({})
    if not isinstance(loaded, dict):
        return LSPConfiguration(
            {}, (f"LSP config '{path}' must contain a server mapping.",)
        )
    servers: dict[str, LSPServer] = {}
    warnings: list[str] = []
    for name, value in loaded.items():
        try:
            if not isinstance(name, str) or not name:
                raise ValueError("server name must be a non-empty string")
            if isinstance(value, _InvalidServer):
                raise ValueError(value.detail)
            servers[name] = _parse_server(value, environment)
        except (TypeError, ValueError) as exc:
            warnings.append(f"LSP server '{name}' was skipped: {exc}")
    return LSPConfiguration(servers, tuple(warnings))


def _read_secure_file(path: Path, root: Path) -> str | None:
    current = root
    info = None
    for part in path.relative_to(root).parts:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError("symbolic links are not allowed")
    assert info is not None
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError("path is not a regular file")
    if info.st_size > MAX_CONFIG_BYTES:
        raise RuntimeError("file exceeds the 256 KiB limit")
    data = path.read_bytes()
    if len(data) > MAX_CONFIG_BYTES:
        raise RuntimeError("file exceeds the 256 KiB limit")
    return data.decode("utf-8")


def _parse_server(value: object, environment: Mapping[str, str]) -> LSPServer:
    if not isinstance(value, dict):
        raise TypeError("configuration must be a mapping")
    allowed = {
        "command",
        "args",
        "extensions",
        "env",
        "initialization_options",
        "settings",
    }
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ValueError("unknown field(s): " + ", ".join(unknown))
    command = value.get("command")
    args = value.get("args", [])
    extensions = value.get("extensions")
    if not isinstance(command, str) or not command:
        raise ValueError("command must be a non-empty string")
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError("args must be a list of strings")
    if not isinstance(extensions, dict) or not extensions:
        raise ValueError("extensions must be a non-empty string mapping")
    if not all(
        isinstance(extension, str)
        and extension.startswith(".")
        and len(extension) > 1
        and isinstance(language, str)
        and bool(language)
        for extension, language in extensions.items()
    ):
        raise ValueError(
            "extensions must map suffixes that start with '.' to non-empty language IDs"
        )
    env = _expand_env(value.get("env", {}), environment)
    initialization_options = value.get("initialization_options", {})
    settings = value.get("settings", {})
    for name, item in (
        ("initialization_options", initialization_options),
        ("settings", settings),
    ):
        if not _is_json_value(item):
            raise ValueError(f"{name} must be JSON-compatible") from None
    return LSPServer(
        command,
        MappingProxyType(dict(extensions)),
        tuple(args),
        MappingProxyType(env),
        initialization_options,
        settings,
    )


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item) for key, item in value.items()
        )
    return False


def _expand_env(value: object, environment: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError("env must map strings to strings")
    expanded: dict[str, str] = {}
    missing: set[str] = set()
    for key, item in value.items():

        def replace(match: re.Match[str]) -> str:
            variable = match.group(1)
            if variable not in environment:
                missing.add(variable)
                return ""
            return environment[variable]

        expanded[key] = _VARIABLE.sub(replace, item)
    if missing:
        raise ValueError(
            "missing environment variable(s): " + ", ".join(sorted(missing))
        )
    return expanded


class LSPManager:
    def __init__(
        self,
        workspace: Path,
        environment: Mapping[str, str],
        *,
        home: Path | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.environment = dict(environment)
        self.configuration = load_lsp_configuration(environment, home=home)
        self._initialized = False
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._loop_ready = threading.Event()
        self._loop_start_lock = threading.Lock()
        self._loop_error: Exception | None = None
        self._connections: dict[str, _Connection] = {}
        self._starting: dict[str, asyncio.Task[_Connection]] = {}
        self._failures: dict[str, str] = {}

    @property
    def server_names(self) -> tuple[str, ...]:
        return tuple(self.configuration.servers)

    @property
    def server_descriptions(self) -> tuple[str, ...]:
        return tuple(
            f"{name}: "
            + ", ".join(
                f"{extension} → {language}"
                for extension, language in server.extensions.items()
            )
            for name, server in self.configuration.servers.items()
        )

    def initialize(self) -> list[str]:
        if self._initialized or self._closed:
            return []
        self._initialized = True
        return list(self.configuration.warnings)

    def execute(self, **arguments: Any) -> Any:
        if self._closed:
            raise RuntimeError("LSP runtime is closed")
        operation = arguments["operation"]
        server_name = str(arguments["server"])
        server = self.configuration.servers.get(server_name)
        if server is None:
            raise ValueError(f"Unknown LSP server '{server_name}'")
        document: tuple[Path, str, str] | None = None
        if operation != "workspace_symbols":
            path, content = self._read_document(str(arguments["path"]))
            document = (path, content, self._language_id(server, path))
            if operation in {"definition", "references", "hover"}:
                _line_prefix(content, int(arguments["line"]), int(arguments["column"]))
        self._start_loop()
        if self._loop is None:
            raise RuntimeError("LSP event loop is unavailable")
        coroutine = self._execute_async(server_name, arguments, document)
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        except Exception:
            coroutine.close()
            raise
        try:
            return future.result(
                timeout=START_TIMEOUT_SECONDS + REQUEST_TIMEOUT_SECONDS + 2
            )
        except FutureTimeoutError as exc:
            future.cancel()
            raise RuntimeError("LSP request timed out") from exc

    def _start_loop(self) -> None:
        with self._loop_start_lock:
            self._start_loop_locked()

    def _start_loop_locked(self) -> None:
        if self._thread is not None:
            if not self._loop_ready.wait(START_TIMEOUT_SECONDS):
                raise RuntimeError("LSP event loop startup timed out")
            if self._loop_error is not None:
                raise RuntimeError(str(self._loop_error)) from self._loop_error
            return

        def run() -> None:
            try:
                loop = asyncio.new_event_loop()
            except Exception as exc:
                self._loop_error = exc
                self._loop_ready.set()
                return
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._loop_ready.set()
            try:
                loop.run_forever()
            finally:
                loop.close()

        self._thread = threading.Thread(
            target=run, name="duckduckcode-lsp", daemon=True
        )
        try:
            self._thread.start()
        except Exception:
            self._thread = None
            raise
        if not self._loop_ready.wait(START_TIMEOUT_SECONDS):
            raise RuntimeError("LSP event loop startup timed out")
        if self._loop_error is not None:
            raise RuntimeError(str(self._loop_error)) from self._loop_error

    async def _execute_async(
        self,
        server_name: str,
        arguments: Mapping[str, Any],
        document: tuple[Path, str, str] | None,
    ) -> Any:
        connection = await self._connection(server_name)
        try:
            async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                operation = str(arguments["operation"])
                if operation == "workspace_symbols":
                    return await connection.request(
                        "workspace/symbol", {"query": arguments["query"]}
                    )
                assert document is not None
                path, content, language_id = document
                uri = await connection.sync_document(path, language_id, content)
                if operation == "document_symbols":
                    return await connection.request(
                        "textDocument/documentSymbol", {"textDocument": {"uri": uri}}
                    )
                position = _lsp_position(
                    content,
                    int(arguments["line"]),
                    int(arguments["column"]),
                    connection.position_encoding,
                )
                params: dict[str, Any] = {
                    "textDocument": {"uri": uri},
                    "position": position,
                }
                methods = {
                    "definition": "textDocument/definition",
                    "references": "textDocument/references",
                    "hover": "textDocument/hover",
                }
                if operation == "references":
                    include = arguments["include_declaration"]
                    params["context"] = {
                        "includeDeclaration": True if include is None else include
                    }
                return await connection.request(methods[operation], params)
        except TimeoutError as exc:
            raise RuntimeError(
                f"LSP request timed out after {REQUEST_TIMEOUT_SECONDS:g} seconds"
            ) from exc

    async def _connection(self, name: str) -> _Connection:
        if name in self._failures:
            raise RuntimeError(self._failures[name])
        existing = self._connections.get(name)
        if existing is not None:
            return existing
        task = self._starting.get(name)
        if task is None:
            task = asyncio.create_task(self._start_connection(name))
            self._starting[name] = task
        try:
            connection = await task
        except Exception as exc:
            message = f"LSP server '{name}' failed: {str(exc) or type(exc).__name__}"
            self._failures[name] = message
            raise RuntimeError(message) from exc
        finally:
            self._starting.pop(name, None)
        return connection

    async def _start_connection(self, name: str) -> _Connection:
        server = self.configuration.servers[name]
        inherited = {
            key: self.environment[key]
            for key in _SAFE_ENVIRONMENT
            if key in self.environment
        }
        connection = _Connection(
            name,
            server,
            self.workspace,
            {**inherited, **dict(server.env)},
        )
        try:
            async with asyncio.timeout(START_TIMEOUT_SECONDS):
                await connection.start()
        except TimeoutError as exc:
            raise RuntimeError(
                f"LSP startup timed out after {START_TIMEOUT_SECONDS:g} seconds"
            ) from exc
        self._connections[name] = connection
        return connection

    @staticmethod
    def _language_id(server: LSPServer, path: Path) -> str:
        for extension in sorted(server.extensions, key=len, reverse=True):
            if path.name.endswith(extension):
                return server.extensions[extension]
        raise ValueError(f"LSP server does not map the extension for '{path.name}'")

    def _read_document(self, value: str) -> tuple[Path, str]:
        path = Path(value)
        if not path.is_absolute():
            path = self.workspace / path
        path = path.resolve()
        if not path.is_relative_to(self.workspace):
            raise ValueError("LSP path must resolve inside the workspace")
        if not path.is_file() or path.is_symlink():
            raise ValueError("LSP path must be a regular file")
        try:
            return path, path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("LSP path must contain UTF-8 text") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
            future.result(timeout=CLOSE_TIMEOUT_SECONDS + 1)
        except Exception:
            pass
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=CLOSE_TIMEOUT_SECONDS + 1)

    async def _shutdown(self) -> None:
        tasks = list(self._starting.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._connections:
            await asyncio.gather(
                *(connection.close() for connection in self._connections.values()),
                return_exceptions=True,
            )


class _ResponseError(RuntimeError):
    pass


@dataclass
class _Document:
    path: Path
    language_id: str
    text: str
    version: int
    opened: bool


class _Connection:
    def __init__(
        self,
        name: str,
        server: LSPServer,
        workspace: Path,
        environment: Mapping[str, str],
    ) -> None:
        self.name = name
        self.server = server
        self.workspace = workspace
        self.environment = dict(environment)
        self.process: asyncio.subprocess.Process | None = None
        self.position_encoding = "utf-16"
        self.capabilities: dict[str, Any] = {}
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._write_lock = asyncio.Lock()
        self._sync_lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._documents: dict[str, _Document] = {}
        self._fatal: str | None = None
        self._closing = False

    async def start(self) -> None:
        try:
            self.process = await asyncio.create_subprocess_exec(
                self.server.command,
                *self.server.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace,
                env=self.environment,
            )
            self._reader_task = asyncio.create_task(self._read_loop())
            self._stderr_task = asyncio.create_task(self._drain_stderr())
            result = await self.request(
                "initialize",
                {
                    "processId": os.getpid(),
                    "clientInfo": {"name": "DuckDuckCode"},
                    "rootPath": str(self.workspace),
                    "rootUri": self.workspace.as_uri(),
                    "workspaceFolders": [
                        {"uri": self.workspace.as_uri(), "name": self.workspace.name}
                    ],
                    "capabilities": {
                        "general": {"positionEncodings": ["utf-8", "utf-16", "utf-32"]},
                        "workspace": {
                            "configuration": True,
                            "workspaceFolders": True,
                        },
                    },
                    "initializationOptions": self.server.initialization_options,
                },
            )
            if not isinstance(result, dict) or not isinstance(
                result.get("capabilities", {}), dict
            ):
                raise RuntimeError("initialize returned invalid capabilities")
            self.capabilities = result.get("capabilities", {})
            encoding = self.capabilities.get("positionEncoding", "utf-16")
            if encoding not in {"utf-8", "utf-16", "utf-32"}:
                raise RuntimeError(f"unsupported position encoding '{encoding}'")
            self.position_encoding = encoding
            await self.notify("initialized", {})
            await self.notify(
                "workspace/didChangeConfiguration", {"settings": self.server.settings}
            )
        except asyncio.CancelledError:
            await self._force_close(asyncio.get_running_loop().time())
            raise
        except BaseException:
            await self._force_close()
            raise

    async def request(self, method: str, params: Any) -> Any:
        if self._fatal is not None:
            raise RuntimeError(self._fatal)
        request_id = self._next_id
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                }
            )
            return await future
        except asyncio.CancelledError:
            asyncio.create_task(self.notify("$/cancelRequest", {"id": request_id}))
            raise
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: Any) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _send(self, message: Mapping[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("LSP process is not running")
        data = json.dumps(
            message, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if len(data) > MAX_MESSAGE_BYTES:
            raise RuntimeError("LSP message exceeds the 16 MiB limit")
        async with self._write_lock:
            self.process.stdin.write(
                b"Content-Length: "
                + str(len(data)).encode("ascii")
                + b"\r\n\r\n"
                + data
            )
            await self.process.stdin.drain()

    async def _read_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            while True:
                message = await _read_frame(self.process.stdout)
                if "method" in message and "id" in message:
                    asyncio.create_task(self._handle_server_request(message))
                elif "method" in message:
                    continue
                elif "id" in message:
                    future = self._pending.get(message["id"])
                    if future is None or future.done():
                        continue
                    if "error" in message:
                        error = message["error"]
                        future.set_exception(
                            _ResponseError(
                                f"LSP error {error.get('code')}: {error.get('message')}"
                                if isinstance(error, dict)
                                else f"LSP error: {error}"
                            )
                        )
                    else:
                        future.set_result(message.get("result"))
        except (EOFError, asyncio.IncompleteReadError):
            self._fail_pending(self._process_error("process exited"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pending(str(exc) or type(exc).__name__)

    async def _handle_server_request(self, message: Mapping[str, Any]) -> None:
        method = message["method"]
        request_id = message["id"]
        params = message.get("params")
        try:
            if method == "workspace/configuration":
                items = params.get("items", []) if isinstance(params, dict) else []
                result = [
                    (
                        _setting(self.server.settings, item.get("section"))
                        if isinstance(item, dict)
                        else None
                    )
                    for item in items
                ]
            elif method == "workspace/workspaceFolders":
                result = [{"uri": self.workspace.as_uri(), "name": self.workspace.name}]
            elif method == "workspace/applyEdit":
                result = {
                    "applied": False,
                    "failureReason": "DuckDuckCode LSP navigation is read-only.",
                }
            elif method in {
                "client/registerCapability",
                "client/unregisterCapability",
                "window/workDoneProgress/create",
                "window/showMessageRequest",
            }:
                result = None
            else:
                await self._send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": "Method not found"},
                    }
                )
                return
            await self._send({"jsonrpc": "2.0", "id": request_id, "result": result})
        except Exception:
            return

    async def _drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        try:
            while line := await self.process.stderr.readline():
                self._stderr_tail.append(line.decode("utf-8", "replace").rstrip())
        except asyncio.CancelledError:
            raise

    def _process_error(self, message: str) -> str:
        suffix = ": " + " | ".join(self._stderr_tail) if self._stderr_tail else ""
        return f"LSP server '{self.name}' {message}{suffix}"

    def _fail_pending(self, message: str) -> None:
        self._fatal = message
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(RuntimeError(message))

    async def sync_document(self, path: Path, language_id: str, text: str) -> str:
        uri = path.as_uri()
        async with self._sync_lock:
            document = self._documents.get(uri)
            open_close, change_kind = _sync_options(self.capabilities)
            if document is None:
                opened = open_close
                document = _Document(path, language_id, text, 1, opened)
                self._documents[uri] = document
                if opened:
                    await self.notify(
                        "textDocument/didOpen",
                        {
                            "textDocument": {
                                "uri": uri,
                                "languageId": language_id,
                                "version": 1,
                                "text": text,
                            }
                        },
                    )
            elif document.text != text:
                document.version += 1
                if change_kind:
                    change: dict[str, Any] = {"text": text}
                    if change_kind == 2:
                        change["range"] = {
                            "start": {"line": 0, "character": 0},
                            "end": _end_position(document.text, self.position_encoding),
                        }
                    await self.notify(
                        "textDocument/didChange",
                        {
                            "textDocument": {
                                "uri": uri,
                                "version": document.version,
                            },
                            "contentChanges": [change],
                        },
                    )
                document.text = text
            return uri

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self.process is None:
            return
        deadline = asyncio.get_running_loop().time() + CLOSE_TIMEOUT_SECONDS
        try:
            async with asyncio.timeout_at(deadline):
                for uri, document in self._documents.items():
                    if document.opened:
                        await self.notify(
                            "textDocument/didClose", {"textDocument": {"uri": uri}}
                        )
                if self.process.returncode is None:
                    await self.request("shutdown", None)
                    await self.notify("exit", None)
                    await self.process.wait()
        except Exception:
            pass
        finally:
            await self._force_close(deadline)

    async def _force_close(self, deadline: float | None = None) -> None:
        process = self.process
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                timeout = (
                    CLOSE_TIMEOUT_SECONDS
                    if deadline is None
                    else max(0, deadline - asyncio.get_running_loop().time())
                )
                await asyncio.wait_for(process.wait(), timeout=timeout)
            except TimeoutError:
                process.kill()
                await process.wait()
        current = asyncio.current_task()
        tasks = [
            task
            for task in (self._reader_task, self._stderr_task)
            if task is not None and task is not current
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


async def _read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    content_length: int | None = None
    header_bytes = 0
    while True:
        line = await reader.readline()
        if not line:
            raise EOFError
        header_bytes += len(line)
        if header_bytes > 64 * 1024:
            raise RuntimeError("LSP headers exceed 64 KiB")
        if line in {b"\r\n", b"\n"}:
            break
        try:
            key, value = line.decode("ascii").split(":", 1)
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("LSP response has invalid headers") from exc
        if key.lower() == "content-length":
            try:
                content_length = int(value.strip())
            except ValueError as exc:
                raise RuntimeError("LSP response has invalid Content-Length") from exc
    if content_length is None or content_length < 0:
        raise RuntimeError("LSP response is missing Content-Length")
    if content_length > MAX_MESSAGE_BYTES:
        raise RuntimeError("LSP message exceeds the 16 MiB limit")
    data = await reader.readexactly(content_length)
    try:
        message = json.loads(data, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("LSP response contains invalid JSON") from exc
    if not isinstance(message, dict):
        raise RuntimeError("LSP response must be a JSON object")
    return message


def _setting(settings: Any, section: Any) -> Any:
    if not isinstance(section, str) or not section:
        return settings
    value = settings
    for part in section.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _sync_options(capabilities: Mapping[str, Any]) -> tuple[bool, int]:
    value = capabilities.get("textDocumentSync", 0)
    if isinstance(value, int) and not isinstance(value, bool):
        return value != 0, value
    if isinstance(value, dict):
        change = value.get("change", 0)
        return bool(value.get("openClose", False)), (
            change if isinstance(change, int) and change in {0, 1, 2} else 0
        )
    return False, 0


def _lines(text: str) -> list[str]:
    return re.split(r"\r\n|\r|\n", text)


def _line_prefix(text: str, line: int, column: int) -> str:
    lines = _lines(text)
    if not 1 <= line <= len(lines):
        raise ValueError(f"LSP line {line} is outside the file")
    value = lines[line - 1]
    if not 1 <= column <= len(value) + 1:
        raise ValueError(f"LSP column {column} is outside line {line}")
    return value[: column - 1]


def _units(text: str, encoding: str) -> int:
    if encoding == "utf-8":
        return len(text.encode("utf-8"))
    if encoding == "utf-16":
        return len(text.encode("utf-16-le")) // 2
    return len(text)


def _lsp_position(text: str, line: int, column: int, encoding: str) -> dict[str, int]:
    return {
        "line": line - 1,
        "character": _units(_line_prefix(text, line, column), encoding),
    }


def _end_position(text: str, encoding: str) -> dict[str, int]:
    lines = _lines(text)
    return {"line": len(lines) - 1, "character": _units(lines[-1], encoding)}
