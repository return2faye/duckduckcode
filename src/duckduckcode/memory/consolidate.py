from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import signal
import stat
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..core.context import ContextManager, Message
from ..core.event import ConversationEvent, DoneEvent, ErrorEvent, ToolCallEvent
from ..providers.openai.client import OpenAIClient
from ..tools.tool import ToolCall, ToolManager, ToolResult
from .long_term import (
    MEMORY_MAX_BYTES,
    MEMORY_MAX_LINES,
    MemoryError,
    MemoryManager,
    MemoryStore,
    ID_RE,
    _atomic_write,
    _reject_duplicate_ids,
    _validate_directory,
    _validate_regular,
    build_memory_block,
    memory_write_locks,
)
from .session import SYNTHETIC_TOOL_ERROR, SessionManager

CONSOLIDATE_AFTER_SECONDS = 7 * 24 * 60 * 60
CONSOLIDATE_MIN_SESSIONS = 5
CONSOLIDATE_TIMEOUT_SECONDS = 60 * 60
CONSOLIDATE_MAX_ITERATIONS = 40
CONSOLIDATE_PROMPT = """You maintain DuckDuckCode's durable memory in staging.

The real memory directories and session logs are read-only to you. Follow these
four phases in order:
1. Locate: list staging, then read both MEMORY.md indexes and relevant memories.
2. Collect signals: search recent workspace sessions selectively; never load every
   session wholesale.
3. Consolidate: merge duplicates, remove disproved facts, resolve conflicts, and
   replace relative dates with absolute dates.
4. Prune indexes: remove dead links, shorten summaries, retain important memories,
   and keep the two MEMORY.md files together within 200 lines and 25KB.

Every memory must retain strict frontmatter and each file must appear exactly once
in its scope's index. User and project indexes must not cross-reference one another.
Use only the provided tools. Finish with a short plain-text completion note and no
tool calls after all edits are complete.
"""


def maybe_consolidate(
    config: Any, manager: MemoryManager, *, now: float | None = None
) -> bool:
    manager.project.ensure()
    lock_path = manager.project.root / ".consolidate-lock"
    created = not lock_path.exists()
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    stream = os.fdopen(descriptor, "r+", encoding="utf-8")
    if created:
        stream.flush()
        os.fsync(stream.fileno())
    metadata = os.fstat(stream.fileno())
    if not stat.S_ISREG(metadata.st_mode):
        stream.close()
        raise MemoryError("consolidate lock is not a regular file")
    current = time.time() if now is None else now
    if current - metadata.st_mtime <= CONSOLIDATE_AFTER_SECONDS:
        stream.close()
        return False
    if (
        count_active_sessions(manager.workspace, metadata.st_mtime)
        < CONSOLIDATE_MIN_SESSIONS
    ):
        stream.close()
        return False
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return False
    before = os.fstat(stream.fileno())
    try:
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        os.fsync(stream.fileno())
        with (
            _alarm(CONSOLIDATE_TIMEOUT_SECONDS),
            memory_write_locks(manager.user, manager.project),
        ):
            _run_staged_consolidation(config, manager)
        os.utime(lock_path, ns=(before.st_atime_ns, int(current * 1_000_000_000)))
        return True
    except Exception:
        os.utime(lock_path, ns=(before.st_atime_ns, before.st_mtime_ns))
        raise
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def count_active_sessions(workspace: Path, since: float) -> int:
    directory = workspace / ".duckduckcode" / "sessions"
    if not directory.exists():
        return 0
    _validate_directory(directory)
    manager = SessionManager(workspace, ContextManager(system_prompt="memory audit"))
    active = 0
    for info in manager.list():
        if info.status != "valid":
            continue
        try:
            records, _ = manager._read(manager.directory / f"{info.id}.jsonl")
            if any(_real_activity(record.as_dict(), since) for record in records):
                active += 1
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError, MemoryError):
            continue
    return active


def lock_holder_alive(lock_path: str | Path) -> bool | None:
    try:
        text = Path(lock_path).read_text(encoding="utf-8").strip()
        pid = int(text)
        if pid <= 0:
            return None
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, UnicodeError, ValueError):
        return None


def _run_staged_consolidation(config: Any, manager: MemoryManager) -> None:
    user_records, _ = manager.user.load()
    project_records, _ = manager.project.load()
    with tempfile.TemporaryDirectory(
        prefix="memory-staging-", dir=manager.project.root
    ) as directory:
        staging = Path(directory)
        staged_user = MemoryStore(staging / "user", "user")
        staged_project = MemoryStore(staging / "project", "project")
        staged_user.publish(user_records)
        staged_project.publish(project_records)
        tools = _consolidation_tools(
            staged_user,
            staged_project,
            manager.workspace / ".duckduckcode" / "sessions",
        )
        client = OpenAIClient(
            api_key=config.openai_api_key,
            model=config.openai_model,
            langsmith_tracing=config.langsmith_tracing,
            langsmith_api_key=config.langsmith_api_key,
            langsmith_project=config.langsmith_project,
        )
        try:
            _run_tool_loop(client, config.reasoning, tools, staging)
        finally:
            client.close()
        _validate_staging(staging)
        new_user, user_index = staged_user.load()
        new_project, project_index = staged_project.load()
        _reject_duplicate_ids(new_user, new_project)
        _, truncated = build_memory_block(user_index, project_index)
        if truncated:
            raise MemoryError(
                f"staged indexes exceed {MEMORY_MAX_LINES} lines/{MEMORY_MAX_BYTES} bytes"
            )
        manager.user.publish(new_user)
        manager.project.publish(new_project)


def _run_tool_loop(
    client: Any, reasoning: Any, tools: ToolManager, staging: Path
) -> None:
    messages = [
        Message("system", CONSOLIDATE_PROMPT),
        Message(
            "user",
            f"Staging root: {staging}\nInspect and consolidate the staged memory now.",
        ),
    ]
    for _ in range(CONSOLIDATE_MAX_ITERATIONS):
        calls: list[ToolCall] = []
        text = ""
        completed = False
        for event in client.stream(
            messages, tools=tools.schemas(), reasoning=reasoning
        ):
            if isinstance(event, ConversationEvent):
                text += event.delta
            elif isinstance(event, ToolCallEvent):
                calls.append(event.tool_call)
            elif isinstance(event, ErrorEvent):
                raise MemoryError(event.message)
            elif isinstance(event, DoneEvent):
                completed = True
                break
        if not completed:
            raise MemoryError("memory consolidator response did not complete")
        if text:
            messages.append(Message("assistant", text))
        if not calls:
            return
        for call in calls:
            messages.append(Message.tool_call(call.call_id, call.name, call.arguments))
            result = tools.execute(call)
            messages.append(Message.tool_result(call.call_id, result.to_model_output()))
    raise MemoryError("memory consolidator exceeded its iteration limit")


def _consolidation_tools(
    user: MemoryStore, project: MemoryStore, sessions: Path
) -> ToolManager:
    staged = (user.root, project.root)
    readable = (*staged, sessions)
    tools = ToolManager()

    def read_file(path: str) -> str:
        target = _safe_path(path, readable, write=False)
        _validate_regular(target)
        return target.read_text(encoding="utf-8")

    def glob_files(pattern: str, path: str) -> str:
        root = _safe_path(path, readable, write=False)
        _validate_directory(root)
        found = []
        for target in root.glob(pattern):
            try:
                safe = _safe_path(str(target), readable, write=False)
                if safe.is_file() and not safe.is_symlink():
                    found.append(str(safe))
            except (OSError, MemoryError):
                continue
        return "\n".join(sorted(found))

    def grep_files(pattern: str, paths: list[str]) -> str:
        expression = re.compile(pattern)
        matches = []
        for name in paths:
            target = _safe_path(name, readable, write=False)
            _validate_regular(target)
            for number, line in enumerate(
                target.read_text(encoding="utf-8").splitlines(), 1
            ):
                if expression.search(line):
                    matches.append(f"{target}:{number}:{line}")
        return "\n".join(matches)

    def write_file(path: str, content: str) -> str:
        target = _safe_path(path, staged, write=True)
        _atomic_write(target, content, 0o600)
        return f"Wrote {target}"

    def edit_file(path: str, old_string: str, new_string: str) -> str:
        target = _safe_path(path, staged, write=True)
        _validate_regular(target)
        content = target.read_text(encoding="utf-8")
        if content.count(old_string) != 1:
            raise MemoryError("old_string must occur exactly once")
        _atomic_write(target, content.replace(old_string, new_string, 1), 0o600)
        return f"Edited {target}"

    def delete_memory(scope: str, id: str) -> str:
        store = {"user": user, "project": project}.get(scope)
        if store is None or not ID_RE.fullmatch(id):
            raise MemoryError("invalid memory scope")
        matches = list(store.root.glob(f"*/{id}.md"))
        if len(matches) != 1:
            raise MemoryError("memory ID was not found exactly once")
        target = _safe_path(str(matches[0]), staged, write=True)
        _validate_regular(target)
        target.unlink()
        return f"Deleted {id}"

    def bash(command: str) -> str:
        if any(
            marker in command for marker in ("|", ">", "<", ";", "`", "$", "&", "\n")
        ):
            raise MemoryError("unsupported shell syntax")
        try:
            arguments = shlex.split(command)
        except ValueError as exc:
            raise MemoryError(str(exc)) from exc
        if not arguments or arguments[0] not in {"ls", "cat", "grep", "rg"}:
            raise MemoryError("unsupported command")
        if arguments[0] == "ls":
            flags = [item for item in arguments[1:] if item.startswith("-")]
            if any(item not in {"-l", "-a", "-la", "-al"} for item in flags):
                raise MemoryError("unsupported ls option")
            names = [item for item in arguments[1:] if not item.startswith("-")] or [
                str(user.root)
            ]
            output = []
            for name in names:
                target = _safe_path(name, readable, write=False)
                if target.is_dir():
                    output.extend(str(item) for item in sorted(target.iterdir()))
                else:
                    output.append(str(target))
            return "\n".join(output)
        if arguments[0] == "cat":
            if len(arguments) < 2 or any(
                item.startswith("-") for item in arguments[1:]
            ):
                raise MemoryError("cat requires files and no options")
            return "\n".join(read_file(name) for name in arguments[1:])
        rest = arguments[1:]
        numbered = False
        if rest and rest[0] == "-n":
            numbered = True
            rest = rest[1:]
        if len(rest) < 2 or any(item.startswith("-") for item in rest):
            raise MemoryError("grep/rg requires [-n] pattern and paths")
        result = grep_files(rest[0], rest[1:])
        return (
            result
            if numbered
            else "\n".join(
                line.split(":", 2)[0] + ":" + line.split(":", 2)[2]
                for line in result.splitlines()
            )
        )

    object_schema = lambda properties, required: {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    tools.register(
        "ReadFile",
        "Read an allowed file",
        object_schema({"path": {"type": "string"}}, ["path"]),
        read_file,
        is_read_only=True,
    )
    tools.register(
        "Glob",
        "Find allowed files",
        object_schema(
            {"pattern": {"type": "string"}, "path": {"type": "string"}},
            ["pattern", "path"],
        ),
        glob_files,
        is_read_only=True,
    )
    tools.register(
        "Grep",
        "Search allowed files",
        object_schema(
            {
                "pattern": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}},
            },
            ["pattern", "paths"],
        ),
        grep_files,
        is_read_only=True,
    )
    tools.register(
        "WriteFile",
        "Write a staging memory file",
        object_schema(
            {"path": {"type": "string"}, "content": {"type": "string"}},
            ["path", "content"],
        ),
        write_file,
    )
    tools.register(
        "EditFile",
        "Edit one exact string in staging",
        object_schema(
            {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            ["path", "old_string", "new_string"],
        ),
        edit_file,
    )
    tools.register(
        "DeleteMemory",
        "Delete one staged memory by ID",
        object_schema(
            {
                "scope": {"type": "string", "enum": ["user", "project"]},
                "id": {"type": "string"},
            },
            ["scope", "id"],
        ),
        delete_memory,
    )
    tools.register(
        "Bash",
        "Restricted read-only ls/cat/grep/rg facade; no shell is invoked",
        object_schema({"command": {"type": "string"}}, ["command"]),
        bash,
        is_read_only=True,
    )
    return tools


def _safe_path(path: str, roots: tuple[Path, ...], *, write: bool) -> Path:
    target = Path(path)
    if not target.is_absolute():
        raise MemoryError("tool paths must be absolute")
    try:
        resolved = (
            (target.parent.resolve() / target.name) if write else target.resolve()
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise MemoryError(f"path cannot be resolved: {path}") from exc
    if not any(resolved.is_relative_to(root.resolve()) for root in roots):
        raise MemoryError("path escapes allowed roots")
    current = resolved
    while current != current.parent:
        if current.exists() and current.is_symlink():
            raise MemoryError("symbolic links are not allowed")
        if any(current == root.resolve() for root in roots):
            break
        current = current.parent
    return resolved


def _real_activity(record: Any, since: float) -> bool:
    if not isinstance(record, dict) or set(record) != {"role", "context", "ts"}:
        raise ValueError("invalid session record")
    context = record["context"]
    if not isinstance(record["ts"], int) or not isinstance(context, dict):
        raise ValueError("invalid session record")
    kind = context.get("type")
    return (
        record["ts"] > since
        and kind != "compaction"
        and context.get("content") != SYNTHETIC_TOOL_ERROR
        and kind in {"message", "tool_call", "tool_result"}
    )


def _validate_staging(staging: Path) -> None:
    allowed_roots = {staging / "user", staging / "project"}
    for path in staging.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise MemoryError("staging contains a symbolic link")
        relative = path.relative_to(staging)
        if path.is_dir():
            if len(relative.parts) == 1 and path in allowed_roots:
                continue
            if len(relative.parts) == 2 and relative.parts[1] in {
                "preference",
                "feedback",
                "project",
                "reference",
            }:
                continue
            raise MemoryError(f"unexpected staging directory: {relative}")
        if not stat.S_ISREG(metadata.st_mode):
            raise MemoryError("staging contains a non-regular file")
        if len(relative.parts) == 2 and relative.parts[1] in {
            "MEMORY.md",
            ".write-lock",
        }:
            continue
        if (
            len(relative.parts) == 3
            and relative.parts[1] in {"preference", "feedback", "project", "reference"}
            and path.suffix == ".md"
        ):
            continue
        raise MemoryError(f"unexpected staging file: {relative}")


@contextmanager
def _alarm(seconds: int) -> Iterator[None]:
    def timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError("memory consolidation exceeded one hour")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _unique_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value
