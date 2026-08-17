from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ..core.context import ContextManager, Message
from ..tools.tool import ToolCall, ToolResult

STALE_SECONDS = 24 * 60 * 60
RETENTION_SECONDS = 30 * 24 * 60 * 60
SYNTHETIC_TOOL_ERROR = (
    "DuckDuckCode recovered this session after the process stopped before the "
    "tool returned a result. The tool was not run again."
)


class SessionPersistenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionRecord:
    role: Literal["user", "assistant", "tool"]
    context: dict[str, Any]
    ts: int

    def as_dict(self) -> dict[str, Any]:
        return {"role": self.role, "context": self.context, "ts": self.ts}


@dataclass(frozen=True)
class SessionInfo:
    id: str
    created_at: int
    last_activity: int
    status: Literal["valid", "invalid"] = "valid"
    active: bool = False


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    records: tuple[SessionRecord, ...]
    token_usage: int
    restored: bool = False
    cleaned: int = 0
    invalid: tuple[str, ...] = ()


class SessionManager:
    def __init__(
        self,
        workspace: str | Path,
        context: ContextManager,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.directory = self.workspace / ".duckduckcode" / "sessions"
        self.context = context
        self._clock = clock or datetime.now
        self._current_id: str | None = None
        self._records: list[SessionRecord] = []

    @property
    def current_session_id(self) -> str | None:
        return self._current_id

    @property
    def current_path(self) -> Path | None:
        return (
            self.directory / f"{self._current_id}.jsonl" if self._current_id else None
        )

    def start(self) -> SessionSnapshot:
        self._ensure_directory()
        now = self._timestamp()
        cleaned = 0
        valid: list[SessionInfo] = []
        invalid: list[str] = []
        for info in self.list():
            if info.status == "invalid":
                invalid.append(info.id)
            elif now - info.last_activity > RETENTION_SECONDS:
                self._path(info.id).unlink()
                cleaned += 1
            else:
                valid.append(info)
        if valid:
            latest = max(
                valid, key=lambda item: (item.last_activity, item.created_at, item.id)
            )
            snapshot = self.resume(latest.id)
            return SessionSnapshot(
                snapshot.session_id,
                snapshot.records,
                snapshot.token_usage,
                restored=True,
                cleaned=cleaned,
                invalid=tuple(sorted(invalid)),
            )
        snapshot = self.create()
        return SessionSnapshot(
            snapshot.session_id,
            snapshot.records,
            snapshot.token_usage,
            cleaned=cleaned,
            invalid=tuple(sorted(invalid)),
        )

    def list(self) -> list[SessionInfo]:
        if not self.directory.exists():
            return []
        infos: list[SessionInfo] = []
        for path in self.directory.iterdir():
            if path.suffix != ".jsonl":
                continue
            session_id = path.stem
            try:
                records, created_at = self._read(path)
            except (OSError, ValueError, UnicodeError, json.JSONDecodeError):
                try:
                    metadata = path.lstat()
                    created_at = _created_at(metadata)
                    last_activity = int(metadata.st_mtime)
                except OSError:
                    created_at = last_activity = 0
                infos.append(
                    SessionInfo(
                        session_id,
                        created_at,
                        last_activity,
                        "invalid",
                        session_id == self._current_id,
                    )
                )
                continue
            infos.append(
                SessionInfo(
                    session_id,
                    created_at,
                    _last_activity(records, created_at),
                    active=session_id == self._current_id,
                )
            )
        return sorted(
            infos, key=lambda item: (item.last_activity, item.id), reverse=True
        )

    def create(self) -> SessionSnapshot:
        self._ensure_directory()
        base = self._now().strftime("%Y%m%d-%H%M%S")
        suffix = 1
        while True:
            session_id = base if suffix == 1 else f"{base}-{suffix}"
            if session_id == self._current_id:
                suffix += 1
                continue
            path = self.directory / f"{session_id}.jsonl"
            try:
                flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(path, flags, 0o600)
            except FileExistsError:
                suffix += 1
                continue
            except OSError as exc:
                raise SessionPersistenceError(
                    f"Could not create session: {exc}"
                ) from exc
            try:
                try:
                    os.fchmod(descriptor, 0o600)
                    os.fsync(descriptor)
                except OSError as exc:
                    raise SessionPersistenceError(
                        f"Could not create session: {exc}"
                    ) from exc
            finally:
                os.close(descriptor)
            self._current_id = session_id
            self._records = []
            self.context.restore([], "", "")
            return self.snapshot()

    def resume(self, session_id: str) -> SessionSnapshot:
        path = self._path(session_id)
        try:
            records, created_at = self._read(path)
        except FileNotFoundError as exc:
            raise ValueError(f"Session '{session_id}' does not exist.") from exc
        except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Session '{session_id}' is invalid: {exc}") from exc
        self._current_id = session_id
        self._records = list(records)
        last_activity = _last_activity(records, created_at)
        self._replay(records)
        self._repair_pending_tool_calls()
        reminder = ""
        if self._timestamp() - last_activity > STALE_SECONDS:
            local = (
                datetime.fromtimestamp(last_activity)
                .astimezone()
                .isoformat(timespec="seconds")
            )
            reminder = (
                f"Session reminder: the last real activity was {local}. More than "
                "24 hours have passed, so code and files may have changed. Re-read "
                "relevant files before relying on prior conversation details."
            )
        self.context.set_reminder(reminder)
        return self.snapshot(restored=True)

    def delete(self, session_id: str | None = None) -> SessionSnapshot:
        target = session_id or self._require_current_id()
        path = self._path(target)
        try:
            self._validate_regular_file(path)
            path.unlink()
        except FileNotFoundError as exc:
            raise ValueError(f"Session '{target}' does not exist.") from exc
        if target == self._current_id:
            return self.create()
        return self.snapshot()

    def snapshot(self, *, restored: bool = False) -> SessionSnapshot:
        return SessionSnapshot(
            self._require_current_id(),
            tuple(self._records),
            sum(_record_usage(record) for record in self._records),
            restored=restored,
        )

    def commit_message(
        self,
        role: Literal["user", "assistant"],
        content: str,
        *,
        status: Literal["completed", "error"] = "completed",
        token_usage: int = 0,
        visible: bool = True,
    ) -> None:
        record = SessionRecord(
            role,
            {
                "type": "message",
                "content": content,
                "status": status,
                "token_usage": token_usage,
                "visible": visible,
            },
            self._timestamp(),
        )
        self._append(record)
        self.context.add_message(
            Message(role, content, status=status, token_usage=token_usage)
        )

    def commit_assistant_stream(
        self,
        index: int,
        status: Literal["completed", "error"],
        token_usage: int = 0,
    ) -> None:
        message = self.context.messages()[index]
        record = SessionRecord(
            "assistant",
            {
                "type": "message",
                "content": message.content,
                "status": status,
                "token_usage": token_usage if status == "completed" else 0,
                "visible": True,
                **(
                    {"reasoning_content": message.reasoning_content}
                    if message.reasoning_content
                    else {}
                ),
            },
            self._timestamp(),
        )
        try:
            self._append(record)
        except Exception:
            self.context.discard_assistant_stream(index)
            raise
        if status == "completed":
            self.context.finish_assistant_stream(index, token_usage)
        else:
            self.context.fail_assistant_stream(index)

    def commit_tool_call(self, tool_call: ToolCall) -> None:
        record = SessionRecord(
            "assistant",
            {
                "type": "tool_call",
                "call_id": tool_call.call_id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
            self._timestamp(),
        )
        self._append(record)
        self.context.add_tool_call(tool_call)

    def commit_tool_result(self, tool_call: ToolCall, result: ToolResult) -> None:
        record = SessionRecord(
            "tool",
            {
                "type": "tool_result",
                "call_id": tool_call.call_id,
                "name": tool_call.name,
                "content": result.content,
                "is_error": result.is_error,
            },
            self._timestamp(),
        )
        self._append(record)
        self.context.add_tool_result(tool_call.call_id, result.to_model_output())

    def commit_compaction(
        self, summary: str, cutoff: int, token_usage: int = 0
    ) -> None:
        summary = summary.strip()
        if not summary or not 0 < cutoff <= len(self.context.messages()):
            raise ValueError("Invalid compaction result")
        record = SessionRecord(
            "assistant",
            {
                "type": "compaction",
                "summary": summary,
                "cutoff": cutoff,
                "token_usage": token_usage,
            },
            self._timestamp(),
        )
        self._append(record)
        self.context.apply_compaction(summary, cutoff)

    def _replay(self, records: list[SessionRecord]) -> None:
        self.context.restore([], "", "")
        for record in records:
            value = record.context
            kind = value["type"]
            if kind == "message":
                self.context.add_message(
                    Message(
                        record.role,
                        value["content"],
                        status=value["status"],
                        token_usage=value["token_usage"],
                        reasoning_content=value.get("reasoning_content", ""),
                    )
                )
            elif kind == "tool_call":
                self.context.add_tool_call(
                    ToolCall(value["call_id"], value["name"], value["arguments"])
                )
            elif kind == "tool_result":
                self.context.add_tool_result(
                    value["call_id"],
                    ToolResult(value["content"], value["is_error"]).to_model_output(),
                )
            else:
                self.context.apply_compaction(value["summary"], value["cutoff"])

    def _repair_pending_tool_calls(self) -> None:
        calls: dict[str, tuple[ToolCall, int]] = {}
        for record in self._records:
            value = record.context
            if value["type"] == "tool_call":
                calls[value["call_id"]] = (
                    ToolCall(value["call_id"], value["name"], value["arguments"]),
                    record.ts,
                )
        pending: dict[str, tuple[ToolCall, int]] = {}
        for message in self.context.messages():
            if message.kind == "tool_call" and message.tool_call_id in calls:
                pending[message.tool_call_id] = calls[message.tool_call_id]
            elif message.kind == "tool_result":
                pending.pop(message.tool_call_id, None)
        for call_id, (tool_call, timestamp) in pending.items():
            result = ToolResult(SYNTHETIC_TOOL_ERROR, is_error=True)
            record = SessionRecord(
                "tool",
                {
                    "type": "tool_result",
                    "call_id": call_id,
                    "name": tool_call.name,
                    "content": result.content,
                    "is_error": True,
                },
                timestamp,
            )
            self._append(record)
            self.context.add_tool_result(call_id, result.to_model_output())

    def _append(self, record: SessionRecord) -> None:
        try:
            _parse_record(record.as_dict(), 0)
            path = self._path(self._require_current_id())
            encoded = (
                json.dumps(record.as_dict(), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            flags = os.O_APPEND | os.O_WRONLY
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise OSError("session path is not a regular file")
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if written == 0:
                        raise OSError("session write returned zero bytes")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except (OSError, TypeError, ValueError) as exc:
            raise SessionPersistenceError(f"Could not persist session: {exc}") from exc
        self._records.append(record)

    def _read(self, path: Path) -> tuple[list[SessionRecord], int]:
        metadata = self._validate_regular_file(path)
        records: list[SessionRecord] = []
        model_message_count = 0
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.endswith("\n"):
                    raise ValueError(f"line {line_number} is incomplete")
                value = json.loads(line, object_pairs_hook=_unique_object)
                record = _parse_record(value, line_number)
                if record.context["type"] == "compaction":
                    cutoff = record.context["cutoff"]
                    if cutoff > model_message_count:
                        raise ValueError(f"line {line_number} has an invalid cutoff")
                    model_message_count -= cutoff
                else:
                    model_message_count += 1
                records.append(record)
        return records, _created_at(metadata)

    def _validate_regular_file(self, path: Path) -> os.stat_result:
        if path.parent != self.directory or path.suffix != ".jsonl":
            raise ValueError("Session path escapes the sessions directory.")
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Session path is not a regular file.")
        return metadata

    def _path(self, session_id: str) -> Path:
        if (
            not session_id
            or session_id in {".", ".."}
            or Path(session_id).name != session_id
            or session_id.endswith(".jsonl")
        ):
            raise ValueError("Invalid session ID.")
        return self.directory / f"{session_id}.jsonl"

    def _ensure_directory(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            metadata = self.directory.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise SessionPersistenceError(
                    "Sessions path is not a regular directory."
                )
            self.directory.chmod(0o700)
        except OSError as exc:
            raise SessionPersistenceError(
                f"Could not prepare sessions directory: {exc}"
            ) from exc

    def _require_current_id(self) -> str:
        if self._current_id is None:
            raise RuntimeError("Session manager has not started.")
        return self._current_id

    def _now(self) -> datetime:
        value = self._clock()
        return value.astimezone() if value.tzinfo is not None else value

    def _timestamp(self) -> int:
        return int(self._now().timestamp())


def _parse_record(value: Any, line_number: int) -> SessionRecord:
    if not isinstance(value, dict) or set(value) != {"role", "context", "ts"}:
        raise ValueError(f"line {line_number} has invalid top-level fields")
    role, context, timestamp = value["role"], value["context"], value["ts"]
    if role not in {"user", "assistant", "tool"}:
        raise ValueError(f"line {line_number} has an invalid role")
    if not isinstance(timestamp, int) or isinstance(timestamp, bool):
        raise ValueError(f"line {line_number} has an invalid timestamp")
    if not isinstance(context, dict):
        raise ValueError(f"line {line_number} has invalid context")
    kind = context.get("type")
    fields = {
        "message": {"type", "content", "status", "token_usage", "visible"},
        "tool_call": {"type", "call_id", "name", "arguments"},
        "tool_result": {"type", "call_id", "name", "content", "is_error"},
        "compaction": {"type", "summary", "cutoff", "token_usage"},
    }
    message_fields = fields["message"]
    if kind == "message" and set(context) == message_fields | {"reasoning_content"}:
        pass
    elif kind not in fields or set(context) != fields[kind]:
        raise ValueError(f"line {line_number} has invalid {kind!r} fields")
    if kind == "message":
        if (
            role not in {"user", "assistant"}
            or not isinstance(context["content"], str)
            or context["status"] not in {"completed", "error"}
            or not _integer(context["token_usage"])
            or context["token_usage"] < 0
            or not isinstance(context["visible"], bool)
            or not isinstance(context.get("reasoning_content", ""), str)
            or ("reasoning_content" in context and role != "assistant")
        ):
            raise ValueError(f"line {line_number} has an invalid message")
    elif kind == "tool_call":
        if (
            role != "assistant"
            or not _nonempty_string(context["call_id"])
            or not _nonempty_string(context["name"])
            or not isinstance(context["arguments"], dict)
        ):
            raise ValueError(f"line {line_number} has an invalid tool call")
    elif kind == "tool_result":
        if (
            role != "tool"
            or not _nonempty_string(context["call_id"])
            or not _nonempty_string(context["name"])
            or not isinstance(context["content"], str)
            or not isinstance(context["is_error"], bool)
        ):
            raise ValueError(f"line {line_number} has an invalid tool result")
    elif (
        role != "assistant"
        or not _nonempty_string(context["summary"])
        or not _integer(context["cutoff"])
        or context["cutoff"] <= 0
        or not _integer(context["token_usage"])
        or context["token_usage"] < 0
    ):
        raise ValueError(f"line {line_number} has an invalid compaction")
    return SessionRecord(role, context, timestamp)


def _last_activity(records: list[SessionRecord], created_at: int) -> int:
    activity = [
        record.ts
        for record in records
        if record.context["type"] != "compaction"
        and record.context.get("content") != SYNTHETIC_TOOL_ERROR
    ]
    return max(activity, default=created_at)


def _record_usage(record: SessionRecord) -> int:
    value = record.context
    return (
        value.get("token_usage", 0) if value["type"] in {"message", "compaction"} else 0
    )


def _created_at(metadata: os.stat_result) -> int:
    return int(getattr(metadata, "st_birthtime", metadata.st_ctime))


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value
