from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal

from ..tools.tool import ToolCall, ToolResult
from .prompts import PLAN_MODE_REMINDER, build_system_prompt

SINGLE_TOOL_RESULT_LIMIT = 80 * 1024
COMBINED_TOOL_RESULT_LIMIT = 100 * 1024
TOOL_RESULT_PREVIEW_BYTES = 2 * 1024
TOOL_RESULT_CHUNK_CHARS = 16_000


@dataclass(frozen=True)
class ReasoningConfig:
    effort: str = "low"


@dataclass(frozen=True)
class Message:
    role: str
    content: str = ""
    kind: Literal["message", "tool_call", "tool_result"] = "message"
    status: Literal["completed", "streaming", "error"] = "completed"
    token_usage: int = 0
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None

    @classmethod
    def tool_call(cls, call_id: str, name: str, arguments: dict[str, Any]) -> Message:
        return cls(
            "assistant",
            kind="tool_call",
            tool_call_id=call_id,
            tool_name=name,
            tool_arguments=arguments,
        )

    @classmethod
    def tool_result(cls, call_id: str, output: str) -> Message:
        return cls("tool", output, kind="tool_result", tool_call_id=call_id)

    def to_openai(self) -> dict[str, Any]:
        if self.kind == "tool_call":
            return {
                "type": "function_call",
                "call_id": self.tool_call_id,
                "name": self.tool_name,
                "arguments": json_dumps(self.tool_arguments or {}),
            }
        if self.kind == "tool_result":
            return {
                "type": "function_call_output",
                "call_id": self.tool_call_id,
                "output": self.content,
            }
        return {"role": self.role, "content": self.content}


class ContextManager:
    def __init__(
        self,
        system_prompt: str | None = None,
        workspace: str | Path | None = None,
        mode_instructions: str = "",
        abstraction: str = "",
        reasoning: ReasoningConfig | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
        tool_result_directory: str | Path | None = None,
    ) -> None:
        self._messages: list[Message] = []
        self.system_prompt = system_prompt or build_system_prompt(
            workspace,
            mode_instructions=mode_instructions,
            tool_result_directory=tool_result_directory,
        )
        self.abstraction = abstraction
        self.reasoning = reasoning or ReasoningConfig()
        self._tool_schemas = tool_schemas or []
        self._tool_result_base = (
            Path(tool_result_directory).resolve()
            if tool_result_directory is not None
            else None
        )
        self._tool_result_session: Path | None = None
        self.mode: Literal["default", "plan"] = "default"

    def add_user(self, content: str) -> None:
        self._messages.append(Message("user", content))

    def add_assistant(self, content: str) -> None:
        self._messages.append(Message("assistant", content))

    def start_assistant_stream(self) -> int:
        self._messages.append(Message("assistant", status="streaming"))
        return len(self._messages) - 1

    def append_assistant_delta(self, index: int, delta: str) -> None:
        message = self._messages[index]
        self._messages[index] = Message(
            message.role,
            message.content + delta,
            message.kind,
            message.status,
            message.token_usage,
            message.tool_call_id,
            message.tool_name,
            message.tool_arguments,
        )

    def finish_assistant_stream(self, index: int, token_usage: int = 0) -> None:
        self._set_assistant_stream_status(index, "completed", token_usage)

    def fail_assistant_stream(self, index: int) -> None:
        self._set_assistant_stream_status(index, "error", 0)

    def _set_assistant_stream_status(
        self, index: int, status: Literal["completed", "error"], token_usage: int
    ) -> None:
        message = self._messages[index]
        self._messages[index] = Message(
            message.role,
            message.content,
            message.kind,
            status,
            token_usage,
            message.tool_call_id,
            message.tool_name,
            message.tool_arguments,
        )

    def add_tool_call(self, tool_call: ToolCall) -> None:
        self._messages.append(
            Message.tool_call(tool_call.call_id, tool_call.name, tool_call.arguments)
        )

    def add_tool_result(self, call_id: str, output: str) -> None:
        self._messages.append(Message.tool_result(call_id, output))

    def compact_tool_results(
        self,
        results: list[tuple[ToolCall, ToolResult]],
    ) -> list[tuple[ToolCall, ToolResult]]:
        if self._tool_result_base is None:
            return list(results)
        compacted = list(results)
        sizes = [_model_output_size(result) for _, result in compacted]
        attempted: set[int] = set()

        for index, size in enumerate(sizes):
            if size > SINGLE_TOOL_RESULT_LIMIT:
                attempted.add(index)
                stored = self._store_tool_result(*compacted[index])
                if stored is not None:
                    compacted[index] = (compacted[index][0], stored)
                    sizes[index] = _model_output_size(stored)

        while sum(sizes) > COMBINED_TOOL_RESULT_LIMIT:
            candidates = [
                index for index in range(len(compacted)) if index not in attempted
            ]
            if not candidates:
                break
            index = max(
                candidates, key=lambda candidate: (sizes[candidate], -candidate)
            )
            attempted.add(index)
            stored = self._store_tool_result(*compacted[index])
            if stored is None:
                continue
            stored_size = _model_output_size(stored)
            if stored_size >= sizes[index]:
                continue
            compacted[index] = (compacted[index][0], stored)
            sizes[index] = stored_size

        return compacted

    def messages(self) -> list[Message]:
        return list(self._messages)

    def model_messages(self) -> list[Message]:
        messages = [Message("system", self.system_prompt)]
        if self.mode == "plan":
            messages.append(Message("system", PLAN_MODE_REMINDER))
        if self.abstraction:
            messages.append(
                Message("system", f"Conversation summary:\n{self.abstraction}")
            )
        return messages + self.messages()

    def set_mode(self, mode: Literal["default", "plan"]) -> None:
        self.mode = mode

    def tool_schemas(self) -> list[dict[str, Any]]:
        return list(self._tool_schemas)

    def set_tool_schemas(self, tool_schemas: list[dict[str, Any]]) -> None:
        self._tool_schemas = list(tool_schemas)

    def close(self) -> None:
        if self._tool_result_session is not None:
            shutil.rmtree(self._tool_result_session, ignore_errors=True)
            self._tool_result_session = None

    def _store_tool_result(
        self,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> ToolResult | None:
        try:
            directory = self._result_session_directory()
            created_at = datetime.now(timezone.utc)
            timestamp = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
            filename = (
                f"{timestamp}__{_filename_part(tool_call.name)}__"
                f"{_filename_part(tool_call.call_id)}.txt"
            )
            path = directory / filename
            content_bytes = len(result.content.encode("utf-8"))
            metadata = json.dumps(
                {
                    "tool": tool_call.name,
                    "call_id": tool_call.call_id,
                    "created_at": created_at.isoformat(),
                    "is_error": result.is_error,
                    "content_bytes": content_bytes,
                    "format": "chunked-jsonl",
                    "chunk_chars": TOOL_RESULT_CHUNK_CHARS,
                },
                ensure_ascii=False,
            )
            with path.open("x", encoding="utf-8") as stream:
                stream.write(metadata)
                stream.write("\n")
                for index, start in enumerate(
                    range(0, len(result.content), TOOL_RESULT_CHUNK_CHARS),
                    start=1,
                ):
                    stream.write(
                        json.dumps(
                            {
                                "chunk": index,
                                "content": result.content[
                                    start : start + TOOL_RESULT_CHUNK_CHARS
                                ],
                            },
                            ensure_ascii=False,
                        )
                    )
                    stream.write("\n")
        except OSError:
            return None

        head, tail = _preview(result.content)
        preview = (
            "Tool result stored on disk.\n"
            f"Tool: {tool_call.name}\n"
            f"Call ID: {tool_call.call_id}\n"
            f"Created at: {created_at.isoformat()}\n"
            f"Size: {content_bytes} bytes\n"
            f"Path: {path}\n\n"
            f"Preview start:\n{head}\n\n"
            f"Preview end:\n{tail}\n\n"
            "The file is chunked JSONL: line 1 is metadata and later lines "
            "contain ordered content chunks. Use ReadFile with this absolute "
            "path and offset/limit to inspect it, starting at offset 2."
        )
        return ToolResult(preview, result.is_error)

    def _result_session_directory(self) -> Path:
        if self._tool_result_session is None:
            assert self._tool_result_base is not None
            self._tool_result_base.mkdir(parents=True, exist_ok=True)
            self._tool_result_session = Path(
                tempfile.mkdtemp(prefix="session-", dir=self._tool_result_base)
            ).resolve()
            self._tool_result_session.chmod(0o700)
        return self._tool_result_session


def json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value)


def _model_output_size(result: ToolResult) -> int:
    return len(result.to_model_output().encode("utf-8"))


def _filename_part(value: str) -> str:
    safe = "".join(
        (
            character
            if character.isascii()
            and (character.isalnum() or character in {".", "_", "-"})
            else "_"
        )
        for character in value
    )
    return safe[:80] or "unknown"


def _preview(content: str) -> tuple[str, str]:
    encoded = content.encode("utf-8")
    head = encoded[:TOOL_RESULT_PREVIEW_BYTES].decode("utf-8", errors="replace")
    tail = encoded[-TOOL_RESULT_PREVIEW_BYTES:].decode("utf-8", errors="replace")
    return head, tail
