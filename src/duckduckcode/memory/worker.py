from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path
from typing import Any

from ..config import Config
from ..core.context import ContextManager, Message
from ..core.event import ConversationEvent, DoneEvent, ErrorEvent, ToolCallEvent
from ..providers.openai.client import OpenAIClient
from .long_term import MemoryError, MemoryManager
from .session import SYNTHETIC_TOOL_ERROR, SessionManager

INPUT_LIMIT = 128 * 1024
TOOL_PREVIEW_LIMIT = 1024
ACTION_TOOL = {
    "type": "function",
    "name": "ApplyMemoryActions",
    "description": (
        "Return all durable-memory changes. Use an empty actions list when nothing "
        "is worth remembering. Do not return memory changes as prose."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["create", "update", "delete"],
                        },
                        "id": {
                            "type": ["string", "null"],
                            "description": (
                                "Existing ID for update/delete; null or empty for create."
                            ),
                        },
                        "category": {
                            "type": "string",
                            "enum": [
                                "preference",
                                "feedback",
                                "project",
                                "reference",
                            ],
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["user", "project"],
                        },
                        "summary": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "body": {"type": "string"},
                    },
                    "required": [
                        "operation",
                        "id",
                        "category",
                        "scope",
                        "summary",
                        "tags",
                        "body",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["actions"],
        "additionalProperties": False,
    },
}
EXTRACTION_PROMPT = """You extract durable memory for a coding agent.

Record only stable, reusable information. Never store credentials, tokens, private
keys, transient task status, or guesses. The categories are:
- preference: user-scope stable coding or response preferences.
- feedback: user-scope explicit corrections, with narrow applicability conditions.
- project: project-scope technical facts about this workspace.
- reference: user or project scope according to where it applies.

Use the supplied inventory to update/delete duplicates, contradictions, or obsolete
facts. The host validates and publishes changes. Call ApplyMemoryActions exactly
once, including with an empty list. Return no free text.
"""
PAYLOAD_LIMIT = (
    INPUT_LIMIT
    - len(EXTRACTION_PROMPT.encode("utf-8"))
    - len(json.dumps(ACTION_TOOL, ensure_ascii=False).encode("utf-8"))
    - 2048
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()
    manager: MemoryManager | None = None
    try:
        workspace = args.workspace.resolve()
        manager = MemoryManager(workspace)
        if args.session.stem != args.session_id:
            raise MemoryError("session ID does not match its path")
        records = read_session_slice(workspace, args.session, args.start, args.end)
        prompt = build_extraction_input(manager, records)
        config = Config.from_env()
        client = OpenAIClient(
            api_key=config.openai_api_key,
            model=config.openai_model,
            langsmith_tracing=config.langsmith_tracing,
            langsmith_api_key=config.langsmith_api_key,
            langsmith_project=config.langsmith_project,
        )
        try:
            actions = request_actions(client, prompt, config.reasoning)
        finally:
            client.close()
        manager.apply_actions(actions, args.session_id)
        from .consolidate import maybe_consolidate

        maybe_consolidate(config, manager)
        manager.write_state(None)
    except Exception as exc:
        if manager is not None:
            try:
                manager.write_state(str(exc))
            except Exception:
                pass


def read_session_slice(
    workspace: Path, session_path: Path, start: int, end: int
) -> list[dict[str, Any]]:
    session_manager = SessionManager(
        workspace, ContextManager(system_prompt="memory worker validation")
    )
    sessions = session_manager.directory
    source = session_path.absolute()
    if (
        source.parent.resolve() != sessions
        or source.suffix != ".jsonl"
        or start < 0
        or end < start
    ):
        raise MemoryError("invalid session slice")
    metadata = source.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise MemoryError("session is not a regular file")
    path = sessions / source.name
    try:
        values = [record.as_dict() for record in session_manager._read(path)[0]]
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise MemoryError(f"invalid session JSONL: {exc}") from exc
    if end > len(values):
        raise MemoryError("session slice exceeds the session")
    return values[start:end]


def build_extraction_input(
    manager: MemoryManager, records: list[dict[str, Any]]
) -> str:
    user_records, user_index = manager.user.load()
    project_records, project_index = manager.project.load()
    inventory = [
        {
            "id": record.id,
            "scope": record.scope,
            "category": record.category,
            "summary": record.summary,
            "tags": list(record.tags),
            "updated_at": record.updated_at,
        }
        for record in sorted(
            [*user_records.values(), *project_records.values()],
            key=lambda item: (item.scope, item.id),
        )
    ]
    transcript = _transcript_items(records)

    def render() -> str:
        return json.dumps(
            {
                "user_MEMORY.md": user_index,
                "project_MEMORY.md": project_index,
                "inventory": inventory,
                "recent_turn": transcript,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    content = render()
    for removable in (
        lambda item: item.get("type") == "tool_result",
        lambda item: item.get("type") == "message"
        and item.get("role") == "assistant"
        and not item.get("final"),
    ):
        while len(content.encode("utf-8")) > PAYLOAD_LIMIT:
            index = next(
                (i for i, item in enumerate(transcript) if removable(item)), None
            )
            if index is None:
                break
            transcript.pop(index)
            content = render()
    if len(content.encode("utf-8")) > PAYLOAD_LIMIT:
        for item in transcript:
            if item.get("type") != "message" or item.get("role") not in {
                "user",
                "assistant",
            }:
                continue
            while (
                len(content.encode("utf-8")) > PAYLOAD_LIMIT
                and len(item.get("content", "").encode("utf-8")) > 256
            ):
                item["content"] = _truncate_utf8(
                    item["content"], 3 * len(item["content"].encode("utf-8")) // 4
                )
                item["truncated"] = True
                content = render()
    while len(content.encode("utf-8")) > PAYLOAD_LIMIT:
        index = next(
            (i for i, item in enumerate(transcript) if item.get("type") == "tool_call"),
            None,
        )
        if index is None:
            break
        transcript.pop(index)
        content = render()
    if len(content.encode("utf-8")) > PAYLOAD_LIMIT:
        raise MemoryError("memory extraction metadata exceeds 128KiB")
    return content


def request_actions(client: Any, content: str, reasoning: Any) -> list[dict[str, Any]]:
    calls = []
    text = ""
    completed = False
    for event in client.stream(
        [Message("system", EXTRACTION_PROMPT), Message("user", content)],
        tools=[ACTION_TOOL],
        reasoning=reasoning,
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
        raise MemoryError("memory extraction response did not complete")
    if text.strip() or len(calls) != 1 or calls[0].name != ACTION_TOOL["name"]:
        raise MemoryError("memory extraction must return exactly one tool call")
    arguments = calls[0].arguments
    if not isinstance(arguments, dict) or set(arguments) != {"actions"}:
        raise MemoryError("invalid memory extraction tool arguments")
    return arguments["actions"]


def _transcript_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    visible_assistants: list[int] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"role", "context", "ts"}:
            raise MemoryError("invalid session record")
        context = record["context"]
        if not isinstance(context, dict):
            raise MemoryError("invalid session record context")
        kind = context.get("type")
        if kind == "message":
            if context.get("status") != "completed" or not context.get("visible"):
                continue
            item = {
                "type": "message",
                "role": record["role"],
                "content": context.get("content", ""),
            }
            items.append(item)
            if record["role"] == "assistant":
                visible_assistants.append(len(items) - 1)
        elif kind == "tool_call":
            items.append(
                {
                    "type": "tool_call",
                    "name": context.get("name"),
                    "arguments": _bounded_json(context.get("arguments"), 8 * 1024),
                }
            )
        elif kind == "tool_result" and context.get("content") != SYNTHETIC_TOOL_ERROR:
            items.append(
                {
                    "type": "tool_result",
                    "name": context.get("name"),
                    "is_error": context.get("is_error"),
                    "preview": _preview(str(context.get("content", ""))),
                }
            )
    if visible_assistants:
        items[visible_assistants[-1]]["final"] = True
    return items


def _preview(content: str) -> str:
    encoded = content.encode("utf-8")
    if len(encoded) <= TOOL_PREVIEW_LIMIT:
        return content
    half = (TOOL_PREVIEW_LIMIT - len("\n...[truncated]...\n")) // 2
    return (
        encoded[:half].decode("utf-8", errors="ignore")
        + "\n...[truncated]...\n"
        + encoded[-half:].decode("utf-8", errors="ignore")
    )


def _truncate_utf8(content: str, limit: int) -> str:
    marker = "\n...[truncated]..."
    encoded = content.encode("utf-8")
    if len(encoded) <= limit:
        return content
    return (
        encoded[: max(0, limit - len(marker.encode()))].decode("utf-8", errors="ignore")
        + marker
    )


def _bounded_json(value: Any, limit: int) -> Any:
    rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(rendered.encode("utf-8")) <= limit:
        return value
    return {"truncated_json": _truncate_utf8(rendered, limit)}


def _unique_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


if __name__ == "__main__":
    main()
