from __future__ import annotations

from collections.abc import Iterator

from .client import Client
from .context import ContextManager
from .event import ConversationEvent, DoneEvent, ErrorEvent, StreamEvent, ToolCallEvent
from .tool import ToolManager


class Agent:
    def __init__(
        self,
        client: Client,
        context: ContextManager | None = None,
        tools: ToolManager | None = None,
        max_tool_rounds: int = 8,
    ) -> None:
        self.client = client
        self.context = context or ContextManager()
        self.tools = tools or ToolManager()
        self.max_tool_rounds = max_tool_rounds

    def stream(self, user_message: str) -> Iterator[StreamEvent]:
        self.context.add_user(user_message)
        self.context.set_tool_schemas(self.tools.schemas())

        for _ in range(self.max_tool_rounds + 1):
            messages = self.context.model_messages()
            assistant_index = self.context.start_assistant_stream()
            tool_called = False
            stream_finished = False

            try:
                for event in self.client.stream(
                    messages,
                    tools=self.context.tool_schemas(),
                    reasoning=self.context.reasoning,
                ):
                    if isinstance(event, ConversationEvent):
                        self.context.append_assistant_delta(
                            assistant_index, event.delta
                        )
                    elif isinstance(event, ToolCallEvent):
                        tool_called = True
                        self.context.add_tool_call(event.tool_call)
                        self.context.add_tool_result(
                            event.tool_call.call_id, self.tools.execute(event.tool_call)
                        )
                    elif isinstance(event, DoneEvent):
                        self.context.finish_assistant_stream(
                            assistant_index, event.token_usage
                        )
                        stream_finished = True
                    elif isinstance(event, ErrorEvent):
                        self.context.fail_assistant_stream(assistant_index)
                        stream_finished = True
                    yield event
            finally:
                if not stream_finished:
                    self.context.fail_assistant_stream(assistant_index)

            if not tool_called:
                break
