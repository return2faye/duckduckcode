from __future__ import annotations

from .client import Client
from .context import ContextManager
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

    def ask(self, user_message: str) -> str:
        self.context.add_user(user_message)
        self.context.set_tool_schemas(self.tools.schemas())

        for _ in range(self.max_tool_rounds):
            response = self.client.generate(
                self.context.model_messages(),
                tools=self.context.tool_schemas(),
                reasoning=self.context.reasoning,
            )
            if not response.tool_calls:
                self.context.add_assistant(response.text)
                return response.text

            for tool_call in response.tool_calls:
                self.context.add_tool_call(tool_call)
                self.context.add_tool_result(tool_call.call_id, self.tools.execute(tool_call))

        raise RuntimeError("Too many tool calls")
