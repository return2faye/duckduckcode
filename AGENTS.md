# DuckDuckCode Agent Guidance

Build this project as a minimal Python coding agent.

- Use `uv` for the virtual environment and dependency management.
- Load startup settings through `Config.from_env()` in `config.py`; avoid reading environment variables from feature classes.
- Keep the main agent flow in `agent.py`.
- Keep provider-neutral client contracts in `client.py`.
- Keep `Message`, system prompt, abstraction summary, tool schemas, and conversation history handling in `context.py`.
- Keep provider-specific message serializers/deserializers in `serialize.py` and inject them into clients.
- Keep streaming event objects in `event.py`; keep provider-specific SSE parsing/handling in `stream.py` and inject handlers into clients.
- During streaming, create one assistant placeholder message, append text deltas into it, and end with `completed` or `error`; keep token usage on `Message` until real token accounting is requested.
- `ContextManager.model_messages()` should put the system prompt first, optional abstraction second, then user/LLM conversation messages, to improve KV-cache reuse.
- Use the OpenAI Responses API through `OpenAIClient`, as one `Client` implementation.
- Use `o4-mini` as the default thinking/reasoning model unless `OPENAI_MODEL` overrides it.
- Keep reasoning effort configurable through `ReasoningConfig`; default to `low`.
- Keep tool registration, tool schemas, and tool execution in `tool.py` via `ToolManager`; attach the manager to `Agent`.
- The agent may receive tool calls from the model; execute them through its `ToolManager`, append tool call/result messages to context, then ask the model again.
- Keep multi-turn state in `ContextManager`; do not add persistent memory until requested.
- Apply the ponytail skill for development: prefer stdlib/native features, reuse existing code, avoid speculative abstractions, and add dependencies only when they are clearly needed.
- Put local secrets in `.env`; keep `.env.example` safe to commit.
- Run `uv run duckduckcode "hello"` for a real API smoke test when `OPENAI_API_KEY` is configured.
