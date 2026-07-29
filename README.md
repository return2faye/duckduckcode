# duckduckcode

Minimal Python coding agent shell using the OpenAI API.

## Setup

```bash
uv sync
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

Defaults:

- `OPENAI_MODEL=o4-mini`
- `OPENAI_REASONING_EFFORT=low`

## Chat

```bash
uv run duckduckcode
```

DuckDuckCode starts the TUI by default. The upper area shows the duck banner, version, working directory, and chat history. The lower input area accepts prompts, and the bottom status bar shows the model and token usage. Press `Esc` to interrupt a response, or press it while idle to exit.

The old line REPL is still available:

```bash
uv run duckduckcode --repl
```

You can also pass one first prompt and exit after the response:

```bash
uv run duckduckcode "Say hello in one sentence."
```

## Structure

- `agent.py`: main multi-turn agent flow
- `backend.py`: JSONL pipe backend used by the TUI frontend
- `client.py`: provider-neutral `Client` abstraction
- `config.py`: startup configuration loaded from environment variables
- `context.py`: `Message` object, system prompt, abstraction, tool schemas, and in-memory `ContextManager`
- `event.py`: internal streaming events such as `ConversationEvent`, `ToolCallEvent`, and `ErrorEvent`
- `openai_client.py`: OpenAI Responses API implementation
- `serialize.py`: provider-specific message serializers and response deserializers
- `stream.py`: OpenAI SSE event parser and event handler
- `tui.py`: curses frontend that talks to the backend through stdin/stdout pipes
- `tool.py`: `ToolManager`, tool schemas, and tool-call execution

`ContextManager` builds the model context: system prompt first, optional abstraction summary second, then user, assistant, tool-call, and tool-result messages. There is no persistent memory layer yet.

`Agent` owns `ToolManager`, passes tool schemas into `ContextManager`, executes returned tool calls, appends the tool call/result messages, then asks the model again for the final answer.

The default model is `o4-mini`, a reasoning model. Reasoning effort defaults to `low`; CLI selection can be added later.

`OpenAIClient.stream()` uses Responses API SSE streaming and yields internal stream events. The parser currently handles text deltas, function tool calls, errors, and completion.

During streaming, `Agent.stream()` creates an empty assistant message, appends text deltas into it, then marks it `completed` or `error`. `Message.token_usage` records returned usage for now; token accounting can be added later.
