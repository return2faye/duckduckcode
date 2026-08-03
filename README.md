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
- `LANGSMITH_TRACING=false`
- `OPENAI_JUDGE_MODEL=gpt-5.6-terra`

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to trace Agent and Judge
Responses API calls to the `LANGSMITH_PROJECT` project.

## Chat

```bash
uv run duckduckcode
```

DuckDuckCode starts the TUI by default. The upper area shows the duck banner, version, working directory, and chat history. The lower input area accepts prompts, and the bottom status bar shows the model and token usage. Press `Esc` to interrupt a response, or press it while idle to exit.

## Structure

- `agent.py`: main multi-turn agent flow
- `backend.py`: JSONL pipe backend used by the TUI frontend
- `client.py`: provider-neutral `Client` abstraction
- `config.py`: startup configuration loaded from environment variables
- `context.py`: `Message` object, system prompt, abstraction, tool schemas, and in-memory `ContextManager`
- `event.py`: internal streaming events such as `ConversationEvent`, `ToolCallEvent`, and `ErrorEvent`
- `eval/`: benchmark loading, isolated execution, local result storage, and judging
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

## Local evaluations

```bash
uv run duckduckcode-eval
uv run duckduckcode-eval --case single-file-bug
uv run duckduckcode-eval --bench path/to/bench.jsonl
uv run duckduckcode-eval --bench evals/benches/context
uv run duckduckcode-eval-report
```

Benches come from JSON/JSONL files under `evals/benches` by default; `--bench`
accepts another file or directory and may be repeated. Repo fixtures live outside
the case record and are materialized at the declared content hash or Git commit.
Runs use isolated temporary workspaces and append results to
`.duckduckcode/evals.sqlite3`; each case makes one independent Judge API call.
Network tool requests are denied during evaluation. See
[evals/README.md](evals/README.md) for the bench contract.

The context suite checks critical-fact retention, latest-instruction precedence,
and pending-work retention. A run fails deterministically when its observed
compaction count differs from `metadata.expected_compactions`; the Judge also
receives each compacted summary and grades whether it kept the required facts
without promoting irrelevant padding. `duckduckcode-eval-report` renders the
latest batch as `.duckduckcode/eval-reports/eval-report.html`, including token usage,
compaction before/after sizes and summaries, complete tool arguments/results/errors,
permission decisions, tests, scores, and diffs. The Judge receives the same tool trace.

SQLite stores local evaluation state only:

- `cases` is the synchronized benchmark catalog: normalized case JSON, source
  hash, and active/inactive status. Missing cases become inactive; history is not
  deleted.
- `evaluations` is append-only run history: batch and case IDs, models, completion
  status, score and reason, final answer, workspace diff, tool events, required
  test results, validation errors, token usage, duration, compaction events and
  summaries, and runtime errors.

Bench JSON/JSONL files and repo fixtures remain regular files. Each temporary
workspace is removed after its case finishes and is not stored in SQLite.

SQLite is used because batch/case filtering, append-only history, schema migration,
and regenerating reports are needed together; Python provides it in the standard
library and writes one local transactional file. It is not the benchmark source of
truth. For a disposable single-run result, JSONL would be enough.
