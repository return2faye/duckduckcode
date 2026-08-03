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

Chats resume the most recently active valid workspace session automatically. Use
`/sessions` to open an arrow-key session picker (Enter resumes, Esc closes),
`/new` to start an empty one, and `/delete-session [id]` twice to confirm
permanent deletion.
Session switching is available only while the agent is idle and outside Plan Mode.

## Structure

- `core/agent.py`: main multi-turn agent flow
- `interfaces/backend.py`: JSONL pipe backend used by the TUI frontend
- `core/client.py`: provider-neutral `Client` abstraction
- `config.py`: startup configuration loaded from environment variables
- `core/context.py`: `Message`, system prompt, abstraction, tool schemas, and `ContextManager`
- `core/event.py`: internal streaming and session lifecycle events
- `eval/`: benchmark loading, isolated execution, local result storage, and judging
- `memory/instruction.py`: project and user instruction loading
- `memory/long_term.py`: validated user/project long-term memory storage and injection
- `memory/session.py`: secure workspace-local JSONL session persistence
- `memory/worker.py`: asynchronous per-loop memory extraction
- `memory/consolidate.py`: locked staging-based periodic memory consolidation
- `providers/openai/`: OpenAI Responses API client, serializers, and SSE handling
- `interfaces/tui.py`: curses frontend that talks to the backend through pipes
- `tools/tool.py`: `ToolManager`, tool schemas, and tool-call execution

`ContextManager` builds the model context: system prompt first, optional abstraction
summary, long-term-memory snapshot, and stale-session reminder next, then user,
assistant, tool-call, and tool-result messages.

`Agent` owns `ToolManager`, passes tool schemas into `ContextManager`, executes returned tool calls, appends the tool call/result messages, then asks the model again for the final answer.

## User and project instructions

DuckDuckCode loads instruction files once when an agent starts, in increasing
priority order:

1. `~/.duckduckcode/DDCODE.md`
2. `<workspace>/DDCODE.md`
3. `<workspace>/.duckduckcode/DDCODE.md`
4. `<workspace>/DDCODE.local.md`

Missing and empty files are skipped. Later files override conflicting earlier
instructions, but cannot override DuckDuckCode's built-in safety or mode rules.
An instruction line containing only `@relative/path` expands that UTF-8 file in
place, relative to the file containing the reference. References stay inside the
workspace (or `~/.duckduckcode` for user instructions), including after resolving
symbolic links. Absolute paths, globs, URLs, and anchors are unsupported. Cycles
fail startup, duplicate references expand once per top-level file, and nesting is
limited to five referenced levels.

Expanded instructions remain in the system prompt, count toward the static token
estimate with tool schemas, and are never summarized or compacted. Restart the
agent to pick up changes. Startup fails if this non-compactable static context
reaches the compaction threshold. `duckduckcode-eval` loads instruction files from
its fixture workspace but excludes the machine's user-level file for reproducible
runs.

## Skills

DuckDuckCode discovers Skills from two directories on startup and before each
user turn:

- user: `~/.duckduckcode/skills/`
- project: `<workspace>/.duckduckcode/skills/`

Entries can be either `skills/name.md` files or `skills/name/SKILL.md`
directories. Project Skills override user Skills with the same `name`; duplicate
names inside one scope are skipped. The agent does not create these directories
or add them to `.gitignore`, so project Skills can be committed.

Each entry is a UTF-8 Markdown file with YAML frontmatter:

```markdown
---
name: debug-tests
description: Debug failing Python tests.
mode: inline
---

Skill instructions go here.
```

`name` is required, must be lowercase kebab-case up to 64 characters, and becomes
the slash command `/<name>`. `description` is required. `mode` is optional and
defaults to `inline`; `fork` runs the Skill in an isolated child Agent. Unknown
frontmatter fields are preserved but unused.

Invalid Skills do not stop startup. Symlinks, non-regular files, duplicate YAML
fields, invalid YAML, empty bodies, invalid modes, command-name conflicts, and
files larger than 256KiB are skipped and reported once per changed warning set
through `ErrorEvent(code="skill")`.

Only the Skill catalog (`name` and `description`) is injected before loading.
When the model or user selects a Skill, the read-only `LoadSkill` tool loads the
full body for the current turn only. Inline Skills enter the parent Agent's
active system block. Fork Skills receive the parent history through the current
user message, abstraction, memory, system instructions, mode, permission policy,
and a separate client and tool runtime; Skills and sessions are disabled in the
child. Child tool activity and usage remain visible, with prefixed call IDs, but
only the final child reply is returned through the parent's `LoadSkill` result.
Child failure becomes an error tool result so the parent can recover. Skill
bodies are not written to sessions, summarized during compaction, or retained
after success, failure, or interruption. `ContextManager.model_messages()` keeps
the main system prompt first, optional abstraction second, then memory, Skill
catalog, active Skills, reminders, mode instructions, and conversation messages.

Directory Skills temporarily grant `ReadFile`, `Glob`, and `Grep` read access to
that Skill directory while active. File Skills do not expose their parent
directory. `WriteFile` and `EditFile` remain limited to the workspace and private
temporary directories; project Skill files follow normal workspace permissions,
while user Skill files are not writable through file tools. `full_access` Bash
keeps its existing behavior.

Use `/skills` to refresh the list and open a multi-select menu for the next
prompt. Each Skill also registers as `/<name>`: the bare command selects it for
the next prompt, and `/<name> <prompt>` sends the prompt immediately with that
Skill selected. Explicit selections are loaded before the first model request;
the model can still call `LoadSkill` for additional matching Skills from the
catalog. Selections clear after the message is accepted. `duckduckcode-eval`
passes `enable_skills=False`, so evaluations do not read user or project Skill
paths.

The default model is `o4-mini`, a reasoning model. Reasoning effort defaults to `low`; CLI selection can be added later.

`OpenAIClient.stream()` uses Responses API SSE streaming and yields internal stream events. The parser currently handles text deltas, function tool calls, errors, and completion.

During streaming, `Agent.stream()` creates an empty assistant message, appends text deltas into it, then marks it `completed` or `error`. `Message.token_usage` records returned usage for now; token accounting can be added later.

## Sessions

Each workspace stores chats in `.duckduckcode/sessions/`. Session files are
UTF-8 JSONL named with local time (`YYYYMMDD-HHMMSS.jsonl`, then `-2`, `-3`,
and so on for same-second collisions). The directory is mode `0700`; files are
mode `0600` and ignored by Git. Every line has exactly `role`, `context`, and
Unix-second `ts` fields. Context records are one of:

- `message`: content, completed/error status, token usage, and TUI visibility
- `tool_call`: call ID, tool name, and arguments
- `tool_result`: call ID, tool name, content, and error status
- `compaction`: summary, model-message cutoff, and token usage

DuckDuckCode writes, flushes, and `fsync`s a record before adding it to model
context. A persistence failure stops the turn or tool chain without advancing
context. Completed and interrupted assistant replies are saved; an unfinished
delta can be lost only if the process hard-crashes mid-stream. On recovery, a
tool call that has no result receives an explicit synthetic error result so the
model never sees a broken tool chain.

Compaction appends a checkpoint before replacing old model messages with its
summary. The source JSONL remains append-only, so the TUI can restore the full
visible history while the model receives the latest summary plus uncompressed
messages. Stored token usage is restored into the TUI total. If restored context
already reaches the compaction threshold, startup attempts automatic compaction
once before accepting input; failures keep the session intact and suggest
`/new`.

When the last real activity is strictly more than 24 hours old, the model gets a
derived reminder to re-read relevant files because workspace code may have
changed. The reminder is not written to disk and does not update activity time.
Before automatic restore, valid sessions strictly older than 30 days are deleted;
empty sessions use file creation time. Invalid JSONL and symbolic links are kept,
reported as invalid, never restored, and never cleaned automatically.

Session storage is intentionally workspace-local and single-process. There are
no cross-process locks, titles, search, export, or configurable retention yet.
Evaluation runs explicitly disable sessions and never read or write this
directory.

## Automatic long-term memory

DuckDuckCode keeps durable memory in two local scopes:

- user: `~/.duckduckcode/memory/`
- project: `<workspace>/.duckduckcode/memory/` (ignored by Git)

Each scope has a generated `MEMORY.md` index and one Markdown file per memory at
`<category>/<id>.md`. Categories are `preference` for stable user preferences,
`feedback` for narrowly applicable lessons from explicit user corrections,
`project` for workspace facts, and `reference` for reusable reference material.
Preferences and feedback are user-scoped, project facts are project-scoped, and
references use whichever scope they apply to.

Every memory file has strict YAML frontmatter with exactly `id`, `category`,
`scope`, `summary`, `tags`, `created_at`, `updated_at`, and `source_session`.
Its body records the fact, applicability boundary, and necessary details. The
index links every file exactly once and repeats the same self-contained summary.
Invalid UTF-8, duplicate fields or IDs, path traversal, symbolic links,
non-regular files, inconsistent metadata, and likely credentials/private keys
are rejected before publication.

At startup and before persisting each new user message, DuckDuckCode refreshes a
snapshot containing the user index first and project index second. Individual
memory files are not injected. The complete tagged snapshot, separator, and any
truncation warning are limited to 200 lines and 25KB of valid UTF-8. When needed,
project index lines receive space before user index lines. Memory is a separate,
non-compactable system message after the optional conversation abstraction and
is explicitly marked as possibly stale background that cannot override safety,
DDCODE instructions, or the current request. Startup fails when the system
prompt, DDCODE instructions, memory, and tool schemas already reach the
compaction threshold.

After a loop has successfully persisted its final assistant reply and has no
pending tools, the agent starts one detached `python -m duckduckcode.memory.worker`
process and immediately returns. The worker rereads only that loop's JSONL
records, keeps visible messages plus bounded tool arguments/result previews, and
limits extraction input to 128KiB. It makes one Responses API call with the
configured model/reasoning and accepts changes only through one strict tool call.
The model decides whether memories are durable, duplicate, contradictory, or
obsolete; the host validates `create`, `update`, and `delete` actions, generates
IDs/timestamps, and atomically regenerates affected indexes. Credentials, tokens,
private keys, and transient task status must never be stored.

Writers take POSIX `flock` locks at each scope's `.write-lock` in sorted absolute
path order. After extraction, consolidation becomes eligible only when the
project `.consolidate-lock` is strictly older than seven days and at least five
distinct valid sessions have real activity since the previous success.
Compaction checkpoints, synthetic repairs, and invalid sessions do not count.
Only one consolidator can hold the non-blocking lock; its PID is diagnostic, the
run times out after one hour, and a failed run restores the prior successful
mtime.

Consolidation copies both scopes to staging while holding both write locks. Its
restricted agent can read staging and workspace sessions, can write only staging
memory, and has a no-shell facade limited to read-only `ls`, `cat`, `grep`, and
`rg`. It locates memories, selectively gathers recent session evidence, resolves
duplicates/conflicts and relative dates, then prunes indexes. The host validates
all staged files, frontmatter, IDs, links, summaries, scopes, and the untruncated
combined index limit before publishing each scope in file/index/delete order.
Failure or timeout leaves real memory unchanged.

A running worker never blocks the next turn; the old valid snapshot remains in
use until refresh. A corrupt replacement also leaves the old snapshot active.
Background failures are written atomically to project memory state and surfaced
once on the next turn through the existing `ErrorEvent(code="memory")`; a later
successful task clears the state. Evaluation runs pass `enable_memory=False`, so
they do not read user memory, create memory directories, or launch workers.

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
