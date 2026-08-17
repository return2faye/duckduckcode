# duckduckcode

Minimal Python coding agent shell using OpenAI or DeepSeek.

## Setup

```bash
uv sync
cp .env.example .env
```

Configure each runtime role independently in `.env`:

```env
AGENT_PROVIDER=openai
AGENT_MODEL=o4-mini
SUBAGENT_PROVIDER=openai
SUBAGENT_MODEL=o4-mini
MEMORY_PROVIDER=openai
MEMORY_MODEL=o4-mini
JUDGE_PROVIDER=openai
JUDGE_MODEL=gpt-5.6-terra
```

Provider values are `openai` or `deepseek`. Set `OPENAI_API_KEY` and/or
`DEEPSEEK_API_KEY` for the providers selected by these roles. DeepSeek defaults
to `deepseek-v4-pro` when a role model is omitted; override its endpoint with
`DEEPSEEK_BASE_URL` if needed. Existing `OPENAI_MODEL` and
`OPENAI_JUDGE_MODEL` values remain OpenAI-only compatibility fallbacks.

Other defaults:

- `OPENAI_REASONING_EFFORT=low`
- `LANGSMITH_TRACING=false`

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` to trace Agent and Judge
calls to the `LANGSMITH_PROJECT` project.

## Chat

Install the command once from this repository (editable installs pick up local
code changes without reinstalling):

```bash
uv tool install --editable .
```

Then start DuckDuckCode from any directory. That directory becomes the
workspace:

```bash
cd /path/to/project
duckduckcode
```

DuckDuckCode starts the TUI by default. The upper area shows the duck banner, version, working directory, and chat history. The lower input area accepts prompts, and the bottom status bar shows the model and token usage. Press `Esc` to interrupt a response, or press it while idle to exit.

Chats resume the most recently active valid workspace session automatically. Use
`/sessions` to open an arrow-key session picker (Enter resumes, Esc closes),
`/new` to start an empty one, and `/delete-session [id]` twice to confirm
permanent deletion.
Session switching is available only while the agent is idle and outside Plan Mode.

Use `/plan` to enter Plan Mode. The agent writes the approval document to
`.duckduckcode/plan.md` and maintains a separate structured task checklist at
`.duckduckcode/checklist.md` through `UpdatePlan`. Approval keeps both files for
execution; successful completion removes them, while failure, interruption, or
the iteration limit preserves them for recovery. An unfinished or invalid
checklist prevents the agent from reporting successful completion.

For long tasks, the main agent may delegate independent repository research,
diagnosis, test design, or review to read-only subagents. The main agent remains
responsible for code changes, integration, verification, and checklist updates.

## Structure

- `core/agent.py`: main multi-turn agent flow
- `interfaces/backend.py`: JSONL pipe backend used by the TUI frontend
- `core/client.py`: provider-neutral `Client` abstraction
- `config.py`: startup configuration loaded from environment variables
- `core/context.py`: `Message`, system prompt, abstraction, tool schemas, and `ContextManager`
- `core/event.py`: internal streaming and session lifecycle events
- `core/mcp.py`: MCP configuration, transports, discovery, and `MCPTool` management
- `core/lsp.py`: LSP configuration, stdio JSON-RPC, document sync, and navigation
- `core/subagent.py`: subagent Definition discovery and worker lifecycle
- `core/worktree.py`: persistent isolated worktrees, recovery, dependency injection, and patches
- `eval/`: benchmark loading, isolated execution, local result storage, and judging
- `memory/instruction.py`: project and user instruction loading
- `memory/long_term.py`: validated user/project long-term memory storage and injection
- `memory/session.py`: secure workspace-local JSONL session persistence
- `memory/worker.py`: asynchronous per-loop memory extraction
- `memory/consolidate.py`: locked staging-based periodic memory consolidation
- `providers/openai/`: OpenAI Responses API client, serializers, and SSE handling
- `providers/deepseek/`: DeepSeek Chat Completions client, serializers, and stream handling
- `interfaces/tui.py`: curses frontend that talks to the backend through pipes
- `tools/tool.py`: the `Tool` protocol, `BuiltinTool`, `ToolManager`, schemas, and execution
- `tools/worktree.py`: persistent worktree listing and protected removal tools

`ContextManager` builds the model context: system prompt first, optional abstraction
summary, long-term-memory snapshot, stale-session and active-checklist reminders
next, then user, assistant, tool-call, and tool-result messages.

`Agent` owns `ToolManager`, which treats built-in and MCP tools as equal `Tool`
implementations. It passes their schemas into `ContextManager`, executes returned
tool calls, appends the tool call/result messages, then asks the model again for
the final answer.

## MCP servers

DuckDuckCode loads MCP servers once at startup from these files, with a project
entry replacing the complete user entry that has the same server name:

1. `~/.duckduckcode/mcp.yaml`
2. `<workspace>/.duckduckcode/mcp.yaml`

The root is a server map. Stdio servers accept only `type`, `command`, `args`,
`env`, and `concurrency_safe_tools`; Streamable HTTP servers accept only `type`,
`url`, `headers`, and `concurrency_safe_tools`:

```yaml
files:
  type: stdio
  command: npx
  args: ["-y", "@modelcontextprotocol/server-filesystem", "."]
  env:
    TOKEN: "${FILES_TOKEN}"
  concurrency_safe_tools: [read_file]

remote:
  type: http
  url: https://example.com/mcp
  headers:
    Authorization: "Bearer ${MCP_TOKEN}"
```

`${VAR}` expansion applies only to `env` and `headers` values, using the
startup environment captured by `Config.from_env()`. A missing variable skips
that server and emits an MCP warning. Empty files are valid. Symlinks,
non-regular files, duplicate YAML fields, unknown transport fields, invalid
HTTP(S) URLs, and files larger than 256 KiB are rejected; one invalid server
does not prevent valid sibling servers from starting.

User configuration is trusted automatically. New or changed project
configuration asks for approval before any connection or process starts. The
approval shows only server names, transports, command/args, or URLs—never env
or header values. “Allow once” lasts for the process; “always allow” atomically
stores the exact file digest in ignored `.duckduckcode/mcp.trust`; denial uses
only the user layer. Non-interactive CLI runs deny an untrusted project layer.

Discovered tools are named `mcp__<server>__<tool>` and use their MCP input
schema with non-strict model validation. `MCPManager` retains each as an
`MCPTool`; `mcp_tools()` provides the ordered discovery catalog for management
code. Model context receives a stable, untrusted catalog containing only MCP
tool names and descriptions. No remote MCP tool is initially callable: the
model must call the read-only local `LoadTools` with exact catalog names to add
complete schemas for the current session. Loaded schemas survive compaction and
are reconstructed when the same session is restored. MCP calls remain
potentially side-effecting: Plan Mode blocks them and other modes use the
normal permission panel; `LoadTools` remains available in Plan Mode and needs
no MCP-call approval. “Always allow” stores an exact, canonical JSON argument
match in project-local permission rules. Search, pagination, unloading,
Resources, and dynamic tool refresh are unsupported.

Servers initialize concurrently with a 10-second per-server limit. Tool calls
have a 60-second limit. A failed, timed-out, or disconnected session does not
stop other servers and is not recreated by DuckDuckCode. The SDK remains
responsible for protocol-level SSE resumption inside a live Streamable HTTP
session. Sessions, HTTP clients, and stdio processes remain open until the
Agent closes.

MCP tools execute serially by default. `concurrency_safe_tools` is an optional
list of exact remote tool names that the user has verified can run concurrently;
unknown names produce a startup warning. DuckDuckCode does not infer this from
MCP annotations because the protocol has no concurrency-safety contract and
annotations are untrusted hints. Large-result storage is calculated across all
results from one model turn, independently of whether those calls ran in
parallel.

The first MCP release intentionally omits prompts, sampling, OAuth, health
checks, manager-level automatic reconnect, configuration hot reload, and
dynamic `tools/list_changed` refresh. Restart DuckDuckCode after changing
configuration or server tool definitions.

## Language servers

DuckDuckCode loads user-level LSP configuration once from
`~/.duckduckcode/lsp.yaml`. The root is a server map; `command` and a non-empty
`extensions` map are required:

```yaml
pyright:
  command: pyright-langserver
  args: ["--stdio"]
  extensions:
    ".py": python
    ".pyi": python
  env:
    PYTHONPATH: "${PYTHONPATH}"
  initialization_options: {}
  settings: {}
```

Install each configured language server yourself and make its command available
on the startup `PATH`. `${VAR}` expansion applies only to `env` values and uses
the environment captured by `Config.from_env()`; a missing variable skips that
server and emits one LSP warning. Empty files are valid. Symlinks, non-regular
files, duplicate or unknown YAML fields, non-JSON-compatible options/settings,
and files larger than 256 KiB are rejected. One invalid server does not hide
valid sibling entries.

When at least one server is valid, the model receives one strict, read-only
`LSP` tool for `definition`, `references`, `hover`, `document_symbols`, and
`workspace_symbols`. Every call selects an exact configured server. Source
paths must resolve to regular UTF-8 files inside the workspace. Input lines and
columns are 1-based Unicode code-point coordinates; returned JSON is the
language server's unmodified result, including file URIs and native 0-based LSP
positions. Because the tool is read-only, it is available in Plan Mode.

Servers start lazily on first use in one background asyncio runtime. Startup is
limited to 10 seconds, navigation requests to 30 seconds, and shutdown to 5
seconds; timed-out requests receive `$/cancelRequest`. A failed or disconnected
server is isolated and is not restarted in the current process. The main Agent
owns the manager, while in-process Fork children register their own `LSP` tool
bound to the same server sessions and do not close them. Definition workers,
evaluation, and memory helpers disable LSP.

The first LSP release intentionally omits diagnostics, completion, rename, code
actions, workspace edits, TCP/socket transports, project-level configuration,
automatic detection, file watchers, hot reload, and automatic reconnect.

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

The main-agent default is OpenAI `o4-mini`. Reasoning effort defaults to `low`.

## Subagents

The model can call one strict `Agent` tool to run a non-interactive subagent. All
seven fields are required; nullable values use JSON `null`:

```json
{
  "prompt": "Inspect the session persistence flow",
  "description": "Trace sessions",
  "subagent_type": "explore",
  "model": null,
  "run_in_background": true,
  "name": null,
  "isolation": true
}
```

`subagent_type` selects a Definition. `null` creates a fork that inherits the
parent abstraction, memory, completed conversation, permission mode, and
persistent permission rules. Subagents use `SUBAGENT_PROVIDER` and
`SUBAGENT_MODEL`; a non-null tool `model` overrides only that model. Forks always run in the
background. Definition subagents receive project startup instructions, their
Definition body, and the assigned prompt, but no parent conversation. They are
limited to `ReadFile`, `Glob`, and `Grep`, minus any tools listed by the
Definition. Subagents cannot load Skills, enter Plan Mode, recurse through
`Agent`, persist sessions or memory, or ask the user for permission; an action
that would require confirmation is denied.

Definitions are Markdown files discovered before each user turn in increasing
priority order:

1. packaged `explore` and `plan` Definitions
2. `~/.duckduckcode/agents/*.md`
3. `<workspace>/.duckduckcode/agents/*.md`

Project entries override user and packaged entries with the same `type`.
Duplicates inside one scope are skipped and reported. Files must be regular
UTF-8 Markdown, no larger than 256KiB, with unique YAML fields and a non-empty
body:

```markdown
---
type: explore
whenToUse: Use for focused repository exploration and evidence gathering.
disallowedTools: []
maxTurns: 20
---

Role, responsibilities, and output constraints go here.
```

`type` is lowercase kebab-case, `whenToUse` is non-empty, `maxTurns` is 1–50,
and `disallowedTools` may name only registered tools. Refreshing Definitions
updates the enum on the existing `Agent` schema; it never creates per-Definition
tools.

At most four workers run at once, each with a 600-second deadline. A foreground
Definition task streams its tool and usage events but not its answer text; press
Ctrl+B to leave that worker running in the background. Ctrl+B is ignored when no
foreground subagent is active. Background completion or failure is delivered as
untrusted hidden task data before the next model request in the session that
created it. Switching sessions does not cancel work or leak results; deleting a
session terminates its workers. Tasks are process-local and are not restored
after restart.

An isolated Definition runs read-only in a temporary workspace snapshot. An
isolated writable fork instead requires a clean Git repository and starts from
the current `HEAD` in a persistent worktree. Its stable ID is derived from the
chat session ID and slug, and its branch is
`worktree-<slug>-<session-hash-8>`. Explicit `name` values must be at most 48 ASCII
characters and match `[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*`; a missing name is
derived safely from `description`. The worktree lives under
`~/.duckduckcode/worktrees/<repo-hash>/<worktree-id>` and is reused only by the
same chat and slug. Its original base commit is never reset when the parent
advances. Workers receive the worktree as their own tool root without changing
the main process cwd, and cannot modify their `.git` pointer or shared Git
metadata. Git commands disable Manager hooks and interactive terminal prompts;
the main repository's effective hooks path is set explicitly as worktree-local
configuration for tools that inspect it.

Ignored local files are absent by default. Users may opt Git-ignored,
non-symlink dependencies into isolated worktrees with
`~/.duckduckcode/worktree.yaml`:

```yaml
copy:
  - .env
  - .duckduckcode/permissions.local.yaml

symlinks:
  - .venv
  - node_modules
```

`copy` accepts regular files; `symlinks` accepts regular files or directories
and creates absolute links. Paths are relative to the workspace, limited and
validated, and both forms are read-only to dedicated write tools and the OS
sandbox. A persisted SHA-256 manifest lets the Manager refresh safe copies and
remove obsolete injections without overwriting externally changed destinations.
Missing, tracked, non-ignored, source-symlink, unsafe, or conflicting entries
are skipped with a warning. Both lists default empty because they may expose
secrets or large local dependencies to a subagent.

After the worker stops, DuckDuckCode captures a binary full-index patch plus the
fixed base commit and changed-file list, reports whether the parent workspace
changed, and marks failed or timed-out output as partial. The patch is cumulative
from the first base across reused tasks. It never applies or merges the patch.
The main Agent must inspect the base, `parent_changed`, `partial`, and patch
before reproducing changes. Large patches use existing on-disk tool-result
storage.

Worktree state and the injection manifest are atomically persisted in ignored
`.duckduckcode/worktrees.json`. Startup validates only Manager-owned state and
Git registration, changes a crashed active session back to idle, and preserves
its checkout and modifications. `/new`, session deletion, task failure, timeout,
and process shutdown do not remove persistent worktrees. `ListWorktrees` reports
their exact IDs, activity, dirtiness, base, and parent status. `RemoveWorktree`
is the only normal removal path: it refuses active worktrees, captures a final
patch, then removes the checkout and branch. A dirty removal always asks once,
even in full-access mode, and an “always allow” response applies only to that
request. Corrupt or externally changed ownership metadata is warned about and
never automatically pruned or reset.

A non-isolated fork may edit the shared workspace. Only one such fork runs at a
time: it holds a whole-task write lease that temporarily makes parent
`WriteFile`, `EditFile`, and `Bash` calls return busy. Read-only Definition tasks
and isolated worktree forks can still run concurrently.

`OpenAIClient.stream()` uses Responses API SSE streaming. `DeepSeekClient.stream()`
uses Chat Completions, converts the shared message/tool representation, assembles
fragmented tool calls, and does not expose `reasoning_content`. Both yield the
same internal text, tool-call, error/completion, and usage events.

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
uv run duckduckcode-retention
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
The separate `duckduckcode-retention` command runs five direct compaction
scenarios with 41 exact assertions; see
[evals/CONTEXT_RETENTION.md](evals/CONTEXT_RETENTION.md).

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

### SWE-bench Live

`duckduckcode-swebench-live` downloads every row of Microsoft's official
`SWE-bench-Live/SWE-bench-Live` `full` split from Hugging Face by default,
checks out each declared `base_commit` from a repository cache, runs the Agent in
a temporary clone, and atomically writes the prediction object accepted by the
official evaluator. Gold `patch` and `test_patch` fields are deliberately not
shown to the Agent. `--split lite|test|verified` is available for cheaper trial
runs; `--dataset-file` accepts an offline official JSON/JSONL export. Duplicate
instance IDs whose complete upstream rows are identical are evaluated once
because the official prediction format is keyed by instance ID; conflicting
duplicates are rejected.

```bash
uv run duckduckcode-swebench-live \
  --repository-cache ~/.cache/duckduckcode/swebench-repos \
  --predictions .duckduckcode/swebench-predictions.json \
  --clone-missing
```

The output keeps completed/error metadata for diagnosis, while the official
runner reads only `model_patch`. Existing predictions are resumable and skipped
unless `--overwrite` is given. Agent network tool calls are denied; repository
cloning occurs only when `--clone-missing` is explicit and Git prompts are
disabled.

Use the upstream [SWE-bench Live evaluator](https://github.com/microsoft/SWE-bench-Live/tree/70ec57e852e3f2d195790fe71f553e272c691833/evaluation)
to score predictions in its instance-specific Docker images. Local inference is
not a substitute for that deterministic Docker score. The upstream project notes
that a typical instance needs about 4 CPUs and 16 GB RAM, with some repositories
requiring substantially more.
