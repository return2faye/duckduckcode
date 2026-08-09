# Bench format

`duckduckcode-eval` reads every `.json` and `.jsonl` file under `evals/benches`
by default. A JSON file contains one case or an array of cases; each JSONL line is
one case. Use `--bench PATH` to load downloaded bench files or directories.

```json
{
  "inputs": {
    "task": "...",
    "repo_fixture": "../fixtures/example",
    "base_commit": "...",
    "conversation_script": ["..."],
    "agent_config": {
      "max_iterations": 30,
      "context_window": 32000,
      "compaction_trigger": 24000,
      "compaction_target": 10000
    }
  },
  "outputs": {
    "required_tests": ["python -m unittest"],
    "allowed_files": ["src/**", "tests/**"],
    "forbidden_files": ["pyproject.toml"],
    "critical_facts": ["..."],
    "required_actions": ["..."]
  },
  "metadata": {
    "id": "optional-stable-id",
    "suite": "compaction",
    "category": "latest-instruction",
    "difficulty": "medium",
    "expected_compactions": 1,
    "language": "python"
  }
}
```

`conversation_script` is a list of later user turns sent to the same Agent after
`task`. `allowed_files` and `forbidden_files` use shell-style path patterns.
Required tests run without network access inside the existing OS sandbox.
`expected_compactions` is enforced exactly. Completed compaction events include
their before/after token estimates and resulting summary so the Judge can assess
fact retention, instruction precedence, and unresolved work.
Every tool call is also recorded with its arguments, result or error, and permission
decision; the Judge and HTML report receive this complete trace.

The repository includes four context cases under `evals/benches/context`:

- `context-compression-quality`: reduces a large noisy tool result while retaining
  three authoritative facts.
- `context-critical-facts`: retains an early numeric contract through compaction.
- `context-latest-instruction`: preserves a newer rule across two compactions.
- `context-pending-action`: keeps unresolved work pending and resumes it later.

Run them and render the latest batch with:

```bash
uv run duckduckcode-eval --bench evals/benches/context
uv run duckduckcode-eval-report
```

`duckduckcode-retention` is the deterministic companion probe: five direct
compaction scenarios check 41 exact markers without an LLM Judge. Its design,
scoring rules, DeepSeek baseline, fixes, and known limits are documented in
[CONTEXT_RETENTION.md](CONTEXT_RETENTION.md).

For a non-Git fixture directory, `base_commit` is the fixture tree SHA-256 used by
the loader. For a local Git repository, it is a commit or ref exported with
`git archive`. Downloading is deliberately separate: place downloaded bench files
and fixtures locally, then pass their file or directory to `--bench`.
