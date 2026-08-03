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

For a non-Git fixture directory, `base_commit` is the fixture tree SHA-256 used by
the loader. For a local Git repository, it is a commit or ref exported with
`git archive`. Downloading is deliberately separate: place downloaded bench files
and fixtures locally, then pass their file or directory to `--bench`.
