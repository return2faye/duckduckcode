from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


@dataclass(frozen=True)
class AgentConfig:
    max_iterations: int
    context_window: int
    compaction_trigger: int
    compaction_target: int


@dataclass(frozen=True)
class BenchCase:
    id: str
    task: str
    repo_fixture: str
    base_commit: str
    conversation_script: tuple[str, ...]
    agent_config: AgentConfig
    required_tests: tuple[str, ...]
    allowed_files: tuple[str, ...]
    forbidden_files: tuple[str, ...]
    critical_facts: tuple[str, ...]
    required_actions: tuple[str, ...]
    metadata: dict[str, Any]
    source_path: Path
    source_hash: str

    @property
    def fixture_path(self) -> Path:
        path = Path(self.repo_fixture)
        return (
            path if path.is_absolute() else self.source_path.parent / path
        ).resolve()

    def record(self) -> dict[str, Any]:
        return {
            "inputs": {
                "task": self.task,
                "repo_fixture": self.repo_fixture,
                "base_commit": self.base_commit,
                "conversation_script": list(self.conversation_script),
                "agent_config": {
                    "max_iterations": self.agent_config.max_iterations,
                    "context_window": self.agent_config.context_window,
                    "compaction_trigger": self.agent_config.compaction_trigger,
                    "compaction_target": self.agent_config.compaction_target,
                },
            },
            "outputs": {
                "required_tests": list(self.required_tests),
                "allowed_files": list(self.allowed_files),
                "forbidden_files": list(self.forbidden_files),
                "critical_facts": list(self.critical_facts),
                "required_actions": list(self.required_actions),
            },
            "metadata": dict(self.metadata),
        }


def load_benches(paths: list[Path] | Path) -> list[BenchCase]:
    requested = [paths] if isinstance(paths, Path) else paths
    files: list[Path] = []
    for requested_path in requested:
        path = requested_path.resolve()
        if path.is_dir():
            files.extend(sorted((*path.rglob("*.json"), *path.rglob("*.jsonl"))))
        elif path.suffix in {".json", ".jsonl"} and path.is_file():
            files.append(path)
        else:
            raise RuntimeError(f"Bench path '{requested_path}' is not JSON or JSONL")
    cases: list[BenchCase] = []
    ids: set[str] = set()
    for path in sorted(set(files)):
        for raw, record_number, multiple in _records(path):
            fallback_id = f"{path.stem}:{record_number}" if multiple else path.stem
            case = _validate_case(raw, path, record_number, fallback_id)
            if case.id in ids:
                raise RuntimeError(f"Duplicate bench case id '{case.id}'")
            ids.add(case.id)
            cases.append(case)
    return cases


def _records(path: Path) -> list[tuple[Any, int, bool]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read bench '{path}': {exc}") from exc
    if path.suffix == ".jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append((json.loads(line), line_number, True))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
        return records
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path}: invalid JSON: {exc}") from exc
    values = loaded if isinstance(loaded, list) else [loaded]
    return [(value, index, len(values) > 1) for index, value in enumerate(values, 1)]


def _validate_case(
    raw: Any, path: Path, record_number: int, fallback_id: str
) -> BenchCase:
    location = f"{path}:{record_number}"
    _exact_object(raw, {"inputs", "outputs", "metadata"}, location)
    inputs = raw["inputs"]
    outputs = raw["outputs"]
    metadata = raw["metadata"]
    _exact_object(
        inputs,
        {"task", "repo_fixture", "base_commit", "conversation_script", "agent_config"},
        f"{location}.inputs",
    )
    _exact_object(
        outputs,
        {
            "required_tests",
            "allowed_files",
            "forbidden_files",
            "critical_facts",
            "required_actions",
        },
        f"{location}.outputs",
    )
    _require_object(metadata, f"{location}.metadata")
    for field in ("suite", "category", "difficulty", "language"):
        _nonempty_string(metadata.get(field), f"{location}.metadata.{field}")
    expected_compactions = metadata.get("expected_compactions")
    if (
        isinstance(expected_compactions, bool)
        or not isinstance(expected_compactions, int)
        or expected_compactions < 0
    ):
        raise RuntimeError(
            f"{location}.metadata.expected_compactions must be a non-negative integer"
        )
    case_id = metadata.get("id", fallback_id)
    _nonempty_string(case_id, f"{location}.metadata.id")
    for field in ("task", "repo_fixture", "base_commit"):
        _nonempty_string(inputs[field], f"{location}.inputs.{field}")
    script = _string_list(
        inputs["conversation_script"], f"{location}.inputs.conversation_script"
    )
    config = _agent_config(inputs["agent_config"], f"{location}.inputs.agent_config")
    required_tests = _string_list(
        outputs["required_tests"], f"{location}.outputs.required_tests"
    )
    allowed_files = _string_list(
        outputs["allowed_files"], f"{location}.outputs.allowed_files"
    )
    forbidden_files = _string_list(
        outputs["forbidden_files"], f"{location}.outputs.forbidden_files"
    )
    for field, patterns in (
        ("allowed_files", allowed_files),
        ("forbidden_files", forbidden_files),
    ):
        for pattern in patterns:
            if (
                PurePosixPath(pattern).is_absolute()
                or PureWindowsPath(pattern).is_absolute()
                or ".." in PurePosixPath(pattern.replace("\\", "/")).parts
            ):
                raise RuntimeError(
                    f"{location}.outputs.{field} has unsafe path '{pattern}'"
                )
    critical_facts = _string_list(
        outputs["critical_facts"], f"{location}.outputs.critical_facts"
    )
    required_actions = _string_list(
        outputs["required_actions"], f"{location}.outputs.required_actions"
    )
    canonical = json.dumps(
        raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return BenchCase(
        case_id,
        inputs["task"],
        inputs["repo_fixture"],
        inputs["base_commit"],
        script,
        config,
        required_tests,
        allowed_files,
        forbidden_files,
        critical_facts,
        required_actions,
        dict(metadata),
        path,
        hashlib.sha256(canonical.encode()).hexdigest(),
    )


def _agent_config(raw: Any, location: str) -> AgentConfig:
    fields = {
        "max_iterations",
        "context_window",
        "compaction_trigger",
        "compaction_target",
    }
    _exact_object(raw, fields, location)
    for field in fields:
        if isinstance(raw[field], bool) or not isinstance(raw[field], int):
            raise RuntimeError(f"{location}.{field} must be an integer")
    config = AgentConfig(**raw)
    if not 1 <= config.max_iterations <= 50:
        raise RuntimeError(f"{location}.max_iterations must be between 1 and 50")
    if not (
        0 < config.compaction_target < config.compaction_trigger < config.context_window
    ):
        raise RuntimeError(f"{location} must satisfy target < trigger < context_window")
    return config


def _exact_object(raw: Any, fields: set[str], location: str) -> None:
    _require_object(raw, location)
    if set(raw) != fields:
        raise RuntimeError(f"{location} must contain exactly {sorted(fields)}")


def _require_object(raw: Any, location: str) -> None:
    if not isinstance(raw, dict):
        raise RuntimeError(f"{location} must be an object")


def _nonempty_string(value: Any, location: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{location} must be a non-empty string")


def _string_list(value: Any, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise RuntimeError(f"{location} must be an array of non-empty strings")
    return tuple(value)
