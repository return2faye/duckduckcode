from __future__ import annotations

import argparse
import difflib
import fnmatch
import hashlib
import io
import json
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langsmith import Client as LangSmithClient
from langsmith.wrappers import wrap_openai
from openai import OpenAI

from ..config import Config
from ..core.event import (
    ContextCompactionEvent,
    ConversationEvent,
    ErrorEvent,
    LoopCompleteEvent,
    PermissionRequestEvent,
    PlanReviewEvent,
    PlanReviewResponse,
    ToolCallEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    UsageEvent,
)
from ..main import build_agent
from ..tools.tool import ToolCall
from .schema import BenchCase, load_benches

DEFAULT_BENCHES = Path("evals/benches")
DEFAULT_DATABASE = Path(".duckduckcode/evals.sqlite3")
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 4},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class JudgeResult:
    score: int
    reason: str
    token_usage: int = 0


def sync_cases(connection: sqlite3.Connection, cases: list[BenchCase]) -> None:
    _initialize_database(connection)
    with connection:
        connection.execute("UPDATE cases SET active = 0")
        connection.executemany(
            """
            INSERT INTO cases (
                id, prompt, files_json, expected_outcome, source_hash,
                active, case_json
            ) VALUES (?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(id) DO UPDATE SET
                prompt = excluded.prompt,
                files_json = excluded.files_json,
                expected_outcome = excluded.expected_outcome,
                source_hash = excluded.source_hash,
                active = 1,
                case_json = excluded.case_json
            """,
            [
                (
                    case.id,
                    case.task,
                    json.dumps({"repo_fixture": case.repo_fixture}),
                    json.dumps(case.record()["outputs"], ensure_ascii=False),
                    case.source_hash,
                    json.dumps(case.record(), ensure_ascii=False, sort_keys=True),
                )
                for case in cases
            ],
        )


def run_evaluations(
    config: Config,
    bench_paths: list[Path] | Path = DEFAULT_BENCHES,
    database_path: Path = DEFAULT_DATABASE,
    case_ids: list[str] | None = None,
) -> int:
    cases = load_benches(bench_paths)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        sync_cases(connection, cases)
        selected = _select_cases(cases, case_ids or [])
        if not selected:
            raise RuntimeError("No active bench cases found")
        batch_id = uuid.uuid4().hex
        passed = 0
        for case in selected:
            result = _run_case(config, case, batch_id)
            _save_evaluation(connection, result)
            passed += result["passed"]
            score = "-" if result["score"] is None else str(result["score"])
            verdict = "PASS" if result["passed"] else "FAIL"
            print(f"{case.id}: {verdict} score={score} {result['reason']}")
    total = len(selected)
    print(f"{passed}/{total} passed ({passed / total:.1%})")
    return 0 if passed == total else 1


def _run_case(config: Config, case: BenchCase, batch_id: str) -> dict[str, Any]:
    started = time.monotonic()
    answers: list[str] = []
    tool_events: list[dict[str, Any]] = []
    compaction_events: list[dict[str, Any]] = []
    test_results: list[dict[str, Any]] = []
    errors: list[str] = []
    token_usage = 0
    compactions = 0
    completed = False
    initial: dict[str, bytes] = {}
    final: dict[str, bytes] = {}

    with tempfile.TemporaryDirectory(prefix="duckduckcode-eval-") as directory:
        workspace = Path(directory)
        agent = None
        try:
            _materialize_fixture(case, workspace)
            initial = _snapshot(workspace)
            agent_config = case.agent_config
            agent = build_agent(
                config,
                workspace,
                max_iterations=agent_config.max_iterations,
                context_window_tokens=agent_config.context_window,
                compaction_trigger_tokens=agent_config.compaction_trigger,
                compaction_target_tokens=agent_config.compaction_target,
                include_user_instructions=False,
            )
            agent.set_permission_mode("accept_edits")
            completed = True
            for turn, prompt in enumerate((case.task, *case.conversation_script), 1):
                answer, turn_completed, turn_usage, turn_compactions, turn_errors = (
                    _run_turn(agent, prompt, turn, tool_events, compaction_events)
                )
                answers.append(answer)
                token_usage += turn_usage
                compactions += turn_compactions
                errors.extend(turn_errors)
                if not turn_completed:
                    completed = False
                    break
            for index, command in enumerate(case.required_tests, 1):
                result = agent.tools.execute(
                    ToolCall(
                        f"required_test_{index}",
                        "Bash",
                        {"command": command, "network_access": False},
                    )
                )
                test_results.append(
                    {
                        "command": command,
                        "content": result.content,
                        "passed": not result.is_error,
                    }
                )
        except Exception as exc:
            completed = False
            errors.append(str(exc))
        finally:
            if agent is not None:
                try:
                    agent.close()
                except Exception as exc:
                    errors.append(f"Agent close failed: {exc}")
            final = _snapshot(workspace)

    changed_files = sorted(
        name
        for name in initial.keys() | final.keys()
        if initial.get(name) != final.get(name)
    )
    validation_errors = _validate_outputs(
        case, changed_files, test_results, compactions
    )
    diff = _workspace_diff(initial, final)
    evidence = {
        "bench": case.record(),
        "final_answers": answers,
        "workspace_diff": diff,
        "changed_files": changed_files,
        "required_test_results": test_results,
        "validation_errors": validation_errors,
        "tool_events": tool_events,
        "tool_errors": [
            event
            for event in tool_events
            if event.get("type") == "tool_result" and event.get("is_error")
        ],
        "agent_completed": completed,
        "agent_errors": errors,
        "actual_compactions": compactions,
        "expected_compactions": case.metadata["expected_compactions"],
        "compaction_events": compaction_events,
    }
    status = "completed" if completed else "agent_error"
    judge_usage = 0
    score: int | None = None
    try:
        judged = _judge(config, evidence)
        score, reason, judge_usage = judged.score, judged.reason, judged.token_usage
    except Exception as exc:
        status = "judge_error"
        reason = f"Judge error: {exc}"
        errors.append(reason)
    passed = bool(
        completed and not validation_errors and score is not None and score >= 3
    )
    return {
        "batch_id": batch_id,
        "case_id": case.id,
        "agent_model": config.openai_model,
        "judge_model": config.openai_judge_model,
        "status": status,
        "score": score,
        "passed": passed,
        "reason": reason,
        "final_answer": answers[-1] if answers else "",
        "workspace_diff": diff,
        "tool_events": json.dumps(tool_events, ensure_ascii=False, default=str),
        "token_usage": token_usage,
        "judge_token_usage": judge_usage,
        "duration_seconds": time.monotonic() - started,
        "error": "\n".join(errors),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_results": json.dumps(test_results, ensure_ascii=False),
        "validation_errors": json.dumps(validation_errors, ensure_ascii=False),
        "compactions": compactions,
        "compaction_events": json.dumps(compaction_events, ensure_ascii=False),
    }


def _run_turn(
    agent: Any,
    prompt: str,
    turn: int,
    tool_events: list[dict[str, Any]],
    compaction_events: list[dict[str, Any]],
) -> tuple[str, bool, int, int, list[str]]:
    answer = ""
    turn_text = ""
    completed = False
    token_usage = 0
    compactions = 0
    errors: list[str] = []
    calls: dict[str, ToolCallEvent] = {}
    stream = agent.stream(prompt)
    try:
        event = next(stream)
        while True:
            response = None
            if isinstance(event, ConversationEvent):
                turn_text += event.delta
            elif isinstance(event, ToolCallEvent):
                calls[event.tool_call.call_id] = event
                tool_events.append(
                    {
                        "type": "tool_call",
                        "turn": turn,
                        "call_id": event.tool_call.call_id,
                        "name": event.tool_call.name,
                        "arguments": event.tool_call.arguments,
                    }
                )
            elif isinstance(event, ToolResultEvent):
                tool_events.append(
                    {
                        "type": "tool_result",
                        "turn": turn,
                        "call_id": event.call_id,
                        "name": event.name,
                        "content": event.content,
                        "is_error": event.is_error,
                    }
                )
            elif isinstance(event, PermissionRequestEvent):
                call = calls.get(event.call_id)
                network = bool(
                    call
                    and call.tool_call.name == "Bash"
                    and call.tool_call.arguments.get("network_access") is True
                )
                response = "deny" if network else "allow_once"
                tool_events.append(
                    {
                        "type": "permission",
                        "turn": turn,
                        "call_id": event.call_id,
                        "decision": response,
                    }
                )
            elif isinstance(event, PlanReviewEvent):
                response = PlanReviewResponse(False, "Evaluations use default mode")
            elif isinstance(event, TurnCompleteEvent):
                if turn_text:
                    answer = turn_text
                turn_text = ""
            elif isinstance(event, UsageEvent):
                token_usage += event.total_tokens
            elif isinstance(event, ContextCompactionEvent):
                compactions += event.status == "completed"
                record = {
                    "turn": turn,
                    "status": event.status,
                    "automatic": event.automatic,
                    "before_tokens": event.before_tokens,
                    "after_tokens": event.after_tokens,
                }
                if event.status == "completed":
                    record["summary"] = getattr(
                        getattr(agent, "context", None), "abstraction", ""
                    )
                compaction_events.append(record)
            elif isinstance(event, ErrorEvent):
                errors.append(event.message)
            elif isinstance(event, LoopCompleteEvent):
                completed = event.reason == "completed"
            event = stream.send(response) if response is not None else next(stream)
    except StopIteration:
        if turn_text:
            answer = turn_text
    finally:
        stream.close()
    return answer, completed, token_usage, compactions, errors


def _validate_outputs(
    case: BenchCase,
    changed_files: list[str],
    test_results: list[dict[str, Any]],
    compactions: int,
) -> list[str]:
    errors = [
        f"Required test failed: {result['command']}"
        for result in test_results
        if not result["passed"]
    ]
    if case.allowed_files:
        errors.extend(
            f"Changed file is not allowed: {name}"
            for name in changed_files
            if not any(
                fnmatch.fnmatchcase(name, pattern) for pattern in case.allowed_files
            )
        )
    errors.extend(
        f"Forbidden file changed: {name}"
        for name in changed_files
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in case.forbidden_files)
    )
    expected = case.metadata["expected_compactions"]
    if compactions != expected:
        errors.append(f"Expected {expected} compactions, observed {compactions}")
    return errors


def _materialize_fixture(case: BenchCase, workspace: Path) -> None:
    source = case.fixture_path
    if not source.is_dir():
        raise RuntimeError(f"Repo fixture '{source}' is not a local directory")
    if (source / ".git").exists():
        resolved = subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "rev-parse",
                "--verify",
                f"{case.base_commit}^{{commit}}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        archive = subprocess.run(
            ["git", "-C", str(source), "archive", "--format=tar", resolved],
            check=True,
            capture_output=True,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(workspace, filter="data")
        return
    actual_hash = _tree_hash(source)
    if case.base_commit != actual_hash:
        raise RuntimeError(
            f"Fixture hash mismatch for '{source}': expected {case.base_commit}, got {actual_hash}"
        )
    shutil.copytree(source, workspace, dirs_exist_ok=True)


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Repo fixture contains a symlink: {path}")
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _judge(config: Config, evidence: dict[str, Any]) -> JudgeResult:
    langsmith_client = None
    client = OpenAI(api_key=config.openai_api_key)
    if config.langsmith_tracing:
        langsmith_client = LangSmithClient(api_key=config.langsmith_api_key)
        client = wrap_openai(
            client,
            tracing_extra={
                "client": langsmith_client,
                "project_name": config.langsmith_project,
                "enabled": True,
            },
        )
    try:
        response = client.responses.create(
            model=config.openai_judge_model,
            instructions=(
                "Grade this coding-agent run against the benchmark outputs. "
                "Score 0-4: 0 no useful work, 1 major failure, 2 partial, "
                "3 correct with only minor issues, 4 fully correct. Judge only "
                "the supplied evidence and explain the decisive reason concisely. "
                "For compaction cases, verify the captured summaries preserve "
                "critical facts, newest instructions, and pending actions while "
                "discarding irrelevant detail. Use the complete tool trace, "
                "including call arguments, results, errors, and permission "
                "decisions, to verify what the agent actually inspected and did."
            ),
            input=json.dumps(evidence, ensure_ascii=False),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "duckduckcode_evaluation",
                    "schema": JUDGE_SCHEMA,
                    "strict": True,
                }
            },
        )
        try:
            result = json.loads(response.output_text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError("Judge returned invalid JSON") from exc
        if not isinstance(result, dict) or set(result) != {"score", "reason"}:
            raise RuntimeError("Judge returned unexpected fields")
        score, reason = result["score"], result["reason"]
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4:
            raise RuntimeError("Judge score must be an integer from 0 to 4")
        if not isinstance(reason, str) or not reason.strip():
            raise RuntimeError("Judge reason must be a non-empty string")
        usage = int(getattr(getattr(response, "usage", None), "total_tokens", 0) or 0)
        return JudgeResult(score, reason.strip(), usage)
    finally:
        try:
            client.close()
        finally:
            if langsmith_client is not None:
                langsmith_client.flush()


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS cases (
            id TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            files_json TEXT NOT NULL,
            expected_outcome TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
            case_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            case_id TEXT NOT NULL REFERENCES cases(id),
            agent_model TEXT NOT NULL,
            judge_model TEXT NOT NULL,
            status TEXT NOT NULL,
            score INTEGER,
            passed INTEGER NOT NULL CHECK (passed IN (0, 1)),
            reason TEXT NOT NULL,
            final_answer TEXT NOT NULL,
            workspace_diff TEXT NOT NULL,
            tool_events TEXT NOT NULL,
            token_usage INTEGER NOT NULL,
            judge_token_usage INTEGER NOT NULL,
            duration_seconds REAL NOT NULL,
            error TEXT NOT NULL,
            created_at TEXT NOT NULL,
            test_results TEXT NOT NULL DEFAULT '[]',
            validation_errors TEXT NOT NULL DEFAULT '[]',
            compactions INTEGER NOT NULL DEFAULT 0,
            compaction_events TEXT NOT NULL DEFAULT '[]'
        );
        """)
    _ensure_column(connection, "cases", "case_json", "TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(
        connection, "evaluations", "test_results", "TEXT NOT NULL DEFAULT '[]'"
    )
    _ensure_column(
        connection, "evaluations", "validation_errors", "TEXT NOT NULL DEFAULT '[]'"
    )
    _ensure_column(
        connection, "evaluations", "compactions", "INTEGER NOT NULL DEFAULT 0"
    )
    _ensure_column(
        connection,
        "evaluations",
        "compaction_events",
        "TEXT NOT NULL DEFAULT '[]'",
    )


def _ensure_column(
    connection: sqlite3.Connection, table: str, column: str, definition: str
) -> None:
    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _select_cases(cases: list[BenchCase], case_ids: list[str]) -> list[BenchCase]:
    if not case_ids:
        return cases
    selected = [case for case in cases if case.id in case_ids]
    missing = sorted(set(case_ids) - {case.id for case in selected})
    if missing:
        raise RuntimeError(f"Unknown active bench case(s): {', '.join(missing)}")
    return selected


def _save_evaluation(connection: sqlite3.Connection, result: dict[str, Any]) -> None:
    columns = tuple(result)
    with connection:
        connection.execute(
            f"INSERT INTO evaluations ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})",
            [result[column] for column in columns],
        )


def _snapshot(workspace: Path) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for path in sorted(workspace.rglob("*")):
        relative = path.relative_to(workspace)
        if _ignored(relative) or not path.is_file():
            continue
        snapshot[relative.as_posix()] = (
            f"symlink:{path.readlink()}".encode()
            if path.is_symlink()
            else path.read_bytes()
        )
    return snapshot


def _ignored(path: Path) -> bool:
    ignored_parts = {
        ".duckduckcode",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
    }
    return bool(ignored_parts.intersection(path.parts) or path.suffix == ".pyc")


def _workspace_diff(before: dict[str, bytes], after: dict[str, bytes]) -> str:
    chunks: list[str] = []
    for name in sorted(before.keys() | after.keys()):
        if before.get(name) == after.get(name):
            continue
        old = before.get(name)
        new = after.get(name)
        chunks.extend(
            difflib.unified_diff(
                [] if old is None else old.decode(errors="replace").splitlines(True),
                [] if new is None else new.decode(errors="replace").splitlines(True),
                fromfile="/dev/null" if old is None else f"a/{name}",
                tofile="/dev/null" if new is None else f"b/{name}",
            )
        )
    return "".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DuckDuckCode benchmarks.")
    parser.add_argument("--case", action="append", dest="case_ids", metavar="ID")
    parser.add_argument(
        "--bench",
        action="append",
        type=Path,
        dest="bench_paths",
        metavar="PATH",
        help="JSON/JSONL bench file or directory; may be repeated",
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()
    try:
        code = run_evaluations(
            Config.from_env(),
            args.bench_paths or DEFAULT_BENCHES,
            args.database,
            args.case_ids,
        )
    except RuntimeError as exc:
        parser.exit(2, f"duckduckcode-eval: error: {exc}\n")
    raise SystemExit(code)
