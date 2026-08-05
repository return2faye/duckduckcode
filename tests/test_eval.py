from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from duckduckcode.config import Config, ModelConfig
from duckduckcode import eval as eval_module
from duckduckcode.eval import JudgeResult, load_benches, run_evaluations, sync_cases
from duckduckcode.eval import runner as runner_module
from duckduckcode.eval.report import DEFAULT_REPORT, generate_report
from duckduckcode.core.event import (
    ContextCompactionEvent,
    ConversationEvent,
    LoopCompleteEvent,
    PermissionRequestEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    UsageEvent,
)
from duckduckcode.tools.tool import ToolCall


def _bench_record(fixture: Path, case_id: str = "case-1") -> dict[str, object]:
    return {
        "inputs": {
            "task": "Fix it",
            "repo_fixture": str(fixture),
            "base_commit": runner_module._tree_hash(fixture),
            "conversation_script": ["Check it once more"],
            "agent_config": {
                "max_iterations": 30,
                "context_window": 32_000,
                "compaction_trigger": 24_000,
                "compaction_target": 10_000,
            },
        },
        "outputs": {
            "required_tests": [],
            "allowed_files": ["source.py"],
            "forbidden_files": ["forbidden.py"],
            "critical_facts": ["source.py contains new"],
            "required_actions": ["Inspect source.py"],
        },
        "metadata": {
            "id": case_id,
            "suite": "test",
            "category": "latest-instruction",
            "difficulty": "medium",
            "expected_compactions": 2,
            "language": "python",
        },
    }


def _write_jsonl(path: Path, *cases: dict[str, object]) -> None:
    path.write_text(
        "".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8"
    )


class BenchSchemaTest(unittest.TestCase):
    def test_loads_json_jsonl_and_directories_with_stable_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            fixture.mkdir()
            fixture.joinpath("source.py").write_text("old\n", encoding="utf-8")
            benches = root / "benches"
            benches.mkdir()
            _write_jsonl(benches / "one.jsonl", _bench_record(fixture, "jsonl-case"))
            json_case = _bench_record(fixture)
            json_case["metadata"].pop("id")
            (benches / "single.json").write_text(
                json.dumps(json_case), encoding="utf-8"
            )

            loaded = load_benches(benches)

            self.assertEqual({case.id for case in loaded}, {"jsonl-case", "single"})
            self.assertEqual(loaded[0].agent_config.compaction_trigger, 24_000)
            self.assertEqual(loaded[0].conversation_script, ("Check it once more",))
            self.assertEqual(len(loaded[0].source_hash), 64)

    def test_rejects_invalid_schema_config_and_file_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            fixture.mkdir()
            fixture.joinpath("source.py").write_text("old\n", encoding="utf-8")
            path = root / "bench.json"
            for change, message in (
                (
                    lambda case: case["inputs"]["agent_config"].update(
                        {"compaction_target": 25_000}
                    ),
                    "target < trigger",
                ),
                (
                    lambda case: case["outputs"].update(
                        {"allowed_files": ["../outside.py"]}
                    ),
                    "unsafe path",
                ),
                (
                    lambda case: case["inputs"].update({"conversation_script": [42]}),
                    "array of non-empty strings",
                ),
            ):
                case = _bench_record(fixture)
                change(case)
                path.write_text(json.dumps(case), encoding="utf-8")
                with self.subTest(message=message):
                    with self.assertRaisesRegex(RuntimeError, message):
                        load_benches(path)

    def test_fixture_hash_detects_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            fixture.mkdir()
            source = fixture / "source.py"
            source.write_text("old\n", encoding="utf-8")
            path = root / "bench.json"
            path.write_text(json.dumps(_bench_record(fixture)), encoding="utf-8")
            case = load_benches(path)[0]
            source.write_text("changed\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Fixture hash mismatch"):
                runner_module._materialize_fixture(case, root / "workspace")

    def test_sync_is_idempotent_marks_inactive_and_migrates_old_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            fixture.mkdir()
            fixture.joinpath("source.py").write_text("old\n", encoding="utf-8")
            path = root / "bench.jsonl"
            _write_jsonl(
                path,
                _bench_record(fixture, "one"),
                _bench_record(fixture, "two"),
            )
            connection = sqlite3.connect(":memory:")
            runner_module._initialize_database(connection)

            sync_cases(connection, load_benches(path))
            sync_cases(connection, load_benches(path))
            _write_jsonl(path, _bench_record(fixture, "one"))
            sync_cases(connection, load_benches(path))

            self.assertEqual(
                connection.execute(
                    "SELECT id, active FROM cases ORDER BY id"
                ).fetchall(),
                [("one", 1), ("two", 0)],
            )
            self.assertIn(
                "case_json",
                {row[1] for row in connection.execute("PRAGMA table_info(cases)")},
            )


class EvalRunTest(unittest.TestCase):
    def test_full_command_runs_script_injects_config_and_appends_history(self) -> None:
        workspaces: list[Path] = []
        decisions: list[str] = []
        evidence: list[dict[str, object]] = []
        build_kwargs: list[dict[str, object]] = []

        class FakeAgent:
            def __init__(self, workspace: Path) -> None:
                self.workspace = workspace
                self.context = type("Context", (), {"abstraction": "kept summary"})()

            def stream(self, prompt):
                self.workspace.joinpath("source.py").write_text(
                    "new\n", encoding="utf-8"
                )
                internal = self.workspace / ".duckduckcode" / "ignored.txt"
                internal.parent.mkdir(exist_ok=True)
                internal.write_text("ignore", encoding="utf-8")
                offline = ToolCall(
                    "offline",
                    "Bash",
                    {"command": "python source.py", "network_access": False},
                )
                yield ToolCallEvent(offline)
                decision = yield PermissionRequestEvent(
                    "offline", "Bash", "python source.py", "ask"
                )
                decisions.append(decision)
                yield ToolResultEvent("offline", "Bash", "ok")
                online = ToolCall(
                    "online", "Bash", {"command": "uv sync", "network_access": True}
                )
                yield ToolCallEvent(online)
                decision = yield PermissionRequestEvent(
                    "online", "Bash", "network:uv sync", "ask"
                )
                decisions.append(decision)
                yield ToolResultEvent("online", "Bash", "denied", True)
                yield ContextCompactionEvent("completed", True, 25_000, 9_000)
                yield TurnCompleteEvent(1)
                yield ConversationEvent(f"fixed:{prompt}")
                yield UsageEvent(7)
                yield TurnCompleteEvent(2)
                yield LoopCompleteEvent("completed", 2)

            def set_permission_mode(self, mode):
                self.permission_mode = mode

            def close(self):
                pass

        def fake_build(config, workspace, **kwargs):
            workspaces.append(workspace)
            build_kwargs.append(kwargs)
            return FakeAgent(workspace)

        def fake_judge(config, supplied):
            evidence.append(supplied)
            return JudgeResult(3, "correct", 2)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            fixture.mkdir()
            fixture.joinpath("source.py").write_text("old\n", encoding="utf-8")
            bench = root / "bench.json"
            bench.write_text(json.dumps(_bench_record(fixture)), encoding="utf-8")
            database = root / "evals.sqlite3"
            config = Config("key")
            with (
                patch.object(runner_module, "build_agent", side_effect=fake_build),
                patch.object(runner_module, "_judge", side_effect=fake_judge),
                patch(
                    "sys.argv",
                    [
                        "duckduckcode-eval",
                        "--bench",
                        str(bench),
                        "--database",
                        str(database),
                    ],
                ),
                patch.object(runner_module.Config, "from_env", return_value=config),
                patch("sys.stdout", new=io.StringIO()) as output,
            ):
                with self.assertRaises(SystemExit) as exit_1:
                    eval_module.main()
                with self.assertRaises(SystemExit) as exit_2:
                    eval_module.main()

            rows = (
                sqlite3.connect(database)
                .execute(
                    "SELECT status, score, passed, final_answer, workspace_diff, "
                    "tool_events, token_usage, judge_token_usage, compactions, "
                    "compaction_events "
                    "FROM evaluations ORDER BY id"
                )
                .fetchall()
            )

        self.assertEqual(exit_1.exception.code, 0)
        self.assertEqual(exit_2.exception.code, 0)
        self.assertEqual(decisions, ["allow_once", "deny"] * 4)
        self.assertEqual(len(set(workspaces)), 2)
        self.assertTrue(all(not workspace.exists() for workspace in workspaces))
        self.assertEqual(build_kwargs[0]["max_iterations"], 30)
        self.assertEqual(build_kwargs[0]["context_window_tokens"], 32_000)
        self.assertEqual(build_kwargs[0]["compaction_trigger_tokens"], 24_000)
        self.assertEqual(build_kwargs[0]["compaction_target_tokens"], 10_000)
        self.assertFalse(build_kwargs[0]["include_user_instructions"])
        self.assertFalse(build_kwargs[0]["enable_sessions"])
        self.assertFalse(build_kwargs[0]["enable_memory"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0][:4],
            ("completed", 3, 1, "fixed:Check it once more"),
        )
        self.assertIn("-old", rows[0][4])
        self.assertIn("+new", rows[0][4])
        self.assertNotIn("ignored.txt", rows[0][4])
        self.assertEqual(rows[0][6:9], (14, 2, 2))
        self.assertEqual(len(json.loads(rows[0][9])), 2)
        self.assertEqual(len(evidence[0]["final_answers"]), 2)
        self.assertEqual(evidence[0]["actual_compactions"], 2)
        self.assertEqual(evidence[0]["compaction_events"][0]["summary"], "kept summary")
        self.assertEqual(
            evidence[0]["tool_events"][0]["arguments"]["command"],
            "python source.py",
        )
        self.assertEqual(evidence[0]["tool_events"][2]["content"], "ok")
        self.assertTrue(evidence[0]["tool_errors"][0]["is_error"])
        self.assertIn("1/1 passed", output.getvalue())

    def test_incomplete_and_file_violation_force_failure_and_judge_error_is_saved(
        self,
    ) -> None:
        class IncompleteAgent:
            def __init__(self, workspace):
                self.workspace = workspace

            def stream(self, prompt):
                self.workspace.joinpath("forbidden.py").write_text(
                    "bad\n", encoding="utf-8"
                )
                yield LoopCompleteEvent("error", 1)

            def set_permission_mode(self, mode):
                pass

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            fixture.mkdir()
            fixture.joinpath("source.py").write_text("old\n", encoding="utf-8")
            bench = root / "bench.json"
            record = _bench_record(fixture)
            record["inputs"]["conversation_script"] = []
            bench.write_text(json.dumps(record), encoding="utf-8")
            database = root / "evals.sqlite3"

            def build(config, workspace, **kwargs):
                return IncompleteAgent(workspace)

            with (
                patch.object(runner_module, "build_agent", side_effect=build),
                patch.object(
                    runner_module, "_judge", return_value=JudgeResult(4, "looked good")
                ),
                patch("sys.stdout", new=io.StringIO()),
            ):
                self.assertEqual(run_evaluations(Config("key"), bench, database), 1)
            with (
                patch.object(runner_module, "build_agent", side_effect=build),
                patch.object(
                    runner_module, "_judge", side_effect=RuntimeError("bad JSON")
                ),
                patch("sys.stdout", new=io.StringIO()),
            ):
                self.assertEqual(run_evaluations(Config("key"), bench, database), 1)
            rows = (
                sqlite3.connect(database)
                .execute(
                    "SELECT status, score, passed, validation_errors, error "
                    "FROM evaluations ORDER BY id"
                )
                .fetchall()
            )

        self.assertEqual(rows[0][:3], ("agent_error", 4, 0))
        self.assertIn("Forbidden file changed", rows[0][3])
        self.assertIn("Expected 2 compactions, observed 0", rows[0][3])
        self.assertEqual(rows[1][:3], ("judge_error", None, 0))
        self.assertIn("bad JSON", rows[1][4])


class JudgeTest(unittest.TestCase):
    def test_uses_strict_schema_and_validates_response(self) -> None:
        response = type(
            "Response",
            (),
            {
                "output_text": '{"score": 3, "reason": "minor issue"}',
                "usage": type("Usage", (), {"total_tokens": 11})(),
            },
        )()
        responses = type(
            "Responses",
            (),
            {
                "create": lambda self, **kwargs: setattr(self, "kwargs", kwargs)
                or response
            },
        )()
        client = type(
            "Client",
            (),
            {
                "responses": responses,
                "close": lambda self: setattr(self, "closed", True),
            },
        )()
        with patch.object(runner_module, "OpenAI", return_value=client):
            result = runner_module._judge(Config("key"), {"agent_completed": True})

        self.assertEqual(result, JudgeResult(3, "minor issue", 11))
        self.assertTrue(client.closed)
        self.assertTrue(responses.kwargs["text"]["format"]["strict"])
        self.assertEqual(responses.kwargs["model"], "gpt-5.6-terra")

        response.output_text = '{"score": 5, "reason": "bad"}'
        with patch.object(runner_module, "OpenAI", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "0 to 4"):
                runner_module._judge(Config("key"), {})

    def test_deepseek_judge_uses_chat_json_output(self) -> None:
        response = type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message",
                                (),
                                {"content": '{"score": 4, "reason": "correct"}'},
                            )()
                        },
                    )()
                ],
                "usage": type("Usage", (), {"total_tokens": 9})(),
            },
        )()
        completions = type(
            "Completions",
            (),
            {
                "create": lambda self, **kwargs: setattr(self, "kwargs", kwargs)
                or response
            },
        )()
        client = type(
            "Client",
            (),
            {
                "chat": type("Chat", (), {"completions": completions})(),
                "close": lambda self: setattr(self, "closed", True),
            },
        )()
        config = Config(
            "openai-key",
            deepseek_api_key="deepseek-key",
            deepseek_base_url="https://deepseek.example/v1",
            judge=ModelConfig("deepseek", "deepseek-judge"),
        )

        with patch.object(runner_module, "OpenAI", return_value=client) as factory:
            result = runner_module._judge(config, {"agent_completed": True})

        self.assertEqual(result, JudgeResult(4, "correct", 9))
        factory.assert_called_once_with(
            api_key="deepseek-key", base_url="https://deepseek.example/v1"
        )
        self.assertEqual(completions.kwargs["model"], "deepseek-judge")
        self.assertEqual(completions.kwargs["response_format"], {"type": "json_object"})
        self.assertTrue(client.closed)


class ReportTest(unittest.TestCase):
    def test_renders_latest_batch_with_compaction_and_escaped_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            fixture.mkdir()
            fixture.joinpath("source.py").write_text("old\n", encoding="utf-8")
            bench = root / "bench.json"
            bench.write_text(json.dumps(_bench_record(fixture)), encoding="utf-8")
            database = root / "evals.sqlite3"
            with sqlite3.connect(database) as connection:
                sync_cases(connection, load_benches(bench))
                runner_module._save_evaluation(
                    connection,
                    {
                        "batch_id": "batch-1",
                        "case_id": "case-1",
                        "agent_model": "agent",
                        "judge_model": "judge",
                        "status": "completed",
                        "score": 4,
                        "passed": 1,
                        "reason": "looks <good>",
                        "final_answer": "done",
                        "workspace_diff": "+new",
                        "tool_events": json.dumps(
                            [
                                {
                                    "turn": 2,
                                    "type": "tool_call",
                                    "call_id": "call-1",
                                    "name": "Bash",
                                    "arguments": {"command": "printf '<ok>'"},
                                },
                                {
                                    "turn": 2,
                                    "type": "tool_result",
                                    "call_id": "call-1",
                                    "name": "Bash",
                                    "content": "<ok>",
                                    "is_error": False,
                                },
                            ]
                        ),
                        "token_usage": 12,
                        "judge_token_usage": 3,
                        "duration_seconds": 1.5,
                        "error": "",
                        "created_at": "2026-08-03T00:00:00+00:00",
                        "test_results": "[]",
                        "validation_errors": "[]",
                        "compactions": 1,
                        "compaction_events": '[{"turn": 2, "status": "completed", "before_tokens": 1200, "after_tokens": 400, "summary": "retain <fact>"}]',
                    },
                )
            report = generate_report(database, root / "report.html")
            rendered = report.read_text(encoding="utf-8")

        self.assertIn("case-1", rendered)
        self.assertEqual(
            DEFAULT_REPORT, Path(".duckduckcode/eval-reports/eval-report.html")
        )
        self.assertIn("1,200 → 400 estimated tokens · saved 800 (66.7%)", rendered)
        self.assertIn("retain &lt;fact&gt;", rendered)
        self.assertNotIn("retain <fact>", rendered)
        self.assertIn("Tool trace (1 calls)", rendered)
        self.assertIn(
            '<details class="event"><summary><b>turn 2 · tool_result · Bash</b>',
            rendered,
        )
        self.assertIn("printf &#x27;&lt;ok&gt;&#x27;", rendered)
        self.assertIn("&lt;ok&gt;", rendered)
        self.assertNotIn("printf '<ok>'", rendered)


if __name__ == "__main__":
    unittest.main()
