from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from duckduckcode.config import Config
from duckduckcode.core.event import ConversationEvent, LoopCompleteEvent
from duckduckcode.eval import swebench_live


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


class SwebenchLiveTest(unittest.TestCase):
    def test_loads_every_row_from_official_full_split(self) -> None:
        records = [
            {
                "instance_id": f"owner__project-{index}",
                "repo": "owner/project",
                "base_commit": str(index) * 40,
                "problem_statement": f"Fix bug {index}",
                "patch": "gold must stay hidden",
            }
            for index in (1, 2)
        ]
        responses = [
            {"sha": "a" * 40},
            {
                "partial": False,
                "num_rows_total": 2,
                "rows": [{"row": records[0]}],
            },
            {
                "partial": False,
                "num_rows_total": 2,
                "rows": [{"row": records[1]}],
            },
        ]

        with (
            patch.object(swebench_live, "_PAGE_SIZE", 1),
            patch.object(swebench_live, "_get_json", side_effect=responses) as get_json,
        ):
            instances, revision = swebench_live.load_official_instances()

        self.assertEqual(revision, "a" * 40)
        self.assertEqual(len(instances), 2)
        self.assertNotIn("patch", instances[0])
        self.assertIn("split=full", get_json.call_args_list[1].args[0])
        self.assertIn("offset=1", get_json.call_args_list[2].args[0])

    def test_official_split_deduplicates_only_identical_instances(self) -> None:
        record = {
            "instance_id": "owner__project-1",
            "repo": "owner/project",
            "base_commit": "1" * 40,
            "problem_statement": "Fix bug",
        }
        responses = [
            {"sha": "a" * 40},
            {
                "partial": False,
                "num_rows_total": 2,
                "rows": [{"row": record}, {"row": dict(record)}],
            },
        ]
        with patch.object(swebench_live, "_get_json", side_effect=responses):
            instances, _ = swebench_live.load_official_instances()
        self.assertEqual(instances, [record])

        responses[1]["rows"][1]["row"]["patch"] = "different gold patch"
        with patch.object(swebench_live, "_get_json", side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, "conflicting duplicate"):
                swebench_live.load_official_instances()

    def test_loads_official_jsonl_fields_and_rejects_unsafe_repo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "cases.jsonl"
            record = {
                "instance_id": "owner__project-1",
                "repo": "owner/project",
                "base_commit": "a" * 40,
                "problem_statement": "Fix the bug",
                "patch": "secret gold patch",
            }
            dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

            self.assertEqual(
                swebench_live.load_instances(dataset),
                [{key: record[key] for key in record if key != "patch"}],
            )
            record["repo"] = "../outside"
            dataset.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "repo is unsafe"):
                swebench_live.load_instances(dataset)

    def test_repository_cache_rejects_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            outside = root / "outside"
            cache.mkdir()
            outside.mkdir()
            cache.joinpath("owner").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(RuntimeError, "escapes through a symlink"):
                swebench_live._repository(
                    {
                        "instance_id": "owner__project-1",
                        "repo": "owner/project",
                        "base_commit": "a" * 40,
                        "problem_statement": "Fix it",
                    },
                    cache.resolve(),
                    False,
                )

    def test_runs_agent_at_base_commit_and_writes_official_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repos" / "owner" / "project"
            repository.mkdir(parents=True)
            _git("init", cwd=repository)
            _git("config", "user.email", "test@example.com", cwd=repository)
            _git("config", "user.name", "Test", cwd=repository)
            repository.joinpath("bug.py").write_text("value = 1\n", encoding="utf-8")
            _git("add", "bug.py", cwd=repository)
            _git("commit", "-m", "base", cwd=repository)
            commit = _git("rev-parse", "HEAD", cwd=repository)
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "instance_id": "owner__project-1",
                        "repo": "owner/project",
                        "base_commit": commit,
                        "problem_statement": "Set value to 2",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            predictions = root / "predictions.json"
            workspaces: list[Path] = []

            class Agent:
                def __init__(self, workspace: Path) -> None:
                    self.workspace = workspace

                def set_permission_mode(self, mode: str) -> None:
                    self.mode = mode

                def stream(self, prompt: str):
                    workspaces.append(self.workspace)
                    self.workspace.joinpath("bug.py").write_text(
                        "value = 2\n", encoding="utf-8"
                    )
                    self.workspace.joinpath("new.bin").write_bytes(b"\x00\xff")
                    metadata = self.workspace / ".duckduckcode"
                    metadata.mkdir()
                    metadata.joinpath("permissions.yaml").write_text(
                        "generated\n", encoding="utf-8"
                    )
                    yield ConversationEvent("fixed")
                    yield LoopCompleteEvent("completed", 1)

                def close(self) -> None:
                    pass

            with patch.object(
                swebench_live,
                "build_agent",
                side_effect=lambda config, workspace, **kwargs: Agent(workspace),
            ):
                code = swebench_live.run_inference(
                    Config("key"), dataset, root / "repos", predictions
                )

            result = json.loads(predictions.read_text(encoding="utf-8"))[
                "owner__project-1"
            ]
            self.assertEqual(code, 0)
            self.assertTrue(result["agent_completed"])
            self.assertIn("diff --git a/bug.py b/bug.py", result["model_patch"])
            self.assertIn("diff --git a/new.bin b/new.bin", result["model_patch"])
            self.assertNotIn(".duckduckcode", result["model_patch"])
            self.assertEqual(result["errors"], [])
            self.assertEqual(result["final_answer"], "fixed")
            self.assertEqual(result["tool_events"], [])
            self.assertTrue(all(not workspace.exists() for workspace in workspaces))


if __name__ == "__main__":
    unittest.main()
