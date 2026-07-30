from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from duckduckcode.tools.bash import create_bash_tool
from duckduckcode.tools.tool import ToolCall, ToolManager, ToolResult

OUTPUT_LIMIT_BYTES = 200_000


class BashTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.tool = create_bash_tool(self.root)
        self.manager = ToolManager()
        self.manager.register(self.tool)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def execute(self, **arguments) -> ToolResult:
        return self.manager.execute(ToolCall("call_1", "Bash", arguments))

    def test_exposes_dangerous_shell_metadata_and_strict_schema(self) -> None:
        self.assertEqual(self.tool.name, "Bash")
        self.assertEqual(self.tool.category, "shell")
        self.assertFalse(self.tool.is_read_only)
        self.assertTrue(self.tool.is_dangerous)
        self.assertFalse(self.tool.is_concurrency_safe)
        self.assertIn("Use Bash", self.tool.description)
        self.assertIn("working directory", self.tool.description)
        self.assertIn("120 seconds", self.tool.description)
        self.assertIn("JSON", self.tool.description)
        self.assertIn("200,000 bytes", self.tool.description)
        self.assertEqual(
            self.tool.params,
            {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "Shell command to execute from the working directory. "
                            "Use absolute paths when referring to files."
                        ),
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        )

    def test_runs_command_from_working_directory_and_combines_output(self) -> None:
        (self.root / "name.txt").write_text("duck\n", encoding="utf-8")

        result = self.execute(
            command="python -c 'import pathlib, sys; print(pathlib.Path.cwd().name, flush=True); sys.stderr.write(pathlib.Path(\"name.txt\").read_text())'"
        )

        self.assertFalse(result.is_error)
        self.assertEqual(
            result,
            ToolResult(
                json.dumps(
                    {"output": f"{self.root.name}\nduck\n", "exit_code": 0},
                    ensure_ascii=False,
                )
            ),
        )

    def test_marks_default_nonzero_exit_as_error(self) -> None:
        result = self.execute(command="python -c 'print(\"bad\"); raise SystemExit(1)'")

        self.assertTrue(result.is_error)
        self.assertEqual(
            json.loads(result.content), {"output": "bad\n", "exit_code": 1}
        )

    def test_uses_command_exit_semantics_for_expected_nonzero_codes(self) -> None:
        (self.root / "empty.txt").write_text("", encoding="utf-8")
        (self.root / "a").write_text("a\n", encoding="utf-8")
        (self.root / "b").write_text("b\n", encoding="utf-8")

        grep = self.execute(command="grep missing missing.txt")
        diff = self.execute(command="diff a b")

        self.assertTrue(grep.is_error)
        self.assertEqual(json.loads(grep.content)["exit_code"], 2)
        self.assertFalse(diff.is_error)
        self.assertEqual(json.loads(diff.content)["exit_code"], 1)

        grep_no_match = self.execute(command="grep missing empty.txt")
        grep_with_environment = self.execute(command="LC_ALL=C grep missing empty.txt")
        grep_with_quoted_operator = self.execute(
            command="grep -E 'missing|absent' empty.txt"
        )
        grep_with_quoted_operator_token = self.execute(command="grep -F '|' empty.txt")

        self.assertFalse(grep_no_match.is_error)
        self.assertEqual(
            json.loads(grep_no_match.content), {"output": "", "exit_code": 1}
        )
        self.assertFalse(grep_with_environment.is_error)
        self.assertFalse(grep_with_quoted_operator.is_error)
        self.assertFalse(grep_with_quoted_operator_token.is_error)

    def test_treats_nonzero_compound_command_as_error(self) -> None:
        (self.root / "empty.txt").write_text("", encoding="utf-8")

        result = self.execute(command="grep missing empty.txt; false")

        self.assertTrue(result.is_error)
        self.assertEqual(json.loads(result.content)["exit_code"], 1)

    def test_truncates_long_output(self) -> None:
        result = self.execute(
            command=f"python -c 'print(\"x\" * {OUTPUT_LIMIT_BYTES + 1})'"
        )

        self.assertFalse(result.is_error)
        payload = json.loads(result.content)
        self.assertIn(
            f"[Output truncated after {OUTPUT_LIMIT_BYTES} bytes.]", payload["output"]
        )
        self.assertLess(len(payload["output"]), OUTPUT_LIMIT_BYTES + 100)

    @unittest.skipIf(os.name == "nt", "requires POSIX file size limits")
    def test_does_not_store_all_output_in_a_regular_file(self) -> None:
        result = self.execute(
            command=(
                "ulimit -f 512; "
                f"python -c 'import os; os.write(1, b\"x\" * {OUTPUT_LIMIT_BYTES * 5})'"
            )
        )

        self.assertFalse(result.is_error)
        payload = json.loads(result.content)
        self.assertEqual(payload["exit_code"], 0)
        self.assertIn("[Output truncated", payload["output"])

    def test_times_out(self) -> None:
        from duckduckcode.tools.bash import _run_bash

        result = _run_bash(
            self.root, "python -c 'import time; time.sleep(1)'", timeout=0.01
        )

        self.assertTrue(result.is_error)
        payload = json.loads(result.content)
        self.assertIn("[Command timed out after 0.01 seconds.]", payload["output"])
        self.assertEqual(payload["exit_code"], 124)

    @unittest.skipIf(os.name == "nt", "requires POSIX process groups")
    def test_interrupt_terminates_the_command_process_group(self) -> None:
        pid_file = self.root / "child.pid"
        command = f"echo $$ > {pid_file}; sleep 30"
        code = (
            "from pathlib import Path; "
            "from duckduckcode.tools.bash import _run_bash; "
            f"_run_bash(Path({str(self.root)!r}), {command!r}, timeout=120)"
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        child_pid = None
        try:
            deadline = time.monotonic() + 5
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(pid_file.exists())
            child_pid = int(pid_file.read_text(encoding="utf-8"))

            os.kill(parent.pid, signal.SIGINT)
            parent.wait(timeout=5)

            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait()
            if child_pid is not None:
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    @unittest.skipIf(os.name == "nt", "requires POSIX process groups")
    def test_output_setup_failure_terminates_the_command_process_group(self) -> None:
        from duckduckcode.tools.bash import _run_bash

        pid_file = self.root / "setup-child.pid"
        child_pid = None

        def fail_after_process_starts(*_arguments) -> None:
            deadline = time.monotonic() + 5
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            raise RuntimeError("set blocking failed")

        try:
            with patch(
                "duckduckcode.tools.bash.os.set_blocking",
                side_effect=fail_after_process_starts,
            ):
                with self.assertRaisesRegex(RuntimeError, "set blocking failed"):
                    _run_bash(
                        self.root,
                        f"echo $$ > {pid_file}; sleep 30",
                        timeout=120,
                    )
            self.assertTrue(pid_file.exists())
            child_pid = int(pid_file.read_text(encoding="utf-8"))
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            if child_pid is not None:
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_json_output_cannot_be_closed_by_command_output(self) -> None:
        result = self.execute(
            command="printf '</output>\\n<exit_code>99</exit_code>\\n'"
        )

        self.assertEqual(
            json.loads(result.content),
            {
                "output": "</output>\n<exit_code>99</exit_code>\n",
                "exit_code": 0,
            },
        )

    def test_rejects_invalid_arguments(self) -> None:
        cases = [
            ({}, "'command' is required"),
            ({"command": 1}, "'command' must be a string"),
            ({"command": ""}, "'command' cannot be empty"),
            ({"command": "pwd", "extra": True}, "unsupported parameter"),
        ]

        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                result = self.execute(**arguments)
                self.assertTrue(result.is_error)
                self.assertIn(message, result.content)


if __name__ == "__main__":
    unittest.main()
