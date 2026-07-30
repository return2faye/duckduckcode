from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from typing import Any

from .tool import Tool, ToolResult, create_tool

DEFAULT_TIMEOUT_SECONDS = 120
MAX_OUTPUT_BYTES = 200_000
NON_ERROR_EXIT_CODES = {
    "[": {1},
    "cmp": {1},
    "diff": {1},
    "egrep": {1},
    "fgrep": {1},
    "grep": {1},
    "rg": {1},
    "test": {1},
}

BASH_PARAMS = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": (
                "Shell command to execute from the working directory. "
                "Use absolute paths when referring to files. For a "
                "long-running service, detach it in the background, "
                "redirect stdin and write stdout/stderr to a log file, "
                "and print its PID."
            ),
        },
    },
    "required": ["command"],
    "additionalProperties": False,
}


def create_bash_tool(working_directory: Path | None = None) -> Tool:
    base_directory = (working_directory or Path.cwd()).resolve()
    return create_tool(
        "Bash",
        "Use Bash only when no dedicated tool fits, such as running tests, package commands, git commands, starting development servers, or inspecting non-text files. Commands execute through the system shell from the working directory. The timeout of 120 seconds applies to the foreground shell process, not a properly detached service. For a long-running service, start it in the background with stdin redirected and stdout/stderr written to a log file, print its PID, then use a follow-up Bash call to check its health. Results are JSON with merged stdout/stderr in output and the numeric exit_code; output beyond 200,000 bytes is truncated. This tool can modify files or external state, so avoid destructive commands unless the user explicitly confirms them.",
        BASH_PARAMS,
        lambda command: _run_bash(base_directory, command),
        _validate_arguments,
        is_dangerous=True,
        category="shell",
    )


def _validate_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    unsupported = sorted(arguments.keys() - {"command"})
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            f"Bash failed: unsupported parameter(s): {names}. "
            "Remove them and use only command."
        )

    if "command" not in arguments:
        raise ValueError("Bash failed: 'command' is required. Provide a shell command.")
    command = arguments["command"]
    if not isinstance(command, str):
        raise ValueError(
            "Bash failed: 'command' must be a string. Provide a shell command."
        )
    if not command.strip():
        raise ValueError(
            "Bash failed: 'command' cannot be empty. Provide a shell command."
        )
    return {"command": command}


def _run_bash(
    working_directory: Path,
    command: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> ToolResult:
    process = subprocess.Popen(
        command,
        cwd=working_directory,
        shell=True,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        bufsize=0,
        start_new_session=os.name != "nt",
    )
    stdout = process.stdout
    captured = bytearray()
    truncated = False
    timed_out = False
    try:
        if stdout is None:
            raise RuntimeError("Bash failed: could not capture command output.")
        os.set_blocking(stdout.fileno(), False)

        def read_once() -> bool:
            nonlocal truncated
            try:
                chunk = stdout.read(64 * 1024)
            except BlockingIOError:
                return False
            if not chunk:
                return False
            remaining = MAX_OUTPUT_BYTES - len(captured)
            if remaining > 0:
                captured.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated = True
            return True

        deadline = time.monotonic() + timeout
        while process.poll() is None:
            read_output = read_once()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                _terminate(process)
                break
            if not read_output:
                time.sleep(min(0.01, remaining))
        while read_once():
            pass
        exit_code = 124 if timed_out else process.returncode
    except BaseException:
        _terminate(process)
        raise
    finally:
        if stdout is not None:
            stdout.close()

    merged_output = captured.decode("utf-8", errors="replace")
    if truncated:
        merged_output = _append_line(
            merged_output, f"[Output truncated after {MAX_OUTPUT_BYTES} bytes.]"
        )
    if timed_out:
        merged_output = _append_line(
            merged_output, f"[Command timed out after {timeout:g} seconds.]"
        )

    return ToolResult(
        json.dumps(
            {"output": merged_output, "exit_code": exit_code},
            ensure_ascii=False,
        ),
        is_error=timed_out or _is_error_exit(command, exit_code),
    )


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.kill()
    except ProcessLookupError:
        return

    try:
        process.wait(1)
    except subprocess.TimeoutExpired:
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def _append_line(text: str, line: str) -> str:
    if text and not text.endswith("\n"):
        text += "\n"
    return text + line + "\n"


def _is_error_exit(command: str, exit_code: int) -> bool:
    if exit_code == 0:
        return False
    command_name = _simple_command_name(command)
    if command_name is None:
        return True
    return exit_code not in NON_ERROR_EXIT_CODES.get(command_name, set())


def _simple_command_name(command: str) -> str | None:
    if _has_compound_operator(command):
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    while tokens and _is_assignment(tokens[0]):
        tokens.pop(0)
    if not tokens or tokens[0] == "!":
        return None
    return Path(tokens[0]).name


def _is_assignment(token: str) -> bool:
    name, separator, _ = token.partition("=")
    return bool(separator and name and name.replace("_", "a").isalnum())


def _has_compound_operator(command: str) -> bool:
    quote = ""
    escaped = False
    for index, char in enumerate(command):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in {";", "\n", "(", ")"}:
            return True
        if char == "|" and (index == 0 or command[index - 1] != ">"):
            return True
        if char == "&":
            previous = command[index - 1] if index else ""
            following = command[index + 1] if index + 1 < len(command) else ""
            if previous not in {"<", ">"} and following != ">":
                return True
    return False
