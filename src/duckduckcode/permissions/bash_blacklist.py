from __future__ import annotations

import re
import shlex

from ..tools.tool import ToolCall

_ASSIGNMENT = r"[A-Za-z_][A-Za-z0-9_]*=\S+"
_PREFIX_BODY = (
    rf"(?:{_ASSIGNMENT}\s+)*"
    rf"(?:(?:command|exec)(?:\s+-\S+)*\s+|"
    rf"env(?:\s+(?:-\S+|{_ASSIGNMENT}))*\s+)*"
)
_COMMAND_PREFIX = rf"^{_PREFIX_BODY}(?:sudo(?:\s+-\S+)*\s+)?"
_RAW_COMMAND_START = rf"(?:^|[;&|()`\n])\s*{_PREFIX_BODY}"
_DEVICE = (
    r"/dev/(?:"
    r"[hsv]d[a-z]\d*|xvd[a-z]\d*|nvme\d+n\d+(?:p\d+)?|"
    r"mmcblk\d+(?:p\d+)?|r?disk\d+(?:s\d+)?|"
    r"md\d+|dm-\d+|loop\d+|root|mapper/\S+|disk/by-id/\S+"
    r")"
)
_GIT_PREFIX = (
    _COMMAND_PREFIX
    + r"(?:\S*/)?git\b"
    + r"(?:\s+(?:(?:-C|-c|--git-dir|--namespace|--work-tree)\s+\S+|--\S+))*"
)
_RAW_PATTERNS = (
    (
        "fork bomb",
        re.compile(
            _RAW_COMMAND_START + r"(?P<fork_name>[:A-Za-z_][A-Za-z0-9_]*)\s*"
            r"\(\s*\)\s*\{\s*(?P=fork_name)\s*\|\s*"
            r"(?P=fork_name)\s*&\s*\}\s*;?\s*(?P=fork_name)"
        ),
    ),
    (
        "remote script pipe",
        re.compile(
            _RAW_COMMAND_START
            + r"(?:\S*/)?(?:curl|wget)\b[^;\n]*\|\s*"
            + r"(?:(?:sudo|command|exec)\s+|env(?:\s+\S+)*\s+)*"
            + r"(?:\S*/)?(?:bash|sh|zsh|ksh|fish|"
            + r"python(?:\d+(?:\.\d+)?)?|perl|ruby|node)\b"
        ),
    ),
)
_COMMAND_PATTERNS = (
    (
        "rm -rf",
        re.compile(
            _COMMAND_PREFIX
            + r"(?:\S*/)?rm\b"
            + r"(?=.*(?:\s--recursive\b|\s-[^-\s]*[rR]\S*))"
            + r"(?=.*(?:\s--force\b|\s-[^-\s]*f\S*))"
        ),
    ),
    (
        "direct disk write",
        re.compile(
            _COMMAND_PREFIX + r"(?:\S*/)?dd\b" + rf"(?=.*\sof\s*=\s*{_DEVICE}(?:\s|$))"
        ),
    ),
    (
        "block device overwrite",
        re.compile(rf"^.*\s(?:\d+\s+)?(?:>>?|&>|>\||>&)\s+{_DEVICE}(?:\s|$)"),
    ),
    (
        "block device overwrite",
        re.compile(
            _COMMAND_PREFIX
            + r"(?:\S*/)?(?:cp|install|mv|shred|tee)\b"
            + rf"(?=.*\s{_DEVICE}(?:\s|$))"
        ),
    ),
    (
        "disk formatting",
        re.compile(
            _COMMAND_PREFIX
            + r"(?:\S*/)?(?:fdisk|mkfs(?:\.\w+)?|parted|sfdisk|wipefs)\b"
            + rf"(?=.*\s{_DEVICE}(?:\s|$))"
        ),
    ),
    (
        "disk formatting",
        re.compile(
            _COMMAND_PREFIX
            + r"(?:\S*/)?diskutil\b"
            + rf"(?=.*\b(?:eraseDisk|eraseVolume|partitionDisk)\b)"
            + rf"(?=.*\s{_DEVICE}(?:\s|$))"
        ),
    ),
    (
        "recursive root permission change",
        re.compile(
            _COMMAND_PREFIX
            + r"(?:\S*/)?(?:chgrp|chmod|chown)\b"
            + r"(?=.*(?:\s--recursive\b|\s-[^-\s]*R\S*))"
            + r".*\s(?:/{1,2}|/\*|/\.)(?:\s|$)"
        ),
    ),
    (
        "privileged execution",
        re.compile(rf"^{_PREFIX_BODY}(?:sudo|su)\b"),
    ),
    (
        "system power control",
        re.compile(_COMMAND_PREFIX + r"(?:\S*/)?(?:halt|poweroff|reboot|shutdown)\b"),
    ),
    (
        "git reset --hard",
        re.compile(_GIT_PREFIX + r"\s+reset\b(?=.*\s--hard(?:\s|$))"),
    ),
    (
        "git clean --force",
        re.compile(
            _GIT_PREFIX + r"\s+clean\b" + r"(?=.*(?:\s--force\b|\s-[^-\s]*f\S*))"
        ),
    ),
    (
        "git push --force",
        re.compile(
            _GIT_PREFIX + r"\s+push\b" + r"(?=.*(?:\s--force(?:-\S+)?\b|\s-f(?:\s|$)))"
        ),
    ),
)
_SHELL_COMMAND = re.compile(
    _COMMAND_PREFIX
    + r"(?:\S*/)?(?:bash|fish|ksh|sh|zsh)\b"
    + r".*\s-[^\s]*c[^\s]*\s+(?P<script>.+)$"
)
_SINGLE_QUOTED_TEXT = re.compile(r"'[^']*'")
_PUNCTUATION = ";&|()`<>\n"
_CONTROL_OPERATORS = frozenset(";&|()`\n")


def check_bash_blacklist(tool_call: ToolCall) -> str | None:
    if tool_call.name != "Bash":
        return None
    command = tool_call.arguments.get("command")
    if not isinstance(command, str):
        return None

    try:
        blocked = _blocked_rule(command)
    except ValueError:
        return "Permission denied: the Bash command could not be checked safely."
    if blocked is None:
        return None
    return f"Permission denied: Bash command matches blocked rule '{blocked}'."


def _blocked_rule(command: str) -> str | None:
    unquoted = _SINGLE_QUOTED_TEXT.sub("", command)
    for name, pattern in _RAW_PATTERNS:
        if pattern.search(unquoted):
            return name

    for tokens in _split_commands(command):
        normalized = " ".join(tokens)
        nested = _SHELL_COMMAND.match(normalized)
        if nested is not None:
            blocked = _blocked_rule(nested.group("script"))
            if blocked is not None:
                return blocked
        for name, pattern in _COMMAND_PATTERNS:
            if pattern.search(normalized):
                return name
    return None


def _split_commands(command: str) -> list[list[str]]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=_PUNCTUATION)
    lexer.commenters = ""
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    commands: list[list[str]] = []
    current: list[str] = []
    for token in lexer:
        if token and set(token) <= _CONTROL_OPERATORS:
            if current:
                commands.append(current)
                current = []
        else:
            current.append(token)
    if current:
        commands.append(current)
    return commands
