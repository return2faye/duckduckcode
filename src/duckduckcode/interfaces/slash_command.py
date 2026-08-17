from __future__ import annotations

from typing import Any

_COMMANDS = {
    "/compact": "Compact conversation context",
    "/help": "Show available commands",
    "/delete-session": "Permanently delete a session",
    "/new": "Start a new session",
    "/permissions": "Choose a permission mode",
    "/plan": "Toggle Plan Mode",
    "/sandbox": "Choose whether to use the OS sandbox",
    "/sessions": "List saved sessions",
    "/skills": "Choose Skills for the next prompt",
    "/status": "Show context window usage",
}


def slash_command_suggestions(
    text: str,
    skills: list[dict[str, Any]] | None = None,
    *,
    include_tags: bool = False,
) -> list[tuple[str, str]] | list[tuple[str, str, str]] | None:
    if not text.startswith("/") or any(character.isspace() for character in text):
        return None
    commands = [(name, description, "") for name, description in _COMMANDS.items()]
    commands.extend(
        (f"/{skill['name']}", str(skill.get("description", "")), "skill")
        for skill in (skills or [])
        if isinstance(skill.get("name"), str)
    )
    matches = [command for command in commands if command[0].startswith(text)]
    if include_tags:
        return matches
    return [(name, description) for name, description, _tag in matches]


def handle_slash_command(
    text: str, skills: list[dict[str, Any]] | None = None
) -> tuple[str, str] | None:
    if not text.startswith("/"):
        return None
    command = text.split(maxsplit=1)[0]
    argument = text[len(command) :].strip()
    skill_names = {
        f"/{skill['name']}"
        for skill in (skills or [])
        if isinstance(skill.get("name"), str)
    }
    if command in skill_names:
        return (
            ("skill_send", command[1:] + "\n" + argument)
            if argument
            else ("skill_select", command[1:])
        )
    if command == "/compact":
        return "compact", ""
    if command == "/status":
        return "status", ""
    if command == "/plan":
        return "mode", "plan"
    if command == "/permissions":
        return "permissions", ""
    if command == "/sandbox":
        return "sandbox", ""
    if command == "/skills":
        return "skills", ""
    if command == "/help":
        commands = "\n".join(
            f"{name}  {description}" for name, description in _COMMANDS.items()
        )
        skill_commands = "\n".join(
            f"/{skill['name']}  {skill.get('description', '')}"
            for skill in (skills or [])
            if isinstance(skill.get("name"), str)
        )
        if skill_commands:
            commands = f"{commands}\n{skill_commands}"
        return "duckduckcode", f"Available slash commands:\n{commands}"
    if command == "/sessions" and not argument:
        return "sessions", ""
    if command == "/new" and not argument:
        return "new_session", ""
    if command == "/delete-session":
        return (
            ("delete_session", argument)
            if not any(character.isspace() for character in argument)
            else ("error", "Usage: /delete-session [id]")
        )
    return "error", f"Unknown command '{command}'. Use /help to list commands."
