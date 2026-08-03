from __future__ import annotations

_COMMANDS = {
    "/compact": "Compact conversation context",
    "/help": "Show available commands",
    "/delete-session": "Permanently delete a session",
    "/new": "Start a new session",
    "/permissions": "Choose a permission mode",
    "/plan": "Toggle Plan Mode",
    "/sessions": "List saved sessions",
    "/status": "Show context window usage",
}


def slash_command_suggestions(text: str) -> list[tuple[str, str]] | None:
    if not text.startswith("/") or any(character.isspace() for character in text):
        return None
    return [
        (name, description)
        for name, description in _COMMANDS.items()
        if name.startswith(text)
    ]


def handle_slash_command(text: str) -> tuple[str, str] | None:
    if not text.startswith("/"):
        return None
    command = text.split(maxsplit=1)[0]
    argument = text[len(command) :].strip()
    if command == "/compact":
        return "compact", ""
    if command == "/status":
        return "status", ""
    if command == "/plan":
        return "mode", "plan"
    if command == "/permissions":
        return "permissions", ""
    if command == "/help":
        commands = "\n".join(
            f"{name}  {description}" for name, description in _COMMANDS.items()
        )
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
