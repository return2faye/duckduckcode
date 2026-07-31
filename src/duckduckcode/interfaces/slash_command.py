from __future__ import annotations

_COMMANDS = {
    "/compact": "Compact conversation context",
    "/help": "Show available commands",
    "/permissions": "Choose a permission mode",
    "/plan": "Toggle Plan Mode",
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
    return "error", f"Unknown command '{command}'. Use /help to list commands."
