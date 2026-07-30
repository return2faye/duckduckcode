from __future__ import annotations

_COMMANDS = {"/help": "Show available commands"}


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
    if command == "/help":
        commands = "\n".join(
            f"{name}  {description}" for name, description in _COMMANDS.items()
        )
        return "duckduckcode", f"Available slash commands:\n{commands}"
    return "error", f"Unknown command '{command}'. Use /help to list commands."
