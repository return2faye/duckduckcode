from __future__ import annotations

import stat
from glob import has_magic
from pathlib import Path

MAX_REFERENCE_DEPTH = 5
SEPARATOR = "\n\n---\n\n"


def load_instructions(
    workspace: str | Path,
    home: str | Path | None = None,
    include_user: bool = True,
) -> str:
    workspace = Path(workspace).resolve()
    user_root = (Path(home) if home is not None else Path.home()) / ".duckduckcode"
    sources = [
        (workspace / "DDCODE.md", workspace),
        (workspace / ".duckduckcode" / "DDCODE.md", workspace),
        (workspace / "DDCODE.local.md", workspace),
    ]
    if include_user:
        sources.insert(0, (user_root / "DDCODE.md", user_root))
    instructions = []
    for source, boundary in sources:
        try:
            source.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise _error(
                f"instruction path cannot be accessed: {exc}", source, 1, [source]
            ) from exc
        resolved_boundary = _resolve(boundary, source, 1, [])
        expanded = _expand(source, resolved_boundary, set(), [], 0, source, 1)
        if expanded:
            instructions.append(expanded)
    return SEPARATOR.join(instructions)


def _expand(
    path: Path,
    boundary: Path,
    visited: set[Path],
    active: list[Path],
    depth: int,
    source: Path,
    line_number: int,
) -> str:
    canonical = _resolve(path, source, line_number, active)
    chain = [*active, canonical]
    if not canonical.is_relative_to(boundary):
        raise _error(
            "instruction path escapes its allowed directory", source, line_number, chain
        )
    if canonical in active:
        raise _error("instruction reference cycle", source, line_number, chain)
    if canonical in visited:
        return ""
    if depth > MAX_REFERENCE_DEPTH:
        raise _error(
            f"instruction references exceed {MAX_REFERENCE_DEPTH} nested levels",
            source,
            line_number,
            chain,
        )
    try:
        mode = canonical.stat().st_mode
    except FileNotFoundError as exc:
        raise _error(
            "instruction file does not exist", source, line_number, chain
        ) from exc
    except OSError as exc:
        raise _error(
            f"instruction path cannot be accessed: {exc}",
            source,
            line_number,
            chain,
        ) from exc
    if not stat.S_ISREG(mode):
        raise _error(
            "instruction path is not a regular file", source, line_number, chain
        )

    visited.add(canonical)
    active.append(canonical)
    try:
        try:
            content = canonical.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise _error(
                "instruction file is not valid UTF-8",
                source,
                line_number,
                active,
            ) from exc
        except OSError as exc:
            raise _error(
                f"instruction file cannot be read: {exc}",
                source,
                line_number,
                active,
            ) from exc

        output = []
        for current_line, text in enumerate(content.splitlines(), 1):
            if not text.startswith("@") or len(text) == 1:
                output.append(text)
                continue
            reference = text[1:]
            _validate_reference(reference, canonical, current_line, active)
            expanded = _expand(
                canonical.parent / reference,
                boundary,
                visited,
                active,
                depth + 1,
                canonical,
                current_line,
            )
            if expanded:
                output.extend(("---", expanded, "---"))
        return "\n".join(output).strip()
    finally:
        active.pop()


def _resolve(path: Path, source: Path, line_number: int, chain: list[Path]) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise _error(
            f"instruction path cannot be resolved: {exc}",
            source,
            line_number,
            [*chain, path],
        ) from exc


def _validate_reference(
    reference: str,
    source: Path,
    line_number: int,
    chain: list[Path],
) -> None:
    if (
        Path(reference).is_absolute()
        or "://" in reference
        or "#" in reference
        or has_magic(reference)
    ):
        raise _error(
            f"unsupported instruction reference: @{reference}",
            source,
            line_number,
            chain,
        )


def _error(
    message: str,
    source: Path,
    line_number: int,
    chain: list[Path],
) -> RuntimeError:
    rendered_chain = " -> ".join(str(path) for path in chain) or str(source)
    return RuntimeError(
        f"{message}; source: {source}; line: {line_number}; chain: {rendered_chain}"
    )
