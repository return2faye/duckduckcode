from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import stat
from typing import Any, Literal

import yaml

from ..tools.tool import ToolResult

MAX_SKILL_BYTES = 256 * 1024
NAME_RE = re.compile(r"(?=.{1,64}\Z)[a-z0-9]+(?:-[a-z0-9]+)*\Z")
MODES = {"inline", "fork"}


class SkillError(RuntimeError):
    pass


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    value: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in value
        except TypeError as exc:
            raise SkillError("YAML field names must be scalar values") from exc
        if duplicate:
            raise SkillError(f"duplicate YAML field: {key}")
        value[key] = loader.construct_object(value_node, deep=deep)
    return value


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    mode: Literal["inline", "fork"]
    scope: Literal["user", "project"]
    path: Path
    root: Path | None
    metadata: dict[str, Any]


class SkillManager:
    def __init__(
        self,
        workspace: str | Path,
        *,
        home: str | Path | None = None,
        builtin_commands: set[str] | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        home_path = Path(home).expanduser() if home is not None else Path.home()
        self.user_root = home_path / ".duckduckcode" / "skills"
        self.project_root = self.workspace / ".duckduckcode" / "skills"
        self.builtin_commands = set(builtin_commands or set())
        self.skills: dict[str, Skill] = {}
        self.errors: tuple[str, ...] = ()
        self._error_signature: tuple[str, ...] = ()
        self.loaded: set[str] = set()
        self.active: dict[str, str] = {}
        self.active_roots: dict[str, Path] = {}

    def refresh(self) -> tuple[list[Skill], str | None]:
        user_skills, user_errors = self._discover_scope(self.user_root, "user")
        project_skills, project_errors = self._discover_scope(
            self.project_root, "project"
        )
        merged = dict(user_skills)
        merged.update(project_skills)
        self.skills = dict(sorted(merged.items()))
        self.errors = tuple(user_errors + project_errors)
        signature = self.errors
        warning = None
        if signature and signature != self._error_signature:
            warning = "Skill discovery warnings:\n" + "\n".join(
                f"- {error}" for error in signature
            )
        self._error_signature = signature
        return list(self.skills.values()), warning

    def list(self) -> tuple[Skill, ...]:
        return tuple(self.skills.values())

    def catalog_block(self) -> str:
        if not self.skills:
            return ""
        lines = [
            "Available Skills for this turn. If a skill clearly matches the user's "
            "current intent, call LoadSkill with its exact name before solving."
        ]
        lines.extend(
            f"- {skill.name}: {skill.description}" for skill in self.skills.values()
        )
        return "\n".join(lines)

    def active_block(self) -> str:
        if not self.active:
            return ""
        blocks = ["Active Skills for this user turn:"]
        for name in sorted(self.active):
            blocks.append(self.active[name])
        return "\n\n".join(blocks)

    def load(self, name: str, task: str) -> ToolResult:
        skill = self.skills.get(name)
        if skill is None:
            return ToolResult(f"Skill '{name}' was not found.", is_error=True)
        if name in self.loaded:
            return ToolResult(f"Skill '{name}' is already loaded.")
        if skill.mode == "fork":
            return ToolResult(
                f"Skill '{name}' must be run by the Agent in fork mode.",
                is_error=True,
            )
        self.active[name] = _skill_block(skill, task)
        self.loaded.add(name)
        if skill.root is not None:
            self.active_roots[name] = skill.root
        return ToolResult(f"Loaded skill '{name}'.")

    def load_fork(
        self, name: str, task: str
    ) -> tuple[Skill | None, str | None, ToolResult | None]:
        skill = self.skills.get(name)
        if skill is None:
            return None, None, ToolResult(f"Skill '{name}' was not found.", True)
        if name in self.loaded:
            return None, None, ToolResult(f"Skill '{name}' is already loaded.")
        if skill.mode != "fork":
            return None, None, ToolResult(f"Skill '{name}' is not a fork Skill.", True)
        try:
            block = _skill_block(skill, task)
        except (OSError, UnicodeError, ValueError, SkillError) as exc:
            return None, None, ToolResult(f"Could not load Skill '{name}': {exc}", True)
        self.loaded.add(name)
        return skill, block, None

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def clear_active(self) -> None:
        self.loaded.clear()
        self.active.clear()
        self.active_roots.clear()

    def _discover_scope(
        self, root: Path, scope: Literal["user", "project"]
    ) -> tuple[dict[str, Skill], list[str]]:
        try:
            if not root.exists():
                return {}, []
            if not root.is_dir() or root.is_symlink():
                return {}, [f"{root} is not a regular skills directory"]
            entries = sorted(root.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            return {}, [f"{root}: {exc}"]

        candidates: list[Path] = []
        for path in entries:
            if path.is_symlink() and (path.suffix == ".md" or path.is_dir()):
                candidates.append(path)
            elif path.is_dir():
                entry = path / "SKILL.md"
                if entry.exists() or entry.is_symlink():
                    candidates.append(entry)
            elif path.suffix == ".md":
                candidates.append(path)

        by_name: dict[str, list[Skill]] = {}
        errors = []
        for path in candidates:
            try:
                skill = _read_skill(path, scope)
                if f"/{skill.name}" in self.builtin_commands:
                    errors.append(f"{path}: skill command /{skill.name} conflicts")
                    continue
                by_name.setdefault(skill.name, []).append(skill)
            except (OSError, UnicodeError, ValueError, SkillError) as exc:
                errors.append(f"{path}: {exc}")

        skills = {}
        for name, matches in by_name.items():
            if len(matches) > 1:
                errors.append(f"{root}: duplicate skill name '{name}'")
                continue
            skills[name] = matches[0]
        return skills, errors


def _read_skill(path: Path, scope: Literal["user", "project"]) -> Skill:
    return _read_skill_document(path, scope)[0]


def _read_skill_document(
    path: Path, scope: Literal["user", "project"]
) -> tuple[Skill, str]:
    _validate_entry(path)
    resolved = path.resolve()
    text = path.read_text(encoding="utf-8")
    metadata, body = _split_frontmatter(text)
    name = metadata.get("name")
    description = metadata.get("description")
    skill_mode = metadata.get("mode", "inline")
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise SkillError("name must be lowercase kebab-case up to 64 characters")
    if not isinstance(description, str) or not description.strip():
        raise SkillError("description is required")
    if skill_mode not in MODES:
        raise SkillError("mode must be inline or fork")
    if not body.strip():
        raise SkillError("body cannot be empty")
    root = resolved.parent if resolved.name == "SKILL.md" else None
    return (
        Skill(
            name,
            description.strip(),
            skill_mode,
            scope,
            resolved,
            root,
            dict(metadata),
        ),
        body.strip(),
    )


def _skill_block(skill: Skill, task: str) -> str:
    current, body = _read_skill_document(skill.path, skill.scope)
    if current.name != skill.name or current.mode != skill.mode:
        raise SkillError("entry changed after discovery; refresh Skills and retry")
    lines = [f"Skill: {skill.name}", f"Path: {skill.path}", f"Task: {task}"]
    if skill.root is not None:
        lines.append(
            f"Resources: {skill.root} (read-only via ReadFile, Glob, and Grep)"
        )
    return "\n".join(lines) + f"\n\n{body}"


def _validate_entry(path: Path) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise SkillError("entry must not be a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise SkillError("entry must be a regular file")
    if info.st_size > MAX_SKILL_BYTES:
        raise SkillError(f"entry exceeds {MAX_SKILL_BYTES} bytes")


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise SkillError("frontmatter is required")
    end = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.rstrip("\r\n") == "---"
        ),
        None,
    )
    if end is None:
        raise SkillError("frontmatter closing marker is required")
    raw = "".join(lines[1:end])
    body = "".join(lines[end + 1 :])
    try:
        metadata = yaml.load(raw, Loader=_UniqueLoader)
    except yaml.YAMLError as exc:
        raise SkillError(f"invalid YAML frontmatter: {exc}") from exc
    if not isinstance(metadata, dict):
        raise SkillError("frontmatter must be a mapping")
    return metadata, body
