from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from typing import Any, Literal

import yaml

from ..tools.tool import MAX_SUBAGENT_SLUG_LENGTH, SUBAGENT_SLUG_RE

MAX_CONFIG_BYTES = 256 * 1024
MAX_STATE_BYTES = 256 * 1024
MAX_WORKTREES = 256
MAX_DEPENDENCIES = 64
MAX_DEPENDENCY_LENGTH = 512
GIT_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class WorktreeConfiguration:
    copy: tuple[str, ...] = ()
    symlinks: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class InjectedFile:
    path: str
    source: str
    kind: Literal["copy", "symlink"]
    digest: str | None = None


@dataclass(frozen=True)
class Worktree:
    id: str
    name: str
    path: Path
    branch: str
    base_commit: str
    created_at: int


@dataclass
class WorktreeSession:
    worktree: Worktree
    owner_session_id: str
    active_task_id: str | None = None
    entered_at: int | None = None
    last_used_at: int | None = None
    injected: dict[str, InjectedFile] = field(default_factory=dict)
    recovered: bool = False
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    state: Literal["creating", "idle", "active"] = "idle"


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def load_worktree_configuration(
    *, home: str | Path | None = None
) -> WorktreeConfiguration:
    home_path = Path(home).expanduser().resolve() if home is not None else Path.home()
    path = home_path / ".duckduckcode" / "worktree.yaml"
    try:
        text = _read_regular_file(path, home_path, MAX_CONFIG_BYTES)
    except (OSError, RuntimeError, UnicodeError) as exc:
        return WorktreeConfiguration(
            warnings=(f"Worktree config '{path}' was skipped: {exc}",)
        )
    if text is None:
        return WorktreeConfiguration()
    try:
        loaded = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        detail = "duplicate YAML key" if "duplicate key" in str(exc) else "invalid YAML"
        return WorktreeConfiguration(
            warnings=(f"Worktree config '{path}' was skipped: {detail}.",)
        )
    if loaded is None:
        return WorktreeConfiguration()
    if not isinstance(loaded, dict):
        return WorktreeConfiguration(
            warnings=(f"Worktree config '{path}' must contain a mapping.",)
        )
    unknown = sorted(str(key) for key in loaded if key not in {"copy", "symlinks"})
    if unknown:
        return WorktreeConfiguration(
            warnings=(
                f"Worktree config '{path}' was skipped: unknown field(s): "
                + ", ".join(unknown),
            )
        )
    values: dict[str, tuple[str, ...]] = {}
    try:
        configured_count = 0
        for name in ("copy", "symlinks"):
            configured = loaded.get(name, [])
            if not isinstance(configured, list) or not all(
                isinstance(value, str) for value in configured
            ):
                raise ValueError(f"{name} must be a list of strings")
            configured_count += len(configured)
            values[name] = tuple(
                dict.fromkeys(_validate_dependency_path(value) for value in configured)
            )
        if configured_count > MAX_DEPENDENCIES:
            raise ValueError(
                f"copy and symlinks may contain at most {MAX_DEPENDENCIES} entries"
            )
        overlap = set(values["copy"]) & set(values["symlinks"])
        if overlap:
            raise ValueError(
                "paths cannot appear in both copy and symlinks: "
                + ", ".join(sorted(overlap))
            )
    except ValueError as exc:
        return WorktreeConfiguration(
            warnings=(f"Worktree config '{path}' was skipped: {exc}.",)
        )
    return WorktreeConfiguration(values["copy"], values["symlinks"])


class WorktreeManager:
    def __init__(
        self,
        workspace: str | Path,
        *,
        home: str | Path | None = None,
        clock: Any | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.home = (
            Path(home).expanduser().resolve() if home is not None else Path.home()
        )
        self.configuration = load_worktree_configuration(home=self.home)
        self.state_path = self.workspace / ".duckduckcode" / "worktrees.json"
        self._clock = clock or time.time
        self._lock = threading.RLock()
        self._sessions: dict[str, WorktreeSession] = {}
        self._active: dict[str, WorktreeSession] = {}
        self._warnings = list(self.configuration.warnings)
        self._state_error: str | None = None
        repository = _git(self.workspace, "rev-parse", "--show-toplevel", check=False)
        self.repository = (
            Path(repository.stdout.strip()).resolve()
            if repository.returncode == 0 and repository.stdout.strip()
            else None
        )
        self.workspace_relative = (
            self.workspace.relative_to(self.repository)
            if self.repository is not None
            and self.workspace.is_relative_to(self.repository)
            else Path(".")
        )
        repository_key = hashlib.sha256(
            str(self.repository or self.workspace).encode("utf-8")
        ).hexdigest()[:16]
        self.root = self.home / ".duckduckcode" / "worktrees" / repository_key
        self.recover()

    @property
    def warnings(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._warnings)

    @property
    def active(self) -> dict[str, WorktreeSession]:
        with self._lock:
            return dict(self._active)

    def enter(self, owner_session_id: str, name: str, task_id: str) -> WorktreeSession:
        with self._lock:
            if not owner_session_id or not task_id:
                raise ValueError("owner_session_id and task_id are required.")
            if (
                len(name) > MAX_SUBAGENT_SLUG_LENGTH
                or SUBAGENT_SLUG_RE.fullmatch(name) is None
            ):
                raise ValueError("Invalid worktree name.")
            repository = self._require_repository()
            if self._state_error is not None:
                raise RuntimeError(self._state_error)
            worktree_id = _worktree_id(owner_session_id, name)
            session = self._sessions.get(worktree_id)
            if session is not None:
                if session.active_task_id is not None:
                    raise RuntimeError(
                        f"Worktree '{worktree_id}' is already active in task "
                        f"'{session.active_task_id}'."
                    )
                if session.error is not None:
                    raise RuntimeError(session.error)
                self._ensure_checkout(session)
            else:
                status = self._repository_status().stdout
                if status:
                    raise RuntimeError(
                        "Git repository must be clean before starting a new "
                        "worktree fork"
                    )
                if len(self._sessions) >= MAX_WORKTREES:
                    raise RuntimeError(
                        f"At most {MAX_WORKTREES} managed worktrees are supported."
                    )
                base_commit = _git(
                    repository, "rev-parse", "--verify", "HEAD"
                ).stdout.strip()
                _secure_directory(self.root, self.home, create=True)
                worktree = Worktree(
                    worktree_id,
                    name,
                    self.root / worktree_id,
                    f"worktree-{worktree_id}",
                    base_commit,
                    int(self._clock()),
                )
                _git(repository, "check-ref-format", "--branch", worktree.branch)
                if (
                    worktree.path.exists()
                    or worktree.path.is_symlink()
                    or worktree.path in self._registered()
                ):
                    raise RuntimeError(
                        f"Refusing to overwrite unmanaged worktree path '{worktree.path}'."
                    )
                if self._branch_exists(worktree.branch):
                    raise RuntimeError(
                        f"Refusing to reset unmanaged branch '{worktree.branch}'."
                    )
                session = WorktreeSession(worktree, owner_session_id, state="creating")
                self._sessions[worktree_id] = session
                try:
                    self._persist()
                    self._add_checkout(session)
                    session.state = "idle"
                except Exception:
                    self._rollback_create(session)
                    raise
            try:
                session.warnings.extend(self._reconcile_dependencies(session))
            except Exception:
                session.state = "idle"
                try:
                    self._persist()
                except RuntimeError as exc:
                    self._warnings.append(str(exc))
                raise
            now = int(self._clock())
            session.active_task_id = task_id
            session.entered_at = now
            session.last_used_at = now
            session.recovered = False
            session.state = "active"
            self._active[worktree_id] = session
            try:
                self._persist()
            except Exception:
                session.active_task_id = None
                session.entered_at = None
                session.state = "idle"
                self._active.pop(worktree_id, None)
                try:
                    self._persist()
                except RuntimeError as exc:
                    self._warnings.append(str(exc))
                raise
            return session

    def leave(self, session: WorktreeSession, *, partial: bool) -> dict[str, object]:
        with self._lock:
            current = self._sessions.get(session.worktree.id)
            if current is not session:
                raise RuntimeError("Worktree session is no longer managed.")
            warnings = list(session.warnings)
            try:
                changes = self._changes(session, partial=partial)
            except Exception as exc:
                warnings.append(f"Could not collect worktree diff: {exc}")
                changes = self._empty_changes(session, partial=True)
            session.active_task_id = None
            session.entered_at = None
            session.last_used_at = int(self._clock())
            session.state = "idle"
            session.warnings.clear()
            self._active.pop(session.worktree.id, None)
            try:
                self._persist()
            except Exception as exc:
                warnings.append(f"Could not persist worktree session: {exc}")
            return {
                "worktree_id": session.worktree.id,
                "warnings": warnings,
                "changes": changes,
            }

    def list(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            result = []
            for worktree_id in sorted(self._sessions):
                session = self._sessions[worktree_id]
                status = self._status(session)
                result.append(
                    {
                        "id": worktree_id,
                        "name": session.worktree.name,
                        "path": str(self._workspace_path(session)),
                        "branch": session.worktree.branch,
                        "base_commit": session.worktree.base_commit,
                        "created_at": session.worktree.created_at,
                        "owner_session_id": session.owner_session_id,
                        "active": session.active_task_id is not None,
                        "active_task_id": session.active_task_id,
                        "dirty": bool(status["files"]),
                        "files": status["files"],
                        "parent_changed": status["parent_changed"],
                        "recovered": session.recovered,
                        "error": session.error,
                    }
                )
            return tuple(result)

    def preflight_remove(self, worktree_id: str) -> dict[str, object]:
        with self._lock:
            session = self._require_session(worktree_id)
            if session.active_task_id is not None:
                raise RuntimeError(
                    f"Worktree '{worktree_id}' is active in task "
                    f"'{session.active_task_id}'."
                )
            validation = self._validate_recovered(session)
            if validation is not None:
                raise RuntimeError(validation)
            status = self._status(session)
            return {
                "id": worktree_id,
                "name": session.worktree.name,
                "base_commit": session.worktree.base_commit,
                "dirty": bool(status["files"]),
                "files": status["files"],
                "parent_changed": status["parent_changed"],
            }

    def remove(self, worktree_id: str) -> dict[str, object]:
        with self._lock:
            session = self._require_session(worktree_id)
            if session.active_task_id is not None:
                raise RuntimeError(
                    f"Worktree '{worktree_id}' is active in task "
                    f"'{session.active_task_id}'."
                )
            validation = self._validate_recovered(session)
            if validation is not None:
                raise RuntimeError(validation)
            changes = (
                self._changes(session, partial=False)
                if session.worktree.path.is_dir()
                else self._empty_changes(session, partial=False)
            )
            repository = self._require_repository()
            if session.worktree.path in self._registered():
                removed = _git(
                    repository,
                    "worktree",
                    "remove",
                    "--force",
                    str(session.worktree.path),
                    check=False,
                )
                if removed.returncode != 0:
                    changes["warnings"] = [
                        removed.stderr.strip()
                        or f"Could not remove worktree '{session.worktree.path}'."
                    ]
                    return changes
            branch = _git(
                repository,
                "branch",
                "-D",
                "--",
                session.worktree.branch,
                check=False,
            )
            if branch.returncode != 0 and self._branch_exists(session.worktree.branch):
                session.error = branch.stderr.strip() or "Could not remove branch."
                warnings = [session.error]
                try:
                    self._persist()
                except RuntimeError as exc:
                    warnings.append(f"Could not persist removal state: {exc}")
                changes["warnings"] = warnings
                return changes
            self._sessions.pop(worktree_id, None)
            self._active.pop(worktree_id, None)
            try:
                self._persist()
            except RuntimeError as exc:
                session.error = f"Worktree was removed but state cleanup failed: {exc}"
                self._sessions[worktree_id] = session
                changes["warnings"] = [session.error]
            return changes

    def recover(self) -> tuple[str, ...]:
        with self._lock:
            if not self.state_path.exists() and not self.state_path.is_symlink():
                return self.warnings
            try:
                text = _read_regular_file(
                    self.state_path, self.workspace, MAX_STATE_BYTES
                )
                assert text is not None
                raw = json.loads(text, object_pairs_hook=_unique_object)
                sessions = self._parse_state(raw)
            except (
                OSError,
                RuntimeError,
                UnicodeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                self._state_error = (
                    f"Worktree state '{self.state_path}' could not be recovered: {exc}"
                )
                self._warnings.append(self._state_error)
                return self.warnings
            changed = False
            self._sessions = sessions
            for session in self._sessions.values():
                if session.active_task_id is not None:
                    session.active_task_id = None
                    session.entered_at = None
                    session.recovered = True
                    changed = True
                if session.state != "idle":
                    session.state = "idle"
                    session.recovered = True
                    changed = True
                try:
                    session.error = self._validate_recovered(session)
                except RuntimeError as exc:
                    session.error = (
                        f"Managed worktree '{session.worktree.id}' could not be "
                        f"validated: {exc}"
                    )
                if session.error:
                    self._warnings.append(session.error)
            if changed:
                try:
                    self._persist()
                except RuntimeError as exc:
                    self._warnings.append(str(exc))
            return self.warnings

    def close(self) -> None:
        with self._lock:
            if not self._active:
                return
            for session in self._active.values():
                session.active_task_id = None
                session.entered_at = None
                session.last_used_at = int(self._clock())
                session.state = "idle"
            self._active.clear()
            try:
                self._persist()
            except RuntimeError as exc:
                self._warnings.append(str(exc))

    def _ensure_checkout(self, session: WorktreeSession) -> None:
        if session.error is not None:
            raise RuntimeError(session.error)
        registered = self._registered().get(session.worktree.path)
        if session.worktree.path.is_dir() and registered == (
            session.worktree.base_commit,
            f"refs/heads/{session.worktree.branch}",
        ):
            return
        if session.worktree.path.exists() or registered is not None:
            raise RuntimeError(
                f"Managed worktree '{session.worktree.id}' has conflicting path or "
                "Git registration."
            )
        if self._branch_exists(session.worktree.branch):
            tip = _git(
                self._require_repository(),
                "rev-parse",
                "--verify",
                session.worktree.branch,
            ).stdout.strip()
            if tip != session.worktree.base_commit:
                raise RuntimeError(
                    f"Refusing to reset worktree branch '{session.worktree.branch}' "
                    "because it contains external commits."
                )
        self._add_checkout(session)

    def _add_checkout(self, session: WorktreeSession) -> None:
        worktree = session.worktree
        worktree.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        added = _git(
            self._require_repository(),
            "worktree",
            "add",
            "-B",
            worktree.branch,
            str(worktree.path),
            worktree.base_commit,
        )
        if added.returncode != 0:
            raise RuntimeError(added.stderr.strip() or "Could not create worktree.")
        worktree.path.chmod(0o700)
        self._configure_hooks(session)

    def _configure_hooks(self, session: WorktreeSession) -> None:
        repository = self._require_repository()
        core_worktree = _git(
            repository, "config", "--get", "core.worktree", check=False
        ).stdout.strip()
        bare = _git(
            repository, "config", "--bool", "--get", "core.bare", check=False
        ).stdout.strip()
        enabled = _git(
            repository,
            "config",
            "--bool",
            "--get",
            "extensions.worktreeConfig",
            check=False,
        ).stdout.strip()
        if enabled != "true":
            if core_worktree or bare == "true":
                raise RuntimeError(
                    "Cannot safely enable extensions.worktreeConfig while shared "
                    "core.worktree or core.bare=true is configured."
                )
            _git(repository, "config", "extensions.worktreeConfig", "true")
        configured = _git(
            repository,
            "config",
            "--path",
            "--get",
            "core.hooksPath",
            check=False,
            disable_hooks=False,
        ).stdout.strip()
        if configured:
            hooks = Path(configured).expanduser()
            if not hooks.is_absolute():
                hooks = self.workspace / hooks
        else:
            output = _git(
                repository,
                "rev-parse",
                "--git-path",
                "hooks",
                disable_hooks=False,
            ).stdout.strip()
            hooks = Path(output)
            if not hooks.is_absolute():
                hooks = repository / hooks
        hooks = hooks.resolve()
        if not hooks.is_dir():
            raise RuntimeError(f"Git hooks path '{hooks}' is not a directory.")
        _git(
            session.worktree.path,
            "config",
            "--worktree",
            "core.hooksPath",
            str(hooks),
            disable_hooks=False,
        )

    def _reconcile_dependencies(self, session: WorktreeSession) -> list[str]:
        warnings: list[str] = []
        desired = {
            **{path: "copy" for path in self.configuration.copy},
            **{path: "symlink" for path in self.configuration.symlinks},
        }
        for relative, injected in list(session.injected.items()):
            if relative in desired:
                continue
            destination = self._workspace_path(session) / relative
            if not _safe_injection_parent(self._workspace_path(session), destination):
                warnings.append(
                    f"Injected path '{relative}' has an unsafe parent and was retained."
                )
                continue
            if _matches_injected(destination, injected, self.workspace):
                _remove_injected(destination)
                session.injected.pop(relative, None)
            else:
                warnings.append(
                    f"Injected path '{relative}' changed outside WorktreeManager and "
                    "was retained."
                )
        for relative, kind in desired.items():
            source = self.workspace / relative
            valid = self._validate_dependency(source, relative, kind)
            if valid is not None:
                warnings.append(valid)
                continue
            destination = self._workspace_path(session) / relative
            if not _safe_injection_parent(self._workspace_path(session), destination):
                warnings.append(
                    f"Worktree dependency '{relative}' was skipped: destination has "
                    "an unsafe parent."
                )
                continue
            previous = session.injected.get(relative)
            if previous is not None and not _matches_injected(
                destination, previous, self.workspace
            ):
                warnings.append(
                    f"Injected path '{relative}' changed outside WorktreeManager and "
                    "was not overwritten."
                )
                continue
            if previous is None and (destination.exists() or destination.is_symlink()):
                warnings.append(
                    f"Worktree dependency '{relative}' was skipped: destination exists."
                )
                continue
            if previous is not None:
                _remove_injected(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if kind == "copy":
                _copy_atomic(source, destination)
                injected = InjectedFile(
                    relative, relative, "copy", _sha256_file(destination)
                )
            else:
                destination.symlink_to(
                    source.resolve(), target_is_directory=source.is_dir()
                )
                injected = InjectedFile(relative, relative, "symlink")
            session.injected[relative] = injected
        return warnings

    def _validate_dependency(
        self, source: Path, relative: str, kind: str
    ) -> str | None:
        try:
            info = source.lstat()
        except FileNotFoundError:
            return f"Worktree dependency '{relative}' was skipped: source is missing."
        if stat.S_ISLNK(info.st_mode):
            return f"Worktree dependency '{relative}' was skipped: source is a symlink."
        if kind == "copy" and not stat.S_ISREG(info.st_mode):
            return (
                f"Worktree dependency '{relative}' was skipped: copy sources must be "
                "regular files."
            )
        if kind == "symlink" and not (
            stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
        ):
            return (
                f"Worktree dependency '{relative}' was skipped: symlink sources must "
                "be regular files or directories."
            )
        resolved = source.resolve()
        if not resolved.is_relative_to(self.workspace):
            return (
                f"Worktree dependency '{relative}' was skipped: source resolves "
                "outside the workspace."
            )
        repository_relative = source.relative_to(self._require_repository()).as_posix()
        if (
            _git(
                self._require_repository(),
                "ls-files",
                "--error-unmatch",
                "--",
                repository_relative,
                check=False,
            ).returncode
            == 0
        ):
            return f"Worktree dependency '{relative}' was skipped: source is tracked."
        if (
            _git(
                self._require_repository(),
                "check-ignore",
                "--quiet",
                "--",
                repository_relative,
                check=False,
            ).returncode
            != 0
        ):
            return (
                f"Worktree dependency '{relative}' was skipped: source is not ignored."
            )
        return None

    def _changes(self, session: WorktreeSession, *, partial: bool) -> dict[str, object]:
        worktree = session.worktree
        pathspec = self._diff_pathspec(session)
        untracked = [
            path
            for path in _git(
                worktree.path,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                *pathspec,
            ).stdout.split("\0")
            if path
        ]
        try:
            if untracked:
                _git(worktree.path, "add", "-N", "--", *untracked)
            common = ("--no-renames", worktree.base_commit, "--", *pathspec)
            patch = _git(
                worktree.path,
                "diff",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "--no-textconv",
                *common,
            ).stdout
            entries = [
                entry
                for entry in _git(
                    worktree.path, "diff", "--name-status", "-z", *common
                ).stdout.split("\0")
                if entry
            ]
        finally:
            if untracked:
                _git(worktree.path, "reset", "--quiet", "--", *untracked, check=False)
        return {
            "worktree_id": worktree.id,
            "branch": worktree.branch,
            "base_commit": worktree.base_commit,
            "parent_changed": self._parent_changed(worktree.base_commit),
            "partial": partial,
            "files": [
                {"status": entries[index], "path": entries[index + 1]}
                for index in range(0, len(entries), 2)
            ],
            "patch": patch,
        }

    def _status(self, session: WorktreeSession) -> dict[str, object]:
        if session.error is not None or not session.worktree.path.is_dir():
            return {"files": [], "parent_changed": True}
        output = _git(
            session.worktree.path,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
            "--",
            *self._diff_pathspec(session),
            check=False,
        ).stdout
        files = [line[3:] for line in output.splitlines() if len(line) >= 4]
        return {
            "files": files,
            "parent_changed": self._parent_changed(session.worktree.base_commit),
        }

    def _diff_pathspec(self, session: WorktreeSession) -> tuple[str, ...]:
        scope = self.workspace_relative.as_posix() or "."
        prefix = "" if self.workspace_relative == Path(".") else f"{scope}/"
        return (
            scope,
            *(f":(exclude){prefix}{relative}" for relative in sorted(session.injected)),
        )

    def _parent_changed(self, base_commit: str) -> bool:
        repository = self._require_repository()
        head = _git(repository, "rev-parse", "--verify", "HEAD", check=False)
        status = self._repository_status(check=False)
        return (
            head.returncode != 0
            or head.stdout.strip() != base_commit
            or bool(status.stdout)
        )

    def _repository_status(
        self, *, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        state_relative = self.state_path.relative_to(
            self._require_repository()
        ).as_posix()
        return _git(
            self._require_repository(),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
            "--",
            ".",
            f":(exclude){state_relative}",
            check=check,
        )

    def _empty_changes(
        self, session: WorktreeSession, *, partial: bool
    ) -> dict[str, object]:
        worktree = session.worktree
        return {
            "worktree_id": worktree.id,
            "branch": worktree.branch,
            "base_commit": worktree.base_commit,
            "parent_changed": True,
            "partial": partial,
            "files": [],
            "patch": "",
        }

    def _workspace_path(self, session: WorktreeSession) -> Path:
        return session.worktree.path / self.workspace_relative

    def workspace_path(self, session: WorktreeSession) -> Path:
        with self._lock:
            return self._workspace_path(session)

    def read_only_paths(self, session: WorktreeSession) -> tuple[Path, ...]:
        with self._lock:
            paths = []
            for injected in session.injected.values():
                destination = self._workspace_path(session) / injected.path
                paths.append(
                    destination if injected.kind == "copy" else destination.resolve()
                )
            return tuple(paths)

    def _registered(self) -> dict[Path, tuple[str, str]]:
        output = _git(
            self._require_repository(), "worktree", "list", "--porcelain"
        ).stdout
        registered: dict[Path, tuple[str, str]] = {}
        current: dict[str, str] = {}
        for line in [*output.splitlines(), ""]:
            if not line:
                if {"worktree", "HEAD", "branch"} <= current.keys():
                    registered[Path(current["worktree"]).resolve()] = (
                        current["HEAD"],
                        current["branch"],
                    )
                current = {}
                continue
            key, _, value = line.partition(" ")
            current[key] = value
        return registered

    def _validate_recovered(self, session: WorktreeSession) -> str | None:
        worktree = session.worktree
        try:
            _secure_directory(self.root, self.home, create=False)
        except RuntimeError as exc:
            return f"Managed worktree root is unsafe: {exc}"
        if worktree.path != (self.root / worktree.id).resolve():
            return f"Managed worktree '{worktree.id}' has an unsafe path."
        if self._branch_exists(worktree.branch):
            tip = _git(
                self._require_repository(),
                "rev-parse",
                "--verify",
                worktree.branch,
            ).stdout.strip()
            if tip != worktree.base_commit:
                return f"Managed worktree '{worktree.id}' branch contains external commits."
        registered = self._registered().get(worktree.path)
        if registered is None and not worktree.path.exists():
            return None
        if registered != (worktree.base_commit, f"refs/heads/{worktree.branch}"):
            return f"Managed worktree '{worktree.id}' has conflicting Git metadata."
        if not worktree.path.is_dir():
            return f"Managed worktree '{worktree.id}' path is not a directory."
        return None

    def _rollback_create(self, session: WorktreeSession) -> None:
        repository = self._require_repository()
        _git(
            repository,
            "worktree",
            "remove",
            "--force",
            str(session.worktree.path),
            check=False,
        )
        if self._branch_exists(session.worktree.branch):
            _git(
                repository,
                "branch",
                "-D",
                "--",
                session.worktree.branch,
                check=False,
            )
        shutil.rmtree(session.worktree.path, ignore_errors=True)
        self._sessions.pop(session.worktree.id, None)
        try:
            self._persist()
        except RuntimeError as exc:
            self._warnings.append(str(exc))

    def _branch_exists(self, branch: str) -> bool:
        return (
            _git(
                self._require_repository(),
                "show-ref",
                "--verify",
                "--quiet",
                f"refs/heads/{branch}",
                check=False,
            ).returncode
            == 0
        )

    def _require_repository(self) -> Path:
        if self.repository is None:
            raise RuntimeError("Isolated fork subagents require a Git repository.")
        return self.repository

    def _require_session(self, worktree_id: str) -> WorktreeSession:
        try:
            return self._sessions[worktree_id]
        except KeyError as exc:
            raise RuntimeError(f"Unknown worktree '{worktree_id}'.") from exc

    def _persist(self) -> None:
        payload = {
            "version": 1,
            "repository": str(self.repository) if self.repository is not None else None,
            "worktrees": [
                _session_json(self._sessions[key]) for key in sorted(self._sessions)
            ],
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        if len(encoded.encode("utf-8")) > MAX_STATE_BYTES:
            raise RuntimeError("Worktree state exceeds the 256 KiB limit.")
        _secure_directory(self.state_path.parent, self.workspace, create=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".worktrees-", dir=self.state_path.parent
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.state_path)
            directory = os.open(self.state_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            Path(temporary).unlink(missing_ok=True)
            raise

    def _parse_state(self, value: Any) -> dict[str, WorktreeSession]:
        if (
            not isinstance(value, dict)
            or set(value) != {"version", "repository", "worktrees"}
            or not _integer(value["version"])
            or value["version"] != 1
            or not isinstance(value["worktrees"], list)
            or len(value["worktrees"]) > MAX_WORKTREES
        ):
            raise ValueError("invalid top-level fields")
        expected_repository = (
            str(self.repository) if self.repository is not None else None
        )
        if value["repository"] != expected_repository:
            raise ValueError("repository path does not match")
        sessions: dict[str, WorktreeSession] = {}
        for raw in value["worktrees"]:
            session = _session_from_json(raw, self.root)
            if session.worktree.id in sessions:
                raise ValueError(f"duplicate worktree id '{session.worktree.id}'")
            sessions[session.worktree.id] = session
        return sessions


def _session_json(session: WorktreeSession) -> dict[str, object]:
    worktree = session.worktree
    return {
        "id": worktree.id,
        "name": worktree.name,
        "path": str(worktree.path),
        "branch": worktree.branch,
        "base_commit": worktree.base_commit,
        "created_at": worktree.created_at,
        "owner_session_id": session.owner_session_id,
        "active_task_id": session.active_task_id,
        "entered_at": session.entered_at,
        "last_used_at": session.last_used_at,
        "status": session.state,
        "injected": [
            {
                "path": item.path,
                "source": item.source,
                "kind": item.kind,
                "digest": item.digest,
            }
            for item in sorted(session.injected.values(), key=lambda item: item.path)
        ],
    }


def _session_from_json(value: Any, root: Path) -> WorktreeSession:
    fields = {
        "id",
        "name",
        "path",
        "branch",
        "base_commit",
        "created_at",
        "owner_session_id",
        "active_task_id",
        "entered_at",
        "last_used_at",
        "status",
        "injected",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("invalid worktree fields")
    for name in ("id", "name", "path", "branch", "base_commit", "owner_session_id"):
        if not isinstance(value[name], str) or not value[name]:
            raise ValueError(f"invalid {name}")
    if (
        len(value["name"]) > MAX_SUBAGENT_SLUG_LENGTH
        or SUBAGENT_SLUG_RE.fullmatch(value["name"]) is None
    ):
        raise ValueError("invalid name")
    if len(value["base_commit"]) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in value["base_commit"]
    ):
        raise ValueError("invalid base_commit")
    if value["id"] != _worktree_id(value["owner_session_id"], value["name"]):
        raise ValueError("worktree id does not match owner and name")
    if value["branch"] != f"worktree-{value['id']}":
        raise ValueError("worktree branch does not match id")
    path = Path(value["path"])
    if not path.is_absolute() or path.resolve() != (root / value["id"]).resolve():
        raise ValueError("unsafe worktree path")
    if not _integer(value["created_at"]) or value["created_at"] < 0:
        raise ValueError("invalid created_at")
    for name in ("entered_at", "last_used_at"):
        if value[name] is not None and (not _integer(value[name]) or value[name] < 0):
            raise ValueError(f"invalid {name}")
    if value["active_task_id"] is not None and not isinstance(
        value["active_task_id"], str
    ):
        raise ValueError("invalid active_task_id")
    if value["status"] not in {"creating", "idle", "active"}:
        raise ValueError("invalid status")
    if (value["status"] == "active") != (value["active_task_id"] is not None):
        raise ValueError("status does not match active_task_id")
    if not isinstance(value["injected"], list):
        raise ValueError("invalid injected manifest")
    injected: dict[str, InjectedFile] = {}
    for raw in value["injected"]:
        if not isinstance(raw, dict) or set(raw) != {
            "path",
            "source",
            "kind",
            "digest",
        }:
            raise ValueError("invalid injected entry")
        relative = _validate_dependency_path(raw["path"])
        source = _validate_dependency_path(raw["source"])
        if relative in injected or source != relative:
            raise ValueError("invalid injected path")
        if raw["kind"] not in {"copy", "symlink"}:
            raise ValueError("invalid injected kind")
        digest = raw["digest"]
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("invalid injected digest")
        if (raw["kind"] == "copy") != (digest is not None):
            raise ValueError("injected digest does not match kind")
        injected[relative] = InjectedFile(relative, source, raw["kind"], digest)
    worktree = Worktree(
        value["id"],
        value["name"],
        path.resolve(),
        value["branch"],
        value["base_commit"],
        value["created_at"],
    )
    return WorktreeSession(
        worktree,
        value["owner_session_id"],
        value["active_task_id"],
        value["entered_at"],
        value["last_used_at"],
        injected,
        state=value["status"],
    )


def _worktree_id(owner_session_id: str, name: str) -> str:
    digest = hashlib.sha256(owner_session_id.encode("utf-8")).hexdigest()[:8]
    return f"{name}-{digest}"


def _validate_dependency_path(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_DEPENDENCY_LENGTH:
        raise ValueError("dependency paths must be strings of 1 to 512 characters")
    if "\\" in value or "\0" in value:
        raise ValueError("dependency paths must not contain backslashes or NUL")
    segments = value.split("/")
    path = PurePosixPath(value)
    if path.is_absolute() or any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError("dependency paths must be safe relative workspace paths")
    return path.as_posix()


def _read_regular_file(path: Path, root: Path, limit: int) -> str | None:
    current = root
    info = None
    for part in path.relative_to(root).parts:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError("symbolic links are not allowed")
    assert info is not None
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError("path is not a regular file")
    if info.st_size > limit:
        raise RuntimeError("file exceeds the 256 KiB limit")
    data = path.read_bytes()
    if len(data) > limit:
        raise RuntimeError("file exceeds the 256 KiB limit")
    return data.decode("utf-8")


def _secure_directory(path: Path, root: Path, *, create: bool) -> None:
    try:
        root_info = root.lstat()
    except FileNotFoundError:
        if not create:
            raise RuntimeError(f"directory '{root}' is missing")
        root.mkdir(parents=True, mode=0o700)
    else:
        if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
            raise RuntimeError(f"'{root}' is not a safe directory")
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not create:
                raise RuntimeError(f"directory '{current}' is missing")
            current.mkdir(mode=0o700)
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"'{current}' is not a safe directory")
    path.chmod(0o700)


def _matches_injected(path: Path, injected: InjectedFile, workspace: Path) -> bool:
    if injected.kind == "symlink":
        return (
            path.is_symlink()
            and path.resolve() == (workspace / injected.source).resolve()
        )
    return (
        path.is_file()
        and not path.is_symlink()
        and injected.digest is not None
        and _sha256_file(path) == injected.digest
    )


def _safe_injection_parent(root: Path, destination: Path) -> bool:
    current = root
    try:
        relative = destination.relative_to(root)
    except ValueError:
        return False
    for part in relative.parts[:-1]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            return False
    return True


def _remove_injected(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _copy_atomic(source: Path, destination: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=".worktree-copy-", dir=destination.parent
    )
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary)
        os.chmod(temporary, stat.S_IMODE(source.stat().st_mode) & ~0o222)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(128 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _git(
    workspace: Path,
    *arguments: str,
    check: bool = True,
    disable_hooks: bool = True,
) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        result = subprocess.run(
            [
                "git",
                *(("-c", "core.hooksPath=/dev/null") if disable_hooks else ()),
                "-C",
                str(workspace),
                *arguments,
            ],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            env=environment,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Git is required for isolated fork subagents") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Git command timed out") from exc
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "Git command failed"
        raise RuntimeError(detail)
    return result
