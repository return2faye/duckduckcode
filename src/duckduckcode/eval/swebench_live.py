from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..config import Config
from ..main import build_agent
from .runner import _run_turn

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_COMMIT = re.compile(r"[0-9a-fA-F]{7,64}")
OFFICIAL_DATASET = "SWE-bench-Live/SWE-bench-Live"
OFFICIAL_SPLIT = "full"
_DATASET_API = "https://huggingface.co/api/datasets"
_ROWS_API = "https://datasets-server.huggingface.co/rows"
_PAGE_SIZE = 100


def load_instances(path: Path) -> list[dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read dataset '{path}': {exc}") from exc
    try:
        raw = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if path.suffix == ".jsonl"
            else json.loads(text)
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Dataset '{path}' is invalid JSON: {exc}") from exc
    records = raw if isinstance(raw, list) else [raw]
    instances = [_validate(record, index) for index, record in enumerate(records, 1)]
    ids = [instance["instance_id"] for instance in instances]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Dataset contains duplicate instance_id values")
    return instances


def load_official_instances(
    dataset: str = OFFICIAL_DATASET, split: str = OFFICIAL_SPLIT
) -> tuple[list[dict[str, str]], str]:
    if dataset != OFFICIAL_DATASET:
        raise RuntimeError(f"Unsupported dataset: {dataset}")
    if split not in {"full", "lite", "test", "verified"}:
        raise RuntimeError(f"Unsupported {OFFICIAL_DATASET} split: {split}")
    metadata = _get_json(f"{_DATASET_API}/{dataset}")
    revision = metadata.get("sha") if isinstance(metadata, dict) else None
    if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
        raise RuntimeError("Official dataset did not report a valid revision")

    records: list[Any] = []
    total: int | None = None
    while total is None or len(records) < total:
        query = urlencode(
            {
                "dataset": dataset,
                "config": "default",
                "split": split,
                "offset": len(records),
                "length": _PAGE_SIZE,
                "revision": revision,
            }
        )
        page = _get_json(f"{_ROWS_API}?{query}")
        if not isinstance(page, dict) or page.get("partial") is True:
            raise RuntimeError("Official dataset server returned a partial response")
        page_total = page.get("num_rows_total")
        rows = page.get("rows")
        if (
            not isinstance(page_total, int)
            or page_total < 1
            or not isinstance(rows, list)
        ):
            raise RuntimeError("Official dataset server returned an invalid page")
        if total is not None and page_total != total:
            raise RuntimeError("Official dataset changed while it was being downloaded")
        total = page_total
        page_records = [row.get("row") for row in rows if isinstance(row, dict)]
        if len(page_records) != len(rows) or not page_records:
            raise RuntimeError("Official dataset server returned an incomplete page")
        records.extend(page_records)
        if len(records) > total:
            raise RuntimeError("Official dataset server returned too many rows")

    instances: list[dict[str, str]] = []
    by_id: dict[str, dict[str, str]] = {}
    raw_by_id: dict[str, Any] = {}
    for index, record in enumerate(records, 1):
        instance = _validate(record, index)
        previous = by_id.get(instance["instance_id"])
        if previous is not None:
            if previous != instance or raw_by_id[instance["instance_id"]] != record:
                raise RuntimeError(
                    "Official dataset contains conflicting duplicate instance_id "
                    f"{instance['instance_id']}"
                )
            continue
        by_id[instance["instance_id"]] = instance
        raw_by_id[instance["instance_id"]] = record
        instances.append(instance)
    return instances, revision


def run_inference(
    config: Config,
    dataset: Path | str,
    repository_cache: Path,
    predictions_path: Path,
    instance_ids: set[str] | None = None,
    *,
    clone_missing: bool = False,
    overwrite: bool = False,
    max_iterations: int = 50,
    split: str = OFFICIAL_SPLIT,
) -> int:
    if isinstance(dataset, Path):
        instances = load_instances(dataset)
        dataset_name = str(dataset)
        dataset_revision = None
    else:
        instances, dataset_revision = load_official_instances(dataset, split)
        dataset_name = dataset
    if instance_ids:
        known = {instance["instance_id"] for instance in instances}
        missing = sorted(instance_ids - known)
        if missing:
            raise RuntimeError(f"Unknown instance(s): {', '.join(missing)}")
        instances = [
            instance
            for instance in instances
            if instance["instance_id"] in instance_ids
        ]
    predictions = _load_predictions(predictions_path)
    repository_cache.mkdir(parents=True, exist_ok=True)
    repository_cache = repository_cache.resolve()
    failures = 0
    for instance in instances:
        instance_id = instance["instance_id"]
        if instance_id in predictions and not overwrite:
            print(f"{instance_id}: SKIP existing prediction")
            continue
        errors: list[str] = []
        completed = False
        patch = ""
        answer = ""
        tool_events: list[dict[str, Any]] = []
        token_usage = 0
        compactions = 0
        try:
            repository = _repository(instance, repository_cache, clone_missing)
            with tempfile.TemporaryDirectory(prefix="duckduckcode-swebench-") as root:
                workspace = Path(root) / "repo"
                _checkout(repository, workspace, instance["base_commit"])
                agent = build_agent(
                    config,
                    workspace,
                    max_iterations=max_iterations,
                    include_user_instructions=False,
                    enable_sessions=False,
                    enable_memory=False,
                    enable_skills=False,
                    enable_subagents=False,
                    enable_mcp=False,
                    enable_lsp=False,
                )
                try:
                    agent.set_permission_mode("accept_edits")
                    answer, completed, token_usage, compactions, errors = _run_turn(
                        agent, instance["problem_statement"], 1, tool_events, []
                    )
                finally:
                    agent.close()
                patch = _git_patch(workspace)
        except Exception as exc:
            errors.append(str(exc))
        predictions[instance_id] = {
            "model_patch": patch,
            "model_name_or_path": config.agent.model,
            "dataset": dataset_name,
            "dataset_split": split if dataset_revision else None,
            "dataset_revision": dataset_revision,
            "completed": completed,
            "errors": errors,
            "final_answer": answer,
            "tool_events": tool_events,
            "token_usage": token_usage,
            "compactions": compactions,
        }
        _write_json(predictions_path, predictions)
        if completed:
            print(f"{instance_id}: COMPLETE patch_bytes={len(patch.encode())}")
        else:
            failures += 1
            print(f"{instance_id}: ERROR {'; '.join(errors) or 'agent incomplete'}")
    return 1 if failures else 0


def _get_json(url: str) -> Any:
    try:
        with urlopen(
            Request(url, headers={"User-Agent": "duckduckcode-swebench-live"}),
            timeout=60,
        ) as response:
            return json.load(response)
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not download official dataset: {exc}") from exc


def _validate(raw: Any, index: int) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise RuntimeError(f"Dataset record {index} must be an object")
    result: dict[str, str] = {}
    for field in ("instance_id", "repo", "base_commit", "problem_statement"):
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"Dataset record {index}.{field} must be non-empty")
        result[field] = value.strip()
    if not _REPO.fullmatch(result["repo"]) or any(
        part in {".", ".."} for part in result["repo"].split("/")
    ):
        raise RuntimeError(f"Dataset record {index}.repo is unsafe")
    if not _COMMIT.fullmatch(result["base_commit"]):
        raise RuntimeError(f"Dataset record {index}.base_commit is not a Git hash")
    return result


def _repository(instance: dict[str, str], cache: Path, clone_missing: bool) -> Path:
    repository = cache.joinpath(*instance["repo"].split("/"))
    repository.parent.mkdir(parents=True, exist_ok=True)
    if not repository.parent.resolve().is_relative_to(cache):
        raise RuntimeError(
            f"Repository cache path escapes through a symlink: {repository}"
        )
    if repository.exists() and not repository.resolve().is_relative_to(cache):
        raise RuntimeError(
            f"Repository cache path escapes through a symlink: {repository}"
        )
    if not repository.exists():
        if not clone_missing:
            raise RuntimeError(
                f"Repository '{instance['repo']}' is missing from {cache}; "
                "rerun with --clone-missing"
            )
        _git(
            "clone",
            "--no-checkout",
            f"https://github.com/{instance['repo']}.git",
            str(repository),
        )
    try:
        _git("-C", str(repository), "rev-parse", "--is-inside-work-tree")
    except RuntimeError as exc:
        raise RuntimeError(f"Repository cache is not Git: {repository}") from exc
    if _git_config(repository, "remote.origin.promisor") == "true":
        if not clone_missing:
            raise RuntimeError(
                f"Repository cache is partial: {repository}; rerun with "
                "--clone-missing to complete it"
            )
        _git(
            "-C",
            str(repository),
            "fetch",
            "--refetch",
            "--no-filter",
            "--no-tags",
            "origin",
        )
        _git("-C", str(repository), "config", "--unset-all", "remote.origin.promisor")
        _git(
            "-C",
            str(repository),
            "config",
            "--unset-all",
            "remote.origin.partialclonefilter",
        )
    commit = instance["base_commit"]
    try:
        _git("-C", str(repository), "rev-parse", "--verify", f"{commit}^{{commit}}")
    except RuntimeError:
        if not clone_missing:
            raise RuntimeError(
                f"Commit {commit} is missing from cached repository {repository}"
            )
        _git("-C", str(repository), "fetch", "--no-tags", "origin", commit)
    return repository


def _checkout(repository: Path, workspace: Path, commit: str) -> None:
    _git("clone", "--no-checkout", str(repository), str(workspace))
    _git("-C", str(workspace), "checkout", "--detach", commit)
    if _git("-C", str(workspace), "status", "--porcelain"):
        raise RuntimeError("Git checkout is incomplete or dirty")


def _git_patch(workspace: Path) -> str:
    _git("-C", str(workspace), "add", "--intent-to-add", "--all")
    return _git(
        "-C",
        str(workspace),
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "--no-textconv",
        "HEAD",
        "--",
        ".",
        ":(exclude).duckduckcode/**",
    )


def _git(*args: str) -> str:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(
        ["git", *args],
        env=environment,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(
            result.stderr.strip() or result.stdout.strip() or "Git failed"
        )
    return result.stdout


def _git_config(repository: Path, name: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "config", "--get", name],
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read predictions '{path}': {exc}") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError("Predictions file must contain an object")
    return loaded


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SWE-bench Live prediction patches with DuckDuckCode."
    )
    parser.add_argument(
        "--dataset",
        default=OFFICIAL_DATASET,
        help=f"Official dataset ID (default: {OFFICIAL_DATASET}).",
    )
    parser.add_argument(
        "--split",
        default=OFFICIAL_SPLIT,
        choices=("full", "lite", "test", "verified"),
        help="Official dataset split; full contains every published Python task.",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        help="Optional offline JSON/JSONL export of the official dataset.",
    )
    parser.add_argument("--repository-cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--instance", action="append", dest="instance_ids")
    parser.add_argument("--clone-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.max_iterations <= 100:
        parser.error("--max-iterations must be between 1 and 100")
    try:
        code = run_inference(
            Config.from_env(),
            args.dataset_file or args.dataset,
            args.repository_cache,
            args.predictions,
            set(args.instance_ids or ()),
            clone_missing=args.clone_missing,
            overwrite=args.overwrite,
            max_iterations=args.max_iterations,
            split=args.split,
        )
    except RuntimeError as exc:
        parser.exit(2, f"duckduckcode-swebench-live: error: {exc}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
