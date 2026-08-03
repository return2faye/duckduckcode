from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path
import tempfile
import unittest

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "duckduckcode"

EXPECTED_MODULES = {
    "__init__.py",
    "config.py",
    "eval/__init__.py",
    "eval/__main__.py",
    "eval/runner.py",
    "eval/report.py",
    "eval/schema.py",
    "main.py",
    "memory/__init__.py",
    "memory/instruction.py",
    "memory/session.py",
    "core/__init__.py",
    "core/agent.py",
    "core/client.py",
    "core/context.py",
    "core/event.py",
    "interfaces/__init__.py",
    "interfaces/backend.py",
    "interfaces/tui.py",
    "permissions/__init__.py",
    "permissions/bash_blacklist.py",
    "permissions/checker.py",
    "permissions/path_sandbox.py",
    "permissions/rule_policy.py",
    "providers/__init__.py",
    "providers/openai/__init__.py",
    "providers/openai/client.py",
    "providers/openai/serialize.py",
    "providers/openai/stream.py",
    "tools/__init__.py",
    "tools/edit_file.py",
    "tools/glob.py",
    "tools/grep.py",
    "tools/os_sandbox.py",
    "tools/read_file.py",
    "tools/tool.py",
    "tools/write_file.py",
}

LEGACY_MODULES = {
    "agent.py",
    "backend.py",
    "client.py",
    "context.py",
    "event.py",
    "openai_client.py",
    "serialize.py",
    "stream.py",
    "tool.py",
    "tui.py",
}

ALLOWED_DEPENDENCIES = {
    "root": set(),
    "config": {"core"},
    "eval": {"config", "core", "eval", "main", "tools"},
    "main": {
        "config",
        "core",
        "interfaces",
        "memory",
        "permissions",
        "providers",
        "tools",
    },
    "memory": {"core", "memory", "tools"},
    "core": {"core", "permissions", "tools"},
    "interfaces": {"core", "interfaces"},
    "permissions": {"permissions", "tools"},
    "providers": {"core", "providers", "tools"},
    "tools": {"tools"},
}


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SOURCE_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(["duckduckcode", *parts])


def _imports(path: Path, module: str) -> set[str]:
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    imported: set[str] = set()

    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            target = node.module or ""
            if node.level:
                target = resolve_name("." * node.level + target, package)
            names = [
                target,
                *(
                    f"{target}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                ),
            ]
        else:
            continue
        imported.update(name for name in names if name.startswith("duckduckcode"))

    return imported


def _graph() -> dict[str, set[str]]:
    modules = {_module_name(path): path for path in SOURCE_ROOT.rglob("*.py")}
    return {
        module: {imported for imported in _imports(path, module) if imported in modules}
        for module, path in modules.items()
    }


def _area(module: str) -> str:
    parts = module.split(".")
    if len(parts) == 1:
        return "root"
    return parts[1]


class ArchitectureTest(unittest.TestCase):
    def test_tracks_submodules_imported_from_packages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.py"
            path.write_text(
                "from . import client\n"
                "from duckduckcode.providers.openai import stream\n",
                encoding="utf-8",
            )

            imported = _imports(path, "duckduckcode.core.example")

        self.assertIn("duckduckcode.core.client", imported)
        self.assertIn("duckduckcode.providers.openai.stream", imported)

    def test_uses_the_package_layout_without_legacy_modules(self) -> None:
        missing = sorted(
            path for path in EXPECTED_MODULES if not (SOURCE_ROOT / path).is_file()
        )
        remaining = sorted(
            path for path in LEGACY_MODULES if (SOURCE_ROOT / path).exists()
        )

        self.assertEqual(missing, [], f"Missing package modules: {missing}")
        self.assertEqual(remaining, [], f"Legacy modules remain: {remaining}")

    def test_package_dependencies_follow_the_allowed_direction(self) -> None:
        violations = []
        for source, dependencies in _graph().items():
            source_area = _area(source)
            allowed = ALLOWED_DEPENDENCIES.get(source_area, set())
            for dependency in dependencies:
                dependency_area = _area(dependency)
                if dependency_area not in allowed:
                    violations.append(f"{source} -> {dependency}")

        self.assertEqual(
            sorted(violations),
            [],
            f"Forbidden package dependencies: {sorted(violations)}",
        )

    def test_internal_import_graph_has_no_cycles(self) -> None:
        graph = _graph()
        visiting: set[str] = set()
        visited: set[str] = set()
        trail: list[str] = []
        cycles: list[str] = []

        def visit(module: str) -> None:
            if module in visited:
                return
            if module in visiting:
                start = trail.index(module)
                cycles.append(" -> ".join([*trail[start:], module]))
                return

            visiting.add(module)
            trail.append(module)
            for dependency in graph[module]:
                visit(dependency)
            trail.pop()
            visiting.remove(module)
            visited.add(module)

        for module in graph:
            visit(module)

        self.assertEqual(cycles, [], f"Import cycles found: {cycles}")


if __name__ == "__main__":
    unittest.main()
