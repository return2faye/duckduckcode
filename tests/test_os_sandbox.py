from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from duckduckcode.tools.os_sandbox import OSSandbox, _seccomp_program


class OSSandboxTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_directory = tempfile.TemporaryDirectory()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace_directory.cleanup)
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace = Path(self.workspace_directory.name)
        self.private_temp = Path(self.temporary_directory.name)

    def test_disabled_sandbox_leaves_full_access_command_unchanged(self) -> None:
        with patch(
            "duckduckcode.tools.os_sandbox.platform.system", return_value="Plan9"
        ):
            sandbox = OSSandbox(self.workspace, self.private_temp, lambda: False)

        self.assertIsNone(sandbox.prepare("pwd", False))

    def test_darwin_uses_seatbelt_and_denies_network_by_default(self) -> None:
        with (
            patch(
                "duckduckcode.tools.os_sandbox.platform.system",
                return_value="Darwin",
            ),
            patch(
                "duckduckcode.tools.os_sandbox.Path.is_file",
                return_value=True,
            ),
        ):
            sandbox = OSSandbox(self.workspace, self.private_temp, lambda: True)
            offline = sandbox.prepare("uv sync", False)
            online = sandbox.prepare("uv sync", True)

        self.assertEqual(offline.argv[0], "/usr/bin/sandbox-exec")
        self.assertIn("(deny network-outbound)", offline.argv[2])
        self.assertNotIn("(deny network-outbound)", online.argv[2])
        self.assertEqual(offline.argv[-3:], ["/bin/sh", "-c", "uv sync"])
        self.assertEqual(
            offline.environment["UV_CACHE_DIR"],
            str(sandbox.temporary_directory / "cache" / "uv"),
        )

    def test_darwin_denies_writes_to_injected_dependencies(self) -> None:
        dependency = self.workspace / ".env"
        with (
            patch(
                "duckduckcode.tools.os_sandbox.platform.system", return_value="Darwin"
            ),
            patch("duckduckcode.tools.os_sandbox.Path.is_file", return_value=True),
        ):
            sandbox = OSSandbox(
                self.workspace,
                self.private_temp,
                lambda: True,
                (dependency,),
            )
            command = sandbox.prepare("true", False)

        self.assertIn('param "READONLY0"', command.argv[2])
        self.assertIn(f"READONLY0={dependency.resolve()}", command.argv)

    def test_linux_uses_bubblewrap_namespaces_and_seccomp(self) -> None:
        with (
            patch(
                "duckduckcode.tools.os_sandbox.platform.system",
                return_value="Linux",
            ),
            patch(
                "duckduckcode.tools.os_sandbox.platform.machine",
                return_value="x86_64",
            ),
            patch(
                "duckduckcode.tools.os_sandbox.Path.is_file",
                return_value=True,
            ),
        ):
            sandbox = OSSandbox(self.workspace, self.private_temp, lambda: True)
            offline = sandbox.prepare("uv sync", False)
            online = sandbox.prepare("uv sync", True)

        try:
            self.assertEqual(offline.argv[0], "/usr/bin/bwrap")
            self.assertNotIn("--unshare-all", offline.argv)
            self.assertNotIn("--unshare-pid", offline.argv)
            self.assertIn("--unshare-user", offline.argv)
            self.assertIn("--unshare-net", offline.argv)
            self.assertIn("--seccomp", offline.argv)
            self.assertGreater(len(os.read(offline.pass_fds[0], 4096)), 0)
            self.assertNotIn("--unshare-net", online.argv)
        finally:
            offline.close()
            online.close()

    def test_linux_read_only_binds_injected_dependencies(self) -> None:
        dependency = self.workspace / "node_modules"
        with (
            patch(
                "duckduckcode.tools.os_sandbox.platform.system", return_value="Linux"
            ),
            patch(
                "duckduckcode.tools.os_sandbox.platform.machine", return_value="x86_64"
            ),
            patch("duckduckcode.tools.os_sandbox.Path.is_file", return_value=True),
        ):
            sandbox = OSSandbox(
                self.workspace,
                self.private_temp,
                lambda: True,
                (dependency,),
            )
            command = sandbox.prepare("true", False)

        try:
            resolved = str(dependency.resolve())
            index = command.argv.index(resolved)
            self.assertEqual(command.argv[index - 1], "--ro-bind-try")
            self.assertEqual(command.argv[index + 1], resolved)
        finally:
            command.close()

    def test_rejects_unknown_seccomp_architecture(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not supported"):
            _seccomp_program("mips")


if __name__ == "__main__":
    unittest.main()
