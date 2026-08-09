from __future__ import annotations

from collections.abc import Callable
import errno
import os
from dataclasses import dataclass
from pathlib import Path
import platform
import struct

_SEATBELT = """\
(version 1)
(allow default)
(deny file-write*)
(allow file-write*
  (subpath (param "WORKSPACE"))
  (subpath (param "TEMP")))
(deny file-write*
  (literal (param "GIT"))
  (subpath (param "GIT"))
  (literal (param "DUCKDUCKCODE"))
  (subpath (param "DUCKDUCKCODE")))
"""
_NETWORK_DENY = """\
(deny network-outbound)
(deny network-inbound)
(deny system-socket)
"""
_AUDIT_ARCH = {
    "x86_64": 0xC000003E,
    "aarch64": 0xC00000B7,
    "arm64": 0xC00000B7,
}
# ponytail: deny high-risk syscalls; use a libseccomp allowlist if stronger
# syscall confinement becomes worth its compatibility cost.
_BLOCKED_SYSCALLS = {
    "x86_64": (
        101,  # ptrace
        155,  # pivot_root
        165,  # mount
        166,  # umount2
        167,  # swapon
        168,  # swapoff
        169,  # reboot
        175,  # init_module
        176,  # delete_module
        246,  # kexec_load
        248,  # add_key
        249,  # request_key
        250,  # keyctl
        272,  # unshare
        298,  # perf_event_open
        304,  # open_by_handle_at
        308,  # setns
        311,  # process_vm_writev
        313,  # finit_module
        321,  # bpf
        323,  # userfaultfd
    ),
    "aarch64": (
        39,  # umount2
        40,  # mount
        41,  # pivot_root
        97,  # unshare
        104,  # kexec_load
        105,  # init_module
        106,  # delete_module
        117,  # ptrace
        142,  # reboot
        217,  # add_key
        218,  # request_key
        219,  # keyctl
        224,  # swapon
        225,  # swapoff
        241,  # perf_event_open
        265,  # open_by_handle_at
        268,  # setns
        271,  # process_vm_writev
        273,  # finit_module
        280,  # bpf
        282,  # userfaultfd
    ),
}


@dataclass(frozen=True)
class SandboxedCommand:
    argv: list[str]
    pass_fds: tuple[int, ...] = ()
    environment: dict[str, str] | None = None

    def close(self) -> None:
        for descriptor in self.pass_fds:
            os.close(descriptor)


class OSSandbox:
    def __init__(
        self,
        workspace: Path,
        temporary_directory: Path,
        enabled: Callable[[], bool],
        read_only_paths: tuple[Path, ...] = (),
    ) -> None:
        self.workspace = workspace.resolve()
        self.temporary_directory = temporary_directory.resolve()
        self.enabled = enabled
        self.read_only_paths = tuple(path.resolve() for path in read_only_paths)
        self.system = platform.system().lower()

    def prepare(self, command: str, network_access: bool) -> SandboxedCommand | None:
        if not self.enabled():
            return None
        if self.system == "darwin":
            return self._darwin(command, network_access)
        if self.system == "linux":
            return self._linux(command, network_access)
        raise RuntimeError(f"OS sandbox is not supported on {platform.system()}.")

    def _darwin(self, command: str, network_access: bool) -> SandboxedCommand:
        executable = Path("/usr/bin/sandbox-exec")
        if not executable.is_file():
            raise RuntimeError("Seatbelt requires /usr/bin/sandbox-exec.")
        read_only = "".join(
            f'(deny file-write* (literal (param "READONLY{index}")) '
            f'(subpath (param "READONLY{index}")))\n'
            for index in range(len(self.read_only_paths))
        )
        profile = _SEATBELT + read_only + ("" if network_access else _NETWORK_DENY)
        argv = [
            str(executable),
            "-p",
            profile,
            "-D",
            f"WORKSPACE={self.workspace}",
            "-D",
            f"TEMP={self.temporary_directory}",
            "-D",
            f"GIT={self.workspace / '.git'}",
            "-D",
            f"DUCKDUCKCODE={self.workspace / '.duckduckcode'}",
        ]
        for index, path in enumerate(self.read_only_paths):
            argv.extend(["-D", f"READONLY{index}={path}"])
        argv.extend(
            [
                "--",
                "/bin/sh",
                "-c",
                command,
            ]
        )
        return SandboxedCommand(
            argv,
            environment=_sandbox_environment(self.temporary_directory),
        )

    def _linux(self, command: str, network_access: bool) -> SandboxedCommand:
        executable = next(
            (
                candidate
                for candidate in (Path("/usr/bin/bwrap"), Path("/bin/bwrap"))
                if candidate.is_file()
            ),
            None,
        )
        if executable is None:
            raise RuntimeError(
                "Linux OS sandbox requires Bubblewrap at /usr/bin/bwrap."
            )
        seccomp_fd = _seccomp_fd()
        argv = [
            str(executable),
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup",
        ]
        if not network_access:
            argv.append("--unshare-net")
        argv.extend(
            [
                "--ro-bind",
                "/",
                "/",
                "--bind",
                str(self.workspace),
                str(self.workspace),
                "--bind",
                str(self.temporary_directory),
                str(self.temporary_directory),
                "--ro-bind-try",
                str(self.workspace / ".git"),
                str(self.workspace / ".git"),
                "--ro-bind-try",
                str(self.workspace / ".duckduckcode"),
                str(self.workspace / ".duckduckcode"),
            ]
        )
        for path in self.read_only_paths:
            argv.extend(["--ro-bind-try", str(path), str(path)])
        argv.extend(
            [
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--chdir",
                str(self.workspace),
                "--seccomp",
                str(seccomp_fd),
                "--",
                "/bin/sh",
                "-c",
                command,
            ]
        )
        return SandboxedCommand(
            argv,
            (seccomp_fd,),
            _sandbox_environment(self.temporary_directory),
        )


def _sandbox_environment(temporary_directory: Path) -> dict[str, str]:
    cache = temporary_directory / "cache"
    return {
        "TMPDIR": str(temporary_directory),
        "XDG_CACHE_HOME": str(cache),
        "UV_CACHE_DIR": str(cache / "uv"),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "npm_config_cache": str(cache / "npm"),
    }


def _seccomp_fd() -> int:
    read_descriptor, write_descriptor = os.pipe()
    try:
        program = _seccomp_program(platform.machine().lower())
        while program:
            written = os.write(write_descriptor, program)
            program = program[written:]
    except BaseException:
        os.close(read_descriptor)
        raise
    finally:
        os.close(write_descriptor)
    return read_descriptor


def _seccomp_program(machine: str) -> bytes:
    machine = "aarch64" if machine == "arm64" else machine
    try:
        audit_arch = _AUDIT_ARCH[machine]
        blocked = _BLOCKED_SYSCALLS[machine]
    except KeyError as exc:
        raise RuntimeError(
            f"Linux seccomp is not supported on architecture '{machine}'."
        ) from exc

    load_word = 0x20
    jump_equal = 0x15
    return_value = 0x06
    kill_process = 0x80000000
    return_errno = 0x00050000 | errno.EPERM
    allow = 0x7FFF0000
    instructions = [
        (load_word, 0, 0, 4),
        (jump_equal, 1, 0, audit_arch),
        (return_value, 0, 0, kill_process),
        (load_word, 0, 0, 0),
    ]
    for syscall in blocked:
        instructions.extend(
            [
                (jump_equal, 0, 1, syscall),
                (return_value, 0, 0, return_errno),
            ]
        )
    instructions.append((return_value, 0, 0, allow))
    return b"".join(struct.pack("=HBBI", *instruction) for instruction in instructions)
