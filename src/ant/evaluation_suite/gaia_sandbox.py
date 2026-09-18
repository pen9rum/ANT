"""A sandboxed Python execution service for the GAIA substrate.

This module replaces the restricted-AST arithmetic evaluator the first
GAIA pass shipped. It is a genuine arbitrary-code-execution primitive
driven by model output, so the whole of this docstring is a security
writeup rather than a feature summary. **Read it before wiring this to a
live run.**

=====================================================================
WHAT THE ISOLATION ACTUALLY IS
=====================================================================

Two layers, both of which must be described honestly because neither is
a container.

--- Layer 1: OS-level, imposed by this file (the parent) -------------

* **Process separation.** `subprocess.Popen([sys.executable, ...])`.
  A genuinely separate OS process with its own interpreter and its own
  address space. There is no `exec()`/`eval()` of model-chosen source in
  the harness process anywhere in this module. If the child corrupts its
  own interpreter, dies, or leaks, the evaluation process is unaffected.

* **Interpreter flags `-S -s -P`.**
    - `-S` skips `site`, so **site-packages is not importable**: the
      child sees the standard library and nothing else. This is also why
      the sandbox is stdlib-only by design (see LIMITS below).
    - `-s` drops the per-user site directory.
    - `-P` (3.11+) stops the script's own directory being prepended to
      `sys.path`, so **this repository is not importable from inside the
      sandbox** even though the child script physically lives in it.
  `-I` was deliberately NOT used: it implies `-E`, which would discard
  `PYTHONHASHSEED=0` and reintroduce hash randomisation, and the
  environment is already fully controlled (below), so `-E` would buy
  nothing and cost determinism.

* **Scrubbed environment.** The child's `env` is constructed from an
  allowlist (`SYSTEMROOT`/`SystemDrive`/`windir`/`COMSPEC` on Windows,
  `PATH` pinned to the system directory, `TMP`/`TEMP`/`TMPDIR` pointed at
  the scratch directory). **No inherited variable reaches it** -- not
  `HF_TOKEN`, not `OPENAI_API_KEY`, not `PYTHONPATH`. Verified by test.

* **Ephemeral scratch directory as cwd.** A fresh `mkdtemp()` per call,
  used as the child's working directory and as its `TMP`, and removed in
  a `finally` block when the call returns. It is never the repository
  checkout and never a caller-supplied path.

* **Wall-clock timeout** via `Popen.communicate(timeout=...)`, followed
  by a kill of the whole process *tree*: `os.killpg` against a session
  created with `start_new_session=True` on POSIX, `TerminateJobObject`
  on Windows. Killing only the direct child would leave grandchildren
  running; both paths kill the group.

* **Memory limit.**
    - POSIX: `resource.setrlimit(RLIMIT_AS)` (plus `RLIMIT_CPU`,
      `RLIMIT_FSIZE`, `RLIMIT_CORE`), installed by the child before it
      reads a single byte of user code.
    - Windows: a **Job Object** (`CreateJobObjectW` +
      `SetInformationJobObject` with `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`)
      carrying `JOB_OBJECT_LIMIT_PROCESS_MEMORY`,
      `JOB_OBJECT_LIMIT_JOB_MEMORY`, `JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 1`
      and `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, applied through `ctypes`.
      The last flag means that if the harness itself crashes, Windows
      tears the sandboxed process down with it.
    - **Fail closed**: if the limit cannot be installed on either
      platform, the call raises instead of running unbounded.

* **The spawn/limit race is closed.** On Windows a Job Object can only
  be attached after `CreateProcess` returns, which normally leaves a
  window in which the child is alive and unconstrained. Here the child
  blocks on `sys.stdin.read()` and the parent does not write the code
  until after the job is attached, so the child has provably not seen
  the code while unconstrained.

--- Layer 2: interpreter-level, imposed by the child -----------------

Implemented in `gaia_sandbox_child.py` as a **PEP 578 audit hook**
(`sys.addaudithook`). Two properties earn it its place: a hook **cannot
be removed** once installed (CPython exposes no `removeaudithook`), and
the events are raised from inside CPython's own C implementations, so
re-routing in pure Python (`getattr(os, "sys" + "tem")`,
`builtins.__dict__["open"]`, `importlib.import_module`) does not evade
it. The policy:

A **second, weaker layer** sits beside it: a `sys.meta_path` finder that
refuses the same blocked modules. It exists because the `import` audit
event does **not** cover every import route -- CPython raises it from
`builtins.__import__` and the `IMPORT_NAME` opcode, but **not** from
`importlib.import_module`, which drops into the pure-Python
`_bootstrap._gcd_import`. That gap was found by a test, not by reading
documentation: `importlib.import_module("subprocess")` sailed past the
audit hook while `import subprocess` was refused. `sys.meta_path` is an
ordinary list and user code can empty it, so this layer is removable --
which is why the blocked list names the private extension modules
(`_ctypes`, `_socket`, `_winapi`, ...) alongside their public wrappers.
`ctypes/__init__.py` reaches its native half with a plain `from _ctypes
import ...` *statement*, which fires the unremovable audit hook, so the
native-code route stays closed even against an importer that has
defeated every removable layer. Verified against that exact attack.

The policy:

* **Filesystem**: `open` and the `os.*`/`shutil.*` path events are
  bounds-checked. Read+write is allowed **only** inside the ephemeral
  scratch directory. Read-only is additionally allowed under
  `sys.base_prefix`/`sys.prefix` -- that is what makes `import json`
  work. Everything else, including this repository, the user's home
  directory, and `.env`, is refused with `SandboxDenied`.
* **Network**: `socket.*` audit events are denied, and `socket`, `ssl`,
  `http`, `urllib`, `requests`, `httpx`, `ftplib`, `smtplib` (etc.) are
  denied at import.
* **Process creation**: `subprocess.*`, `os.system`, `os.exec*`,
  `os.spawn*`, `os.fork*`, `os.startfile` denied as events, and
  `subprocess`/`multiprocessing`/`pty` denied at import.
* **Native code**: `ctypes` denied at import and `ctypes.*` denied as
  events, because `ctypes` is the one stdlib module that converts Python
  into arbitrary machine code -- exactly the thing an audit hook cannot
  see past.
* **Environment mutation**: `os.putenv`/`os.unsetenv`/`os.chdir` denied.

=====================================================================
WHAT THIS DOES **NOT** PROTECT AGAINST -- read this part twice
=====================================================================

I am not fully confident this isolation is robust against a determined
adversary, and this list is written so that nobody has to take my word
for it either way.

1. **This is not a container, a VM, a seccomp filter, or a jail.** There
   is no kernel-enforced syscall restriction, no namespace, no chroot.
   The child runs as the same OS user with the same privileges as the
   harness. If the interpreter-level guard is bypassed, nothing below it
   stops a syscall.

2. **The network guard is interpreter-level only.** There is no
   firewall rule and no network namespace. It rests entirely on the
   import denial plus `socket.*` audit events. Any path to raw syscalls
   -- a CPython bug, an unaudited C extension, a native-code trick I did
   not anticipate -- bypasses it completely. If genuine network
   isolation matters, run this inside a container with networking
   disabled; that is the correct fix and this module is not it.

3. **Audit-hook coverage is only as complete as CPython's event list,
   and I already found one gap the hard way.** `importlib.import_module`
   does not raise the `import` event (see the second-layer note above).
   That one is closed; I cannot prove there is not another. Any
   operation that raises no audit event is invisible to this guard, and
   the event list drifts between CPython releases.

4. **A CPython escape is a real, unpatched-by-me risk.** Audit hooks
   have had bypasses historically. The native-code route is closed at
   three independent points (`ctypes` and `_ctypes` at import, plus the
   `ctypes.*` events) and I attacked it deliberately -- popping the
   meta_path finder, clearing `sys.meta_path` entirely, and reaching
   `ctypes`/`_ctypes` through `importlib.import_module` -- and every
   attempt was refused. **That is evidence, not proof.** Those are the
   attacks I thought of. Treat "an adversarial model escapes this
   sandbox" as possible, not impossible.

   Note one specific thing I did NOT verify: every escape probe above
   was run on **Windows**. The mechanisms are portable and the
   `_ctypes` import block is platform-independent, but the same probes
   have **not** been executed on Linux, and this project's sweeps run on
   Linux (DeltaAI). Re-run `tests/test_gaia_sandbox.py` there before
   trusting the Linux behaviour; the POSIX `resource.setrlimit` path in
   particular has had no live exercise at all.

5. **`SandboxDenied` is catchable.** User code can wrap a denied call in
   `try/except` and retry or probe. The operation still never happens --
   the hook raises *before* the effect -- but the sandbox is a wall, not
   a tripwire, and it does not report probing back to the caller.

6. **CPU is bounded only by the wall clock** (plus `RLIMIT_CPU` on
   POSIX). A busy loop consumes one core for the whole timeout. There is
   no CPU-share cgroup.

7. **Windows has no `RLIMIT_AS` equivalent inside the process.** The
   memory cap there is entirely the Job Object's. If the `ctypes` calls
   into `kernel32` fail (they raise, and the call fails closed), no cap
   exists -- which is why failure is fatal rather than a warning.

8. **TOCTOU on path checks.** The hook resolves paths with
   `os.path.realpath` at check time. A symlink swapped between the check
   and the open is theoretically exploitable. The only writable
   directory is one we created microseconds earlier, which makes this
   narrow, not absent.

9. **Disk usage inside scratch is capped only on POSIX**
   (`RLIMIT_FSIZE`). On Windows the Job Object caps memory, not disk.

10. **Determinism is best-effort.** `PYTHONHASHSEED=0` is set and the
    child is otherwise clean, but user code can read the clock, so two
    runs of the same code are not guaranteed byte-identical. The
    registry's call log records only the code and a bounded summary, so
    log determinism does not depend on this.

**My honest assessment**: this is a solid *defence in depth against an
LLM that does something careless or mildly adversarial*, which is the
actual threat model of an evaluation harness. It is **not** a safe place
to run code from a genuinely hostile party. For that, put it in a
container. I would not object to this being run as-is on a benchmark
sweep; I would object to it being described as "sandboxed" without the
qualifications above.

=====================================================================
DELIBERATE FUNCTIONAL LIMITS (not security, but they shape results)
=====================================================================

* **Stdlib only.** `-S` means no `numpy`, no `pandas`. This is a
  fairness and reproducibility choice as much as a security one: with
  site-packages visible, the substrate's capability would depend on
  whichever machine ran the sweep. `math`, `statistics`, `fractions`,
  `decimal`, `datetime`, `itertools`, `re`, `json`, `csv` cover GAIA's
  computational needs. This limit binds **every** method identically.
* **No attachment access from inside the sandbox.** The scratch
  directory starts empty; the attachment is reached through
  `inspect_file`/`inspect_table`, which apply the modality policy. That
  keeps the modality policy un-bypassable -- otherwise `run_python`
  would be an image reader.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

#: The child half. Executed, never imported.
CHILD_SCRIPT = Path(__file__).resolve().with_name("gaia_sandbox_child.py")

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MEMORY_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_OUTPUT_CHARS = 20_000
#: A model can emit a lot of code; this only stops a pathological paste.
MAX_CODE_CHARS = 200_000


def sandbox_interpreter() -> str:
    """The interpreter the child runs under: the BASE interpreter, not
    the virtualenv's `python.exe`.

    This is not cosmetic. On Windows a venv's `Scripts/python.exe` is a
    *trampoline* that re-execs the base interpreter as a second process,
    which collides head-on with the Job Object's
    `ActiveProcessLimit = 1` and makes every sandboxed call fail with
    "Unable to create process". Found the hard way; pinned by a test.
    Using the base interpreter also removes the venv from the picture
    entirely, which is the right thing for isolation independently of
    the bug -- `-S` already means site-packages is unreachable, so the
    venv contributes nothing but a moving part.
    """
    base = getattr(sys, "_base_executable", None)
    if base and Path(base).is_file():
        return str(base)
    return sys.executable


class SandboxError(RuntimeError):
    """The sandbox itself could not be established or run. Distinct from
    user code raising an exception, which is a normal, reported result
    (`SandboxResult.ok is False` with an `error`), not a harness fault."""


@dataclass(frozen=True)
class SandboxResult:
    """Everything one execution produced. Frozen and free of wall-clock
    fields on purpose, matching `gaia_tools.ToolCall`'s determinism rule:
    a caller that wants timing reads `UsageStats`, not this."""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    value_repr: str | None = None
    error: str | None = None
    timed_out: bool = False
    killed: bool = False
    exit_code: int | None = None

    def render(self) -> str:
        """A single human/LLM-readable string. Deterministic ordering,
        and it never returns "" for a successful empty run -- a silent
        blank is indistinguishable from a broken tool."""
        parts: list[str] = []
        if self.stdout:
            parts.append(self.stdout.rstrip("\n"))
        if self.value_repr is not None:
            parts.append(self.value_repr)
        if self.stderr:
            parts.append(f"[stderr] {self.stderr.rstrip()}")
        if self.error:
            parts.append(f"[error] {self.error}")
        return "\n".join(parts) if parts else "[no output]"


# --------------------------------------------------------------------
# Windows Job Object support. Isolated here so the POSIX path never
# touches ctypes and so a reviewer can read the whole Windows mechanism
# in one screen.
# --------------------------------------------------------------------

_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9


def _create_windows_job(memory_bytes: int):  # pragma: no cover - Windows only
    """Creates a Job Object carrying the memory and process-count caps.

    Raises `SandboxError` on any failure: a sandbox that silently runs
    without its memory cap is exactly the kind of quiet degradation this
    substrate refuses everywhere else.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise SandboxError(
            f"CreateJobObjectW failed (error {ctypes.get_last_error()}); refusing to run "
            "sandboxed code without a memory limit."
        )

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = (
        _JOB_OBJECT_LIMIT_PROCESS_MEMORY
        | _JOB_OBJECT_LIMIT_JOB_MEMORY
        | _JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )
    info.BasicLimitInformation.ActiveProcessLimit = 1
    info.ProcessMemoryLimit = memory_bytes
    info.JobMemoryLimit = memory_bytes
    ok = kernel32.SetInformationJobObject(
        job, _JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS, ctypes.byref(info), ctypes.sizeof(info)
    )
    if not ok:
        kernel32.CloseHandle(job)
        raise SandboxError(
            f"SetInformationJobObject failed (error {ctypes.get_last_error()}); refusing to "
            "run sandboxed code without a memory limit."
        )
    return kernel32, job


def _build_child_environment(scratch: Path) -> dict[str, str]:
    """An ALLOWLIST, never a copy of `os.environ`.

    The harness process holds `HF_TOKEN`, `OPENAI_API_KEY` and whatever
    else `.env` carries. Inheriting the environment would hand every one
    of them to model-chosen code, which is a credential-exfiltration
    primitive regardless of how good the rest of the sandbox is. Only
    the variables an interpreter genuinely needs to start are copied.
    """
    env: dict[str, str] = {
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        # Anything that consults the temp directory lands in scratch.
        "TMPDIR": str(scratch),
        "TMP": str(scratch),
        "TEMP": str(scratch),
    }
    if os.name == "nt":
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        env.update(
            {
                "SystemRoot": system_root,
                "SystemDrive": os.environ.get("SystemDrive", "C:"),
                "windir": os.environ.get("windir", system_root),
                "COMSPEC": os.environ.get("COMSPEC", rf"{system_root}\system32\cmd.exe"),
                # Deliberately minimal: enough for the interpreter's own
                # DLL resolution, nothing that would locate a tool.
                "PATH": rf"{system_root}\system32;{system_root}",
            }
        )
    else:
        env["PATH"] = "/usr/bin:/bin"
        env["LC_ALL"] = "C.UTF-8"
    return env


def run_python(
    code: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    memory_bytes: int = DEFAULT_MEMORY_BYTES,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> SandboxResult:
    """Executes `code` in the isolated child interpreter.

    Never raises for ordinary user-code failure -- a syntax error, an
    exception, a denied operation and a timeout all come back as a
    `SandboxResult` with `ok=False`, because an agent needs to *read*
    the failure to correct itself. `SandboxError` is reserved for the
    sandbox failing to establish, which is a harness fault and must be
    loud.
    """
    if not isinstance(code, str):
        raise SandboxError(f"code must be a str, got {type(code).__name__}.")
    if len(code) > MAX_CODE_CHARS:
        return SandboxResult(
            ok=False,
            error=f"Program is {len(code)} chars; the limit is {MAX_CODE_CHARS}.",
        )
    if not CHILD_SCRIPT.is_file():
        raise SandboxError(f"Sandbox child script is missing at {CHILD_SCRIPT}.")

    # mkdtemp (not TemporaryDirectory) so the cleanup is explicit and can
    # tolerate a child that left an undeletable handle behind on Windows.
    scratch = Path(tempfile.mkdtemp(prefix="gaia-sandbox-"))
    result_path = scratch / "__gaia_result__.json"
    argv = [
        sandbox_interpreter(),
        "-S",  # no site-packages: stdlib only (see module docstring)
        "-s",  # no per-user site directory
        "-P",  # script directory NOT on sys.path -> this repo is unreachable
        str(CHILD_SCRIPT),
        str(scratch),
        str(result_path),
        str(int(max_output_chars)),
        str(int(memory_bytes)),
        str(max(1, int(timeout_seconds) + 1)),
    ]

    kernel32 = job = None
    process = None
    try:
        if os.name == "nt":
            kernel32, job = _create_windows_job(memory_bytes)

        popen_kwargs: dict[str, object] = {}
        if os.name == "posix":
            # Its own session, so a timeout kills the whole tree rather
            # than orphaning grandchildren.
            popen_kwargs["start_new_session"] = True
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        process = subprocess.Popen(  # noqa: S603 - argv is fully constructed here
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(scratch),
            env=_build_child_environment(scratch),
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,  # type: ignore[arg-type]
        )

        if job is not None and kernel32 is not None:  # pragma: no cover - Windows only
            # THE RACE-CLOSING STEP. The child is blocked on stdin and
            # has not been handed the code yet, so attaching the job
            # here is not "after the fact" in any meaningful sense.
            if not kernel32.AssignProcessToJobObject(job, int(process._handle)):  # noqa: SLF001
                import ctypes

                error = ctypes.get_last_error()
                process.kill()
                raise SandboxError(
                    f"AssignProcessToJobObject failed (error {error}); refusing to run "
                    "sandboxed code without a memory limit."
                )

        try:
            stdout, stderr = process.communicate(input=code, timeout=timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            _kill_tree(process, kernel32, job)
            stdout, stderr = process.communicate()
            timed_out = True

        return _collect(
            result_path=result_path,
            stdout=stdout or "",
            stderr=stderr or "",
            exit_code=process.returncode,
            timed_out=timed_out,
            timeout_seconds=timeout_seconds,
            memory_bytes=memory_bytes,
            max_output_chars=max_output_chars,
        )
    finally:
        if process is not None and process.poll() is None:  # pragma: no cover - defensive
            _kill_tree(process, kernel32, job)
        if job is not None and kernel32 is not None:  # pragma: no cover - Windows only
            # Closing the handle triggers KILL_ON_JOB_CLOSE, which is the
            # backstop if every explicit kill above somehow missed.
            kernel32.CloseHandle(job)
        shutil.rmtree(scratch, ignore_errors=True)


def _kill_tree(process: subprocess.Popen, kernel32, job) -> None:
    """Kills the sandboxed process AND anything it managed to spawn."""
    if os.name == "nt" and kernel32 is not None and job is not None:  # pragma: no cover
        kernel32.TerminateJobObject(job, 1)
    elif os.name == "posix":  # pragma: no cover - POSIX only
        try:
            os.killpg(os.getpgid(process.pid), 9)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.kill()
    except (ProcessLookupError, OSError):
        pass


def _collect(
    *,
    result_path: Path,
    stdout: str,
    stderr: str,
    exit_code: int | None,
    timed_out: bool,
    timeout_seconds: float,
    memory_bytes: int,
    max_output_chars: int,
) -> SandboxResult:
    """Turns the child's side-channel result file into a `SandboxResult`.

    The file's ABSENCE is the signal that the child never finished --
    killed by the timeout, by the memory cap, or by a crash. That is why
    the child writes it only after user code has stopped, and why the
    result is not parsed out of stdout (user code controls stdout and
    could forge any sentinel placed there).
    """
    if timed_out:
        return SandboxResult(
            ok=False,
            stdout=stdout[:max_output_chars],
            stderr=stderr[:max_output_chars],
            error=f"Execution exceeded the {timeout_seconds:g}s wall-clock limit and was killed.",
            timed_out=True,
            killed=True,
            exit_code=exit_code,
        )
    if not result_path.is_file():
        # Do NOT assert a cause here. The memory cap is one way to get
        # here, but so is a program that wrecked its own interpreter
        # badly enough that the child could not even report the failure
        # (clearing `sys.meta_path` does exactly that). Naming a single
        # likely cause reads as a diagnosis and sends the next reader
        # down the wrong path, so the child's own stderr is surfaced and
        # the possibilities are listed instead.
        detail = stderr.strip().splitlines()[-1] if stderr.strip() else ""
        return SandboxResult(
            ok=False,
            stdout=stdout[:max_output_chars],
            stderr=stderr[:max_output_chars],
            error=(
                f"The sandboxed interpreter died before producing a result (exit code "
                f"{exit_code}). Possible causes: the "
                f"{memory_bytes // (1024 * 1024)} MiB memory limit, or a program that "
                f"broke its own interpreter."
                + (f" Last stderr line: {detail}" if detail else "")
            ),
            killed=True,
            exit_code=exit_code,
        )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SandboxError(f"Sandbox produced an unreadable result file: {exc}") from exc

    return SandboxResult(
        ok=bool(payload.get("ok")),
        stdout=str(payload.get("stdout") or "")[:max_output_chars],
        stderr=str(payload.get("stderr") or "")[:max_output_chars],
        value_repr=payload.get("value_repr"),
        error=payload.get("error"),
        exit_code=exit_code,
    )
