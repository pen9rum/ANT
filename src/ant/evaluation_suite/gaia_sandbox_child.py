"""The CHILD half of the GAIA Python sandbox. Never imported by the
parent process -- it is executed as a script, in a separate interpreter,
by `gaia_sandbox.run_python()`.

READ `gaia_sandbox.py`'s module docstring first: it is the security
writeup for this pair of files and states precisely what this mechanism
does and does NOT protect against. This file implements only the
INTERPRETER-LEVEL half of the defence (a PEP 578 audit hook plus resource
limits); the OS-level half (process separation, memory cap, wall-clock
kill, scrubbed environment, ephemeral cwd) lives in the parent.

It deliberately imports NOTHING from `ant` and nothing outside the
standard library, because it runs with `-S -s -P`: no site-packages, no
user site, and no script directory on `sys.path`. If this file ever grew
an `ant.` import it would silently stop working rather than silently
becoming insecure, which is the failure mode we want.

INVOCATION CONTRACT (kept narrow on purpose):
    python -S -s -P gaia_sandbox_child.py <scratch_dir> <result_path> \
        <max_output_chars> <memory_bytes> <cpu_seconds>
    stdin  -- the user code, then EOF. Reading it is what releases the
              child; the parent only writes after it has attached the
              Windows Job Object, which closes the assign-after-spawn
              race without needing CREATE_SUSPENDED.
    stdout/stderr -- captured, but NOT the result channel (user code can
              write anything it likes there, including a forged
              sentinel). The result is a JSON file at <result_path>,
              written after user code has finished.
"""

from __future__ import annotations

import ast
import builtins
import io
import json
import os
import sys
import traceback

# --------------------------------------------------------------------
# Argument parsing. Anything malformed here is a harness bug, and the
# child FAILS CLOSED -- it exits rather than running user code with an
# unknown or absent limit.
# --------------------------------------------------------------------

if len(sys.argv) != 6:
    sys.stderr.write("gaia_sandbox_child: wrong argument count\n")
    raise SystemExit(64)

SCRATCH_DIR = os.path.realpath(sys.argv[1])
RESULT_PATH = os.path.realpath(sys.argv[2])
MAX_OUTPUT_CHARS = int(sys.argv[3])
MEMORY_BYTES = int(sys.argv[4])
CPU_SECONDS = int(sys.argv[5])


# --------------------------------------------------------------------
# 1. Resource limits (POSIX). On Windows this is a no-op HERE and the
#    equivalent cap is imposed by the parent's Job Object -- see
#    gaia_sandbox.py. Failing to install a limit is fatal: a sandbox that
#    quietly runs unbounded is worse than one that refuses to run.
# --------------------------------------------------------------------


def _install_resource_limits() -> None:
    try:
        import resource  # POSIX only.
    except ImportError:
        return  # Windows: the parent's Job Object owns this.
    # RLIMIT_AS caps the address space, which is what actually stops a
    # runaway allocation in CPython. RLIMIT_DATA is not enough (mmap'd
    # arenas are not counted against it on Linux).
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    # A CPU-time backstop UNDER the parent's wall-clock timeout, so a
    # pure busy-loop dies on its own instead of burning a core until the
    # parent's kill lands.
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    # Bound what the scratch directory can be filled with. Writes inside
    # scratch are permitted by policy; unbounded writes are not.
    resource.setrlimit(resource.RLIMIT_FSIZE, (MEMORY_BYTES, MEMORY_BYTES))
    # No core dumps: they would land outside the scratch directory.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


try:
    _install_resource_limits()
except Exception as exc:  # pragma: no cover - platform-dependent
    sys.stderr.write(f"gaia_sandbox_child: could not install resource limits: {exc!r}\n")
    raise SystemExit(65) from exc


# --------------------------------------------------------------------
# 2. The PEP 578 audit hook.
#
#    This is the interpreter-level half of the sandbox. Two properties
#    make it worth having: a hook CANNOT be removed once installed (there
#    is no `sys.removeaudithook`), and CPython raises the events below
#    from inside the C implementations, so a pure-Python re-route
#    (`getattr(os, "sys" + "tem")`, `builtins.__dict__["open"]`, ...)
#    does not evade it.
#
#    It is NOT a syscall filter. See gaia_sandbox.py for the honest
#    limits of that distinction.
# --------------------------------------------------------------------

_ALLOWED_READ_ROOTS = tuple(
    os.path.realpath(root)
    for root in dict.fromkeys(  # de-duplicated, order preserved
        [
            sys.base_prefix,
            sys.prefix,
            sys.base_exec_prefix,
            sys.exec_prefix,
            os.path.dirname(os.__file__ or ""),
            os.path.dirname(sys.executable or ""),
        ]
    )
    if root
)

#: Import-time denial list. Blocking the module is a cheaper and more
#: legible guard than trying to audit every call it could make, and it
#: also blocks the pure-Python wrappers (`urllib`, `requests`) that sit
#: on top of the primitives. `ctypes` is on here because it is the one
#: stdlib module that turns Python into arbitrary native code, which is
#: precisely the thing an audit hook cannot see past.
#
#: The PRIVATE `_`-prefixed extension modules are listed alongside their
#: public wrappers, and that is the load-bearing half rather than
#: belt-and-braces. `importlib.import_module("ctypes")` can dodge the
#: meta_path finder, but `ctypes/__init__.py` reaches its native half
#: with a plain `from _ctypes import ...` STATEMENT, which fires the
#: unremovable `import` audit event. Blocking `_ctypes` therefore closes
#: the native-code route even for an importer that has defeated every
#: removable layer. Same argument for `_socket` under `socket`.
_BLOCKED_IMPORT_PREFIXES = (
    "ctypes",
    "_ctypes",
    "socket",
    "_socket",
    "ssl",
    "_ssl",
    "_winapi",
    "msvcrt",
    "_multiprocessing",
    "_asyncio",
    "_overlapped",
    "_posixshmem",
    "select",
    "selectors",
    "asyncio",
    "http",
    "urllib",
    "urllib3",
    "requests",
    "httpx",
    "aiohttp",
    "ftplib",
    "smtplib",
    "poplib",
    "imaplib",
    "nntplib",
    "telnetlib",
    "socketserver",
    "xmlrpc",
    "webbrowser",
    "subprocess",
    "_posixsubprocess",
    "multiprocessing",
    "concurrent.futures.process",
    "mmap",
    "pty",
    "tty",
    "curses",
    "dbm",
    "sqlite3",
    "_sqlite3",
    "distutils",
    "setuptools",
    "pip",
    "importlib.metadata",
)

#: Audit events denied outright, by exact name or by prefix. These are
#: the process-creation, network and native-code families.
_BLOCKED_EVENT_PREFIXES = (
    "socket.",
    "ctypes.",
    "subprocess.",
    "os.system",
    "os.exec",
    "os.spawn",
    "os.posix_spawn",
    "os.fork",
    "os.forkpty",
    "os.startfile",
    "os.putenv",
    "os.unsetenv",
    "os.chdir",
    "os.fchdir",
    "os.setuid",
    "os.setgid",
    "os.chown",
    "os.chmod",
    "os.chroot",
    "pty.spawn",
    "webbrowser.open",
    "urllib.Request",
    "ftplib.",
    "smtplib.",
    "imaplib.",
    "poplib.",
    "nntplib.",
    "telnetlib.",
    "http.client.",
    "cpython.run_file",
    "cpython.run_command",
    "builtins.input",
    "winreg.",
    "msvcrt.",
)
# NOTE on what is deliberately NOT in that list: `glob.glob`,
# `sys._getframe`, `object.__getattr__` and the `exec`/`compile` events.
# `glob` reaches the filesystem only through `os.scandir`, which IS
# bounds-checked below, so denying the event as well would buy nothing
# and break legitimate scratch-directory listing. The others are
# introspection, not capability -- denying them breaks ordinary
# tracebacks and `ast`-driven code without closing any path out.

#: Events whose FIRST argument is a path and which must therefore be
#: bounds-checked rather than denied outright.
_PATH_EVENTS = {
    "os.listdir",
    "os.scandir",
    "os.walk",
    "os.fwalk",
    "os.lstat",
    "os.truncate",
    "os.utime",
    "os.getxattr",
    "os.listxattr",
    "os.removexattr",
    "os.setxattr",
    "shutil.copyfile",
    "shutil.copymode",
    "shutil.copystat",
    "shutil.copytree",
    "shutil.move",
    "shutil.rmtree",
    "shutil.unpack_archive",
    "shutil.make_archive",
}

#: Events whose first argument is a path AND which always write.
_WRITE_PATH_EVENTS = {
    "os.mkdir",
    "os.rmdir",
    "os.remove",
    "os.rename",
    "os.link",
    "os.symlink",
    "os.add_dll_directory",
}


class SandboxDenied(PermissionError):
    """Raised inside the sandboxed interpreter when the audit hook
    refuses an operation. A subclass of `PermissionError` so that
    ordinary `except OSError` code in a user program sees something
    plausible rather than an exotic type."""


def _as_path(value: object) -> str | None:
    """Audit event path arguments arrive as `str`, `bytes`, `os.PathLike`
    or an already-open file descriptor (`int`). Only the first three are
    bounds-checkable; an `int` refers to a descriptor this process
    already legitimately holds."""
    if isinstance(value, int):
        return None
    try:
        return os.path.realpath(os.fsdecode(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def _check_path(event: str, path: str | None, *, writing: bool) -> None:
    if path is None:
        return
    if _within(path, SCRATCH_DIR):
        return
    if not writing and any(_within(path, root) for root in _ALLOWED_READ_ROOTS):
        # Reading the standard library is what makes `import json` work.
        # It is read-only and is never the interesting attack target;
        # the repository checkout, the user's home directory and `.env`
        # are all outside these roots and therefore refused.
        return
    verb = "write" if writing else "read"
    raise SandboxDenied(
        f"gaia sandbox: refused to {verb} {path!r} via {event!r}. The only writable "
        f"location is this call's ephemeral scratch directory."
    )


def _is_write_mode(args: tuple) -> bool:
    """`open` fires with `(path, mode, flags)`. `io.open` fills `mode`,
    `os.open` fills `flags`; either may be None, so both are consulted
    and anything ambiguous is treated as a WRITE (fail closed)."""
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else None
    if isinstance(mode, str):
        return any(character in mode for character in "wax+")
    if isinstance(flags, int):
        write_bits = (
            getattr(os, "O_WRONLY", 0)
            | getattr(os, "O_RDWR", 0)
            | getattr(os, "O_CREAT", 0)
            | getattr(os, "O_APPEND", 0)
            | getattr(os, "O_TRUNC", 0)
        )
        return bool(flags & write_bits)
    return True


def _audit_hook(event: str, args: tuple) -> None:
    if event == "import":
        module = args[0] if args else ""
        if isinstance(module, str):
            for prefix in _BLOCKED_IMPORT_PREFIXES:
                if module == prefix or module.startswith(prefix + "."):
                    raise SandboxDenied(
                        f"gaia sandbox: import of {module!r} is blocked. This sandbox "
                        f"has no network and no subprocess capability."
                    )
        return

    if event == "open":
        _check_path(event, _as_path(args[0] if args else None), writing=_is_write_mode(args))
        return

    if event in _WRITE_PATH_EVENTS:
        _check_path(event, _as_path(args[0] if args else None), writing=True)
        return

    if event in _PATH_EVENTS:
        _check_path(event, _as_path(args[0] if args else None), writing=False)
        return

    for prefix in _BLOCKED_EVENT_PREFIXES:
        if event == prefix or event.startswith(prefix):
            raise SandboxDenied(f"gaia sandbox: operation {event!r} is blocked.")


sys.addaudithook(_audit_hook)


# --------------------------------------------------------------------
# 2b. A `sys.meta_path` finder, because the `import` audit event does
#     NOT cover every import route.
#
#     Found by a test, not by reading docs: CPython raises the `import`
#     audit event from `PyImport_ImportModuleLevelObject`, i.e. from
#     `builtins.__import__` and the `IMPORT_NAME` opcode. It does NOT
#     raise it from `importlib.import_module`, which drops straight into
#     the pure-Python `_bootstrap._gcd_import`. So
#     `importlib.import_module("subprocess")` sailed past the hook above
#     while `import subprocess` was refused.
#
#     A meta_path finder sits in the one place every route converges on,
#     so it closes that. It is, however, WEAKER than the audit hook in
#     one specific way: `sys.meta_path` is an ordinary list and user code
#     can clear it. That residual is documented in gaia_sandbox.py's
#     limits section rather than papered over. The unremovable backstop
#     for anything that gets past this is the audit hook's denial of the
#     dangerous OPERATIONS themselves (`subprocess.Popen`, `socket.*`,
#     `ctypes.*`, `os.system`, ...), which no import route can evade.
# --------------------------------------------------------------------


class _BlockedModuleFinder:
    """Refuses a blocked module at find_spec time, for every import route."""

    @staticmethod
    def find_spec(fullname, path=None, target=None):  # noqa: ANN001, ARG004
        for prefix in _BLOCKED_IMPORT_PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                raise SandboxDenied(
                    f"gaia sandbox: import of {fullname!r} is blocked. This sandbox "
                    f"has no network and no subprocess capability."
                )
        return None


sys.meta_path.insert(0, _BlockedModuleFinder)


# --------------------------------------------------------------------
# 3. Run the user program.
#
#    REPL-ish semantics on purpose: if the last top-level statement is a
#    bare expression, its value is reported. GAIA answers are usually a
#    single number or string, and forcing every caller to remember a
#    trailing `print()` would turn a formatting slip into a wrong answer.
# --------------------------------------------------------------------

_result: dict[str, object] = {
    "ok": False,
    "stdout": "",
    "stderr": "",
    "value_repr": None,
    "error": "child did not complete",
}


def _run(code: str) -> None:
    tree = ast.parse(code, filename="<gaia-sandbox>", mode="exec")
    tail: ast.Expression | None = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop()
        assert isinstance(last, ast.Expr)
        tail = ast.Expression(body=last.value)
        ast.copy_location(tail, last)

    namespace: dict[str, object] = {"__name__": "__main__", "__builtins__": builtins}
    exec(compile(tree, "<gaia-sandbox>", "exec"), namespace)  # noqa: S102 - the point
    if tail is not None:
        value = eval(compile(tail, "<gaia-sandbox>", "eval"), namespace)  # noqa: S307
        if value is not None:
            _result["value_repr"] = repr(value)[:MAX_OUTPUT_CHARS]


def main() -> int:
    code = sys.stdin.read()
    out, err = io.StringIO(), io.StringIO()
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        _run(code)
        _result["ok"] = True
        _result["error"] = None
    except SandboxDenied as exc:
        _result["error"] = f"SandboxDenied: {exc}"
    except BaseException as exc:  # noqa: BLE001 - user code may raise anything
        # The traceback is the useful half of a failed computation, so it
        # is returned rather than collapsed to a type name. It is user
        # code's OWN traceback; no harness frame is above `_run`.
        _result["error"] = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip() or repr(exc)
        _result["traceback"] = traceback.format_exc()[-MAX_OUTPUT_CHARS:]
    finally:
        sys.stdout, sys.stderr = real_out, real_err
        _result["stdout"] = out.getvalue()[:MAX_OUTPUT_CHARS]
        _result["stderr"] = err.getvalue()[:MAX_OUTPUT_CHARS]

    # The result channel. Inside scratch, so the hook permits it; written
    # only after user code has stopped running, so user code cannot race
    # it. If the child is killed (timeout / OOM) this file never appears
    # and the parent reports the kill instead -- absence is unambiguous.
    with open(RESULT_PATH, "w", encoding="utf-8") as handle:
        json.dump(_result, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
