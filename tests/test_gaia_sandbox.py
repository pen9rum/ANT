"""Tests for the GAIA sandboxed Python execution service.

These are the tests for a security-sensitive component, so they are
written as ASSERTIONS ABOUT THE ISOLATION rather than as feature tests:
each escape attempt below is a thing the sandbox is claimed to prevent,
and a green suite is the evidence for that claim. What the suite does
NOT and cannot prove is stated in `gaia_sandbox`'s module docstring --
in particular, passing these tests does not make the sandbox safe
against a determined adversary, because they exercise the paths I
thought of.

Every test runs real code in a real subprocess. No network, no LLM.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from ant.evaluation_suite.gaia_sandbox import (
    CHILD_SCRIPT,
    MAX_CODE_CHARS,
    SandboxError,
    SandboxResult,
    _build_child_environment,
    run_python,
    sandbox_interpreter,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Short limits so the suite stays fast; the defaults are exercised
# separately by `test_registry_owns_the_execution_budget`.
FAST = {"timeout_seconds": 20.0, "memory_bytes": 256 * 1024 * 1024}


def run(code: str, **kwargs) -> SandboxResult:
    return run_python(code, **{**FAST, **kwargs})


# --------------------------------------------------------------------
# It actually executes Python
# --------------------------------------------------------------------


def test_a_trailing_expression_is_reported_without_an_explicit_print():
    """GAIA answers are usually one value. Forcing a trailing `print()`
    would turn a formatting slip into a wrong answer."""
    result = run("2 + 2")
    assert result.ok
    assert result.value_repr == "4"
    assert "4" in result.render()


def test_stdout_is_captured():
    result = run("print('hello')")
    assert result.ok
    assert result.stdout.strip() == "hello"


def test_multi_statement_programs_run():
    result = run("total = 0\nfor i in range(10):\n    total += i\ntotal")
    assert result.ok
    assert result.value_repr == "45"


def test_the_standard_library_is_available():
    result = run(
        "import math, statistics, fractions, decimal, datetime, itertools, re, json, csv\n"
        "math.gcd(84, 36)"
    )
    assert result.ok
    assert result.value_repr == "12"


def test_a_statement_ending_program_reports_no_value_but_still_succeeds():
    result = run("x = 1")
    assert result.ok
    assert result.value_repr is None
    assert result.render() == "[no output]"


# --------------------------------------------------------------------
# Failure is REPORTED, never raised, and never silently empty
# --------------------------------------------------------------------


def test_a_user_exception_is_reported_not_raised():
    """An agent has to be able to READ its own failure to correct
    itself; raising would turn a debugging loop into a dead end."""
    result = run("1 / 0")
    assert not result.ok
    assert "ZeroDivisionError" in (result.error or "")


def test_a_syntax_error_is_reported_not_raised():
    result = run("def (:")
    assert not result.ok
    assert "SyntaxError" in (result.error or "")


def test_render_never_returns_an_empty_string():
    """A silent blank is indistinguishable from a broken tool."""
    assert run("pass").render() == "[no output]"
    assert run("1/0").render()


def test_an_oversized_program_is_refused_without_spawning():
    result = run_python("x = 1\n" * MAX_CODE_CHARS, **FAST)
    assert not result.ok
    assert "limit" in (result.error or "")


def test_non_string_code_is_a_harness_fault_not_a_user_failure():
    with pytest.raises(SandboxError):
        run_python(123)  # type: ignore[arg-type]


# --------------------------------------------------------------------
# ISOLATION: process separation
# --------------------------------------------------------------------


def test_execution_happens_in_a_separate_process():
    result = run("import os\nos.getpid()")
    assert result.ok
    assert int(result.value_repr or "-1") != os.getpid()


def test_the_child_cannot_mutate_the_harness_process():
    """The point of process separation: anything the child does to its
    own interpreter stays there."""
    run("import sys\nsys.modules.clear()")
    assert sys.modules  # the harness is untouched
    assert run("2 + 2").ok


def test_the_child_script_exists_and_imports_nothing_from_ant():
    """The child runs with `-P`, so this repository is not on its
    `sys.path`. An `ant.` import here would fail at runtime rather than
    fail safe, so it is pinned statically."""
    source = CHILD_SCRIPT.read_text(encoding="utf-8")
    assert "import ant" not in source
    assert "from ant" not in source


def test_the_base_interpreter_is_used_not_the_venv_trampoline():
    """A venv's Scripts/python.exe on Windows re-execs the base
    interpreter as a SECOND process, which collides with the Job
    Object's ActiveProcessLimit=1 and makes every call fail with
    "Unable to create process". Found the hard way."""
    interpreter = sandbox_interpreter()
    assert Path(interpreter).is_file()
    base = getattr(sys, "_base_executable", None)
    if base and Path(base).is_file():
        assert interpreter == str(base)


# --------------------------------------------------------------------
# ISOLATION: the environment carries no secrets
# --------------------------------------------------------------------


def test_the_child_environment_is_an_allowlist_not_a_copy(tmp_path: Path):
    env = _build_child_environment(tmp_path)
    assert "HF_TOKEN" not in env
    assert "OPENAI_API_KEY" not in env
    assert "PYTHONPATH" not in env
    assert env["TMP"] == str(tmp_path)


def test_a_secret_in_the_harness_environment_is_not_visible_to_the_child(monkeypatch):
    """The concrete exfiltration case: the harness holds HF_TOKEN and
    OPENAI_API_KEY, and model-chosen code must not be able to read
    them."""
    monkeypatch.setenv("HF_TOKEN", "hf_sentinel_must_not_leak")
    monkeypatch.setenv("OPENAI_API_KEY", "sk_sentinel_must_not_leak")
    result = run("import os\nsorted(os.environ)")
    assert result.ok
    assert "HF_TOKEN" not in (result.value_repr or "")
    assert "OPENAI_API_KEY" not in (result.value_repr or "")
    assert "sentinel" not in (result.value_repr or "")


# --------------------------------------------------------------------
# ISOLATION: filesystem
# --------------------------------------------------------------------


def test_the_repository_dotenv_cannot_be_read():
    result = run(f"open({str(REPO_ROOT / '.env')!r}).read()")
    assert not result.ok
    assert "SandboxDenied" in (result.error or "")


def test_this_projects_own_source_cannot_be_read():
    target = REPO_ROOT / "src" / "ant" / "evaluation_suite" / "gaia_sandbox.py"
    result = run(f"open({str(target)!r}).read()")
    assert not result.ok
    assert "SandboxDenied" in (result.error or "")


def test_the_users_home_directory_cannot_be_listed():
    result = run(f"import os\nos.listdir({str(Path.home())!r})")
    assert not result.ok
    assert "SandboxDenied" in (result.error or "")


def test_writing_outside_the_scratch_directory_is_refused():
    target = REPO_ROOT / "SANDBOX_ESCAPE_PROOF.txt"
    result = run(f"open({str(target)!r}, 'w').write('escaped')")
    assert not result.ok
    assert "SandboxDenied" in (result.error or "")
    assert not target.exists()


def test_deleting_a_repository_file_is_refused():
    target = REPO_ROOT / "pyproject.toml"
    result = run(f"import os\nos.remove({str(target)!r})")
    assert not result.ok
    assert "SandboxDenied" in (result.error or "")
    assert target.exists()


def test_low_level_os_open_is_bounds_checked_too():
    """`os.open` raises the same `open` audit event as `io.open`, so
    dropping to the low-level call is not a way round the check."""
    result = run(f"import os\nos.open({str(REPO_ROOT / '.env')!r}, os.O_RDONLY)")
    assert not result.ok
    assert "SandboxDenied" in (result.error or "")


def test_pathlib_is_bounds_checked_too():
    result = run(f"import pathlib\npathlib.Path({str(REPO_ROOT / '.env')!r}).read_text()")
    assert not result.ok
    assert "SandboxDenied" in (result.error or "")


def test_the_scratch_directory_is_writable_and_readable():
    result = run("open('note.txt','w').write('ok')\nopen('note.txt').read()")
    assert result.ok
    assert result.value_repr == "'ok'"


def test_the_scratch_directory_starts_empty_every_call():
    """Ephemeral per call: nothing survives from a previous execution,
    so one task cannot leave state for another."""
    first = run("open('carry.txt','w').write('x')\nimport os\nsorted(os.listdir('.'))")
    assert first.ok
    second = run("import os\n[n for n in os.listdir('.') if n == 'carry.txt']")
    assert second.ok
    assert second.value_repr == "[]"


def test_the_scratch_directory_is_removed_after_the_call():
    result = run("import os\nos.getcwd()")
    assert result.ok
    path = Path((result.value_repr or "''").strip("'\""))
    assert not path.exists()


def test_the_attachment_directory_is_not_reachable_from_the_sandbox(tmp_path: Path):
    """If run_python could open the attachment it would be a way around
    the modality policy -- a method could read an image's bytes and gain
    a perception capability the substrate withholds from everyone."""
    secret = tmp_path / "attachment.png"
    secret.write_bytes(b"pretend image bytes")
    result = run(f"open({str(secret)!r}, 'rb').read()")
    assert not result.ok
    assert "SandboxDenied" in (result.error or "")


# --------------------------------------------------------------------
# ISOLATION: process creation and native code
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "import os\nos.system('echo pwned')",
        "import subprocess",
        "import subprocess\nsubprocess.run(['echo','pwned'])",
        "import multiprocessing",
        "__import__('subprocess')",
        "import importlib\nimportlib.import_module('subprocess')",
    ],
)
def test_process_creation_is_blocked(code):
    result = run(code)
    assert not result.ok
    assert "SandboxDenied" in (result.error or "")


def test_ctypes_is_blocked_because_it_defeats_the_audit_hook():
    """`ctypes` is the one stdlib module that turns Python into
    arbitrary native code, which is precisely what an audit hook cannot
    see past."""
    result = run("import ctypes")
    assert not result.ok
    assert "SandboxDenied" in (result.error or "")


def test_reaching_the_blocked_module_through_a_computed_name_still_fails():
    """The audit hook fires inside CPython's import machinery, so
    building the module name at runtime does not evade it."""
    result = run("__import__('sub' + 'process')")
    assert not result.ok
    assert "SandboxDenied" in (result.error or "")


def test_importlib_import_module_is_blocked_too():
    """REGRESSION. `importlib.import_module` does NOT raise the `import`
    audit event -- it drops into the pure-Python `_gcd_import`. This
    escaped the first version of the hook and is why the meta_path
    finder exists."""
    result = run("import importlib\nimportlib.import_module('subprocess')")
    assert not result.ok
    assert "SandboxDenied" in (result.error or "")


def test_removing_the_meta_path_finder_does_not_unblock_a_statement_import():
    """The two layers are complementary: the finder is removable, the
    audit hook is not."""
    result = run("import sys\nsys.meta_path.pop(0)\nimport ctypes")
    assert not result.ok
    assert "SandboxDenied" in (result.error or "")


def test_the_native_code_route_survives_defeating_every_removable_layer():
    """THE IMPORTANT ONE. An attacker who knows the design pops the
    meta_path finder AND uses the audit-event-free `import_module`
    route. It still fails, because `ctypes/__init__.py` reaches its
    native half with a `from _ctypes import ...` STATEMENT, and
    `_ctypes` is on the blocked list guarded by the unremovable hook.

    Evidence, not proof -- see the module docstring's limits section."""
    result = run(
        "import sys, importlib\n"
        "sys.meta_path.pop(0)\n"
        "c = importlib.import_module('ctypes')\n"
        "c.memmove"
    )
    assert not result.ok
    assert "_ctypes" in (result.error or "")


def test_the_private_extension_modules_are_blocked_alongside_their_wrappers():
    for module in ("_ctypes", "_socket"):
        result = run(
            "import sys, importlib\n"
            "sys.meta_path.pop(0)\n"
            f"importlib.import_module({module!r})"
        )
        assert not result.ok
        assert "SandboxDenied" in (result.error or "")


def test_a_program_that_wrecks_its_own_interpreter_is_reported_not_diagnosed():
    """Clearing `sys.meta_path` breaks the child badly enough that it
    cannot write its own result file. The harness must report that
    honestly rather than asserting a cause it does not know."""
    # `fractions` is not already in `sys.modules`, so importing it after
    # clearing meta_path genuinely exercises the broken import system --
    # unlike, say, `json`, which the child imported at startup and would
    # come straight back out of the module cache.
    result = run("import sys\nsys.meta_path.clear()\nimport fractions")
    assert not result.ok
    assert result.killed
    assert "Possible causes" in (result.error or "")
    assert run("2 + 2").ok  # and the harness keeps working


# --------------------------------------------------------------------
# ISOLATION: network
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "import socket",
        "import ssl",
        "import urllib.request",
        "import http.client",
        "import ftplib",
        "import socket\nsocket.gethostbyname('example.com')",
        "import urllib.request\nurllib.request.urlopen('http://example.com')",
    ],
)
def test_network_access_is_blocked(code):
    result = run(code)
    assert not result.ok
    assert "SandboxDenied" in (result.error or "")


# --------------------------------------------------------------------
# ISOLATION: resource limits
# --------------------------------------------------------------------


def test_a_wall_clock_timeout_kills_the_program():
    result = run("while True:\n    pass", timeout_seconds=3.0)
    assert not result.ok
    assert result.timed_out
    assert result.killed
    assert "wall-clock" in (result.error or "")


def test_the_harness_survives_a_timeout_and_keeps_working():
    run("while True:\n    pass", timeout_seconds=3.0)
    assert run("2 + 2").ok


def test_the_memory_limit_binds():
    """Allocated in chunks rather than one huge object so this exercises
    the LIMIT rather than a single allocation the allocator would refuse
    anyway."""
    result = run(
        "chunks = []\nfor _ in range(400):\n    chunks.append(bytearray(8 * 1024 * 1024))\n"
        "len(chunks)",
        memory_bytes=128 * 1024 * 1024,
        timeout_seconds=30.0,
    )
    assert not result.ok
    assert "MemoryError" in (result.error or "") or result.killed


# --------------------------------------------------------------------
# Site-packages are deliberately unreachable
# --------------------------------------------------------------------


@pytest.mark.parametrize("module", ["numpy", "pymupdf", "pytest", "ant"])
def test_site_packages_and_this_repo_are_not_importable(module):
    """`-S -s -P`. This is a fairness/reproducibility choice as much as
    a security one: with site-packages visible, the substrate's
    capability would depend on which machine ran the sweep."""
    result = run(f"import {module}")
    assert not result.ok
    assert "ModuleNotFoundError" in (result.error or "")


# --------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------


def test_identical_programs_produce_identical_results():
    first, second = run("sum(range(1000))"), run("sum(range(1000))")
    assert first == second


def test_hash_randomisation_is_disabled():
    """`PYTHONHASHSEED=0` survives because the child is launched with
    `-S -s -P` and NOT `-I` (which implies `-E` and would discard it)."""
    first = run("hash('gaia')")
    second = run("hash('gaia')")
    assert first.value_repr == second.value_repr


def test_a_sandbox_result_carries_no_wall_clock_field():
    """Same determinism rule as `gaia_tools.ToolCall`: a timing field
    here would make two identical runs diff."""
    fields = set(SandboxResult.__dataclass_fields__)
    assert not (fields & {"duration", "duration_s", "elapsed", "started_at", "timestamp"})
