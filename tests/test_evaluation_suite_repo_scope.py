from __future__ import annotations

from pathlib import Path

from ant.evaluation_suite.repo_scope import (
    LOCKFILE_BASENAMES,
    MAX_TEXT_FILE_BYTES,
    EvalRepoEnvironment,
)
from ant.indexing import discover_territories


def _build_representative_repo(root: Path) -> None:
    """A synthetic repo mixing every category the repo-file-universe audit
    cares about: previously-covered extensions, previously-excluded-but-
    legitimate text (.rst/.sql/.cs -- standing in for any language not on
    the old closed allowlist), a known lockfile, an oversized text file, a
    genuinely binary file, a notebook, and files under every IGNORED_DIRS
    category -- so a single pass over this tree exercises every rule in
    EvalRepoEnvironment.iter_files() at once."""
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "guide.rst").write_text("Guide\n=====\n", encoding="utf-8")
    (root / "fixtures").mkdir()
    (root / "fixtures" / "query.sql").write_text("SELECT 1;\n", encoding="utf-8")
    (root / "src" / "Program.cs").write_text("class Program {}\n", encoding="utf-8")
    (root / "README").write_text("A readme with no extension.\n", encoding="utf-8")

    (root / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    (root / "huge.txt").write_text("x" * (MAX_TEXT_FILE_BYTES + 1024), encoding="utf-8")
    (root / "image.bin").write_bytes(b"\x89PNG\x00\x01\x02\x00binarydata\x00")
    (root / "notebook.ipynb").write_text('{"cells": []}\n', encoding="utf-8")

    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "node_modules" / "pkg" / "index.js").write_text("module.exports = {};\n", "utf-8")
    (root / "build").mkdir()
    (root / "build" / "output.txt").write_text("generated\n", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "main.cpython-313.pyc").write_bytes(b"\x00\x01\x02")


def test_eval_repo_environment_includes_previously_excluded_text_formats(tmp_path: Path) -> None:
    _build_representative_repo(tmp_path)
    env = EvalRepoEnvironment(tmp_path)
    relative = {str(p.relative_to(env.root)).replace("\\", "/") for p in env.iter_files()}

    assert "src/main.py" in relative
    assert "docs/guide.rst" in relative
    assert "fixtures/query.sql" in relative
    assert "src/Program.cs" in relative
    assert "README" in relative


def test_eval_repo_environment_still_excludes_lockfiles_binaries_oversized_and_notebooks(
    tmp_path: Path,
) -> None:
    _build_representative_repo(tmp_path)
    env = EvalRepoEnvironment(tmp_path)
    relative = {str(p.relative_to(env.root)).replace("\\", "/") for p in env.iter_files()}

    assert "package-lock.json" not in relative
    assert "huge.txt" not in relative
    assert "image.bin" not in relative
    assert "notebook.ipynb" not in relative


def test_eval_repo_environment_still_excludes_ignored_dirs(tmp_path: Path) -> None:
    _build_representative_repo(tmp_path)
    env = EvalRepoEnvironment(tmp_path)
    relative = {str(p.relative_to(env.root)).replace("\\", "/") for p in env.iter_files()}

    assert not any(item.startswith(".git/") for item in relative)
    assert not any(item.startswith("node_modules/") for item in relative)
    assert not any(item.startswith("build/") for item in relative)
    assert not any(item.startswith("__pycache__/") for item in relative)


def test_eval_repo_environment_lockfile_basenames_are_the_documented_fixed_set() -> None:
    assert "package-lock.json" in LOCKFILE_BASENAMES
    assert "Cargo.lock" in LOCKFILE_BASENAMES
    assert "yarn.lock" in LOCKFILE_BASENAMES


def test_matched_react_and_retrieval_import_the_same_eval_repo_environment_class() -> None:
    """Parity by construction: if all three call sites hold a reference to
    the identical class object, their file universes cannot diverge --
    there is only one iter_files() implementation for all three to run."""
    from ant.agents import ant_adapter, matched_react, retrieval
    from ant.evaluation_suite.repo_scope import EvalRepoEnvironment as canonical

    assert ant_adapter.EvalRepoEnvironment is canonical
    assert matched_react.EvalRepoEnvironment is canonical
    assert retrieval.EvalRepoEnvironment is canonical


def test_discover_territories_does_not_drop_or_add_files_relative_to_iter_files(
    tmp_path: Path,
) -> None:
    """The direct parity guarantee the audit asked for: BEFORE ANT applies
    WorkerCard territory specialization, the set of files
    discover_territories partitions into territories is EXACTLY the set
    EvalRepoEnvironment.iter_files() produced -- ANT's own territory logic
    (frozen core, untouched) neither silently drops nor adds anything, so
    the same expanded file universe this test proves Matched ReAct/
    Retrieval receive is exactly what ANT's own indexing starts from too.
    """
    _build_representative_repo(tmp_path)
    env = EvalRepoEnvironment(tmp_path)
    iter_files_relative = {
        str(p.relative_to(env.root)).replace("\\", "/") for p in env.iter_files()
    }

    territories = discover_territories(env)
    territory_files_relative = {
        file for territory in territories for file in territory.files
    }

    assert territory_files_relative == iter_files_relative


def test_eval_repo_environment_is_a_real_repo_environment_subclass(tmp_path: Path) -> None:
    """Liskov substitutability: discover_territories's own type hint is
    `repo: RepoEnvironment`; this must hold for real, not just duck-type,
    since that's what makes constructing EvalRepoEnvironment in
    evaluation-suite call sites a safe substitution rather than a type
    violation papered over at runtime."""
    from ant.environment import RepoEnvironment

    env = EvalRepoEnvironment(tmp_path)
    assert isinstance(env, RepoEnvironment)


def test_ant_environment_repo_module_itself_is_never_imported_for_modification() -> None:
    """EvalRepoEnvironment must read ant.environment.repo's own IGNORED_DIRS
    (reuse, not re-derive) but never assign to any of its module-level
    names -- this is the check that the frozen-core-freeze rule (a new
    evaluation-side subclass, never a monkeypatch of the core module) is
    actually being honored, not just asserted in a docstring."""
    import inspect

    from ant.evaluation_suite import repo_scope as repo_scope_module

    source = inspect.getsource(repo_scope_module)
    assert "ant.environment.repo import IGNORED_DIRS" in source
    assert "setattr(" not in source
    # TEXT_EXTENSIONS is discussed in comments (explaining what this
    # module replaces) but never imported or referenced as live code.
    assert "TEXT_EXTENSIONS" not in repo_scope_module.__dict__
    assert "import TEXT_EXTENSIONS" not in source
