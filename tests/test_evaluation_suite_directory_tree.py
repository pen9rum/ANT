from __future__ import annotations

import subprocess
from pathlib import Path

from ant.evaluation_suite.directory_tree import build_directory_structure


def _init_repo_with_files(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for rel_path, content in files.items():
        full = repo / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "init"], cwd=repo, check=True)
    return repo


def test_known_real_files_appear_in_the_rendered_tree(tmp_path: Path) -> None:
    repo = _init_repo_with_files(
        tmp_path,
        {
            "README.md": "hi",
            "src/core/bar.py": "x = 1",
            "src/core/bar_helper.py": "y = 2",
        },
    )
    tree = build_directory_structure(repo)
    assert "README.md" in tree
    assert "bar.py" in tree
    assert "bar_helper.py" in tree
    assert "core/" in tree
    assert "src/" in tree


def test_nonexistent_files_do_not_appear(tmp_path: Path) -> None:
    repo = _init_repo_with_files(tmp_path, {"README.md": "hi", "src/core/bar.py": "x = 1"})
    tree = build_directory_structure(repo)
    assert "totally_fabricated_module.py" not in tree
    assert "docling_core" not in tree
    assert "fs42" not in tree


def test_untracked_file_is_excluded(tmp_path: Path) -> None:
    repo = _init_repo_with_files(tmp_path, {"README.md": "hi"})
    (repo / "untracked.py").write_text("never git-added", encoding="utf-8")
    tree = build_directory_structure(repo)
    assert "README.md" in tree
    assert "untracked.py" not in tree


def test_output_is_deterministic_across_repeated_calls(tmp_path: Path) -> None:
    repo = _init_repo_with_files(
        tmp_path,
        {"b.py": "1", "a.py": "2", "sub/z.py": "3", "sub/a.py": "4"},
    )
    first = build_directory_structure(repo)
    second = build_directory_structure(repo)
    assert first == second


def test_never_depends_on_a_model_name_or_answer_argument() -> None:
    import inspect

    sig = inspect.signature(build_directory_structure)
    param_names = set(sig.parameters)
    assert param_names == {"repo_dir", "max_chars"}


def test_bounded_representation_truncates_large_trees(tmp_path: Path) -> None:
    files = {f"pkg/module_{i:04d}.py": "x = 1" for i in range(2000)}
    repo = _init_repo_with_files(tmp_path, files)
    tree = build_directory_structure(repo, max_chars=500)
    assert len(tree) <= 500 + len("\n... (99999 more characters of the directory tree omitted) ...")
    assert "more characters of the directory tree omitted" in tree


def test_truncation_never_cuts_an_entry_mid_line(tmp_path: Path) -> None:
    files = {f"pkg/module_{i:04d}.py": "x = 1" for i in range(500)}
    repo = _init_repo_with_files(tmp_path, files)
    tree = build_directory_structure(repo, max_chars=300)
    body = tree.rsplit("\n... (", 1)[0]
    for line in body.splitlines():
        assert line == "" or line.strip()  # no partial/garbled line


def test_empty_repo_produces_empty_string(tmp_path: Path) -> None:
    repo = tmp_path / "empty_repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    tree = build_directory_structure(repo)
    assert tree == ""


def test_ordering_is_directories_first_then_alphabetical(tmp_path: Path) -> None:
    repo = _init_repo_with_files(
        tmp_path,
        {"zzz_dir/inner.py": "1", "aaa_file.py": "2"},
    )
    tree = build_directory_structure(repo)
    lines = tree.splitlines()
    # A directory sorts before a file at the same level regardless of name.
    assert lines.index("zzz_dir/") < lines.index("aaa_file.py")
