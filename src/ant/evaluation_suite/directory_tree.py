from __future__ import annotations

import subprocess
from pathlib import Path

DEFAULT_MAX_CHARS = 8000


def build_directory_structure(repo_dir: Path, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """A real, deterministic directory tree for a pinned git checkout,
    rendered as indented ASCII text -- the same shape (a labeled
    "Directory Structure" section) RepoProbe's own official scoring
    pipeline supplies via `repomix-output.md` when repomix has actually
    been run (see evaluator.py's `_parse_repomix_repo_info`, which
    otherwise silently falls back to an EMPTY string -- exactly this
    evaluation suite's own prior fidelity gap). This function is the
    evaluation-side, dependency-free (no Node.js/repomix toolchain)
    substitute: it supplies the SAME kind of information (real files
    that exist at the pinned commit), not a fabricated or heuristic
    approximation.

    Uses `git ls-files` (not a filesystem walk) so the listing is
    exactly the VCS-tracked file set at whatever commit `repo_dir` is
    currently checked out to -- deterministic (git's own sort order is
    stable), gold-answer-independent, and identical regardless of which
    method's answer is being scored (never method-specific: this
    function takes only a repo path, never a model name or answer).

    Bounded: truncated to `max_chars` characters (a fixed, disclosed cap,
    not stripped mid-entry) with a trailing marker noting how much was
    omitted, so a very large repository cannot blow out the scoring
    prompt's own size.
    """
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = sorted(line for line in result.stdout.splitlines() if line.strip())

    tree: dict = {}
    for path in paths:
        parts = path.split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault(parts[-1], None)

    lines: list[str] = []

    def _render(node: dict, prefix: str) -> None:
        entries = sorted(node.items(), key=lambda item: (item[1] is None, item[0].lower()))
        for name, child in entries:
            if child is None:
                lines.append(f"{prefix}{name}")
            else:
                lines.append(f"{prefix}{name}/")
                _render(child, prefix + "  ")

    _render(tree, "")
    rendered = "\n".join(lines)

    if len(rendered) <= max_chars:
        return rendered

    truncated = rendered[:max_chars]
    # Cut back to the last full line so no entry is shown half-written.
    last_newline = truncated.rfind("\n")
    if last_newline != -1:
        truncated = truncated[:last_newline]
    omitted_chars = len(rendered) - len(truncated)
    return f"{truncated}\n... ({omitted_chars} more characters of the directory tree omitted) ..."
