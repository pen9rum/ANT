from __future__ import annotations

import subprocess
from pathlib import Path

DEFAULT_MAX_CHARS = 16000


def build_directory_structure(repo_dir: Path, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """A real, deterministic directory tree for a pinned git checkout,
    rendered as indented ASCII text -- the same shape (a labeled
    "Directory Structure" section) RepoProbe's own official scoring
    pipeline supplies via `repomix-output.md` when repomix has actually
    been run (see evaluator.py's own `_parse_repomix_repo_info`, which
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

    Bounded, PER TOP-LEVEL DIRECTORY: when the full tree exceeds
    max_chars, a naive linear/alphabetical truncation would let whichever
    top-level directory happens to sort first consume the entire budget,
    silently hiding every other top-level directory from the judge
    (confirmed live: agent-framework's real, alphabetically-early
    dotnet/ sample tree alone exceeded an 8000-char cap, so python/ --
    where every RepoProbe-Python question's own code actually lives --
    was never shown at all, and a real citation into it was flagged as
    "does not appear in the provided repository structure"). Instead,
    each top-level entry gets an equal share of the budget, so every
    top-level directory is represented at least partially regardless of
    where it sorts.
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

    full = _render(tree, "")
    if len(full) <= max_chars:
        return full

    top_level = sorted(tree.items(), key=lambda item: (item[1] is None, item[0].lower()))
    if not top_level:
        return ""
    per_entry_budget = max(200, max_chars // len(top_level))

    parts_out: list[str] = []
    for name, child in top_level:
        if child is None:
            parts_out.append(name)
        else:
            parts_out.append(_bounded_subtree(name, child, per_entry_budget))
    combined = "\n".join(parts_out)

    if len(combined) <= max_chars:
        return combined
    return _truncate_at_line_boundary(combined, max_chars)


def _render(node: dict, prefix: str) -> str:
    lines: list[str] = []
    entries = sorted(node.items(), key=lambda item: (item[1] is None, item[0].lower()))
    for name, child in entries:
        if child is None:
            lines.append(f"{prefix}{name}")
        else:
            lines.append(f"{prefix}{name}/")
            lines.append(_render(child, prefix + "  "))
    return "\n".join(line for line in lines if line)


_MARKER_TEMPLATE = "\n  ... ({} more characters omitted) ..."
# Reserves enough room for the marker itself (generously sized for up to a
# 7-digit omitted-character count) so appending it never pushes this
# entry's own rendered text back over its budget.
_MARKER_RESERVE = len(_MARKER_TEMPLATE.format(9_999_999))


def _bounded_subtree(name: str, child: dict, budget: int) -> str:
    """Renders one top-level directory's own subtree, truncated to
    `budget` characters ON ITS OWN (marker included) -- so this entry's
    own omission (if any) never eats into a sibling top-level entry's
    share, and never overflows its own budget once the marker is
    appended."""
    header = f"{name}/"
    body = _render(child, "  ")
    full = f"{header}\n{body}" if body else header
    if len(full) <= budget:
        return full
    content_budget = max(50, budget - _MARKER_RESERVE)
    truncated = _truncate_at_line_boundary(full, content_budget)
    omitted_chars = len(full) - len(truncated)
    return f"{truncated}{_MARKER_TEMPLATE.format(omitted_chars)}"


def _truncate_at_line_boundary(text: str, max_chars: int) -> str:
    truncated = text[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline != -1:
        truncated = truncated[:last_newline]
    return truncated
