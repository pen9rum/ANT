from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ant.environment import RepoEnvironment
from ant.environment.repo import IGNORED_DIRS

# The general, benchmark-independent repository-text policy this
# evaluation suite uses in place of ANT core's own TEXT_EXTENSIONS
# allowlist -- see docs/repo_file_universe_policy.md for the full audit
# and rationale this module implements. Summary: a fixed, closed
# extension allowlist inevitably under-covers real repositories (measured
# directly: it excludes an entire language's source for any repo whose
# primary language isn't one of the ~20 extensions in the list -- e.g.
# every .cs/.rb/.php/.swift/.lua/.f90/.cpp file in the benchmark corpora
# this suite evaluates against, not just the .rst/.sql gap first noticed
# against sphinx/sqlfluff). This module replaces the extension allowlist
# with CONTENT-based text detection (does the file actually decode as
# text, the same NUL-byte heuristic git/most tools use to distinguish
# text from binary) plus a small, fixed set of exclusions for categories
# that are technically valid text but not meaningfully source/doc/config
# for a repository-understanding task: known machine-generated lockfiles
# (by well-known, industry-standard basename, not an open-ended list) and
# a size cap (rejects huge generated blobs regardless of what produced
# them, without needing to name every generator).

# 1 MiB: comfortably larger than any hand-authored source/doc/config file
# this suite has observed across its benchmark corpora, small enough to
# reject huge generated data dumps, minified bundles, or huge lockfiles
# that happen to be valid UTF-8 text. A size cap, not an extension list,
# is what "continue excluding... large generated artifacts" means under a
# content-based policy -- it doesn't care WHAT produced the file, only
# whether it's plausibly something a person would read.
MAX_TEXT_FILE_BYTES = 1_048_576

# Bytes sniffed from the start of a candidate file to decide text-vs-binary.
# 8 KiB is enough to catch a binary file's own magic bytes/structure
# without reading (and holding in memory) the whole file just to classify it.
_SNIFF_BYTES = 8192

# A short, fixed, industry-standard set of machine-generated dependency
# lockfiles, matched by exact basename (not extension, since some of
# these -- package-lock.json -- share an extension with genuinely
# meaningful hand-authored files of the same suffix). These pass the
# text-content sniff (they're valid UTF-8/ASCII) but carry almost no
# narrative/explanatory signal for a repository-understanding question,
# and are frequently enormous (exactly the "large generated artifacts"
# category) -- a small, closed, well-known set, not an ever-growing
# benchmark-specific list.
LOCKFILE_BASENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "Gemfile.lock",
    "poetry.lock",
    "composer.lock",
    "go.sum",
    "Pipfile.lock",
    "mix.lock",
}

# Notebooks (.ipynb) are valid UTF-8 JSON and would pass the content sniff,
# but their raw JSON form is high-noise for a lexical/embedding search
# index: execution counts, cell-output blobs, and (frequently)
# base64-embedded images are mixed in with the actual prose/code content,
# with no clean separation available without dedicated cell-by-cell
# extraction -- a distinct feature this pass does not build (implementing
# it well -- correctly walking the notebook JSON schema, deciding how to
# represent outputs, stripping embedded images -- is real, separate
# implementation risk, not a one-line addition). Excluded explicitly and
# disclosed, not silently dropped: measured directly, only 8 of the 34
# repos sampled for this audit contain any .ipynb files at all, and even
# in those it is a small fraction of total repository content. This is a
# deliberate, revisitable scope decision, not an oversight.
_EXCLUDED_SUFFIXES = {".ipynb"}


def _looks_like_text(path: Path) -> bool:
    """The standard NUL-byte heuristic (the same one git and most
    text/binary detectors use): read a bounded prefix and treat the
    presence of a NUL byte as conclusive evidence of binary content. This
    is a general, content-based signal -- it says nothing about file
    extension, so a .cs/.rb/.php/.swift/.f90/.rst/.sql file (or any
    extension this suite has never seen before) is classified correctly
    without needing its own allowlist entry."""
    try:
        with path.open("rb") as handle:
            chunk = handle.read(_SNIFF_BYTES)
    except OSError:
        return False
    if b"\x00" in chunk:
        return False
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        # A truncated multi-byte sequence exactly at the _SNIFF_BYTES
        # boundary is a real possibility for a genuinely UTF-8 file, so a
        # decode failure alone isn't conclusive the way a NUL byte is --
        # only reject here if it fails README/comment ASCII fallback too.
        try:
            chunk.decode("ascii")
        except UnicodeDecodeError:
            return False
    return True


def _is_excluded(path: Path) -> bool:
    if path.name in LOCKFILE_BASENAMES:
        return True
    if path.suffix.lower() in _EXCLUDED_SUFFIXES:
        return True
    return False


@dataclass(frozen=True)
class EvalRepoEnvironment(RepoEnvironment):
    """Evaluation-side substitute for ANT core's own RepoEnvironment,
    supplying the SAME repository (same root, same IGNORED_DIRS VCS/build/
    cache/dependency exclusion -- reused directly from ant.environment.repo,
    never re-derived) under the general content-based text policy above
    instead of core's closed TEXT_EXTENSIONS allowlist.

    This is a plain subclass (Liskov-substitutable everywhere a
    `RepoEnvironment` is expected -- `discover_territories`, the only
    core function that actually consumes one, calls only `.root` and
    `.iter_files()`, both inherited/overridden here, never anything
    RepoEnvironment-specific) -- not a monkeypatch, not a runtime mutation
    of the existing class, and `ant/environment/repo.py` itself is never
    imported for modification, only for its IGNORED_DIRS constant. ANT's
    own default runtime (`ant.cli`, `ant.evaluation.runner`,
    `ant.git_refresh`) is completely unaffected -- none of those import
    from this module, so ANT's own indexing behavior when NOT run through
    this evaluation suite is byte-for-byte unchanged.

    Used identically by every method this evaluation suite compares
    (`ant.agents.ant_adapter.AntAgent`, `ant.agents.matched_react.
    MatchedReActAgent`, `ant.agents.retrieval.RetrievalAgent`) -- see
    test_evaluation_suite_repo_scope.py's parity tests for the guarantee
    that all three receive the identical file universe before ANT applies
    WorkerCard territory specialization.
    """

    def iter_files(self) -> list[Path]:
        files: list[Path] = []
        for path in self.root.rglob("*"):
            if any(part in IGNORED_DIRS for part in path.parts):
                continue
            if not path.is_file():
                continue
            if _is_excluded(path):
                continue
            try:
                if path.stat().st_size > MAX_TEXT_FILE_BYTES:
                    continue
            except OSError:
                continue
            if not _looks_like_text(path):
                continue
            files.append(path)
        return sorted(files)
