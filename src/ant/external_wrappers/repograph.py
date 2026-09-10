from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ant.benchmarks.base import TaskExample
from ant.domain import Evidence

REPOGRAPH_REPO_URL = "https://github.com/ozyyshr/RepoGraph"
REPOGRAPH_COMMIT = "6c3977d87845993bf2c0359b4ac752278d7f3c45"


class RepoGraphNotCheckedOut(RuntimeError):
    pass


class RepoGraphTool:
    """A callable `extra_tools["search_repograph"]` entry for
    `ant.agents.matched_react.MatchedReActAgent` -- see that class's own
    `extra_tools` docstring. This is NOT a standalone agent: RepoGraph
    itself is a repository structural-graph construction/query tool
    (nodes = def/ref lines, edges = invoke/contain relations, built via
    tree-sitter+ast, no LLM involved), never a QA system on its own --
    confirmed by direct inspection of the official repo (no independent
    "ask a question" entrypoint; the paper's own experiments plug it into
    Agentless/SWE-agent/AutoCodeRover as one more tool/context source, not
    as a replacement for the agent loop itself). The evaluated system this
    wrapper is meant to back is precisely "Matched ReAct + RepoGraph" --
    never presented as "RepoGraph" alone.

    Answers, per invocation, the question this suite's own faithful-
    integration standard requires: given a target function/class name,
    what are its immediate predecessors (callers) and successors
    (callees) in RepoGraph's own precomputed structural graph -- the same
    query shape RepoGraph's own SWE-agent integration
    (SWE-agent/config/commands/_code_graph.py's `repomap.py <dir> <func>`)
    uses in practice, reproduced faithfully against the CORE repograph/
    package's own CodeGraph object (not SWE-agent's separate near-
    duplicate RepoMap class).

    Graph construction/query is delegated to a pinned external checkout's
    own isolated venv via subprocess (`repograph/_ant_query.py`, a new
    driver script, NOT part of the official release -- runs from inside
    `repograph/`'s own directory since `construct_graph.py`'s own `from
    utils import create_structure` is a same-directory import). The graph
    itself is cached per-repo (keyed by the repo's absolute path) after
    first construction -- also a new addition, since the official
    `__main__` entrypoint always rebuilds from scratch with no cache,
    which would be far too slow to call from inside a per-step agent loop.

    FIVE real, disclosed correctness bugs in the officially released
    construct_graph.py were found and fixed with minimal, targeted
    patches to make graph construction succeed at all against a real
    multi-file repository (see
    third_party/manifests/repograph/manifest.json's local_modifications
    for the exact diffs and justification for each -- none change the
    graph's own construction ALGORITHM, only fix crashes/type mismatches
    that prevented it from completing on real repos). Confirmed live:
    graph construction now completes successfully against qibo (134
    files) and returns real query results.

    RepoGraph's own file discovery is Python-only (`find_files` filters
    to `.py`) -- a real, disclosed scope limitation for non-Python repos
    (see the manifest for what this means for RepoProbe applicability).

    FIDELITY AUDIT (found after 0/20 natural search_repograph usage on the
    SWE-QA-Pro pilot): direct inspection of the official host-agent
    integration this baseline is meant to match --
    SWE-agent/config/default.yaml -- shows it is NOT merely a tool schema
    entry. Two things are present there that this evaluation suite had
    been omitting:
      1. `system_template` appends the tool's docstring/signature/
         arguments directly after `{command_docs}` (the other tools'
         auto-generated docs) -- a schema-level parity this wrapper's
         `TOOL_DESCRIPTION` already matches in spirit (one description
         string handed to MatchedReActAgent's own `extra_tool_descriptions`
         injection point).
      2. `instance_template`'s own numbered tips list -- METHOD-INTRINSIC
         usage instruction, not just a schema entry -- says (verbatim):
         "6. Before you proceed to edit, always look up for related
         context using `search_repo` commands." and "7. Always try to use
         the most related and key FUNCTIONS or CLASSES as search terms
         when leveraging search commands." This suite's own baseline had
         NO equivalent instruction -- `search_repograph` was exposed as
         just one more tool among nine, with no guidance that it should
         be consulted before finishing or how to pick a search term. This
         is a genuine, disclosed baseline-fidelity defect, not merely "the
         model chose not to use an available tool" -- the official
         integration does not leave that choice unguided.

    Fix: `TOOL_DESCRIPTION` below is a faithful, minimal port of tips #6
    and #7 above (adapted only for terminology -- "finish and answer"
    in place of "proceed to edit", since this is a QA setting with no
    edit step, and this tool's own name in place of `search_repo`) --
    not new, invented prompting aimed at maximizing usage. Scoped
    entirely to this one baseline's own tool-description string; nothing
    about MatchedReActAgent itself, or any other baseline, changed.
    """

    name = "repograph"

    #: The exact text every orchestration script should pass as
    #: `extra_tool_descriptions={"search_repograph": RepoGraphTool.TOOL_DESCRIPTION}`
    #: -- see the FIDELITY AUDIT note above for where this text comes from
    #: and why it is not arbitrary. Defined once here so every caller (the
    #: RepoProbe-Python and SWE-QA-Pro pilots, and any future run) gets the
    #: identical, correctly-ported description rather than each
    #: hand-typing a slightly different one.
    TOOL_DESCRIPTION = (
        "query a precomputed repository structural code graph for a function or class "
        "NAME (not a natural-language query); returns its known callers, callees, and "
        "definition location from static analysis. Before you finish and answer, always "
        "look up related context using this tool. Always try to use the most related and "
        "key functions or classes as search terms."
    )

    def __init__(
        self,
        checkout_root: Path | None = None,
        venv_python: Path | None = None,
        timeout_seconds: int = 600,
    ) -> None:
        self.checkout_root = checkout_root or Path("third_party/checkouts/repograph")
        self.venv_python = venv_python or (self.checkout_root / ".venv" / "Scripts" / "python.exe")
        self.timeout_seconds = timeout_seconds

    def _require_checkout(self) -> None:
        if not (self.checkout_root / "repograph" / "construct_graph.py").exists():
            raise RepoGraphNotCheckedOut(
                f"RepoGraph is not checked out at {self.checkout_root}. To run this: "
                f"git clone {REPOGRAPH_REPO_URL} {self.checkout_root} && cd {self.checkout_root} "
                f"&& git checkout {REPOGRAPH_COMMIT}, apply the five local modifications "
                "documented in third_party/manifests/repograph/manifest.json, create an "
                "ISOLATED venv (.venv) and `pip install tree-sitter==0.21.3 "
                "tree-sitter-languages==1.10.2 grep-ast==0.3.2 networkx==3.2.1 pygments==2.18.0 "
                "tqdm diskcache`, and create repograph/_ant_query.py (see this class's own "
                "docstring)."
            )
        if not (self.checkout_root / "repograph" / "_ant_query.py").exists():
            raise RepoGraphNotCheckedOut(
                f"{self.checkout_root}/repograph/_ant_query.py is missing -- the small, "
                "not-part-of-the-official-release driver script this wrapper depends on."
            )
        if not self.venv_python.exists():
            raise RepoGraphNotCheckedOut(
                f"Checkout found at {self.checkout_root} but its own isolated venv does not "
                f"exist at {self.venv_python}."
            )

    def query(self, symbol: str, example: TaskExample, repo_dir: Path) -> list[Evidence]:
        """The `extra_tools` callable's own inner logic. Not called
        directly with the `(query, example)` two-arg signature
        `MatchedReActAgent.extra_tools` expects -- see `as_extra_tool()`
        below, which binds `repo_dir` via a closure so this matches that
        signature exactly."""
        self._require_checkout()
        payload = json.dumps({"repo_dir": str(repo_dir), "query": symbol})
        try:
            result = subprocess.run(
                [str(self.venv_python), "_ant_query.py"],
                cwd=self.checkout_root / "repograph",
                input=payload,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            # RepoGraph's own get_tags_raw() deep-copies the ENTIRE repo
            # structure dict once PER FILE processed (confirmed by direct
            # reading of construct_graph.py -- s = deepcopy(self.structure)
            # inside a per-file loop) -- a genuine, disclosed, method-
            # inherent O(n^2)-ish scalability limitation of the released
            # implementation, not something this wrapper works around
            # algorithmically. Confirmed live: construction on a
            # 2600-.py-file repo (sglang, RepoProbe-Python) did not
            # complete within a 600s timeout. A timeout here degrades to
            # "no evidence found" -- the same posture as any other tool
            # failure -- rather than crashing the whole agent run; the
            # repository stays in the evaluated set, RepoGraph simply
            # never contributes evidence for it. See
            # third_party/manifests/repograph/manifest.json for the full,
            # disclosed applicability finding.
            return []
        marker = '{"found"'
        start = result.stdout.rfind(marker)
        if result.returncode != 0 or start == -1:
            # A tool failure is real, comparable overhead for a ReAct
            # agent (same posture as a malformed decision) -- return no
            # evidence rather than crash the whole question.
            return []
        try:
            payload_out = json.loads(result.stdout[start:])
        except json.JSONDecodeError:
            return []
        if not payload_out.get("found"):
            return []

        evidence: list[Evidence] = []
        for item in payload_out.get("results", []):
            fname = item.get("fname") or ""
            try:
                rel_path = str(Path(fname).resolve().relative_to(repo_dir.resolve()))
            except ValueError:
                rel_path = fname
            line = item.get("line")
            if isinstance(line, list) and len(line) == 2:
                line_start, line_end = line
            elif isinstance(line, int):
                line_start = line_end = line
            else:
                line_start = line_end = 0
            evidence.append(
                Evidence(
                    path=rel_path,
                    line_start=max(1, int(line_start) + 1),
                    line_end=max(1, int(line_end) + 1),
                    quote=str(item.get("info", ""))[:1000],
                    reason=(
                        f"RepoGraph {item.get('relation', 'related')} of {symbol!r} "
                        f"({item.get('category', 'symbol')})"
                    ),
                )
            )
        return evidence

    def as_extra_tool(self, repo_dir: Path):
        """Binds `repo_dir` into the exact
        `Callable[[str, TaskExample], list[Evidence]]` shape
        `MatchedReActAgent(extra_tools={...})` expects."""

        def _tool(query: str, example: TaskExample) -> list[Evidence]:
            return self.query(query, example, repo_dir)

        return _tool
