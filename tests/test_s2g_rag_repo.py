"""Tests for the repo-QA S2G-RAG adapter (ant.external_wrappers.s2g_rag_repo).

Same discipline as tests/test_s2g_rag.py: the ONLY mocked boundary is the
LLM call site. Retrieval is REAL -- a real `LocalSearchTool` BM25+RRF pass
over a real, on-disk miniature repository fixture built in `tmp_path` and
discovered through the real `EvalRepoEnvironment` -- because it is local
and free. Zero paid calls.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from pathlib import Path

import pytest

from ant.benchmarks.base import TaskExample
from ant.external_wrappers import s2g_rag as s2g_module
from ant.external_wrappers import s2g_rag_repo as repo_module
from ant.external_wrappers.s2g_rag import (
    MAX_GAP_FACTS_FOR_QUERY,
    MAX_TURNS,
    TOP_DOCS,
    S2GRAGAdapter,
    build_query_from_missing,
    split_wiki_sentences,
)
from ant.external_wrappers.s2g_rag_repo import (
    RepoChunk,
    RepoCorpusRetriever,
    S2GRAGRepoAdapter,
)

# ===========================================================================
# A real miniature repository, on disk. Files are topically overlapping (as
# a real repo's are) so BM25 genuinely ranks and genuinely returns a full
# top-6 of distinct chunks.
# ===========================================================================

QUESTION = "How does the scheduler decide which task to run next?"

REPO_FILES: dict[str, str] = {
    "pkg/scheduler.py": '''
"""Task scheduler entry points."""

import heapq


class Scheduler:
    """Chooses which task runs next."""

    def __init__(self, policy):
        self.policy = policy
        self._heap = []

    def submit(self, task):
        """Add a task to the pending heap."""
        heapq.heappush(self._heap, (self.policy.rank(task), task))

    def next_task(self):
        """Pop the highest-priority task from the heap."""
        if not self._heap:
            return None
        _, task = heapq.heappop(self._heap)
        return task
''',
    "pkg/policy.py": '''
"""Ranking policies used by the scheduler to order tasks."""


class PriorityPolicy:
    """Ranks a task by its declared priority, lower runs first."""

    def rank(self, task):
        """Return the sort key the scheduler heap orders on."""
        return (-task.priority, task.submitted_at)


class FifoPolicy:
    """Ranks purely by submission order."""

    def rank(self, task):
        return task.submitted_at
''',
    "pkg/task.py": '''
"""Task model consumed by the scheduler."""


class Task:
    """A unit of work with a priority the policy reads."""

    def __init__(self, name, priority, submitted_at):
        self.name = name
        self.priority = priority
        self.submitted_at = submitted_at
''',
    "pkg/worker.py": '''
"""Worker loop that drains the scheduler."""


class Worker:
    """Pulls the next task from a scheduler and runs it."""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def run_once(self):
        """Run a single task if the scheduler has one pending."""
        task = self.scheduler.next_task()
        if task is None:
            return False
        return True
''',
    "docs/scheduling.md": """
# Scheduling

The scheduler decides which task to run next by asking the configured
policy to rank each task. The default policy ranks by priority.

## Policies

A policy exposes a single `rank` method. The scheduler never inspects a
task directly; all ordering decisions go through the policy.
""",
    "docs/overview.md": """
# Overview

This package contains a scheduler, a worker, and a task model. See the
scheduling document for how the next task is chosen.
""",
    "pkg/metrics.py": '''
"""Counters describing scheduler behaviour."""

TASKS_SUBMITTED = "tasks_submitted_total"
TASKS_RUN = "tasks_run_total"


def record_submit(registry):
    """Increment the submitted-task counter."""
    registry.increment(TASKS_SUBMITTED)
''',
    "pkg/config.py": '''
"""Configuration for the scheduler and its policy."""

DEFAULT_POLICY = "priority"


def load_policy_name(env):
    """Read which policy the scheduler should run with."""
    return env.get("SCHEDULER_POLICY", DEFAULT_POLICY)
''',
    "README.md": """
# minirepo

A miniature scheduler package used to exercise retrieval.
""",
    "pkg/__init__.py": '"""Package marker."""\n',
}

# A gold-bearing string that must never reach any prompt. Placed in the
# metadata fields this track's benchmarks actually carry.
SECRET = "SECRET_CHECKLIST_VALUE_MUST_NEVER_APPEAR"

JUDGE_INSUFFICIENT = json.dumps(
    {
        "sufficient": False,
        "gap_items": [
            {
                "category": "bridge_entity",
                "target": "Scheduler",
                "slot": "next_task",
                "description": "which component the scheduler asks to order tasks",
            }
        ],
    }
)
JUDGE_SUFFICIENT = json.dumps({"sufficient": True, "gap_items": []})
SELECTOR_PICK = json.dumps({"evidence_global_ids": [1, 2, 3]})
ANSWER_TEXT = (
    "Answer: The scheduler delegates ordering to the configured policy's rank method.\n"
    "Rationale: pkg/scheduler.py pushes (policy.rank(task), task) onto a heap and "
    "next_task pops the smallest key, so pkg/policy.py's PriorityPolicy.rank decides "
    "the order."
)


class _ScriptedProvider:
    """Same convention as tests/test_s2g_rag.py's own stub, with a hard
    call cap so a runaway loop fails loudly instead of hanging."""

    def __init__(self, scripted: list[str], *, cap: int = 64) -> None:
        self._scripted = list(scripted)
        self._calls = 0
        self._cap = cap
        self.prompts: list[str] = []

    def responses_text(self, prompt: str, max_output_tokens: int = 512):
        from ant.domain import TokenUsage
        from ant.providers.openai_provider import ResponseResult

        self._calls += 1
        if self._calls > self._cap:
            raise AssertionError(f"runaway loop: more than {self._cap} LLM calls")
        self.prompts.append(prompt)
        text = self._scripted.pop(0) if self._scripted else self._default_for(prompt)
        return ResponseResult(text=text, usage=TokenUsage(), raw={})

    def _default_for(self, prompt: str) -> str:
        if prompt.startswith(s2g_module.SUFF_SYSTEM_PROMPT):
            return JUDGE_INSUFFICIENT
        if prompt.startswith(s2g_module.SELECTOR_SYSTEM_PROMPT):
            return SELECTOR_PICK
        return ANSWER_TEXT

    def drain_call_count(self) -> int:
        count = self._calls
        self._calls = 0
        return count

    def drain_usage(self):
        from ant.domain import TokenUsage

        return TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)

    def judge_prompts(self) -> list[str]:
        return [p for p in self.prompts if p.startswith(s2g_module.SUFF_SYSTEM_PROMPT)]

    def selector_prompts(self) -> list[str]:
        return [p for p in self.prompts if p.startswith(s2g_module.SELECTOR_SYSTEM_PROMPT)]

    def answer_prompts(self) -> list[str]:
        return [p for p in self.prompts if p.startswith(repo_module.repo_force_answer_prompt)]


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "minirepo"
    for relative, text in REPO_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.lstrip("\n"), encoding="utf-8")
    return root.resolve()


def _example(benchmark: str = "repoprobe", metadata: dict | None = None) -> TaskExample:
    return TaskExample(
        benchmark=benchmark,
        task_id="minirepo-0",
        question=QUESTION,
        reference=SECRET,
        metadata={"repo": "acme/minirepo", "repo_short_name": "minirepo", **(metadata or {})},
    )


def _run(monkeypatch, provider, repo_root: Path, **kwargs):
    monkeypatch.setattr(s2g_module, "_ZeroTemperatureProvider", lambda model: provider)
    return S2GRAGRepoAdapter(**kwargs).run(_example(), repo_root)


def _paths(doc_ids: list[str]) -> set[str]:
    """Source file of each chunk id, with separators normalized. Chunk ids
    embed whatever separator `EvalRepoEnvironment` produced (a backslash on
    Windows) -- identical to what every other repo-QA baseline's evidence
    `path` carries, so this is a test-comparison concern, not a defect."""
    return {d.rsplit(":", 1)[0].replace("\\", "/") for d in doc_ids}


# ===========================================================================
# Structural: ONE algorithm, not a fork. Registration. Isolation.
# ===========================================================================


def test_repo_adapter_is_a_subclass_not_a_fork() -> None:
    assert issubclass(S2GRAGRepoAdapter, S2GRAGAdapter)
    # The turn loop and both judge/extractor roles are INHERITED, never
    # redefined -- this is the guard against the two substrates drifting.
    for shared in ("run", "_judge", "_select_evidence", "_answer"):
        assert shared not in vars(S2GRAGRepoAdapter), (
            f"{shared} must stay inherited; overriding it would fork the algorithm"
        )
    # Only the three declared substrate hooks (plus identity/config) differ.
    assert set(vars(S2GRAGRepoAdapter)) >= {
        "_build_retriever",
        "_assemble_final_answer",
        "ANSWER_SYSTEM_PROMPT",
    }


def test_both_agents_registered_under_distinct_names() -> None:
    from ant.evaluation_suite.registry import get_agent

    assert isinstance(get_agent("s2g_rag_repo"), S2GRAGRepoAdapter)
    assert get_agent("s2g_rag").name == "s2g_rag"
    assert S2GRAGRepoAdapter.name != S2GRAGAdapter.name


def test_frozen_algorithm_hyperparameters_are_shared_with_the_document_track() -> None:
    agent = S2GRAGRepoAdapter()
    assert (agent.max_turns, agent.top_docs, agent.evidence_top_k) == (4, 6, 6)
    assert MAX_GAP_FACTS_FOR_QUERY == 3
    assert agent.ANSWER_SYSTEM_PROMPT is repo_module.repo_force_answer_prompt
    assert agent.answer_max_output_tokens == 8192


def _imported_module_paths(module) -> set[str]:
    """Every module path this module actually imports, via AST -- immune to
    whatever its prose docstring happens to mention."""
    tree = ast.parse(inspect.getsource(module))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_module_never_couples_to_graphrag_repodistill_or_antman_internals() -> None:
    names = set(vars(repo_module))
    for forbidden in (
        "LocalCoordinator",
        "NeedGraph",
        "AutonomousWorker",
        "evolve_workers",
        "RepoGraph",
        "GraphRAG",
        "RepoDistill",
        "build_embedding_index",
    ):
        assert forbidden not in names
    # Nothing is imported from a graph/RepoDistill/coordination module --
    # checked against real import statements, not prose.
    for imported in _imported_module_paths(repo_module):
        lowered = imported.lower()
        assert "repodistill" not in lowered
        assert "graphrag" not in lowered
        assert "repograph" not in lowered
        assert "coordinator" not in lowered
    # The retrieval primitive really is the shared one every repo-QA
    # baseline uses, not a private index.
    assert "ant.tools.local" in _imported_module_paths(repo_module)
    assert "ant.evaluation_suite.repo_scope" in _imported_module_paths(repo_module)


def test_no_max_output_tokens_below_openai_api_minimum() -> None:
    source = inspect.getsource(repo_module)
    for match in re.finditer(r"MAX_OUTPUT_TOKENS\s*=\s*(\d+)", source):
        assert int(match.group(1)) >= 16


# ===========================================================================
# Shared candidate pool: identical to what the repo-QA Sparse baseline sees.
# ===========================================================================


def test_file_universe_is_identical_to_the_sparse_repo_baselines(repo_root: Path) -> None:
    """The fairness requirement, asserted against the real thing rather than
    by reading comments: the file universe is byte-identical to the one
    `ant.agents.retrieval.RetrievalAgent` builds for the same repository."""
    from ant.evaluation_suite.repo_scope import EvalRepoEnvironment

    environment = EvalRepoEnvironment(repo_root)
    baseline_files = [str(p.relative_to(environment.root)) for p in environment.iter_files()]
    retriever = RepoCorpusRetriever(repo_root)
    assert retriever.files == baseline_files
    assert len(retriever.files) == len(REPO_FILES)


def test_retrieval_unit_is_a_retrieval_regions_chunk_not_a_whole_file(
    repo_root: Path,
) -> None:
    retriever = RepoCorpusRetriever(repo_root)
    titles, texts, doc_ids = retriever.search(QUESTION, [], k=TOP_DOCS)
    assert len(doc_ids) == TOP_DOCS
    for doc_id in doc_ids:
        assert re.fullmatch(r".+:\d+-\d+", doc_id), doc_id
    # A chunk, not the file: strictly smaller than its own source file.
    path = doc_ids[0].rsplit(":", 1)[0]
    whole_file = (repo_root / path).read_text(encoding="utf-8")
    assert len(texts[0]) < len(whole_file)
    # RepoChunk structurally carries no relevance/gold field, exactly like
    # DocumentRecord on the document track.
    assert {f.name for f in RepoChunk.__dataclass_fields__.values()} == {
        "doc_id",
        "title",
        "text",
    }


def test_retriever_returns_the_upstream_sentinel_when_nothing_matches(
    repo_root: Path,
) -> None:
    retriever = RepoCorpusRetriever(repo_root)
    titles, texts, doc_ids = retriever.search("zzzzqqqxxnomatchtoken", [], k=TOP_DOCS)
    assert titles == ["No results found."]
    assert texts == ["No results found."]
    assert doc_ids == [""]


def test_split_wiki_sentences_gives_line_level_units_on_code() -> None:
    """Disclosed emergent behavior, pinned by a test so it can never change
    silently: on code the `[\\n]+` alternative makes each line a unit."""
    chunk = "def rank(self, task):\n    return (-task.priority, task.submitted_at)"
    assert split_wiki_sentences(chunk) == [
        "def rank(self, task):",
        "return (-task.priority, task.submitted_at)",
    ]


# ===========================================================================
# 1. Initial retrieval against the real repository fixture.
# ===========================================================================


def test_initial_retrieval_runs_against_the_shared_repository_corpus(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    provider = _ScriptedProvider([JUDGE_INSUFFICIENT, SELECTOR_PICK, JUDGE_SUFFICIENT, ANSWER_TEXT])
    result = _run(monkeypatch, provider, repo_root)

    first = result.trajectory[0]
    assert first["turn"] == 0
    assert len(first["retrieved_doc_ids"]) == TOP_DOCS
    # BM25 really ranked: the scheduler/policy/scheduling sources outrank
    # README/__init__ noise.
    assert _paths(first["retrieved_doc_ids"]) & {
        "pkg/scheduler.py",
        "pkg/policy.py",
        "docs/scheduling.md",
    }
    assert result.usage.tool_calls == 1
    assert result.metadata["retrieval_is_local_and_free"] is True
    assert result.metadata["corpus_substrate"] == "repository_chunks"


# ===========================================================================
# 2. Insufficiency triggers another retrieval round.
# ===========================================================================


def test_insufficient_judgment_triggers_a_second_repo_retrieval_round(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    provider = _ScriptedProvider(
        [
            JUDGE_INSUFFICIENT,
            SELECTOR_PICK,
            JUDGE_INSUFFICIENT,
            SELECTOR_PICK,
            JUDGE_SUFFICIENT,
            ANSWER_TEXT,
        ]
    )
    result = _run(monkeypatch, provider, repo_root)

    retrieving = [t for t in result.trajectory if "retrieval_query" in t]
    assert len(retrieving) == 2
    assert result.metadata["n_judge_calls"] == 3
    assert result.metadata["n_selector_calls"] == 2
    assert result.metadata["n_retrieval_calls"] == 2
    # remove_repeat_docs: round 2 brings genuinely new chunks.
    assert not set(retrieving[0]["retrieved_doc_ids"]) & set(retrieving[1]["retrieved_doc_ids"])
    # Append-only evidence context.
    sizes = [t["evidence_context_chars"] for t in result.trajectory]
    assert sizes == sorted(sizes)
    assert sizes[-1] > 0


# ===========================================================================
# 3. Gap items become the next repo-scoped retrieval query.
# ===========================================================================


def test_gap_items_become_the_next_repo_scoped_query(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    second_gap = json.dumps(
        {
            "sufficient": False,
            "gap_items": [
                {
                    "category": "relation",
                    "target": "PriorityPolicy",
                    "slot": "rank",
                    "description": "how PriorityPolicy ranks a task",
                }
            ],
        }
    )
    provider = _ScriptedProvider(
        [
            JUDGE_INSUFFICIENT,
            SELECTOR_PICK,
            second_gap,
            SELECTOR_PICK,
            JUDGE_SUFFICIENT,
            ANSWER_TEXT,
        ]
    )
    result = _run(monkeypatch, provider, repo_root)

    assert result.trajectory[0]["retrieval_query"] == f"{QUESTION} Scheduler next_task"
    assert result.trajectory[1]["retrieval_query"] == f"{QUESTION} PriorityPolicy rank"


def test_repo_gap_phrases_genuinely_change_the_bm25_ranking(repo_root: Path) -> None:
    """Behavioral, not structural: the target+slot phrase really moves the
    policy source into the top-2 where the bare question does not."""
    retriever = RepoCorpusRetriever(repo_root, remove_repeat_docs=False)
    _, _, bare = retriever.search("How is a unit of work ordered?", [], k=2)
    _, _, steered = retriever.search(
        build_query_from_missing(
            "How is a unit of work ordered?",
            [{"target": "PriorityPolicy", "slot": "rank", "description": ""}],
        ),
        [],
        k=2,
    )
    assert "pkg/policy.py" not in _paths(bare)
    assert "pkg/policy.py" in _paths(steered)


def test_gap_to_query_mechanism_is_the_shared_k3_one() -> None:
    gaps = [{"target": f"T{i}", "slot": f"s{i}", "description": f"d{i}"} for i in range(5)]
    assert build_query_from_missing(QUESTION, gaps) == f"{QUESTION} T0 s0 T1 s1 T2 s2"
    fallback = [{"target": "", "slot": "", "description": "the ranking policy"}]
    assert build_query_from_missing(QUESTION, fallback) == f"{QUESTION} the ranking policy"
    assert build_query_from_missing(QUESTION, []) == QUESTION


# ===========================================================================
# 4. Sufficient evidence terminates before the max-turn cap.
# ===========================================================================


def test_sufficient_judgment_terminates_before_the_repo_max_turn_cap(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    provider = _ScriptedProvider([JUDGE_INSUFFICIENT, SELECTOR_PICK, JUDGE_SUFFICIENT, ANSWER_TEXT])
    result = _run(monkeypatch, provider, repo_root)

    assert result.termination_reason == "judge_declared_sufficient"
    assert result.metadata["turns_used"] == 2 < MAX_TURNS + 1
    assert result.metadata["n_judge_calls"] == 2
    assert result.metadata["n_retrieval_calls"] == 1
    assert len(provider.answer_prompts()) == 1


# ===========================================================================
# 5. The max-turn cap terminates the loop (no hang).
# ===========================================================================


def test_max_turn_cap_terminates_on_a_repo_corpus(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    provider = _ScriptedProvider([], cap=64)
    result = _run(monkeypatch, provider, repo_root)

    assert result.termination_reason == "max_turns_reached"
    assert result.metadata["turns_used"] == MAX_TURNS + 1 == 5
    assert result.metadata["n_judge_calls"] == 5
    assert result.metadata["n_retrieval_calls"] == MAX_TURNS == 4
    assert result.metadata["n_answer_calls"] == 1
    assert len(provider.answer_prompts()) == 1
    # UNLIKE the 10-document multi-hop QA pools, a repository does NOT
    # exhaust after two rounds -- there are far more chunks than 4 x 6, so
    # every retrieval round is productive and every one costs an Extractor
    # call. This is the key repo-vs-document cost difference.
    assert result.metadata["n_selector_calls"] == MAX_TURNS == 4
    assert len(provider.prompts) == 5 + 4 + 1 == 10


def test_repo_pool_does_not_exhaust_across_all_four_rounds(repo_root: Path) -> None:
    retriever = RepoCorpusRetriever(repo_root)
    seen: list[str] = []
    for _ in range(MAX_TURNS):
        _, _, doc_ids = retriever.search(QUESTION, seen, k=TOP_DOCS)
        assert doc_ids != [""], "repo pool exhausted early -- fixture too small"
        seen.extend(doc_ids)
    assert len(seen) == len(set(seen)) == MAX_TURNS * TOP_DOCS


# ===========================================================================
# 6. Repo-QA's own gold-leakage surface: reference AND RepoProbe's checklist.
# ===========================================================================


def test_no_gold_answer_or_checklist_metadata_reaches_any_repo_prompt(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    provider = _ScriptedProvider([], cap=64)
    monkeypatch.setattr(s2g_module, "_ZeroTemperatureProvider", lambda model: provider)

    example = TaskExample(
        benchmark="repoprobe",
        task_id="minirepo-0",
        question=QUESTION,
        reference=SECRET,  # gold answer
        metadata={
            "repo": "acme/minirepo",
            "repo_short_name": "minirepo",
            "commit": "deadbeef",
            "checklist": SECRET,  # RepoProbe's gold scoring rubric
            "taxonomy": SECRET,
            "difficulty": SECRET,
            "cluster": SECRET,
            "qa_type": SECRET,
        },
    )
    result = S2GRAGRepoAdapter().run(example, repo_root)

    assert provider.prompts, "vacuous unless prompts were built"
    for prompt in provider.prompts:
        assert SECRET not in prompt
    assert SECRET not in json.dumps(result.model_dump(), default=str)


def test_repo_run_path_reads_no_gold_bearing_field_at_all() -> None:
    """Structural: the repo substrate's own `_build_retriever` reads NOTHING
    off the TaskExample (the whole candidate pool is `environment_root`),
    and the inherited run() reads only question/benchmark/task_id."""
    hook_source = inspect.getsource(S2GRAGRepoAdapter._build_retriever)
    assert not re.search(r"example\.metadata\[", hook_source)
    assert not re.search(r"example\.reference", hook_source)

    run_source = inspect.getsource(S2GRAGAdapter.run) + hook_source
    accessed = set(re.findall(r"example\.(\w+)", run_source))
    assert accessed <= {"metadata", "question", "benchmark", "task_id"}
    assert "reference" not in accessed

    # AST-based, so the module's own prose (which legitimately NAMES these
    # fields to explain that they are never read) cannot confound it: no
    # real subscript and no real .get() anywhere in this module addresses a
    # gold-bearing key, and no attribute access reads `.reference`.
    forbidden = {"checklist", "reference", "taxonomy", "difficulty", "cluster", "qa_type"}
    tree = ast.parse(inspect.getsource(repo_module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            assert node.slice.value not in forbidden, f"subscript reads {node.slice.value!r}"
        if isinstance(node, ast.Attribute):
            assert node.attr != "reference"
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            assert node.args[0].value not in forbidden


def test_retrieved_evidence_never_carries_a_relevance_label(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    provider = _ScriptedProvider([JUDGE_INSUFFICIENT, SELECTOR_PICK, JUDGE_SUFFICIENT, ANSWER_TEXT])
    result = _run(monkeypatch, provider, repo_root)
    for item in result.evidence:
        assert set(item) == {"doc_id", "title"}


# ===========================================================================
# 7. Predictions reach the existing repo-QA scorers with the right contract.
# ===========================================================================


def test_prediction_reaches_the_repoprobe_scorer_with_the_right_contract(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    from ant.benchmarks import repoprobe as repoprobe_module

    provider = _ScriptedProvider([JUDGE_INSUFFICIENT, SELECTOR_PICK, JUDGE_SUFFICIENT, ANSWER_TEXT])
    result = _run(monkeypatch, provider, repo_root)

    seen: list[dict] = []

    def fake_call_judge(*, system: str, user: str, max_output_tokens: int = 1024):
        seen.append({"system": system, "user": user})

        class _R:
            text = json.dumps(
                {"knowledge_score": 7.0, "clarity_score": 1.0, "total_score": 8.0,
                 "knowledge_max": 9.0, "clarity_max": 1.0, "hallucination": False}
            )
            estimated_cost_usd = 0.0

        return _R()

    monkeypatch.setattr(repoprobe_module, "call_judge", fake_call_judge)
    monkeypatch.setattr(repoprobe_module, "_load_scoring_template", lambda: (
        "{description}|{repo_info_section}|{reference_text}|{scoring_criteria}|"
        "{model_name}|{model_answer}"
    ))
    adapter = repoprobe_module.RepoProbeAdapter()
    monkeypatch.setattr(adapter, "prepare_environment", lambda example: repo_root)
    monkeypatch.setattr(repoprobe_module, "build_directory_structure", lambda root: "tree")

    example = _example(benchmark="repoprobe", metadata={"checklist": "a checklist"})
    metric = adapter.score(example, result)

    # The judge saw exactly this method's own final_answer, under this
    # method's own name, and the suite's 3x stabilization protocol ran.
    assert len(seen) == 3
    assert result.final_answer in seen[0]["user"]
    assert "s2g_rag_repo" in seen[0]["user"]
    assert metric.benchmark == "repoprobe"
    assert metric.task_id == "minirepo-0"
    assert metric.native_score == 8.0
    assert metric.normalized_score == 80.0
    assert metric.metadata["generation_model"] == "gpt-4.1"


def test_prediction_reaches_the_sweqa_pro_scorer_with_the_right_contract(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    from ant.benchmarks import sweqa_pro as sweqa_module

    provider = _ScriptedProvider([JUDGE_INSUFFICIENT, SELECTOR_PICK, JUDGE_SUFFICIENT, ANSWER_TEXT])
    monkeypatch.setattr(s2g_module, "_ZeroTemperatureProvider", lambda model: provider)
    example = _example(benchmark="sweqa_pro")
    result = S2GRAGRepoAdapter().run(example, repo_root)

    seen: list[str] = []

    def fake_call_judge(*, system: str, user: str, max_output_tokens: int = 1024):
        seen.append(user)

        class _R:
            text = json.dumps(
                {"correctness": 8, "completeness": 7, "relevance": 9,
                 "clarity": 8, "reasoning": 7}
            )
            estimated_cost_usd = 0.0

        return _R()

    monkeypatch.setattr(sweqa_module, "call_judge", fake_call_judge)
    metric = sweqa_module.SweQaProAdapter().score(example, result)

    assert len(seen) == 3
    assert result.final_answer in seen[0]
    assert metric.benchmark == "sweqa_pro"
    assert metric.native_score == 39.0  # 8+7+9+8+7
    assert set(metric.submetrics) == {
        "correctness", "completeness", "relevance", "clarity", "reasoning"
    }
    assert metric.metadata["generation_model"] == "gpt-4.1"


def test_final_answer_keeps_the_rationale_for_rubric_judges(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    provider = _ScriptedProvider([JUDGE_INSUFFICIENT, SELECTOR_PICK, JUDGE_SUFFICIENT, ANSWER_TEXT])
    result = _run(monkeypatch, provider, repo_root)
    assert result.final_answer.startswith(
        "The scheduler delegates ordering to the configured policy's rank method."
    )
    assert "pkg/scheduler.py" in result.final_answer  # the rationale survived
    # The document track still drops the rationale (EM/F1 scores a span).
    assert S2GRAGAdapter()._assemble_final_answer("span", "why", "raw") == "span"


def test_unparseable_answer_falls_back_to_the_raw_response(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    raw = "The scheduler asks the policy to rank tasks."  # no Answer:/Rationale:
    provider = _ScriptedProvider([JUDGE_INSUFFICIENT, SELECTOR_PICK, JUDGE_SUFFICIENT, raw])
    result = _run(monkeypatch, provider, repo_root)
    assert result.final_answer == raw


def test_agent_result_shape_matches_the_suite_contract(
    monkeypatch: pytest.MonkeyPatch, repo_root: Path
) -> None:
    provider = _ScriptedProvider([JUDGE_INSUFFICIENT, SELECTOR_PICK, JUDGE_SUFFICIENT, ANSWER_TEXT])
    result = _run(monkeypatch, provider, repo_root)

    assert result.method == "s2g_rag_repo"
    assert result.benchmark == "repoprobe"
    assert result.usage.llm_calls == 4
    assert result.usage.tool_calls == 1
    json.dumps(result.model_dump(), default=str)
    official = {d["official_setting"] for d in result.metadata["deviations"]}
    assert any(s.startswith("S2G-Judge is Llama") for s in official)
    assert any("top_docs=6 counts" in s for s in official)
    # The document track's Wikipedia-corpus entry is replaced, not stacked.
    wiki_doc_track_entry = "Retrieval over a whole-Wikipedia Pyserini BM25 index (or"
    assert not any(s.startswith(wiki_doc_track_entry) for s in official)
