"""Tests for the repo-QA Dense Retrieval baseline
(ant.agents.dense_retrieval_repo). Embedding calls are REAL (local, free --
fastembed/ONNX, no network, no API cost) so ranking behavior is genuinely
exercised, not mocked away; only the LLM answer-generation call site
(CountingOpenAIProvider) is monkeypatched, the same mock-the-external-
boundary convention used throughout this suite. These MUST pass before any
paid call against RepoProbe-Python or SWE-QA-Pro.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ant.agents import dense_retrieval_repo as dense_module
from ant.agents.dense_retrieval_repo import TOP_K, DenseRetrievalRepoAgent
from ant.benchmarks.base import TaskExample


class _RecordingProvider:
    """Records the (question, evidence) args synthesize() receives -- lets
    tests assert exactly what reached the LLM, without a real API call."""

    def __init__(self) -> None:
        self.synthesize_calls: list[dict] = []
        self._calls = 0

    def synthesize(self, *, question, evidence, **kwargs):
        self.synthesize_calls.append({"question": question, "evidence": list(evidence)})
        self._calls += 1
        return "a synthesized answer"

    def drain_call_count(self) -> int:
        count = self._calls
        self._calls = 0
        return count

    def drain_usage(self):
        from ant.domain import TokenUsage

        return TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)

    def drain_retry_log(self) -> list[dict]:
        return []


def _write_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    files: dict[str, str],
    question: str,
    index_root: Path | None = None,
):
    root = _write_repo(tmp_path / "repo", files)
    provider = _RecordingProvider()
    monkeypatch.setattr(dense_module, "CountingOpenAIProvider", lambda model: provider)
    example = TaskExample(
        benchmark="test_repo_bench", task_id="t1", question=question, reference=""
    )
    agent = DenseRetrievalRepoAgent(index_root=index_root or (tmp_path / ".ant-dense"))
    result = agent.run(example, root)
    return result, provider


# --- structural invariants: no BM25/RRF/ReAct/ANTMAN coordination imports ---


def test_module_never_imports_bm25_rrf_or_search_tool() -> None:
    names = set(vars(dense_module))
    forbidden_names = (
        "BM25Index",
        "_bm25_channel_rank",
        "_reciprocal_rank_fusion",
        "LocalSearchTool",
    )
    for forbidden in forbidden_names:
        assert forbidden not in names


def test_module_never_imports_antman_coordination_or_react() -> None:
    names = set(vars(dense_module))
    for forbidden in ("LocalCoordinator", "NeedGraph", "AutonomousWorker", "evolve_workers"):
        assert forbidden not in names


# --- behavioral: ranking is genuinely embedding-based, not lexical ---


def test_ranking_is_semantic_not_lexical(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    files = {
        "a.txt": "The feline slept peacefully on the warm windowsill all afternoon.",
        "b.txt": "Cats cats cats: a numeric list of cats -- cats cats cats cats cats cats.",
    }
    result, provider = _run(monkeypatch, tmp_path, files, "Where did the cat rest?")
    assert provider.synthesize_calls, "synthesize() must have been called"
    evidence = provider.synthesize_calls[0]["evidence"]
    assert evidence, "dense retrieval must return at least one hit"
    assert evidence[0].path == "a.txt"
    for e in evidence:
        assert "Dense semantic match" in e.reason


# --- corpus coverage: not .py-only, matches Sparse Retrieval's file universe ---


def test_corpus_is_not_py_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A non-.py file (.rst, matching sphinx's real content) must be
    # searchable -- this is the exact gap build_symbol_index/dense_search
    # would silently introduce if reused here instead of _retrieval_regions
    # over EvalRepoEnvironment.iter_files().
    files = {
        "docs/guide.rst": "The quantum circuit simulator supports noise modeling.",
        "src/other.py": "def unrelated():\n    return 1\n",
    }
    result, provider = _run(
        monkeypatch, tmp_path, files, "Does the simulator support noise modeling?"
    )
    evidence = provider.synthesize_calls[0]["evidence"]
    # Path.relative_to(...) yields OS-native separators (backslash on
    # Windows) -- normalize before comparing, not a corpus-coverage issue.
    assert any(Path(e.path).as_posix() == "docs/guide.rst" for e in evidence)


# --- top-k parity with Sparse Retrieval ---


def test_top_k_matches_sparse_retrieval_default() -> None:
    assert TOP_K == 8
    assert DenseRetrievalRepoAgent().top_k == 8


def test_retrieves_at_most_top_k_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    text = "\n\n".join(f"Paragraph number {i} talks about topic {i}." for i in range(12))
    files = {"a.txt": text}
    result, provider = _run(monkeypatch, tmp_path, files, "What does paragraph five discuss?")
    evidence = provider.synthesize_calls[0]["evidence"]
    assert len(evidence) <= 8


# --- no gold/reference leakage into generation ---


def test_no_gold_reference_reaches_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = "SECRET_GOLD_VALUE_MUST_NEVER_APPEAR"
    root = _write_repo(
        tmp_path / "repo", {"a.txt": "The feline slept peacefully on the warm windowsill."}
    )
    provider = _RecordingProvider()
    monkeypatch.setattr(dense_module, "CountingOpenAIProvider", lambda model: provider)
    example = TaskExample(
        benchmark="test_repo_bench",
        task_id="t1",
        question="Where did the cat rest?",
        reference=secret,
        metadata={"checklist": secret},
    )
    agent = DenseRetrievalRepoAgent(index_root=tmp_path / ".ant-dense")
    agent.run(example, root)
    for call in provider.synthesize_calls:
        assert secret not in call["question"]
        for e in call["evidence"]:
            assert secret not in e.quote
            assert secret not in e.reason


# --- same answer-generation contract as Sparse Retrieval (RetrievalAgent) ---


def test_uses_exactly_one_synthesize_call_no_condensation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    files = {"a.txt": "The feline slept peacefully on the warm windowsill all afternoon."}
    result, provider = _run(monkeypatch, tmp_path, files, "Where did the cat rest?")
    assert len(provider.synthesize_calls) == 1
    assert result.final_answer == "a synthesized answer"
    assert result.usage.llm_calls == 1  # synthesize only -- no condense_to_answer_span call


# --- per-repo disk-cached index: reused across questions against the same repo ---


def test_index_is_cached_and_reused_across_questions_against_the_same_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    embed_calls: list[list[str]] = []
    real_embed = dense_module.DenseEmbedder.embed

    def _counting_embed(self, texts):
        embed_calls.append(list(texts))
        return real_embed(self, texts)

    monkeypatch.setattr(dense_module.DenseEmbedder, "embed", _counting_embed)
    files = {"a.txt": "The feline slept peacefully on the warm windowsill all afternoon."}
    index_root = tmp_path / ".ant-dense"

    _run(monkeypatch, tmp_path, files, "Where did the cat rest?", index_root=index_root)
    n_calls_after_first = len(embed_calls)
    _run(monkeypatch, tmp_path, files, "A completely different question?", index_root=index_root)
    n_calls_after_second = len(embed_calls)

    # Second run's corpus-build should be a cache hit: only the query
    # embedding call happens, not a full re-embed of the file's chunks.
    assert n_calls_after_second == n_calls_after_first + 1


def test_index_is_cached_even_when_the_repo_contains_an_empty_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Regression test for a real bug found live on RepoProbe-Python: the
    # cache-hit check used to compare {entry.path for entry in
    # cached.entries} against the current file set -- but a file that is
    # empty/whitespace-only (e.g. a package's __init__.py) never produces
    # any _retrieval_regions chunk, so it never appears as an entry path
    # even though it legitimately belongs to the file universe. That made
    # the equality check fail on every single call, silently re-embedding
    # the whole repo from scratch for every question against it.
    embed_calls: list[list[str]] = []
    real_embed = dense_module.DenseEmbedder.embed

    def _counting_embed(self, texts):
        embed_calls.append(list(texts))
        return real_embed(self, texts)

    monkeypatch.setattr(dense_module.DenseEmbedder, "embed", _counting_embed)
    files = {
        "a.txt": "The feline slept peacefully on the warm windowsill all afternoon.",
        "pkg/__init__.py": "",  # empty -- zero _retrieval_regions chunks
    }
    index_root = tmp_path / ".ant-dense"

    _run(monkeypatch, tmp_path, files, "Where did the cat rest?", index_root=index_root)
    n_calls_after_first = len(embed_calls)
    _run(monkeypatch, tmp_path, files, "A completely different question?", index_root=index_root)
    n_calls_after_second = len(embed_calls)

    assert n_calls_after_second == n_calls_after_first + 1, (
        "cache was not reused -- the whole repo was re-embedded despite an unchanged file set"
    )


def test_index_cache_rebuilds_when_the_repo_file_set_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index_root = tmp_path / ".ant-dense"
    _run(
        monkeypatch,
        tmp_path,
        {"a.txt": "The feline slept peacefully on the warm windowsill."},
        "Where did the cat rest?",
        index_root=index_root,
    )
    result, provider = _run(
        monkeypatch,
        tmp_path,
        {
            "a.txt": "The feline slept peacefully on the warm windowsill.",
            "b.txt": "Quarterly revenue increased due to strong regional sales growth.",
        },
        "What drove revenue growth?",
        index_root=index_root,
    )
    evidence = provider.synthesize_calls[0]["evidence"]
    assert any(e.path == "b.txt" for e in evidence)


# --- determinism ---


def test_retrieval_is_deterministic_across_repeated_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    files = {
        "a.txt": "The feline slept peacefully on the warm windowsill all afternoon.",
        "b.txt": "Quarterly revenue increased due to strong regional sales growth.",
    }
    result1, provider1 = _run(
        monkeypatch, tmp_path, files, "Where did the cat rest?", index_root=tmp_path / "idx1"
    )
    result2, provider2 = _run(
        monkeypatch, tmp_path, files, "Where did the cat rest?", index_root=tmp_path / "idx2"
    )
    ids1 = [
        e.path + f":{e.line_start}-{e.line_end}" for e in provider1.synthesize_calls[0]["evidence"]
    ]
    ids2 = [
        e.path + f":{e.line_start}-{e.line_end}" for e in provider2.synthesize_calls[0]["evidence"]
    ]
    scores1 = [round(e.dense_score, 6) for e in provider1.synthesize_calls[0]["evidence"]]
    scores2 = [round(e.dense_score, 6) for e in provider2.synthesize_calls[0]["evidence"]]
    assert ids1 == ids2
    assert scores1 == scores2


def test_agent_is_registered() -> None:
    from ant.evaluation_suite.registry import get_agent

    agent = get_agent("dense_retrieval")
    assert isinstance(agent, DenseRetrievalRepoAgent)
