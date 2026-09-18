"""Tests for the RepoDistill (No-Training) baseline
(`ant.external_wrappers.repodistill` + `_graph` + `_caba`).

WHAT IS REAL AND WHAT IS MOCKED, and why -- same convention as
`test_dense_retrieval_repo.py`:

  REAL (local, free, no network, no API cost):
    - tree-sitter parsing and dependency-graph construction
    - DenseEmbedder embeddings (fastembed/ONNX) in the indexing test
    - GraphRAG retrieval math (Equations 1 and 2)
    - CABA segmentation/MMR math
    - one end-to-end exercise of the REAL Qwen2.5-Coder-0.5B perplexity
      scorer, skipped if the model is not in the local HF cache

  MOCKED:
    - the OpenAI boundary ONLY (`CountingOpenAIProvider`), exactly as
      every other baseline's tests in this suite mock it
    - benchmark dataset network fetches, so benchmark-loading tests are
      deterministic and offline

  NEVER RUN HERE: any paid inference against the real RepoProbe-Python
  (108q) or SWE-QA-Pro (80q) manifests. These tests exist precisely so
  that no paid call is needed to validate the implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ant.benchmarks.base import TaskExample
from ant.external_wrappers import repodistill as rd
from ant.external_wrappers import repodistill_caba as caba
from ant.external_wrappers import repodistill_graph as rg

_MANIFEST_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "manifests"


# ===========================================================================
# Fixtures: a tiny synthetic repo with real, parseable Python structure
# ===========================================================================

_TINY_REPO = {
    "pkg/__init__.py": "",
    "pkg/storage.py": (
        "class BlobStore:\n"
        '    """Persists binary blobs to local disk."""\n'
        "\n"
        "    def write_blob(self, key, payload):\n"
        "        path = self._resolve(key)\n"
        "        path.write_bytes(payload)\n"
        "        return path\n"
        "\n"
        "    def _resolve(self, key):\n"
        "        return self.root / key\n"
    ),
    "pkg/service.py": (
        "from pkg.storage import BlobStore\n"
        "\n"
        "\n"
        "def upload_document(store, key, data):\n"
        '    """Upload a document into the blob store."""\n'
        "    validated = validate_payload(data)\n"
        "    return store.write_blob(key, validated)\n"
        "\n"
        "\n"
        "def validate_payload(data):\n"
        "    if not data:\n"
        "        raise ValueError('empty payload')\n"
        "    return data\n"
    ),
    "docs/readme.md": "Unrelated prose about gardening and tomatoes.\n",
}


def _write_repo(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    return _write_repo(tmp_path / "repo", _TINY_REPO)


class _StubScorer:
    """Deterministic stand-in for Qwen2.5-Coder-0.5B.

    Loading a real 0.5B causal LM for every unit test would make the suite
    minutes slower for no additional assurance about the ALGORITHM (which
    is what these tests check). The real model is exercised separately in
    `test_real_qwen_perplexity_scorer_segments_a_function`.

    Token counting is deliberately whitespace-based and perplexity is a
    pure function of line content, so every assertion below is exactly
    reproducible.
    """

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def line_perplexities(self, text: str) -> list[float]:
        # A spike on any line containing "SPIKE", flat elsewhere -- lets a
        # test place a block boundary at a known line.
        return [100.0 if "SPIKE" in line else 1.0 for line in text.splitlines()]

    def perplexity(self, text: str, prefix: str = "") -> float:
        # Conditioning on a prefix containing "relevant" reduces
        # perplexity, so AMI(b, q) = PPL(q) - PPL(q|b) is positive for
        # exactly those blocks.
        return 5.0 if "relevant" in prefix else 10.0

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vectors.append([float(text.count("a")), float(text.count("b")), 1.0])
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class _RecordingProvider:
    """Records every prompt that reaches the OpenAI boundary, and replays
    canned responses -- so tests can assert exactly what the LLM saw."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.prompts: list[str] = []
        self.responses = list(responses or [])
        self._calls = 0

    def responses_text(self, prompt: str, max_output_tokens: int = 512):
        self.prompts.append(prompt)
        self._calls += 1
        if self.responses:
            text = self.responses.pop(0)
        else:
            text = json.dumps(
                {
                    "compressed_documents": {f"Document_{i}": 0.5 for i in range(20)},
                    "updated_summary": "a running memory summary",
                }
            )

        class _Result:
            def __init__(self, value: str) -> None:
                self.text = value

        return _Result(text)

    def drain_call_count(self) -> int:
        count = self._calls
        self._calls = 0
        return count

    def drain_usage(self):
        from ant.domain import TokenUsage

        return TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120)

    def drain_retry_log(self) -> list[dict]:
        return []


def _run_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    repo_root: Path,
    example: TaskExample,
    provider: _RecordingProvider | None = None,
    scorer: object | None = None,
):
    provider = provider or _RecordingProvider()
    agent = rd.RepoDistillAdapter(index_root=tmp_path / ".ant-repodistill")
    monkeypatch.setattr(agent, "_make_provider", lambda: provider)
    monkeypatch.setattr(agent, "_make_scorer", lambda: scorer or _StubScorer())
    return agent.run(example, repo_root), provider


# ===========================================================================
# PART D.1 -- RepoProbe repository loads correctly via the existing adapter
# ===========================================================================


def test_repoprobe_examples_load_through_the_existing_benchmark_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RepoProbe adapter is used UNMODIFIED. Network fetches are
    stubbed so this is deterministic and offline; the assertion is on the
    `TaskExample` contract RepoDistill consumes."""
    from ant.benchmarks import repoprobe

    repos_info = [
        {
            "name": "Tencent-Hunyuan/FieldStation42",
            "primary_language": "Python",
            "snapshot": {"git": {"head": "deadbeef"}},
        }
    ]
    csv_text = (
        "question_id,taxonomy,difficulty,question,answer,checklist\n"
        "FieldStation42-0,architecture,hard,How does scheduling work?,"
        "GOLD_ANSWER_TEXT,GOLD_CHECKLIST_TEXT\n"
    )

    def fake_fetch(url: str) -> str:
        return json.dumps(repos_info) if url.endswith("repos_info.json") else csv_text

    monkeypatch.setattr(repoprobe, "_fetch_text", fake_fetch)
    adapter = repoprobe.RepoProbeAdapter()
    adapter._repos_info = None

    examples = adapter.load_examples()
    assert len(examples) == 1
    example = examples[0]
    assert example.benchmark == "repoprobe"
    assert example.task_id == "FieldStation42-0"
    assert example.question == "How does scheduling work?"
    # These are the fields RepoDistill must NEVER read (see D.4).
    assert example.reference == "GOLD_ANSWER_TEXT"
    assert example.metadata["checklist"] == "GOLD_CHECKLIST_TEXT"
    assert example.metadata["commit"] == "deadbeef"


def test_repoprobe_python_frozen_manifest_is_the_expected_108_question_set() -> None:
    manifest = json.loads(
        (_MANIFEST_ROOT / "repoprobe" / "sample_manifest_repoprobe_python_full.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["benchmark"] == "repoprobe"
    assert manifest["total_questions"] == 108
    assert len(manifest["task_ids"]) == 108
    assert len(set(manifest["task_ids"])) == 108
    assert len(manifest["included_repositories"]) == 8


# ===========================================================================
# PART D.2 -- SWE-QA-Pro repository loads correctly via the existing adapter
# ===========================================================================


def test_sweqa_pro_examples_load_through_the_existing_benchmark_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ant.benchmarks import sweqa_pro

    class _Raw:
        id = "05533ae903375e6b"
        repo = "qiboteam/qibo"
        question = "How does the circuit executor dispatch backends?"
        answer = "GOLD_ANSWER_TEXT"
        metadata = {"commit_id": "abc1234", "cluster": "architecture", "qa_type": "how"}

    monkeypatch.setattr(sweqa_pro, "load_examples", lambda *a, **k: [_Raw()])
    adapter = sweqa_pro.SweQaProAdapter()

    examples = adapter.load_examples()
    assert len(examples) == 1
    example = examples[0]
    assert example.benchmark == "sweqa_pro"
    assert example.task_id == "05533ae903375e6b"
    assert example.metadata["repo"] == "qiboteam/qibo"
    assert example.metadata["commit_id"] == "abc1234"
    assert example.reference == "GOLD_ANSWER_TEXT"


def test_sweqa_pro_frozen_manifest_is_the_expected_80_question_set() -> None:
    manifest = json.loads(
        (_MANIFEST_ROOT / "sweqa_pro" / "sample_manifest_sweqa_pro_80.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["benchmark"] == "sweqa_pro"
    assert manifest["total_task_ids"] == 80
    assert len(manifest["task_ids"]) == 80
    assert len(set(manifest["task_ids"])) == 80
    assert manifest["total_repositories"] == 8


# ===========================================================================
# PART D.3 -- indexing/preprocessing on a real small repo
# ===========================================================================


def test_tree_sitter_graph_build_extracts_units_contain_and_invoke_edges(
    tiny_repo: Path,
) -> None:
    files = [str(p.relative_to(tiny_repo)) for p in tiny_repo.rglob("*") if p.is_file()]
    graph = rg.build_repository_graph(tiny_repo, files)

    qualnames = {unit.qualname for unit in graph.units.values()}
    assert "BlobStore" in qualnames
    assert "BlobStore.write_blob" in qualnames
    assert "BlobStore._resolve" in qualnames
    assert "upload_document" in qualnames
    assert "validate_payload" in qualnames

    by_qualname = {unit.qualname: unit for unit in graph.units.values()}

    # CONTAIN: class -> its methods (paper 2.1.1).
    store_id = by_qualname["BlobStore"].node_id
    contained = {graph.units[c].qualname for c in graph.contain.get(store_id, [])}
    assert contained == {"BlobStore.write_blob", "BlobStore._resolve"}

    # INVOKE: same-file resolution (upload_document -> validate_payload).
    upload_id = by_qualname["upload_document"].node_id
    invoked = {graph.units[c].qualname for c in graph.invoke.get(upload_id, [])}
    assert "validate_payload" in invoked

    # INVOKE: same-file resolution inside a class (write_blob -> _resolve).
    write_id = by_qualname["BlobStore.write_blob"].node_id
    assert "BlobStore._resolve" in {
        graph.units[c].qualname for c in graph.invoke.get(write_id, [])
    }

    # Metadata the paper says to record is populated.
    unit = by_qualname["validate_payload"]
    assert unit.path.replace("\\", "/") == "pkg/service.py"
    assert unit.line_start > 0 and unit.line_end >= unit.line_start
    assert "raise ValueError" in unit.snippet

    # Non-Python files contribute no nodes (documented deviation 2).
    assert all(unit.path.endswith(".py") for unit in graph.units.values())


def test_graph_survives_a_round_trip_through_the_disk_cache(
    tiny_repo: Path, tmp_path: Path
) -> None:
    files = [str(p.relative_to(tiny_repo)) for p in tiny_repo.rglob("*") if p.is_file()]
    graph = rg.build_repository_graph(tiny_repo, files)
    index_dir = tmp_path / "idx"
    graph.save(index_dir)
    reloaded = rg.RepoGraph.load(index_dir)
    assert reloaded is not None
    assert reloaded.units.keys() == graph.units.keys()
    assert reloaded.contain == graph.contain
    assert reloaded.invoke == graph.invoke


def test_unparseable_file_does_not_abort_the_repo_graph_build(tmp_path: Path) -> None:
    root = _write_repo(
        tmp_path / "repo",
        {"good.py": "def alpha():\n    return 1\n", "broken.py": "def (((:\n  ???\n"},
    )
    graph = rg.build_repository_graph(root, ["good.py", "broken.py"])
    assert "alpha" in {unit.name for unit in graph.units.values()}


def test_full_indexing_preprocessing_builds_graph_and_embedding_index(
    tiny_repo: Path, tmp_path: Path
) -> None:
    """The one-time-per-repo preprocessing step, end to end, with REAL
    local embeddings (fastembed/ONNX -- free, no network at inference)."""
    from ant.retrieval.dense import DenseEmbedder

    files = [str(p.relative_to(tiny_repo)) for p in tiny_repo.rglob("*") if p.is_file()]
    index_dir = tmp_path / "idx"
    graph, node_ids, vectors = rd.ensure_repo_graph_and_index(
        tiny_repo, files, DenseEmbedder(), index_dir
    )
    assert graph.n_nodes >= 5
    assert node_ids == sorted(graph.units)
    assert vectors.shape[0] == len(node_ids)
    assert vectors.shape[1] > 0
    # L2-normalized rows, as EmbeddingIndex requires.
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-4)

    # Second call is a pure cache hit: no re-embedding.
    embed_calls: list[list[str]] = []
    real_embed = DenseEmbedder.embed

    class _CountingEmbedder(DenseEmbedder):
        def embed(self, texts):
            embed_calls.append(list(texts))
            return real_embed(self, texts)

    graph2, node_ids2, vectors2 = rd.ensure_repo_graph_and_index(
        tiny_repo, files, _CountingEmbedder(), index_dir
    )
    assert embed_calls == []
    assert node_ids2 == node_ids
    np.testing.assert_allclose(vectors2, vectors)


def test_graphrag_retrieval_ranks_semantically_and_expands_the_graph(
    tiny_repo: Path, tmp_path: Path
) -> None:
    from ant.retrieval.dense import DenseEmbedder

    files = [str(p.relative_to(tiny_repo)) for p in tiny_repo.rglob("*") if p.is_file()]
    embedder = DenseEmbedder()
    graph, node_ids, vectors = rd.ensure_repo_graph_and_index(
        tiny_repo, files, embedder, tmp_path / "idx"
    )
    [query_list] = embedder.embed(["How is a document uploaded to the blob store?"])
    candidates, anchors = rg.retrieve_candidates(
        graph, "How is a document uploaded to the blob store?", node_ids,
        vectors, np.asarray(query_list, dtype=np.float32),
    )
    assert candidates
    assert anchors
    # Candidate set contains more than bare anchors -- Reasoning Chain
    # Mining actually fired (first-order subtrees and/or multi-hop paths).
    assert {c.kind for c in candidates} - {"anchor"}
    # Equation 2 was applied with lambda = 0.5.
    for candidate in candidates:
        expected = 0.5 * candidate.similarity + 0.5 * candidate.max_member_similarity
        assert candidate.score == pytest.approx(expected)
    # Ranked descending, deterministically.
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_first_order_subtree_width_is_capped_by_query_similarity() -> None:
    """Deviation 6: a hub anchor must not drag its whole neighbourhood
    into one candidate. The retained neighbours must be the most
    query-similar ones, not an arbitrary prefix."""
    n_neighbours = rg.MAX_SUBTREE_NEIGHBOURS + 10
    units = {
        "hub": rg.CodeUnit("hub", "h.py", "function", "hub", "hub", 1, 2, "def hub(): pass")
    }
    for i in range(n_neighbours):
        node_id = f"n{i:03d}"
        units[node_id] = rg.CodeUnit(
            node_id, "n.py", "function", node_id, node_id, 1, 2, "def n(): pass"
        )
    graph = rg.RepoGraph(units=units, invoke={"hub": [f"n{i:03d}" for i in range(n_neighbours)]})

    node_ids = sorted(units)
    # Similarity increases with the neighbour's index, so the LAST ones
    # are the most similar -- a naive "first N" cap would pick the wrong
    # set and this test would fail.
    vectors = np.zeros((len(node_ids), 2), dtype=np.float32)
    for i, node_id in enumerate(node_ids):
        angle = 0.0 if node_id == "hub" else (1.0 - int(node_id[1:]) / (n_neighbours * 2))
        vectors[i] = [np.cos(angle), np.sin(angle)]
    query = np.asarray([1.0, 0.0], dtype=np.float32)

    candidates, _ = rg.retrieve_candidates(graph, "q", node_ids, vectors, query, top_k=1)
    subtree = next(c for c in candidates if c.kind == "subtree")
    assert len(subtree.node_ids) == rg.MAX_SUBTREE_NEIGHBOURS + 1  # anchor + capped neighbours
    kept = [n for n in subtree.node_ids if n != "hub"]
    expected = [f"n{i:03d}" for i in range(n_neighbours - rg.MAX_SUBTREE_NEIGHBOURS, n_neighbours)]
    assert sorted(kept) == sorted(expected)


def test_rerank_equation_uses_paper_lambda_and_is_not_plain_similarity() -> None:
    """Equation 2 must genuinely combine both terms -- a regression here
    would silently turn GraphRAG into plain dense retrieval."""
    assert rg.RERANK_LAMBDA == 0.5
    graph = rg.RepoGraph(
        units={
            "a": rg.CodeUnit("a", "a.py", "function", "a", "a", 1, 2, "def a(): pass"),
            "b": rg.CodeUnit("b", "b.py", "function", "b", "b", 1, 2, "def b(): pass"),
        },
        invoke={"a": ["b"]},
    )
    node_ids = ["a", "b"]
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    candidates, anchors = rg.retrieve_candidates(
        graph, "q", node_ids, vectors, query, top_k=2
    )
    subtree = next(c for c in candidates if c.kind == "subtree")
    # centroid of orthogonal unit vectors -> sim 1/sqrt(2); max member -> 1.0
    assert subtree.similarity == pytest.approx(1 / np.sqrt(2), rel=1e-5)
    assert subtree.max_member_similarity == pytest.approx(1.0)
    assert subtree.score == pytest.approx(0.5 * (1 / np.sqrt(2)) + 0.5 * 1.0, rel=1e-5)


# ===========================================================================
# CABA -- paper equations 3 and 4
# ===========================================================================


def test_paper_hyperparameters_are_the_published_values() -> None:
    assert caba.SEGMENTATION_ALPHA == 0.2  # paper 2.2
    assert caba.MMR_LAMBDA == 0.1  # paper Equation 3
    assert rg.RERANK_LAMBDA == 0.5  # paper Equation 2
    assert rd.CAPO_CHUNK_TOKENS == 10_000  # paper 3, Implementation
    assert rd.VALID_BUDGET_RATES == (0.0, 0.25, 0.5, 0.75, 1.0)  # paper 2.3.1
    assert caba.PERPLEXITY_MODEL == "Qwen/Qwen2.5-Coder-0.5B"  # paper 2.2


def test_block_segmentation_starts_a_block_at_a_perplexity_spike() -> None:
    text = "line one\nline two\nSPIKE here\nline four\n"
    blocks = caba.segment_into_blocks(text, _StubScorer())
    starts = [block.line_start for block in blocks]
    assert 1 in starts
    assert 3 in starts, f"expected a block boundary at the spiking line, got {starts}"


def test_mmr_selection_respects_the_token_budget_and_prefers_informative_blocks() -> None:
    blocks = [
        caba.Block(1, 1, "relevant aaa", 2),
        caba.Block(2, 2, "irrelevant bbb", 2),
        caba.Block(3, 3, "irrelevant ccc", 2),
    ]
    selected = caba.select_blocks_mmr(blocks, "the query", token_budget=2, scorer=_StubScorer())
    assert len(selected) == 1
    # AMI is positive only for the "relevant" block (stub PPL(q|b) < PPL(q)).
    assert selected[0].text == "relevant aaa"


def test_mmr_selection_emits_blocks_in_source_order() -> None:
    blocks = [
        caba.Block(1, 1, "irrelevant aaa", 1),
        caba.Block(2, 2, "relevant bbb", 1),
    ]
    selected = caba.select_blocks_mmr(blocks, "q", token_budget=10, scorer=_StubScorer())
    assert [b.line_start for b in selected] == [1, 2]


def test_compress_unit_honours_the_two_saturating_budget_rates() -> None:
    snippet = "def f():\n    return 1\n"
    scorer = _StubScorer()
    assert caba.compress_unit(snippet, "q", 0.0, scorer) == ""
    assert caba.compress_unit(snippet, "q", 1.0, scorer) == snippet


def test_compress_unit_at_an_intermediate_rate_shrinks_the_text() -> None:
    snippet = "\n".join(f"    step_{i} = compute_{i}()" for i in range(12))
    scorer = _StubScorer()
    out = caba.compress_unit(snippet, "q", 0.25, scorer)
    assert scorer.count_tokens(out) <= scorer.count_tokens(snippet) * 0.25 + 1


@pytest.mark.parametrize("budget_rate", [0.25, 0.5, 0.75])
def test_real_qwen_perplexity_scorer_segments_a_function(budget_rate: float) -> None:
    """Exercises the PAPER'S OWN model (Qwen2.5-Coder-0.5B) end to end --
    local, free, CPU. Skipped when the model is not in the HF cache so the
    suite never blocks on a multi-hundred-MB download."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    try:
        transformers.AutoConfig.from_pretrained(
            caba.PERPLEXITY_MODEL, local_files_only=True
        )
    except Exception:  # noqa: BLE001
        pytest.skip(f"{caba.PERPLEXITY_MODEL} is not in the local HuggingFace cache")

    scorer = caba.get_shared_perplexity_scorer()
    snippet = (
        "def upload_document(store, key, data):\n"
        "    validated = validate_payload(data)\n"
        "    if not validated:\n"
        "        raise ValueError('empty')\n"
        "    return store.write_blob(key, validated)\n"
    )
    perplexities = scorer.line_perplexities(snippet)
    assert len(perplexities) == len(snippet.splitlines())
    assert any(p == p and p > 0 for p in perplexities)  # at least one non-nan, positive

    out = caba.compress_unit(snippet, "How is a document uploaded?", budget_rate, scorer)
    assert scorer.count_tokens(out) <= scorer.count_tokens(snippet)
    assert out.strip(), "a non-zero budget must retain something"


# ===========================================================================
# CAPO -- turn mechanics and the verbatim Appendix A.1.2 prompts
# ===========================================================================


def test_prompts_are_the_verbatim_appendix_templates() -> None:
    """Guards the two prompts against well-meaning "improvement". Every
    assertion below quotes Table 4 of the paper exactly."""
    prompt = rd.CONTEXT_COMPRESSION_PROMPT
    assert prompt.startswith(
        "You are provided with a problem, a chunk of code context and a previous "
        "memory for previous chunks."
    )
    for level in (
        "- 0% : Fully filtered (empty)",
        "- 25% : Aggressive compression (essential information only)",
        "- 50% : Balanced compression (core content retained)",
        "- 75% : Light compression (key context preserved)",
        "- 100% : Original text (no compression)",
    ):
        assert level in prompt
    assert "<problem> {problem} </problem>" in prompt
    assert "<memory> {memory} </memory>" in prompt
    assert "<chunk> {chunk} </chunk>" in prompt
    assert '"compressed_documents"' in prompt
    assert '"updated_summary"' in prompt
    assert "max 200 words" in prompt

    answer = rd.ANSWER_GENERATION_PROMPT
    assert answer.startswith(
        "You are presented with a problem, and a repository memory. Your task is to "
        "directly answer the problem based on the provided repository memory and "
        "problem statement."
    )
    assert "<problem> {problem} </problem>" in answer
    assert (
        "Provide the final answer concisely and directly, without code snippets, "
        "extra explanations or commentary." in answer
    )


def test_compression_prompt_formats_without_brace_errors() -> None:
    rendered = rd.CONTEXT_COMPRESSION_PROMPT.format(
        problem="q?", memory="m", chunk="Document_0 (a.py:1-2, f):\ndef f(): pass"
    )
    assert '{"compressed_documents": {"Document_0": x' in rendered
    assert "<problem> q? </problem>" in rendered


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.75, 0.75),  # paper's own Appendix A.1.3 example form
        (1, 1.0),
        (0, 0.0),
        (75, 0.75),  # percentage form, as the prompt's option list writes it
        ("50%", 0.5),
        (0.6, 0.5),  # off-schema -> snapped to nearest legal level
        ("nonsense", None),
        (None, None),
        (True, None),
        (-1, None),
    ],
)
def test_budget_parsing_accepts_both_paper_sanctioned_forms(raw, expected) -> None:
    assert rd._parse_budget_value(raw) == expected


def test_chunk_packing_respects_the_10k_token_chunk_size() -> None:
    scorer = _StubScorer()
    units = [
        rg.CodeUnit(f"n{i}", "a.py", "function", f"f{i}", f"f{i}", 1, 1, "word " * 4000)
        for i in range(5)
    ]
    chunks = rd.pack_units_into_chunks(units, scorer)
    assert len(chunks) == 3  # 4000-token units: 2 per 10k chunk, then 1
    for chunk in chunks:
        assert sum(scorer.count_tokens(u.snippet) for u in chunk) <= rd.CAPO_CHUNK_TOKENS


def test_one_capo_turn_per_chunk_plus_exactly_one_answer_call(
    monkeypatch: pytest.MonkeyPatch, tiny_repo: Path, tmp_path: Path
) -> None:
    example = TaskExample(
        benchmark="test_bench", task_id="t1", question="How is a document uploaded?", reference=""
    )
    result, provider = _run_agent(monkeypatch, tmp_path, tiny_repo, example)

    n_turns = result.metadata["n_capo_turns"]
    assert n_turns >= 1
    assert len(provider.prompts) == n_turns + 1
    # The last prompt is the answer-generation prompt; all earlier ones
    # are compression turns.
    assert provider.prompts[-1].startswith("You are presented with a problem")
    for prompt in provider.prompts[:-1]:
        assert prompt.startswith("You are provided with a problem")
    assert result.usage.llm_calls == n_turns + 1
    assert rd.expected_paid_calls(0) == (1, 2)


def test_memory_summary_carries_across_capo_turns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _write_repo(
        tmp_path / "repo",
        {
            f"m{i}.py": f"def f{i}():\n    return {i}\n" + ("    # pad\n" * 5)
            for i in range(6)
        },
    )
    responses = [
        json.dumps(
            {"compressed_documents": {f"Document_{i}": 1 for i in range(20)},
             "updated_summary": f"SUMMARY_AFTER_TURN_{t}"}
        )
        for t in range(6)
    ]
    provider = _RecordingProvider(responses=[*responses, "final answer"])

    # Force many small chunks so there is more than one turn.
    monkeypatch.setattr(rd, "CAPO_CHUNK_TOKENS", 5)
    example = TaskExample(benchmark="b", task_id="t", question="what?", reference="")
    result, provider = _run_agent(monkeypatch, tmp_path, root, example, provider=provider)

    assert result.metadata["n_capo_turns"] >= 2
    # Turn 2's prompt must carry turn 1's summary in its <memory> slot.
    assert "<memory> SUMMARY_AFTER_TURN_0 </memory>" in provider.prompts[1]
    # The final answer prompt carries the last summary into repository memory.
    assert "SUMMARY_AFTER_TURN" in provider.prompts[-1]


def test_capo_turn_cap_bounds_paid_calls_and_is_flagged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _write_repo(
        tmp_path / "repo",
        {f"m{i}.py": f"def f{i}():\n    return {i}\n" + ("    # pad\n" * 4) for i in range(20)},
    )
    monkeypatch.setattr(rd, "CAPO_CHUNK_TOKENS", 3)
    monkeypatch.setattr(rd, "MAX_CAPO_TURNS", 2)
    example = TaskExample(benchmark="b", task_id="t", question="what?", reference="")
    result, provider = _run_agent(monkeypatch, tmp_path, root, example)

    assert result.metadata["n_capo_turns"] == 2
    assert result.metadata["capo_turn_cap_hit"] is True
    assert result.termination_reason == "capo_turn_cap_reached"
    assert result.usage.llm_calls == 3  # 2 capped turns + 1 answer call

    # Units the cap cut off must be DROPPED, not silently carried into the
    # answer call at 100% retention -- otherwise the cap would not actually
    # bound the final (paid) call's input size.
    evaluated = {
        node_id for turn in result.trajectory for node_id in turn["document_ids"]
    }
    assert set(result.metadata["retention_budgets"]) == evaluated
    assert result.metadata["n_retrieved_units"] == len(evaluated)
    answer_prompt = provider.prompts[-1]
    for unit_id in result.metadata["retention_budgets"]:
        assert unit_id.split("::")[0] in answer_prompt


def test_zero_budget_documents_are_dropped_from_repository_memory(
    monkeypatch: pytest.MonkeyPatch, tiny_repo: Path, tmp_path: Path
) -> None:
    provider = _RecordingProvider(
        responses=[
            json.dumps(
                {"compressed_documents": {f"Document_{i}": 0 for i in range(20)},
                 "updated_summary": "nothing relevant"}
            ),
            "final answer",
        ]
    )
    example = TaskExample(benchmark="b", task_id="t", question="what?", reference="")
    result, provider = _run_agent(monkeypatch, tmp_path, tiny_repo, example, provider=provider)
    answer_prompt = provider.prompts[-1]
    assert "nothing relevant" in answer_prompt
    assert "def upload_document" not in answer_prompt
    assert result.evidence == []


def test_malformed_budget_json_fails_open_to_full_retention(
    monkeypatch: pytest.MonkeyPatch, tiny_repo: Path, tmp_path: Path
) -> None:
    """A bad compression response must never silently delete retrieved
    context -- it falls back to 100% retention and records the failure."""
    provider = _RecordingProvider(responses=["not json at all", "final answer"])
    example = TaskExample(benchmark="b", task_id="t", question="what?", reference="")
    result, provider = _run_agent(monkeypatch, tmp_path, tiny_repo, example, provider=provider)
    assert all(rate == 1.0 for rate in result.metadata["retention_budgets"].values())
    assert result.trajectory[0]["budget_parse_failures"] > 0
    assert result.evidence


# ===========================================================================
# PART D.4 -- no gold relevant-file / reference metadata reaches the pipeline
# ===========================================================================


def test_no_gold_reference_or_checklist_reaches_the_pipeline(
    monkeypatch: pytest.MonkeyPatch, tiny_repo: Path, tmp_path: Path
) -> None:
    """Structural analogue of Dense Retrieval's own
    `test_no_gold_reference_reaches_generation`. Every gold-bearing field
    on TaskExample is poisoned with a unique sentinel; none may appear in
    ANY prompt that reaches the LLM boundary, nor in the retrieved
    evidence, nor in the result metadata."""
    sentinels = {
        "reference": "SENTINEL_GOLD_ANSWER_MUST_NEVER_APPEAR",
        "checklist": "SENTINEL_CHECKLIST_MUST_NEVER_APPEAR",
        "relevant_files": "SENTINEL_GOLD_FILE_MUST_NEVER_APPEAR",
        "gold_trajectory": "SENTINEL_TRAJECTORY_MUST_NEVER_APPEAR",
    }
    example = TaskExample(
        benchmark="b",
        task_id="t",
        question="How is a document uploaded?",
        reference=sentinels["reference"],
        metadata={
            "checklist": sentinels["checklist"],
            "relevant_files": [sentinels["relevant_files"]],
            "gold_trajectory": sentinels["gold_trajectory"],
            "answer": sentinels["reference"],
        },
    )
    result, provider = _run_agent(monkeypatch, tmp_path, tiny_repo, example)

    assert provider.prompts, "the LLM boundary must actually have been reached"
    blob = json.dumps(
        {
            "prompts": provider.prompts,
            "answer": result.final_answer,
            "evidence": result.evidence,
            "trajectory": result.trajectory,
            "metadata": result.metadata,
        }
    )
    for name, sentinel in sentinels.items():
        assert sentinel not in blob, f"gold field {name!r} leaked into the RepoDistill pipeline"


def test_retrieval_signature_cannot_see_a_task_example_at_all() -> None:
    """Defense in depth: GraphRAG's own retrieval entry point takes a bare
    question string plus repo-derived structures -- there is no parameter
    through which benchmark gold metadata could be passed even by
    accident."""
    import inspect

    parameters = set(inspect.signature(rg.retrieve_candidates).parameters)
    assert parameters == {
        "graph",
        "question",
        "node_ids",
        "node_vectors",
        "query_vector",
        "top_k",
        "rerank_lambda",
        "limit",
    }
    build_parameters = set(inspect.signature(rg.build_repository_graph).parameters)
    assert build_parameters == {"root", "relative_paths"}


def test_graph_build_is_a_pure_function_of_repo_content(tiny_repo: Path) -> None:
    """The same checkout must produce the same graph regardless of which
    question (or which benchmark) is being asked -- the graph cannot be
    conditioned on task metadata because it never receives any."""
    files = [str(p.relative_to(tiny_repo)) for p in tiny_repo.rglob("*") if p.is_file()]
    first = rg.build_repository_graph(tiny_repo, files)
    second = rg.build_repository_graph(tiny_repo, list(reversed(files)))
    assert first.units.keys() == second.units.keys()
    assert first.invoke == second.invoke
    assert first.contain == second.contain


# ===========================================================================
# PART D.5 -- the final prediction reaches the existing scorer, right shape
# ===========================================================================


def test_agent_result_satisfies_the_shared_agent_result_contract(
    monkeypatch: pytest.MonkeyPatch, tiny_repo: Path, tmp_path: Path
) -> None:
    from ant.agents.base import AgentResult

    example = TaskExample(
        benchmark="repoprobe", task_id="FieldStation42-0", question="How is a document uploaded?",
        reference="gold",
    )
    provider = _RecordingProvider(
        responses=[
            json.dumps({"compressed_documents": {"Document_0": 1}, "updated_summary": "s"}),
            "  The document is uploaded via upload_document().  ",
        ]
    )
    result, _ = _run_agent(monkeypatch, tmp_path, tiny_repo, example, provider=provider)

    assert isinstance(result, AgentResult)
    assert result.benchmark == "repoprobe"
    assert result.task_id == "FieldStation42-0"
    assert result.method == "repodistill"
    assert result.final_answer == "The document is uploaded via upload_document()."
    assert result.usage.llm_calls >= 2
    assert result.metadata["generation_model"] == "gpt-4.1"
    assert result.metadata["variant"] == "no_training"
    # JSON-serializable, as the harness requires for output/runs/... dumps.
    json.dumps(result.model_dump())


def test_prediction_reaches_repoprobe_score_with_the_right_contract(
    monkeypatch: pytest.MonkeyPatch, tiny_repo: Path, tmp_path: Path
) -> None:
    """The EXISTING, unmodified `bench.score(...)` is called with this
    agent's AgentResult -- no special judge, no RepoDistill-specific
    scoring path. The judge call itself is mocked (it is a paid call)."""
    from ant.benchmarks import repoprobe

    example = TaskExample(
        benchmark="repoprobe",
        task_id="FieldStation42-0",
        question="How is a document uploaded?",
        reference="gold answer",
        metadata={
            "checklist": "gold checklist",
            "repo": "x/y",
            "commit": "c",
            "repo_short_name": "y",
        },
    )
    result, _ = _run_agent(monkeypatch, tmp_path, tiny_repo, example)

    seen: dict = {}

    class _JudgeResult:
        text = json.dumps(
            {
                "total_score": 7.5,
                "knowledge_score": 6.5,
                "knowledge_max": 9,
                "clarity_score": 1.0,
                "clarity_max": 1,
                "hallucination": False,
            }
        )
        estimated_cost_usd = 0.0

    def fake_call_judge(*, system: str, user: str, max_output_tokens: int = 1024):
        seen["user"] = user
        return _JudgeResult()

    monkeypatch.setattr(repoprobe, "call_judge", fake_call_judge)
    monkeypatch.setattr(
        repoprobe.RepoProbeAdapter, "prepare_environment", lambda self, ex: tiny_repo
    )
    monkeypatch.setattr(repoprobe, "build_directory_structure", lambda path: "pkg/\n  service.py")

    metric = repoprobe.RepoProbeAdapter().score(example, result)

    assert metric.benchmark == "repoprobe"
    assert metric.task_id == "FieldStation42-0"
    assert metric.native_score == pytest.approx(7.5)
    assert metric.metadata["generation_model"] == "gpt-4.1"
    # The scorer received THIS agent's answer, and the gold artifacts the
    # scorer itself is entitled to (the agent never saw them).
    assert result.final_answer in seen["user"]
    assert "repodistill" in seen["user"]


def test_prediction_reaches_sweqa_pro_score_with_the_right_contract(
    monkeypatch: pytest.MonkeyPatch, tiny_repo: Path, tmp_path: Path
) -> None:
    from ant.benchmarks import sweqa_pro

    example = TaskExample(
        benchmark="sweqa_pro",
        task_id="05533ae903375e6b",
        question="How is a document uploaded?",
        reference="gold answer",
        metadata={"repo": "qiboteam/qibo", "commit_id": "abc"},
    )
    result, _ = _run_agent(monkeypatch, tmp_path, tiny_repo, example)

    seen: dict = {}

    class _JudgeResult:
        text = json.dumps(
            {"correctness": 8, "completeness": 7, "relevance": 9, "clarity": 8, "reasoning": 7}
        )
        estimated_cost_usd = 0.0

    def fake_call_judge(*, system: str, user: str, max_output_tokens: int = 1024):
        seen["user"] = user
        return _JudgeResult()

    monkeypatch.setattr(sweqa_pro, "call_judge", fake_call_judge)
    metric = sweqa_pro.SweQaProAdapter().score(example, result)

    assert metric.benchmark == "sweqa_pro"
    assert metric.task_id == "05533ae903375e6b"
    assert metric.native_score == pytest.approx(39.0)  # 8+7+9+8+7
    assert set(metric.submetrics) == {
        "correctness", "completeness", "relevance", "clarity", "reasoning"
    }
    assert metric.metadata["generation_model"] == "gpt-4.1"
    assert result.final_answer in seen["user"]


# ===========================================================================
# Registration and isolation
# ===========================================================================


def test_agent_is_registered_under_its_own_name() -> None:
    from ant.evaluation_suite.registry import get_agent

    assert isinstance(get_agent("repodistill"), rd.RepoDistillAdapter)


def test_module_never_reaches_into_other_baselines_or_antman_internals() -> None:
    """Isolation guard: RepoDistill must not import ANTMAN's coordinator/
    worker machinery, Matched ReAct, ChainRAG, RepoGraph, or the other
    retrieval baselines. It is an additive, standalone adapter."""
    names = set(vars(rd)) | set(vars(rg)) | set(vars(caba))
    for forbidden in (
        "LocalCoordinator",
        "NeedGraph",
        "AutonomousWorker",
        "evolve_workers",
        "AntAgent",
        "MatchedReActAgent",
        "ChainRAGAdapter",
        "RepoGraphTool",
        "DenseRetrievalRepoAgent",
        "RetrievalAgent",
        "LocalSearchTool",
        "BM25Index",
    ):
        assert forbidden not in names, f"{forbidden} must not be reachable from RepoDistill"


def test_no_training_variant_never_references_training_machinery() -> None:
    source = Path(rd.__file__).read_text(encoding="utf-8")
    for forbidden in ("GRPOTrainer", "SFTTrainer", "torch.optim", "AdamW", "from trl"):
        assert forbidden not in source
