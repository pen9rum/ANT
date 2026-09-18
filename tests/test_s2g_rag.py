"""Tests for the S2G-RAG adapter (ant.external_wrappers.s2g_rag).

ZERO paid calls: the ONLY mocked boundary is the LLM call site (a
deterministic scripted stub, the same `_ScriptedProvider` convention
tests/test_chainrag.py and tests/test_longagent.py already use). Retrieval
is REAL -- real `LocalSearchTool` BM25+RRF over real materialized document
files in tmp_path -- because it is local and free, so ranking behavior is
genuinely exercised rather than stubbed.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.document_scope import DocumentRecord, materialize_documents
from ant.external_wrappers import s2g_rag as s2g_module
from ant.external_wrappers.s2g_rag import (
    MAX_GAP_FACTS_FOR_QUERY,
    MAX_SENTS_PER_DOC,
    MAX_TURNS,
    TOP_DOCS,
    S2GRAGAdapter,
    SharedCorpusRetriever,
    append_evidence_context,
    build_query_from_missing,
    build_selector_user_prompt,
    merge_evidence_only,
    parse_selector_output,
    should_force_first_retrieval,
)

# ===========================================================================
# Fixtures: a real, materialized, per-question candidate document collection.
# ===========================================================================

# Ten documents -- deliberately the same shape a real HotpotQA/2Wiki
# example has (`num_documents == 10` in every frozen manifest row), so the
# document-pool exhaustion behavior these tests assert is the behavior the
# real benchmark will produce, not an artifact of a toy 3-document fixture.
# Topically-overlapping distractors, like a real HotpotQA distractor set:
# they share vocabulary with the question (so BM25 genuinely ranks them and
# genuinely returns a full top-6) without containing the answer.
_DISTRACTOR_TEXTS = [
    ("Barrow residence",
     "The Barrow residence is a timber house.\n\nNo cat has ever lived at the Barrow residence."),
    ("Pemberly residence",
     "The Pemberly residence is a listed building.\n\nIts color scheme was never recorded."),
    ("Cat breeds",
     "The tabby cat is a common domestic cat.\n\nA cat often lives indoors near a window."),
    ("Whiskers (disambiguation)",
     "Whiskers may refer to a racehorse.\n\nWhiskers is also a brand of paint thinner."),
    ("Ashcombe village",
     "Ashcombe is a village with a parish church.\n\nThe village has no permanent residence hall."),
    ("Paint colors",
     "Paint color names include vermilion and ochre.\n\nA color may be mixed from pigments."),
    ("Windowsill gardening",
     "A windowsill can host herbs.\n\nThe windowsill of a residence gets morning light."),
    ("Mason guilds",
     "A local mason belonged to a guild.\n\nMasons built many a residence in 1899."),
]

DOCS = [
    DocumentRecord(
        doc_id="doc0",
        title="Whiskers the Cat",
        text=(
            "Whiskers the cat lives on the windowsill of the Ashcombe residence.\n\n"
            "Whiskers was adopted in 2011 and rarely leaves the windowsill."
        ),
    ),
    DocumentRecord(
        doc_id="doc1",
        title="Ashcombe residence",
        text=(
            "The Ashcombe residence is painted cerulean.\n\n"
            "The Ashcombe residence was built in 1899 by a local mason."
        ),
    ),
    *[
        DocumentRecord(doc_id=f"doc{i + 2}", title=title, text=text)
        for i, (title, text) in enumerate(_DISTRACTOR_TEXTS)
    ],
]
ALL_DOC_IDS = {d.doc_id for d in DOCS}

QUESTION = "What color is the residence where Whiskers the cat lives?"

JUDGE_INSUFFICIENT_BRIDGE = json.dumps(
    {
        "sufficient": False,
        "gap_items": [
            {
                "category": "bridge_entity",
                "target": "Whiskers the cat",
                "slot": "residence name",
                "description": "which residence Whiskers lives in",
            }
        ],
    }
)
JUDGE_SUFFICIENT = json.dumps({"sufficient": True, "gap_items": []})
SELECTOR_PICK = json.dumps({"evidence_global_ids": [1, 2, 3]})
ANSWER_TEXT = "Answer: cerulean\nRationale: Whiskers lives at Ashcombe, which is cerulean."


class _ScriptedProvider:
    """Returns pre-scripted responses in call order -- same convention as
    tests/test_chainrag.py's own `_ScriptedProvider`. A hard call cap makes
    a runaway loop fail loudly instead of hanging the suite."""

    def __init__(self, scripted: list[str], *, default: str = "{}", cap: int = 64) -> None:
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
            return JUDGE_INSUFFICIENT_BRIDGE
        if prompt.startswith(s2g_module.SELECTOR_SYSTEM_PROMPT):
            return SELECTOR_PICK
        return ANSWER_TEXT

    @property
    def total_calls(self) -> int:
        return self._calls

    def drain_call_count(self) -> int:
        count = self._calls
        self._calls = 0
        return count

    def drain_usage(self):
        from ant.domain import TokenUsage

        return TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)

    # Prompt-role helpers used by several assertions below.
    def judge_prompts(self) -> list[str]:
        return [p for p in self.prompts if p.startswith(s2g_module.SUFF_SYSTEM_PROMPT)]

    def selector_prompts(self) -> list[str]:
        return [p for p in self.prompts if p.startswith(s2g_module.SELECTOR_SYSTEM_PROMPT)]

    def answer_prompts(self) -> list[str]:
        return [p for p in self.prompts if p.startswith(s2g_module.force_answer_prompt)]


@pytest.fixture
def environment_root(tmp_path: Path) -> Path:
    root = tmp_path / "docenv"
    materialize_documents(DOCS, root)
    return root.resolve()


def _example(metadata: dict | None = None, benchmark: str = "hotpotqa") -> TaskExample:
    return TaskExample(
        benchmark=benchmark,
        task_id="t1",
        question=QUESTION,
        reference=json.dumps(["cerulean"]),
        metadata={"documents": [d.model_dump() for d in DOCS], **(metadata or {})},
    )


def _run(monkeypatch, provider, environment_root: Path, **kwargs):
    monkeypatch.setattr(s2g_module, "_ZeroTemperatureProvider", lambda model: provider)
    agent = S2GRAGAdapter(**kwargs)
    return agent.run(_example(), environment_root)


# ===========================================================================
# Structural invariants / registration / frozen upstream defaults.
# ===========================================================================


def test_agent_is_registered() -> None:
    from ant.evaluation_suite.registry import get_agent

    assert isinstance(get_agent("s2g_rag"), S2GRAGAdapter)


def test_frozen_upstream_defaults_match_the_repo_cli() -> None:
    """inference/inference_bm25.py parse_args(): --max_turns 4, --top_docs 6;
    main_batch's own call site: return_top_k=6, max_sents_per_doc=40;
    build_query_from_missing: max_facts=3 for every non-TriviaQA dataset."""
    assert MAX_TURNS == 4
    assert TOP_DOCS == 6
    assert s2g_module.EVIDENCE_TOP_K == 6
    assert MAX_SENTS_PER_DOC == 40
    assert MAX_GAP_FACTS_FOR_QUERY == 3


def test_no_max_output_tokens_below_openai_api_minimum() -> None:
    source = inspect.getsource(s2g_module)
    for match in re.finditer(r"MAX_OUTPUT_TOKENS\s*=\s*(\d+)", source):
        assert int(match.group(1)) >= 16


def test_module_never_binds_antman_coordination_names() -> None:
    names = set(vars(s2g_module))
    for forbidden in ("LocalCoordinator", "NeedGraph", "AutonomousWorker", "evolve_workers"):
        assert forbidden not in names


# ===========================================================================
# 1. Initial retrieval works against the shared candidate collection.
# ===========================================================================


def test_initial_retrieval_runs_against_the_shared_document_collection(
    monkeypatch: pytest.MonkeyPatch, environment_root: Path
) -> None:
    provider = _ScriptedProvider(
        [JUDGE_INSUFFICIENT_BRIDGE, SELECTOR_PICK, JUDGE_SUFFICIENT, ANSWER_TEXT]
    )
    result = _run(monkeypatch, provider, environment_root)

    first = result.trajectory[0]
    assert first["turn"] == 0
    assert first["retrieved_doc_ids"], "turn 0 must retrieve"
    # Every retrieved id comes from THIS question's own candidate collection.
    assert set(first["retrieved_doc_ids"]) <= ALL_DOC_IDS
    assert len(first["retrieved_doc_ids"]) == TOP_DOCS
    # BM25 really ranked: the two on-topic docs beat the 8 distractors.
    assert first["retrieved_doc_ids"][0] in {"doc0", "doc1"}
    assert result.usage.tool_calls == 1
    assert result.metadata["retrieval_is_local_and_free"] is True


def test_retriever_only_ever_sees_question_visible_documentrecords(
    environment_root: Path,
) -> None:
    retriever = SharedCorpusRetriever(environment_root, DOCS)
    assert set(retriever.corpus) == ALL_DOC_IDS
    # DocumentRecord has no gold field at all -- structurally, not by omission.
    assert set(DocumentRecord.model_fields) == {"doc_id", "title", "text"}
    titles, texts, doc_ids = retriever.search("Ashcombe residence painted", [], k=TOP_DOCS)
    assert "doc1" in doc_ids
    assert len(doc_ids) <= TOP_DOCS


def test_retriever_returns_upstream_sentinel_when_nothing_survives(
    environment_root: Path,
) -> None:
    retriever = SharedCorpusRetriever(environment_root, DOCS, remove_repeat_docs=True)
    titles, texts, doc_ids = retriever.search(
        "Ashcombe residence", sorted(ALL_DOC_IDS), k=TOP_DOCS
    )
    assert titles == ["No results found."]
    assert texts == ["No results found."]
    assert doc_ids == [""]


# ===========================================================================
# 2. Insufficiency triggers another retrieval round.
# ===========================================================================


def test_insufficient_judgment_triggers_a_second_retrieval_round(
    monkeypatch: pytest.MonkeyPatch, environment_root: Path
) -> None:
    provider = _ScriptedProvider(
        [
            JUDGE_INSUFFICIENT_BRIDGE,  # turn 0 judge
            SELECTOR_PICK,  # turn 0 selector
            JUDGE_INSUFFICIENT_BRIDGE,  # turn 1 judge -- still insufficient
            SELECTOR_PICK,  # turn 1 selector
            JUDGE_SUFFICIENT,  # turn 2 judge -- now sufficient
            ANSWER_TEXT,  # final answer
        ]
    )
    result = _run(monkeypatch, provider, environment_root)

    retrieving_turns = [t for t in result.trajectory if "retrieval_query" in t]
    assert len(retrieving_turns) == 2, "a second insufficiency must retrieve again"
    assert result.metadata["n_judge_calls"] == 3
    assert result.metadata["n_selector_calls"] == 2
    assert result.metadata["n_retrieval_calls"] == 2
    assert result.metadata["n_answer_calls"] == 1
    # Evidence Context is append-only: it never shrinks across turns.
    sizes = [t["evidence_context_chars"] for t in result.trajectory]
    assert sizes == sorted(sizes)
    assert sizes[-1] > 0


def test_evidence_context_is_append_only_never_overwritten() -> None:
    first = append_evidence_context("", "block A")
    second = append_evidence_context(first, "block B")
    assert second.startswith("block A")
    assert "block B" in second
    # Exact duplicates are not re-appended (upstream's own guard).
    assert append_evidence_context(second, "block B") == second


def test_first_turn_overconfident_sufficiency_is_overridden_into_one_retrieval(
    monkeypatch: pytest.MonkeyPatch, environment_root: Path
) -> None:
    """Upstream's `should_force_first_retrieval`: a turn-0 'sufficient' on an
    EMPTY evidence context is forced to retrieve at least once."""
    assert should_force_first_retrieval(0, "", True) is True
    assert should_force_first_retrieval(0, "some evidence", True) is False
    assert should_force_first_retrieval(1, "", True) is False

    provider = _ScriptedProvider(
        [JUDGE_SUFFICIENT, SELECTOR_PICK, JUDGE_SUFFICIENT, ANSWER_TEXT]
    )
    result = _run(monkeypatch, provider, environment_root)
    assert result.trajectory[0]["forced_first_retrieval"] is True
    assert result.trajectory[0]["retrieved_doc_ids"]
    # With no gap items (upstream clears them on the override), the query is
    # the bare original question.
    assert result.trajectory[0]["retrieval_query"] == QUESTION


# ===========================================================================
# 3. Gap items become the next retrieval query, by the paper's own mechanism.
# ===========================================================================


def test_gap_to_query_uses_target_plus_slot_when_both_present() -> None:
    gaps = [
        {
            "category": "bridge_entity",
            "target": "Whiskers",
            "slot": "residence",
            "description": "ignored",
        }
    ]
    assert build_query_from_missing(QUESTION, gaps) == f"{QUESTION} Whiskers residence"


def test_gap_to_query_falls_back_to_description_when_target_or_slot_missing() -> None:
    gaps = [{"category": "attribute", "target": "", "slot": "", "description": "the paint color"}]
    assert build_query_from_missing(QUESTION, gaps) == f"{QUESTION} the paint color"
    gaps_partial = [{"target": "Ashcombe", "slot": "", "description": "the paint color"}]
    assert build_query_from_missing(QUESTION, gaps_partial) == f"{QUESTION} the paint color"


def test_gap_to_query_caps_at_k_phrases_and_appends_to_the_original_question() -> None:
    gaps = [
        {"target": f"E{i}", "slot": f"s{i}", "description": f"d{i}"} for i in range(6)
    ]
    query = build_query_from_missing(QUESTION, gaps)
    assert query == f"{QUESTION} E0 s0 E1 s1 E2 s2"  # exactly K=3 phrases
    assert query.startswith(QUESTION)
    assert "E3" not in query
    # Explicit override still honored (upstream's own max_facts parameter).
    assert build_query_from_missing(QUESTION, gaps, max_facts=1) == f"{QUESTION} E0 s0"


def test_no_gap_items_means_the_query_is_the_bare_question() -> None:
    assert build_query_from_missing(QUESTION, []) == QUESTION
    assert build_query_from_missing(QUESTION, [{"target": "", "slot": "", "description": ""}]) == (
        QUESTION
    )


def test_gap_items_reach_the_next_retrieval_query_end_to_end(
    monkeypatch: pytest.MonkeyPatch, environment_root: Path
) -> None:
    second_gap = json.dumps(
        {
            "sufficient": False,
            "gap_items": [
                {
                    "category": "attribute",
                    "target": "Ashcombe residence",
                    "slot": "paint color",
                    "description": "what color the Ashcombe residence is painted",
                }
            ],
        }
    )
    provider = _ScriptedProvider(
        [
            JUDGE_INSUFFICIENT_BRIDGE,
            SELECTOR_PICK,
            second_gap,
            SELECTOR_PICK,
            JUDGE_SUFFICIENT,
            ANSWER_TEXT,
        ]
    )
    result = _run(monkeypatch, provider, environment_root)

    turn0, turn1 = result.trajectory[0], result.trajectory[1]
    assert turn0["retrieval_query"] == f"{QUESTION} Whiskers the cat residence name"
    assert turn1["retrieval_query"] == f"{QUESTION} Ashcombe residence paint color"
    # remove_repeat_docs: the second round brings genuinely NEW documents.
    assert not set(turn0["retrieved_doc_ids"]) & set(turn1["retrieved_doc_ids"])


def test_gap_phrases_genuinely_change_the_bm25_ranking(environment_root: Path) -> None:
    """Behavioral, not merely structural: the target+slot phrase the judge
    emitted really does move the colour-bearing document up the shared
    BM25+RRF ranking, so the gap-to-query mechanism is doing retrieval work
    rather than just being threaded through."""
    retriever = SharedCorpusRetriever(environment_root, DOCS, remove_repeat_docs=False)
    _, _, bare = retriever.search("What color is it?", [], k=3)
    _, _, steered = retriever.search(
        build_query_from_missing(
            "What color is it?",
            [{"target": "Ashcombe residence", "slot": "paint color", "description": ""}],
        ),
        [],
        k=3,
    )
    assert "doc1" not in bare
    assert "doc1" in steered


# ===========================================================================
# 4. Sufficient evidence terminates the loop BEFORE the max-turn cap.
# ===========================================================================


def test_sufficient_judgment_terminates_before_the_max_turn_cap(
    monkeypatch: pytest.MonkeyPatch, environment_root: Path
) -> None:
    provider = _ScriptedProvider(
        [JUDGE_INSUFFICIENT_BRIDGE, SELECTOR_PICK, JUDGE_SUFFICIENT, ANSWER_TEXT]
    )
    result = _run(monkeypatch, provider, environment_root)

    assert result.termination_reason == "judge_declared_sufficient"
    assert result.metadata["turns_used"] == 2  # turn 0 and turn 1 only
    assert result.metadata["turns_used"] < MAX_TURNS + 1
    assert result.metadata["n_judge_calls"] == 2
    assert result.metadata["n_retrieval_calls"] == 1
    assert len(provider.answer_prompts()) == 1
    assert result.final_answer == "cerulean"


# ===========================================================================
# 5. The max-turn cap itself terminates the loop (no hang, no infinite loop).
# ===========================================================================


def test_max_turn_cap_terminates_even_when_the_judge_never_reports_sufficient(
    monkeypatch: pytest.MonkeyPatch, environment_root: Path
) -> None:
    # Empty script -> the stub's role-aware default always says INSUFFICIENT.
    provider = _ScriptedProvider([], cap=64)
    result = _run(monkeypatch, provider, environment_root)

    assert result.termination_reason == "max_turns_reached"
    # turn = 0..max_turns inclusive -> max_turns + 1 judge calls.
    assert result.metadata["turns_used"] == MAX_TURNS + 1 == 5
    assert result.metadata["n_judge_calls"] == 5
    # Retrieval only while turn < max_turns -> exactly max_turns rounds.
    assert result.metadata["n_retrieval_calls"] == MAX_TURNS == 4
    assert result.metadata["n_answer_calls"] == 1
    # Judge calls are the cap-bound worst case: max_turns + 1 = 5.
    assert len(provider.judge_prompts()) == 5
    assert len(provider.answer_prompts()) == 1
    # Evidence-Extractor calls are bounded by DOCUMENT-POOL EXHAUSTION, not
    # by max_turns: with a 10-document pool and top_docs=6 and
    # remove_repeat_docs=True, round 1 takes 6 and round 2 takes the
    # remaining 4; rounds 3 and 4 retrieve nothing, so upstream's own
    # "no usable sentences -> no selector call" branch fires and NO LLM
    # call is made for them. This is the real per-question worst case for
    # HotpotQA/2WikiMultihopQA (10 docs) in this suite.
    assert len(provider.selector_prompts()) == 2
    assert result.metadata["n_selector_calls"] == 2
    assert len(provider.prompts) == 5 + 2 + 1 == 8


def test_max_turn_cap_is_configurable_and_still_bounded(
    monkeypatch: pytest.MonkeyPatch, environment_root: Path
) -> None:
    provider = _ScriptedProvider([], cap=32)
    result = _run(monkeypatch, provider, environment_root, max_turns=1)
    assert result.metadata["turns_used"] == 2
    assert result.metadata["n_retrieval_calls"] == 1
    assert len(provider.prompts) == 2 + 1 + 1  # 2 judge + 1 selector + 1 answer


# ===========================================================================
# 6. Structural leakage test: no gold/supporting-fact ever reaches any prompt.
# ===========================================================================


def test_no_gold_or_supporting_fact_metadata_reaches_retrieval_judge_or_answer(
    monkeypatch: pytest.MonkeyPatch, environment_root: Path
) -> None:
    secret = "SECRET_GOLD_VALUE_MUST_NEVER_APPEAR"
    provider = _ScriptedProvider([], cap=64)
    monkeypatch.setattr(s2g_module, "_ZeroTemperatureProvider", lambda model: provider)

    example = TaskExample(
        benchmark="hotpotqa",
        task_id="t1",
        question=QUESTION,
        reference=json.dumps([secret]),
        metadata={
            "documents": [d.model_dump() for d in DOCS],
            "supporting_doc_ids": [secret],
            "gold_answer_secret": secret,
            "answerable": True,
        },
    )
    result = S2GRAGAdapter().run(example, environment_root)

    assert provider.prompts, "the test is vacuous unless prompts were built"
    for prompt in provider.prompts:
        assert secret not in prompt
    assert secret not in json.dumps(result.model_dump(), default=str)


def test_run_reads_no_gold_bearing_field_of_the_task_example(
    monkeypatch: pytest.MonkeyPatch, environment_root: Path
) -> None:
    """Structural: run() reads `example.metadata["documents"]`,
    `example.question`, `example.benchmark` and `example.task_id` only --
    never `reference`, never any supporting-fact key."""
    # run() plus every hook it delegates the TaskExample to -- the complete
    # set of places this adapter can touch the example at all.
    source = inspect.getsource(S2GRAGAdapter.run) + inspect.getsource(
        S2GRAGAdapter._build_retriever
    )
    accessed = set(re.findall(r"example\.(\w+)", source))
    assert accessed <= {"metadata", "question", "benchmark", "task_id"}
    assert "reference" not in accessed
    metadata_keys = set(re.findall(r'example\.metadata\[\s*"(\w+)"\s*\]', source))
    assert metadata_keys == {"documents"}

    # Scan executable source only: the module docstring and the DEVIATIONS
    # table legitimately NAME these fields to explain that they are not read.
    module_source = inspect.getsource(s2g_module).replace(s2g_module.__doc__ or "", "")
    for forbidden in ("supporting_doc_ids", "is_supporting", "supporting_facts", "answer_aliases"):
        assert not re.search(rf'\[\s*["\']{forbidden}["\']\s*\]', module_source)
        assert not re.search(rf"\.get\(\s*[\"']{forbidden}[\"']", module_source)
    # `reference` is never read anywhere in the module's executable source.
    assert not re.search(r"\.reference\b", module_source)


def test_selector_and_evidence_blocks_are_built_only_from_retrieved_text() -> None:
    titles = ["Ashcombe residence"]
    texts = ["The Ashcombe residence is painted cerulean. It was built in 1899."]
    prompt, id_map = build_selector_user_prompt(QUESTION, titles, texts, [])
    assert prompt is not None
    assert "MISSING FACTS TO FILL:\nNone." in prompt
    assert "You may select up to 6 sentences." in prompt
    assert set(id_map) == {1, 2}

    per_doc = parse_selector_output(json.dumps({"evidence_global_ids": [1, 99, "2"]}), id_map, 1)
    assert per_doc == [[1, 2]]
    block = merge_evidence_only(titles, texts, per_doc)
    assert block.startswith("[Ashcombe residence] EVIDENCE:")
    assert "cerulean" in block


def test_selector_is_skipped_entirely_when_a_turn_retrieved_nothing() -> None:
    sentinel = ["No results found."]
    prompt, id_map = build_selector_user_prompt(QUESTION, sentinel, sentinel, [])
    assert prompt is None and id_map == {}


# ===========================================================================
# 7. Predictions reach the existing QA scorer with the right shape/contract.
# ===========================================================================


def test_prediction_reaches_the_existing_qa_scorer_with_the_right_contract(
    monkeypatch: pytest.MonkeyPatch, environment_root: Path
) -> None:
    from ant.benchmarks import _hotpot_style

    provider = _ScriptedProvider(
        [JUDGE_INSUFFICIENT_BRIDGE, SELECTOR_PICK, JUDGE_SUFFICIENT, ANSWER_TEXT]
    )
    result = _run(monkeypatch, provider, environment_root)

    seen: list[tuple] = []
    real_score_qa = _hotpot_style.score_qa

    def spy(prediction: str, ground_truths: list[str]):
        seen.append((prediction, list(ground_truths)))
        return real_score_qa(prediction, ground_truths)

    monkeypatch.setattr(_hotpot_style, "score_qa", spy)
    metric = _hotpot_style.score_hotpot_style(_example(), result, benchmark_name="hotpotqa")

    assert seen == [("cerulean", ["cerulean"])]
    assert metric.benchmark == "hotpotqa"
    assert metric.task_id == "t1"
    assert metric.native_score == 1.0
    assert metric.normalized_score == 100.0
    assert metric.submetrics == {"exact_match": 1.0, "f1": 1.0}
    assert metric.metadata["prediction"] == "cerulean"
    assert metric.metadata["generation_model"] == "gpt-4.1"
    assert metric.metadata["scoring_method"] == "official_em_f1"


def test_musique_adapter_scores_an_s2g_result_unchanged(
    monkeypatch: pytest.MonkeyPatch, environment_root: Path
) -> None:
    from ant.benchmarks.musique import MuSiQueAdapter

    provider = _ScriptedProvider(
        [JUDGE_INSUFFICIENT_BRIDGE, SELECTOR_PICK, JUDGE_SUFFICIENT, ANSWER_TEXT]
    )
    monkeypatch.setattr(s2g_module, "_ZeroTemperatureProvider", lambda model: provider)
    example = _example(benchmark="musique")
    result = S2GRAGAdapter().run(example, environment_root)

    metric = MuSiQueAdapter().score(example, result)
    assert metric.benchmark == "musique"
    assert metric.native_score == 1.0
    assert metric.submetrics["exact_match"] == 1.0


def test_agent_result_shape_matches_the_suite_contract(
    monkeypatch: pytest.MonkeyPatch, environment_root: Path
) -> None:
    provider = _ScriptedProvider(
        [JUDGE_INSUFFICIENT_BRIDGE, SELECTOR_PICK, JUDGE_SUFFICIENT, ANSWER_TEXT]
    )
    result = _run(monkeypatch, provider, environment_root)

    assert result.method == "s2g_rag"
    assert result.benchmark == "hotpotqa"
    assert result.task_id == "t1"
    assert isinstance(result.final_answer, str) and result.final_answer
    assert result.usage.llm_calls == 4
    assert result.usage.tool_calls == 1
    # JSON-serializable, as run_suite/trajectory dumping requires.
    json.dumps(result.model_dump(), default=str)
    assert result.metadata["deviations"][0]["official_setting"].startswith("S2G-Judge is Llama")


def test_unparseable_judge_output_is_treated_as_insufficient_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, environment_root: Path
) -> None:
    provider = _ScriptedProvider(
        ["not json at all", SELECTOR_PICK, JUDGE_SUFFICIENT, ANSWER_TEXT]
    )
    result = _run(monkeypatch, provider, environment_root)
    assert result.trajectory[0]["sufficient"] is False
    assert result.trajectory[0]["gap_items"] == []
    assert result.trajectory[0]["retrieval_query"] == QUESTION
    assert result.final_answer == "cerulean"


def test_fenced_json_judge_output_is_parsed(
    monkeypatch: pytest.MonkeyPatch, environment_root: Path
) -> None:
    fenced = "```json\n" + JUDGE_INSUFFICIENT_BRIDGE + "\n```"
    provider = _ScriptedProvider([fenced, SELECTOR_PICK, JUDGE_SUFFICIENT, ANSWER_TEXT])
    result = _run(monkeypatch, provider, environment_root)
    assert result.trajectory[0]["gap_items"][0]["target"] == "Whiskers the cat"
