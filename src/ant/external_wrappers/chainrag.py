"""ChainRAG: paper-faithful adaptation. See
docs/chainrag_fidelity_audit.md for the full audit (paper + pinned
official source read before implementation).

Paper: Zhu et al., "Mitigating Lost-in-Retrieval Problems in Retrieval
Augmented Multi-Hop Question Answering," ACL 2025 Main. arXiv:2502.14245.
Official repo: https://github.com/nju-websoft/ChainRAG, pinned commit
0f115b1fa03698b5d9cd2662a26dfdfde62d074a (GPL-3.0).

Every prompt and every piece of algorithmic decision logic below is
reproduced VERBATIM from the pinned source (`retrieval.py`,
`sentence_graph.py`) -- only the backbone model (GPT-4.1 vs upstream's
gpt-4o-mini), temperature (0 vs upstream's 0.2), and retry mechanism
(this suite's own transient-only policy vs upstream's tenacity
decorator) are substituted, per the governing spec's Section 5/14. The
embedding model (`text-embedding-3-small`, real OpenAI API calls) and
reranker (`BAAI/bge-reranker-large` via `FlagEmbedding.FlagReranker`,
local) are preserved unchanged -- this is a dense-retrieval-and-rerank
method, and swapping either would no longer be ChainRAG.

Corpus adaptation (the only unavoidable one, per the audit's own
"Corpus adaptation" section): our frozen examples are a LIST of
DocumentRecords, not upstream's own single pre-flattened context string.
Each document's text is sentence-split independently and the resulting
sentence lists are concatenated in document order, giving the same flat
ordered sentence sequence upstream's own single-text `build_graph` would
produce, while preserving a sentence -> source-document mapping for the
post-hoc supporting-document-recall diagnostic (never used at inference
time).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import spacy
from rank_bm25 import BM25Okapi

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.retry_policy import call_with_transient_retry
from ant.evaluation_suite.usage import UsageStats

PAPER_CITATION = (
    'Zhu, Sun, Cheng, Xu, Hu, Wang, Wang, Sun, Qu. "Mitigating Lost-in-Retrieval '
    'Problems in Retrieval Augmented Multi-Hop Question Answering." ACL 2025 Main, '
    "pp. 22362-22375. arXiv:2502.14245."
)
IMPLEMENTATION_LABEL = "paper-faithful ChainRAG adaptation"
UPSTREAM_COMMIT = "0f115b1fa03698b5d9cd2662a26dfdfde62d074a"
UPSTREAM_REPO_NOTE = (
    "https://github.com/nju-websoft/ChainRAG (GPL-3.0), pinned at the commit above "
    "(2025-10-28, latest at audit time). Every prompt and decision-logic branch below "
    "is reproduced verbatim from retrieval.py/sentence_graph.py at that commit. See "
    "docs/chainrag_fidelity_audit.md for the full audit."
)

# --- Frozen upstream hyperparameters (see fidelity audit's own table -- all
# taken directly from the pinned source, none tuned against this evaluation). ---
CANDIDATE_POOL_K = 100
SEED_K = 7
SIMILARITY_EDGE_K = 10
POSITION_WINDOW = 3
ENTITY_IMPORTANCE_PERCENTILE = 40
MAX_CONTEXT_WORDS = 3000
MAX_HOPS = 3
REFERENCE_PRONOUNS = (
    "this",
    "that",
    "the",
    "these",
    "those",
    "it",
    "they",
    "he",
    "she",
    "his",
    "her",
    "its",
    "their",
)

# --- Disclosed model substitution (Section 5): GPT-4.1 / temperature=0, not
# upstream's gpt-4o-mini / temperature=0.2. ---
DEFAULT_MODEL = "gpt-4.1"

EMBEDDING_MODEL = "text-embedding-3-small"
# Public OpenAI pricing, standard tier, verified 2026-09 (not a local/free model --
# unlike this suite's own BGE-small dense baseline, every embedding call here is a
# real, billed OpenAI API call).
EMBEDDING_PRICE_PER_MILLION_TOKENS = 0.02
RERANKER_MODEL = "BAAI/bge-reranker-large"

_NLP_MODEL_NAME = "en_core_web_sm"
_nlp: spacy.language.Language | None = None


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(_NLP_MODEL_NAME)
    return _nlp


_shared_reranker: Any = None
_shared_reranker_attempted = False


def _get_shared_reranker() -> Any:
    """Loaded once per process, shared across every example -- a fixed,
    stateless pretrained scorer (FlagReranker.compute_score has no memory
    across calls), exactly like `ant.retrieval.dense.get_shared_embedder`'s
    own one-load-per-process convention. Reusing it across examples never
    leaks information between them."""
    global _shared_reranker, _shared_reranker_attempted
    if not _shared_reranker_attempted:
        _shared_reranker_attempted = True
        from FlagEmbedding import FlagReranker

        _shared_reranker = FlagReranker(model_name_or_path=RERANKER_MODEL, use_fp16=False)
    return _shared_reranker


class _ZeroTemperatureProvider(CountingOpenAIProvider):
    """Same override pattern already used by
    `ant.evaluation_suite.answer_extraction._ZeroTemperatureProvider`."""

    def _responses_kwargs(self, prompt: str, max_output_tokens: int) -> dict:
        kwargs = super()._responses_kwargs(prompt, max_output_tokens)
        kwargs["temperature"] = 0
        return kwargs


@dataclass
class _EmbeddingUsage:
    calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0


def _embed_texts(
    client: Any, texts: list[str], usage: _EmbeddingUsage, batch_size: int = 20
) -> list[np.ndarray]:
    """Real OpenAI `text-embedding-3-small` calls, batched at 20 (matching
    upstream's own `custom_embedding`). Retried only via this suite's own
    frozen transient-provider policy -- never upstream's tenacity decorator.
    """
    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        def _call(batch: list[str] = batch) -> Any:
            return client.embeddings.create(
                model=EMBEDDING_MODEL, input=batch, encoding_format="float"
            )

        response, outcome = call_with_transient_retry(_call)
        if response is None:
            assert outcome.final_error is not None
            raise outcome.final_error
        usage.calls += 1
        usage.tokens += response.usage.total_tokens
        usage.cost_usd += (
            response.usage.total_tokens / 1_000_000 * EMBEDDING_PRICE_PER_MILLION_TOKENS
        )
        vectors.extend(np.array(item.embedding, dtype=np.float32) for item in response.data)
    return vectors


# ===========================================================================
# Sentence graph construction -- verbatim from sentence_graph.py.
# ===========================================================================


def _split_sentences_per_document(documents: list[Any]) -> tuple[list[str], list[str]]:
    """The only corpus adaptation (see module docstring): sentence-splits
    each document independently, in document order, preserving a
    sentence -> source-document-id mapping for the post-hoc diagnostic
    only. Produces the same flat ordered sentence list upstream's own
    single-text `build_graph` would, for the exact same total content."""
    nlp = _get_nlp()
    sentences: list[str] = []
    source_doc_ids: list[str] = []
    for doc in documents:
        spacy_doc = nlp(doc.text)
        for sent in spacy_doc.sents:
            text = sent.text.strip()
            if text:
                sentences.append(text)
                source_doc_ids.append(doc.doc_id)
    return sentences, source_doc_ids


def _compute_similarity_matrix(sentence_embeddings: list[np.ndarray]) -> np.ndarray:
    embeddings = np.array(sentence_embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_normalized = embeddings / norms
    return np.dot(embeddings_normalized, embeddings_normalized.T)


def _calculate_bm25_importance(ent_lists: list[list[str]]) -> tuple[set[str], dict[str, float]]:
    entity_docs = []
    all_entities: set[str] = set()
    for ents in ent_lists:
        entity_docs.append(ents)
        all_entities.update(ents)

    if not all_entities:
        return set(), {}

    bm25 = BM25Okapi(entity_docs)
    entity_scores: dict[str, float] = {}
    for entity in all_entities:
        scores = bm25.get_scores([entity])
        entity_scores[entity] = float(np.mean(scores))

    scores = list(entity_scores.values())
    threshold = float(np.percentile(scores, ENTITY_IMPORTANCE_PERCENTILE))
    high_importance = {ent for ent, score in entity_scores.items() if score > threshold}
    return high_importance, entity_scores


def _build_sentence_graph(
    sentences: list[str],
    similarity_matrix: np.ndarray,
    ent_lists: list[list[str]],
    entity_scores: dict[str, float],
    k: int = SIMILARITY_EDGE_K,
) -> nx.Graph:
    graph = nx.Graph()
    n = len(sentences)
    k = min(k, max(0, n - 1))
    sentence_ents = [set(ents) for ents in ent_lists]
    graph.add_nodes_from(range(n))

    if k > 0:
        for i in range(n):
            similarities = similarity_matrix[i].copy()
            similarities[i] = -1
            top_k_indices = np.argpartition(similarities, -k)[-k:]
            graph.add_edges_from(
                (i, int(j), {"weight": float(similarities[j]), "label": "similarity"})
                for j in top_k_indices
            )

    for i in range(n):
        for j in range(max(0, i - POSITION_WINDOW), min(n, i + POSITION_WINDOW + 1)):
            if i != j:
                graph.add_edge(i, j, weight=1.0 / (abs(i - j) + 1), label="position")

    for i in range(n):
        for j in range(i + 1, n):
            common = sentence_ents[i] & sentence_ents[j]
            if common:
                weight = sum(entity_scores.get(ent, 0.0) for ent in common)
                graph.add_edge(i, j, weight=weight, label="entity", entities=list(common))

    return graph


# ===========================================================================
# Question decomposition -- verbatim prompts/parsing from sentence_graph.py.
# ===========================================================================

_JUDGE_PROMPT = """You are a helpful AI assistant that determines if a question requires multiple steps to answer.

        Guidelines for identifying multi-hop questions:
        1. The question requires finding and connecting multiple pieces of information
        2. The answer cannot be found in a single direct statement
        3. You need to find intermediate information to reach the final answer

        Output format should be a JSON object with only one fields:
        - "is_multi_hop": boolean (true/false)

        Example:
        Question: "Who is the paternal grandmother of Marie Of Brabant, Queen Of France?"
        Output: {"is_multi_hop": false}
        Question: "Who is Archibald Acheson, 4Th Earl Of Gosford's paternal grandfather?"
        Output: {"is_multi_hop": false}
        Question: "Who was the wife of the person who founded Microsoft?"
        Output: {"is_multi_hop": true}"""

_DECOMPOSE_SYSTEM_PROMPT = """You are a helpful AI assistant that helps break down questions into minimal necessary sub-questions.
        Guidelines:
        1. Only break down the question if it requires finding and connecting multiple distinct pieces of information
        2. Each sub-question should target a specific, essential piece of information
        3. Avoid generating redundant or overlapping sub-questions
        4. For questions about impact/significance, focus on:
        - What was the thing/event
        - What was its impact/significance
        5. For comparison questions between two items (A vs B):
        - First identify the specific attribute being compared for each item
        - Then ask about that attribute for each item separately
        - For complex comparisons, add a final question to compare the findings
        6.**Logical Progression**:
        Sub-questions should have clear relationships, such as:
        - **Parallel**: Independent sub-questions that both contribute to answering the original question.
        Example:
        Original: "What are the causes and consequences of climate change on global ecosystems?"
        Output: ["What are the main causes of climate change?", "What are the major consequences of climate change on global ecosystems?"]
        - **Sequential**: Sub-questions that build upon each other step-by-step.
        Example:
        Original: "What university, founded in 1890, is known for its groundbreaking work in economics?"
        Output: ["Which universities were founded in 1890?", "Which of these universities is known for its groundbreaking work in economics?"]
        - **Comparative**: Questions that compare attributes between items.
        Example 1:
        Original: "Which film has the director who was born earlier, The Secret Invasion or The House Of The Seven Hawks?"
        Output: ["Who directed The Secret Invasion and when was this director born?", "Who directed The House Of The Seven Hawks and when was this director born?"]
        Example 2:
        Original: "Do both films The Reincarnation Of Golden Lotus and I'll Get By (Film) have directors from the same country?"
        Output: ["Who directed The Reincarnation Of Golden Lotus and which country is he/she from?", "Who directed I'll Get By (Film) and which country is he/she from?"]

        7. Keep the total number of sub-questions minimal (usually 2 at most)

        Output format should be a JSON array of sub-questions. For example:
        Original: "Were the wireless earbuds Apple introduced in 2016 revolutionary for the market?"
        Output: ["What wireless earbuds did Apple introduce in 2016?", "How did these earbuds impact the wireless earbud market?"]

        Remember: Each sub-question must be necessary and distinct. Do not create redundant questions. For comparison questions, focus on gathering the specific information needed for the comparison in the most efficient way."""

_FORCE_ANSWER_PROMPT = """Based on the given context, you must provide an answer with fewest words to the question.
        Only give me the answer and do not output any other words.

        Context: {context}
        Question: {question}

        Provide your best possible answer:"""

_ANSWER_WITH_SUBQ_PROMPT = """Based on the answers to the sub-questions, use the fewest words possible to answer the original question.
        Only give me the answer and do not output any other words.
        Original Question: {original_question}
        Sub-questions and their answers:
        {sub_answers}
        Provide the shortest possible answer:"""

_ANSWER_WITH_CONTEXT_PROMPT = """Based on the following context, use the fewest words possible to answer the original question.
        Only give me the answer and do not output any other words.
        Context: {context}
        Question: {question}

        Answer:"""

_CAN_ANSWER_PROMPT = """Based on the following context, can you answer the question?
        Please respond with 'yes' or 'no' only.

        Question: {question}
        Context: {context}

        Can you answer the question based on this context? (yes/no):"""


def _extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0]
    cleaned = cleaned.strip()
    if not cleaned.startswith("{"):
        cleaned = "{" + cleaned.split("{", 1)[1]
    if not cleaned.endswith("}"):
        cleaned = cleaned.rsplit("}", 1)[0] + "}"
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        cleaned = re.sub(r"[^\{\}\:\"\,\w\.\-\_\s]", "", cleaned)
        return json.loads(cleaned)


def _extract_json_array(text: str) -> list:
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        if cleaned.startswith("[") and cleaned.endswith("]"):
            items = cleaned[1:-1].split('","')
            return [item.strip("\"'") for item in items]
        raise


@dataclass
class _CallLog:
    n_llm_calls: int = 0
    n_embedding_calls: int = 0


def decompose_question(question: str, llm: Any, calls: _CallLog) -> list[str]:
    try:
        judge_full_prompt = (
            f"{_JUDGE_PROMPT}\n\nQuestion: {question}\n\nIs this a multi-hop question?"
        )
        try:
            response = llm(judge_full_prompt, max_output_tokens=200)
            calls.n_llm_calls += 1
            judgment = _extract_json_object(response)
            if not judgment.get("is_multi_hop", True):
                return [question]
        except Exception:  # noqa: BLE001 -- malformed judge output is a safe-fallback case
            return [question]

        full_prompt = (
            f"{_DECOMPOSE_SYSTEM_PROMPT}\n\nQuestion: {question}\n\n"
            "Break down this question into minimal necessary sub-questions:"
        )
        try:
            response = llm(full_prompt, max_output_tokens=500)
            calls.n_llm_calls += 1
            sub_questions = _extract_json_array(response)
            if isinstance(sub_questions, list) and sub_questions:
                return sub_questions
            return [question]
        except Exception:  # noqa: BLE001 -- malformed decompose output is a safe-fallback case
            return [question]
    except Exception:  # noqa: BLE001 -- matches upstream's own outermost safety net
        return [question]


# ===========================================================================
# Main planner -- verbatim decision logic from retrieval.py's QuestionPlanner.
# ===========================================================================


class _QuestionPlanner:
    def __init__(self, llm: Any, reranker: Any) -> None:
        self.llm = llm
        self.reranker = reranker
        self.sentences: list[str] = []
        self.sentence_embeddings: list[np.ndarray] = []
        self.sentence_graph: nx.Graph | None = None
        self.similarity_matrix: np.ndarray | None = None

    def build_graph(self, sentences: list[str], embeddings: list[np.ndarray]) -> None:
        self.sentences = sentences
        self.sentence_embeddings = embeddings
        nlp = _get_nlp()
        ent_lists = [[ent.text for ent in doc.ents] for doc in nlp.pipe(sentences)]
        self.similarity_matrix = _compute_similarity_matrix(embeddings)
        _high_importance, entity_scores = _calculate_bm25_importance(ent_lists)
        self.sentence_graph = _build_sentence_graph(
            sentences, self.similarity_matrix, ent_lists, entity_scores
        )

    def get_n_hop_neighbors(self, sentence: str, n_hops: int) -> set[str]:
        try:
            if self.sentence_graph is None:
                return set()
            sent_idx = self.sentences.index(sentence)
            ego = nx.ego_graph(self.sentence_graph, sent_idx, radius=n_hops)
            return {self.sentences[node] for node in ego.nodes() if node != sent_idx}
        except Exception:  # noqa: BLE001 -- matches upstream's own broad catch
            return set()

    def sort_context(self, question: str, context_sentences: list[str]) -> list[str]:
        if not context_sentences:
            return []
        try:
            pairs = [[question, sent] for sent in context_sentences]
            scores = self.reranker.compute_score(pairs)
            if not isinstance(scores, list):
                scores = [scores]
            scored = sorted(zip(scores, context_sentences, strict=True), reverse=True)
            return [sent for _, sent in scored]
        except Exception:  # noqa: BLE001 -- matches upstream's own broad catch
            return context_sentences

    def sort_section(
        self, question: str, sentences: list[tuple[float, str]], k: int
    ) -> list[tuple[float, str]]:
        try:
            pairs = [[question, sent] for _, sent in sentences]
            scores = self.reranker.compute_score(pairs)
            if not isinstance(scores, list):
                scores = [scores]
            scored = [(score, sent) for score, (_, sent) in zip(scores, sentences, strict=True)]
            scored.sort(reverse=True)
            return scored[:k]
        except Exception:  # noqa: BLE001 -- matches upstream's own broad catch
            return sentences[:k]

    def find_top_k_sentences(
        self, q_embedding: np.ndarray, question: str, k: int
    ) -> list[tuple[float, str]]:
        q_norm = np.linalg.norm(q_embedding)
        if q_norm == 0:
            return [(0.0, s) for s in self.sentences[:k]]
        scores = []
        for idx, emb in enumerate(self.sentence_embeddings):
            norm = np.linalg.norm(emb)
            similarity = 0.0 if norm == 0 else float(np.dot(q_embedding, emb) / (q_norm * norm))
            scores.append((similarity, self.sentences[idx]))
        scores.sort(reverse=True)
        top_pool = scores[: min(CANDIDATE_POOL_K, len(scores))]
        return self.sort_section(question, top_pool, k)

    def can_answer_question(self, question: str, context_sentences: list[str]) -> bool:
        response = (
            self.llm(
                _CAN_ANSWER_PROMPT.format(question=question, context="\n".join(context_sentences)),
                max_output_tokens=16,
            )
            .strip()
            .lower()
        )
        return "yes" in response

    def force_answer(self, sub_question: str, context_sentences: list[str]) -> str:
        response = self.llm(
            _FORCE_ANSWER_PROMPT.format(context=" ".join(context_sentences), question=sub_question),
            max_output_tokens=100,
        )
        return response.strip()

    def answer_original_question(self, question: str, sub_results: list[dict]) -> dict[str, str]:
        sub_answers_text = "\n".join(
            f"Sub-question: {r['sub_question']}\nAnswer: {r['answer']}" for r in sub_results
        )
        all_contexts: list[str] = []
        for r in sub_results:
            all_contexts.extend(r["context"])
        unique_contexts = list(dict.fromkeys(all_contexts))
        sorted_contexts = self.sort_context(question, unique_contexts)

        answer_with_subq = self.llm(
            _ANSWER_WITH_SUBQ_PROMPT.format(
                original_question=question, sub_answers=sub_answers_text
            ),
            max_output_tokens=100,
        ).strip()
        answer_with_context = self.llm(
            _ANSWER_WITH_CONTEXT_PROMPT.format(
                context="\n".join(sorted_contexts), question=question
            ),
            max_output_tokens=100,
        ).strip()
        return {
            "answer_with_subquestions": answer_with_subq,
            "answer_without_subquestions": answer_with_context,
        }

    def plan(self, question: str, embed_query_fn, calls: _CallLog) -> dict[str, Any]:
        sub_questions = decompose_question(question, self.llm, calls)
        sub_results: list[dict] = []
        previous_answers: dict[str, str] = {}
        previous_context_summary: str | None = None

        for sub_q in sub_questions:
            try:
                modified_sub_q = sub_q
                for prev_q, prev_ans in previous_answers.items():
                    if any(ref in sub_q.lower() for ref in REFERENCE_PRONOUNS):
                        should_replace = (
                            self.llm(
                                f"""
                        Previous question: {prev_q}
                        Previous answer: {prev_ans}
                        Current question: {sub_q}
                        Does the current question refer to the answer of the previous question? Answer yes or no.
                        """,
                                max_output_tokens=16,
                            )
                            .strip()
                            .lower()
                        )
                        calls.n_llm_calls += 1
                        if should_replace == "yes":
                            modified_sub_q = self.llm(
                                f"""
                            Rewrite the following question to be self-contained by replacing pronouns or references with the actual entities they refer to.
                            Previous question: {prev_q}
                            Previous answer: {prev_ans}
                            Current question: {sub_q}
                            Rewritten question:
                            """,
                                max_output_tokens=100,
                            ).strip()
                            calls.n_llm_calls += 1
                            break

                q_embedding = embed_query_fn(modified_sub_q)
                top_k_sentences = self.find_top_k_sentences(q_embedding, modified_sub_q, k=SEED_K)
                context = [sent for _, sent in top_k_sentences]

                if previous_context_summary:
                    context.append(f"Previous context summary: {previous_context_summary}")

                processed = set(context)
                current_words = sum(len(s.split()) for s in context)

                calls.n_llm_calls += 1
                if self.can_answer_question(modified_sub_q, context):
                    answer = self.force_answer(modified_sub_q, context)
                    calls.n_llm_calls += 1
                    if "unable" not in answer.lower() and "don't know" not in answer.lower():
                        previous_answers[modified_sub_q] = answer
                        sub_results.append(
                            {
                                "sub_question": sub_q,
                                "modified_sub_question": modified_sub_q,
                                "context": context,
                                "answer": answer,
                            }
                        )
                        previous_context_summary = (
                            f"For the question '{modified_sub_q}', the answer is: {answer}"
                        )
                        continue

                one_hop = set()
                for _, seed in top_k_sentences:
                    one_hop.update(self.get_n_hop_neighbors(seed, n_hops=1))
                for neighbor in one_hop:
                    if neighbor not in processed and current_words < MAX_CONTEXT_WORDS:
                        context.append(neighbor)
                        processed.add(neighbor)
                        current_words += len(neighbor.split())
                        if current_words >= MAX_CONTEXT_WORDS:
                            break

                context = self.sort_context(modified_sub_q, context)
                calls.n_llm_calls += 1
                if self.can_answer_question(modified_sub_q, context):
                    answer = self.force_answer(modified_sub_q, context)
                    calls.n_llm_calls += 1
                    if "unable" not in answer.lower() and "don't know" not in answer.lower():
                        previous_answers[modified_sub_q] = answer
                        sub_results.append(
                            {
                                "sub_question": sub_q,
                                "modified_sub_question": modified_sub_q,
                                "context": context,
                                "answer": answer,
                            }
                        )
                        previous_context_summary = (
                            f"For the question '{modified_sub_q}', the answer is: {answer}"
                        )
                        continue

                current_hop = 2
                while current_hop <= MAX_HOPS and current_words < MAX_CONTEXT_WORDS:
                    higher_hop = set()
                    for _, seed in top_k_sentences:
                        higher_hop.update(self.get_n_hop_neighbors(seed, n_hops=current_hop))
                    for neighbor in higher_hop:
                        if neighbor not in processed and current_words < MAX_CONTEXT_WORDS:
                            context.append(neighbor)
                            processed.add(neighbor)
                            current_words += len(neighbor.split())
                            if current_words >= MAX_CONTEXT_WORDS:
                                break
                    if current_words >= MAX_CONTEXT_WORDS:
                        break
                    current_hop += 1

                context = self.sort_context(modified_sub_q, context)
                if not any(r["sub_question"] == sub_q for r in sub_results):
                    answer = self.force_answer(modified_sub_q, context)
                    calls.n_llm_calls += 1
                    previous_answers[modified_sub_q] = answer
                    sub_results.append(
                        {
                            "sub_question": sub_q,
                            "modified_sub_question": modified_sub_q,
                            "context": context,
                            "answer": answer,
                        }
                    )
                    previous_context_summary = (
                        f"For the question '{modified_sub_q}', the answer is: {answer}"
                    )

            except Exception:  # noqa: BLE001 -- matches upstream's own per-sub-question safety net
                sub_results.append(
                    {
                        "sub_question": sub_q,
                        "modified_sub_question": sub_q,
                        "context": [],
                        "answer": "Unable to process this sub-question due to error",
                    }
                )

        final_answers = self.answer_original_question(question, sub_results)
        calls.n_llm_calls += 2
        return {
            "original_question": question,
            "sub_questions": sub_questions,
            "sub_results": sub_results,
            "final_answer_with_subquestions": final_answers["answer_with_subquestions"],
            "final_answer_without_subquestions": final_answers["answer_without_subquestions"],
        }


class ChainRAGAdapter:
    """Paper-faithful ChainRAG. `environment_root` is unused (like
    `DirectDocumentAgent`) -- ChainRAG builds its own in-memory sentence
    graph from `example.metadata["documents"]` text content directly, no
    file-based search."""

    name = "chainrag"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        del environment_root
        from ant.evaluation_suite.document_scope import DocumentRecord

        documents = [DocumentRecord(**d) for d in example.metadata["documents"]]

        started = time.time()
        provider = _ZeroTemperatureProvider(model=self.model)
        embedding_client = provider.client()
        embedding_usage = _EmbeddingUsage()
        reranker = _get_shared_reranker()
        calls = _CallLog()

        def _llm(prompt: str, max_output_tokens: int = 400) -> str:
            return provider.responses_text(prompt, max_output_tokens=max_output_tokens).text

        sentences, source_doc_ids = _split_sentences_per_document(documents)
        sentence_embeddings = (
            _embed_texts(embedding_client, sentences, embedding_usage) if sentences else []
        )

        planner = _QuestionPlanner(llm=_llm, reranker=reranker)
        if sentences:
            planner.build_graph(sentences, sentence_embeddings)

        def _embed_query(text: str) -> np.ndarray:
            [vec] = _embed_texts(embedding_client, [text], embedding_usage)
            return vec

        if sentences:
            result = planner.plan(example.question, _embed_query, calls)
        else:
            result = {
                "original_question": example.question,
                "sub_questions": [example.question],
                "sub_results": [],
                "final_answer_with_subquestions": "Unable to provide an answer due to error",
                "final_answer_without_subquestions": "Unable to provide an answer due to error",
            }

        # Precommitted canonical field (governing spec Section 9) -- chosen
        # BEFORE any scoring, matching upstream's own field name exactly.
        final_answer = result["final_answer_with_subquestions"]

        llm_calls = provider.drain_call_count()
        token_usage = provider.drain_usage()
        elapsed = time.time() - started

        retrieved_seed_paths = {
            source_doc_ids[sentences.index(sent)]
            for r in result["sub_results"]
            for sent in r["context"]
            if sent in sentences
        }

        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=final_answer,
            trajectory=[
                {
                    "sub_questions": result["sub_questions"],
                    "sub_results": [
                        {
                            "sub_question": r["sub_question"],
                            "modified_sub_question": r["modified_sub_question"],
                            "n_context_sentences": len(r["context"]),
                            "answer": r["answer"],
                        }
                        for r in result["sub_results"]
                    ],
                    "final_answer_with_subquestions": result["final_answer_with_subquestions"],
                    "final_answer_without_subquestions": result[
                        "final_answer_without_subquestions"
                    ],
                }
            ],
            usage=UsageStats(
                llm_calls=llm_calls,
                tool_calls=embedding_usage.calls,
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                total_tokens=token_usage.total_tokens,
                estimated_cost_usd=token_usage.estimated_cost_usd + embedding_usage.cost_usd,
                wall_clock_seconds=elapsed,
                unique_files_inspected=len(set(source_doc_ids)),
            ),
            termination_reason="chainrag_complete",
            metadata={
                "generation_model": self.model,
                "implementation_label": IMPLEMENTATION_LABEL,
                "upstream_commit": UPSTREAM_COMMIT,
                "upstream_repo_note": UPSTREAM_REPO_NOTE,
                "embedding_model": EMBEDDING_MODEL,
                "reranker_model": RERANKER_MODEL,
                "n_sentences_indexed": len(sentences),
                "n_sub_questions": len(result["sub_questions"]),
                "n_rewrites": sum(
                    1
                    for r in result["sub_results"]
                    if r["modified_sub_question"] != r["sub_question"]
                ),
                "n_llm_calls_logical": calls.n_llm_calls,
                "n_embedding_calls": embedding_usage.calls,
                "embedding_tokens": embedding_usage.tokens,
                "embedding_cost_usd": embedding_usage.cost_usd,
                "generation_cost_usd": token_usage.estimated_cost_usd,
                "retrieved_source_doc_ids": sorted(retrieved_seed_paths),
                "raw_answer_with_subquestions": result["final_answer_with_subquestions"],
                "raw_answer_without_subquestions": result["final_answer_without_subquestions"],
                "sub_results_full": result["sub_results"],
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(ChainRAGAdapter())
