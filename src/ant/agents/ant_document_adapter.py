"""ANT-on-documents adapter: the smallest possible bridge between the
frozen, unmodified ANT coordination core (`LocalCoordinator.ask()`,
`build_worker_cards`, the Need Graph / runtime revision / recovery
machinery inside `local.py`) and the long-document/multi-document
evaluation substrate (`ant.evaluation_suite.document_scope`).

Category classification (per the long-context evaluation spec's explicit
requirement to classify every change before implementing it):

  A. interface/generalization-only refactor: NONE needed. `LocalCoordinator`,
     `build_worker_cards`, `AutonomousWorker`, the Need Graph, runtime
     revision, and recovery logic are all imported and used completely
     unmodified -- zero lines of `ant/coordinator/local.py`,
     `ant/indexing/cards.py`, or any other core file are touched by this
     module or by this evaluation pass.
  B. repository-specific implementation cleanup: NONE needed.
  C. actual algorithmic behavior change: NONE needed, and therefore none
     was implemented. (See docs/long_context_dataset_audit.md's sibling
     finding: `self.repo_root` appears in exactly two places in all of
     `local.py` -- constructing `LocalSearchTool` and `AutonomousWorker`,
     both of which operate on any directory of text files, not
     specifically Python source -- and `Territory`/`WorkerCard` are plain
     schema-only Pydantic models with no code-specific required fields.)

The ONE thing this adapter does that `ant_adapter.py` (the repository
baseline) does not is skip `discover_territories` entirely -- that
function's own clustering heuristic (`_natural_root`, directory-based
grouping) is repository-directory-structure-specific and does not apply
to a flat directory of materialized document files. In its place,
`_document_territories` builds `Territory` objects directly from document
boundaries: ONE territory per materialized document, using only the
document's own doc_id/title/relative filename (never gold/supporting-fact
annotations, never the question, never any other benchmark metadata) --
deterministic and task-independent by construction. Everything downstream
of that (`build_worker_cards`, `IndexStore`, `LocalCoordinator`, `.ask()`)
is the exact same call sequence `ant_adapter.py` already uses.

Tool-primitive asymmetry, disclosed rather than hidden: `AutonomousWorker`
(inside frozen `local.py`) always calls `LocalSearchTool.search()`/
`dense_search()`, and its reasoner-driven tool loop can additionally call
`navigate`/`references`/`callers`/`callees`/`assignments`/`imports`/
`subclasses` -- all of which are code-symbol tools that gracefully return
empty results on prose text (zero AST symbols found) rather than crash.
Since `AutonomousWorker`'s tool loop lives inside frozen `local.py`, this
adapter cannot give ANT's own workers the new `view`/`navigate-chunk`
document primitives (`ant.tools.document_tools`) without a Category-C-
adjacent core change -- so ANT's real, meaningfully-used information
primitive on documents is `search` only. This is reported explicitly in
this evaluation pass's behavioral-validation answer (Section 14.B), not
smoothed over as if full three-way tool parity with Matched ReAct/
Retrieval were achieved.
"""
from __future__ import annotations

from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.coordinator import LocalCoordinator
from ant.domain.models import Territory
from ant.evaluation_suite.answer_contract import (
    apply_context_authoritative_regrounding,
    condense_to_answer_span,
)
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.document_scope import DocumentRecord, EvalDocumentEnvironment
from ant.evaluation_suite.natural_qa_synthesis import (
    format_evidence_block,
    is_natural_multihop_qa_benchmark,
    synthesize_natural_multihop_answer,
)
from ant.evaluation_suite.usage import UsageStats
from ant.evaluation_suite.vllm_provider import VLLMChatCompletionsProvider
from ant.indexing import build_worker_cards
from ant.memory import IndexStore


def _document_territories(environment: EvalDocumentEnvironment) -> list[Territory]:
    """One Territory per materialized document -- the natural, deterministic
    document boundary already present in the benchmark's own structure.
    Built ONLY from doc_id/title/relative-filename; never reads supporting-
    fact annotations, the question, or any other example metadata, so
    territory construction is identical regardless of which document(s)
    happen to be gold-relevant for this particular question.
    """
    territories: list[Territory] = []
    for document in environment.ordered_documents():
        relative = environment.relative_path_for(document.doc_id)
        territories.append(
            Territory(
                id=f"doc-{document.doc_id}",
                root=document.doc_id,
                files=[relative],
                summary=(
                    f"Document: {document.title}"
                    if document.title
                    else f"Document {document.doc_id}"
                ),
            )
        )
    return territories


class AntDocumentAgent:
    """Thin adapter around the frozen ANT runtime for the document/
    multi-document evaluation track -- the document-substrate counterpart
    to `ant.agents.ant_adapter.AntAgent`. Registered under a distinct name
    (`ant_document`, not `ant`) because it needs an example's own
    `metadata["documents"]` to construct `EvalDocumentEnvironment`
    (document text isn't recoverable from `environment_root` alone the way
    a repository's file tree is) -- keeping `ant_adapter.py`'s existing
    repository-only `AntAgent.run(example, environment_root)` signature
    and behavior completely untouched.
    """

    name = "ant_document"

    def __init__(
        self,
        model: str = "gpt-4.1",
        max_rounds: int = 6,
        search_top_k: int = 4,
        index_root: Path | None = None,
        worker_model: str | None = None,
        worker_base_url: str | None = None,
        worker_max_context_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.max_rounds = max_rounds
        # Default 4 reproduces prior behavior exactly for every existing
        # caller -- a disclosed passthrough to LocalCoordinator.ask()'s own
        # search_top_k, for controlled boundary-expansion experiments only.
        self.search_top_k = search_top_k
        self.index_root = index_root or Path(".ant/eval-suite-documents")
        # Same coordination/execution model split as ant_adapter.AntAgent
        # -- see that class's own docstring and
        # ant.evaluation_suite.vllm_provider for the full rationale. Both
        # left unset (the default) reproduces the frozen single-GPT-4.1-
        # provider runtime exactly.
        self.worker_model = worker_model
        self.worker_base_url = worker_base_url
        self.worker_max_context_tokens = worker_max_context_tokens

    def _index_path_for(self, example: TaskExample, environment_root: Path) -> Path:
        return self.index_root / example.benchmark / environment_root.name

    def _ensure_indexed(
        self, environment: EvalDocumentEnvironment, index_path: Path
    ) -> list[Territory]:
        territories = _document_territories(environment)
        if (index_path / "workers.json").exists():
            # Stale-index guard: an index built for this same index_path
            # earlier is only safe to reuse if it still covers exactly the
            # CURRENT environment's own searchable file set. A prior
            # index was found, live, to predate a later re-materialization
            # of the same environment directory with a different (larger)
            # document set -- IndexStore.load_workers() kept silently
            # returning the old, smaller file set forever, making every
            # document added after the index was built permanently
            # unsearchable, with zero error or warning. Comparing file
            # SETS (not counts) both catches a mismatched size and stays
            # correct if the set changed by the same count via different
            # documents. Deterministic and mechanical -- no LLM call,
            # no behavior change for the (overwhelmingly common) case
            # where nothing has actually changed.
            existing_workers = IndexStore(index_path).load_workers()
            existing_files = {file for worker in existing_workers for file in worker.files}
            current_files = {file for territory in territories for file in territory.files}
            if existing_files == current_files:
                return territories
        # build_worker_cards is frozen core, called completely unmodified --
        # see module docstring's Category A/B/C classification.
        workers = build_worker_cards(environment.root, territories)
        IndexStore(index_path).save(territories, workers)
        return territories

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        documents = [DocumentRecord(**d) for d in example.metadata["documents"]]
        environment = EvalDocumentEnvironment(environment_root, documents)
        index_path = self._index_path_for(example, environment_root)
        territories = self._ensure_indexed(environment, index_path)
        workers = IndexStore(index_path).load_workers()

        provider = CountingOpenAIProvider(model=self.model)

        assert (self.worker_model is None) == (self.worker_base_url is None), (
            "worker_model and worker_base_url must be set together (or both left unset)"
        )
        worker_provider = provider
        if self.worker_model is not None and self.worker_base_url is not None:
            worker_provider = VLLMChatCompletionsProvider(
                model=self.worker_model,
                base_url=self.worker_base_url,
                max_context_tokens=self.worker_max_context_tokens,
            )

        coordinator = LocalCoordinator(
            environment_root,
            workers,
            reasoner=provider,
            synthesizer=provider,
            index_path=index_path,
            worker_reasoner=worker_provider,
            # memory_routes / cross_repo_experience deliberately omitted --
            # same "ordinary clean runtime" shape as ant_adapter.AntAgent.
        )
        state = coordinator.ask(
            example.question, max_rounds=self.max_rounds, search_top_k=self.search_top_k
        )
        # Shared short-answer contract (Part A of the long-context spec) --
        # applied STRICTLY after coordinator.ask() has already returned its
        # complete, unmodified result. This is the critical property for
        # ANT specifically: `question` is reused throughout frozen
        # local.py for root-Need text, worker instructions, coverage-need
        # normalization, and lexical term extraction
        # (TOKEN_RE.findall(question)), so injecting the contract INTO
        # that string (the way a prompt-template change would) risks
        # measurably altering search-term extraction and worker routing --
        # a real algorithmic effect disguised as "just data". Condensing
        # `state.answer` after the fact touches none of that: routing,
        # Need Graph structure, reroutes, and recovery are provably
        # byte-identical to a run with this call deleted; only the string
        # returned to the harness changes.
        raw_answer = state.answer
        # Benchmark-scoped final-answer synthesis (fixes a benchmark-policy
        # leakage -- see ant.evaluation_suite.natural_qa_synthesis's module
        # docstring): state.answer is the frozen core coordinator's OWN
        # synthesis, developed/tuned against SWE-QA-Pro's abstention-
        # tolerant repository-QA setting. For the three standard multi-hop
        # QA benchmarks, that abstention-prone text is not used as the
        # final answer's basis at all -- a SEPARATE, evidence-driven
        # synthesis step (never reading state.answer, never touching
        # routing/Need-Graph/retrieval) is used instead. Every other
        # benchmark (including SWE-QA-Pro, which uses a completely
        # different adapter, ant.agents.ant_adapter.AntAgent, and never
        # imports this module) keeps the exact prior behavior.
        grounded_answer = raw_answer
        if is_natural_multihop_qa_benchmark(example.benchmark):
            assert example.metadata.get("answer_contract_condition") is None, (
                "answer_contract_condition (Condition B regrounding) is not defined for the "
                "natural-multihop-QA synthesis branch"
            )
            evidence_block = format_evidence_block(state.evidence)
            final_answer = synthesize_natural_multihop_answer(
                provider, example.question, evidence_block
            )
        else:
            # Single-Needle Contamination Study Condition B -- post-hoc,
            # against the SAME evidence state.evidence the frozen
            # coordinator already gathered on its own. Same non-negotiable
            # property as condensation below: this runs strictly after
            # coordinator.ask() returns, so routing/Need-Graph/recovery are
            # provably unaffected -- the question string ANT's own
            # internals saw is never touched.
            if example.metadata.get("answer_contract_condition") == "B":
                evidence_block = "\n".join(
                    f"[{item.path}:{item.line_start}-{item.line_end}] {item.quote}"
                    for item in state.evidence
                )
                grounded_answer = apply_context_authoritative_regrounding(
                    provider, example.question, raw_answer, evidence_block
                )
            final_answer = condense_to_answer_span(provider, example.question, grounded_answer)
        llm_calls = provider.drain_call_count()

        # See ant_adapter.AntAgent.run's identical block for why this is
        # drained/reported separately from `provider`'s own usage above.
        worker_usage_metadata = None
        if worker_provider is not provider:
            worker_usage = worker_provider.drain_usage()
            worker_usage_metadata = {
                "model": self.worker_model,
                "base_url": self.worker_base_url,
                "max_context_tokens": self.worker_max_context_tokens,
                "llm_calls": worker_provider.drain_call_count(),
                "input_tokens": worker_usage.input_tokens,
                "output_tokens": worker_usage.output_tokens,
                "total_tokens": worker_usage.total_tokens,
                "wall_clock_seconds": worker_usage.latency_ms / 1000.0,
            }

        # Behavioral-diagnostic counts for Section 13/14 of the long-context
        # evaluation spec -- disclosed operationalizations, not invented ad
        # hoc: see module docstring and each comment below for exactly what
        # each counts and from which core-owned trajectory field.
        activated_worker_ids = {
            worker_id
            for round_ in state.rounds
            for ne in round_.node_executions
            for worker_id in ne.worker_ids
        }
        need_nodes_created = {
            node_id for round_ in state.rounds for node_id in round_.graph_delta.created_nodes
        }
        # "Need revision": a round's graph_delta recording either a
        # dependency change or a new children list for an EXISTING node
        # (i.e. structural graph rewiring beyond simple initial creation).
        need_revisions = sum(
            len(round_.graph_delta.dependency_changes) + len(round_.graph_delta.created_children)
            for round_ in state.rounds
        )
        # "Reroute": a need_id whose recovery-state shows more than one
        # distinct worker was ever tried against it -- i.e. the coordinator
        # moved off its first-assigned worker for that need at runtime.
        reroutes = sum(
            1
            for tried in state.final_recovery_state.tried_workers_by_node.values()
            if len(tried) > 1
        )
        recovery_events = len(state.final_recovery_state.stuck_episodes)

        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=final_answer,
            trajectory=[round_.model_dump() for round_ in state.rounds],
            evidence=[item.model_dump() for item in state.evidence],
            usage=UsageStats(
                llm_calls=llm_calls,
                input_tokens=state.usage.input_tokens,
                output_tokens=state.usage.output_tokens,
                total_tokens=state.usage.total_tokens,
                estimated_cost_usd=state.usage.estimated_cost_usd,
                wall_clock_seconds=state.usage.latency_ms / 1000.0,
                tool_calls=sum(
                    len(obs.actions)
                    for round_ in state.rounds
                    for ne in round_.node_executions
                    for obs in ne.observations
                ),
                unique_files_inspected=len({item.path for item in state.evidence}),
            ),
            termination_reason=(
                "unresolved_needs_remain" if state.unresolved_needs else "all_needs_resolved"
            ),
            metadata={
                "generation_model": self.model,
                "worker_usage": worker_usage_metadata,
                "final_need_graph_size": len(state.final_need_graph),
                "facet_rescue": state.facet_rescue.model_dump() if state.facet_rescue else None,
                "total_territories": len(territories),
                "workers_available": len(workers),
                "workers_activated": len(activated_worker_ids),
                "need_nodes_created": len(need_nodes_created),
                "need_revisions": need_revisions,
                "reroutes": reroutes,
                "recovery_events": recovery_events,
                "evidence_count": len(state.evidence),
                "raw_answer_before_condensation": raw_answer,
                "answer_contract_condition": example.metadata.get("answer_contract_condition"),
                "grounded_answer_after_regrounding": grounded_answer
                if grounded_answer != raw_answer
                else None,
                "final_synthesis_policy": (
                    "natural_multihop_qa"
                    if is_natural_multihop_qa_benchmark(example.benchmark)
                    else "swe_qa_pro_default"
                ),
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(AntDocumentAgent())
