"""Every tunable numeric constant behind ANT's scoring, routing, and
evolution heuristics, collected in one documented place.

Before this module existed, these numbers were scattered as inline literals
across `score_evidence()` (evidence reranking), `LocalCoordinator._rank_worker_scores()`
(worker routing), `evolve_workers()` (specialization/birth/merge thresholds),
and the eval runner (what counts as a "high-quality" route). None of them are
learned or derived from a formal model -- they were chosen empirically and
have not individually been validated for sensitivity. Centralizing them here
does not change any of their values or any system behavior; it exists so
that:

1. A single file answers "where did this number come from" for every
   heuristic constant in the pipeline, instead of requiring a grep across
   five modules.
2. A sensitivity/ablation pass (recommended before treating any one of these
   values as load-bearing in a paper claim) has one surface to sweep instead
   of five.
3. An experiment can construct an alternate `ScoringConfig` and pass it
   through explicitly (see `score_evidence(config=...)`, `evolve_workers`'s
   existing keyword arguments) without editing source.

See docs/scoring_heuristics.md for the rationale behind each group and
which downstream metric each one most directly affects.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvidenceScoringConfig:
    """Weights used by score_evidence() (ant.retrieval.relevance) to rerank
    candidate evidence within a worker's territory."""

    class_definition_bonus: int = 10
    function_definition_bonus: int = 8
    definition_reason_bonus: int = 6
    source_path_bonus: int = 5
    python_file_bonus: int = 3
    low_value_path_penalty: int = 10
    low_value_part_penalty: int = 10
    symbol_stem_match_bonus: int = 6
    quote_term_overlap_bonus: int = 3
    quote_substring_bonus: int = 1
    # value += round(dense_score * dense_score_weight); dense_score is a
    # 0..1 cosine similarity, so this is the max a pure-dense (zero lexical
    # overlap) match can contribute.
    dense_score_weight: int = 12


@dataclass(frozen=True)
class WorkerRoutingConfig:
    """Weights used by LocalCoordinator._rank_worker_scores() (and its
    helpers) in ant.coordinator.local to select which worker(s) handle a
    need."""

    # _relevant_symbol_weights: a symbol match's bonus is this, divided by
    # how many workers it matches (floor 1) -- a symbol unique to one worker
    # keeps close to full weight, one that matches most of the colony
    # (typically the repo's own package name) is worth almost nothing.
    relevant_symbol_base_weight: int = 6

    # A need's declared source_worker_id gets a bonus toward staying with
    # that worker: larger when the need's scope is "local" (the same worker
    # should keep handling it) than when it's a looser hint, plus up to
    # `source_worker_overlap_cap` more for how many of its own suggested
    # terms/hits this worker actually matched.
    source_worker_local_bonus: int = 8
    source_worker_global_bonus: int = 1
    source_worker_overlap_cap: int = 4

    # A worker already visited this conversation is penalized against being
    # picked again, so a second round doesn't just re-ask the same worker --
    # unless it's the need's own local source_worker_id, which is penalized
    # much less (a local need often legitimately needs a follow-up round
    # with the same worker).
    seen_worker_local_penalty: int = 2
    seen_worker_global_penalty: int = 8
    seen_worker_no_need_penalty: int = 4

    # _source_path_bonus / _test_path_penalty: applied when at least this
    # fraction of a worker's files fall under a source-like / test-like path
    # segment, not per-file -- a worker that's mostly source with a few test
    # files shouldn't flip a large penalty on and off file by file.
    source_path_ratio_threshold: float = 0.5
    source_path_bonus: int = 2
    test_path_ratio_threshold: float = 0.5
    test_path_penalty: int = 20

    # _memory_route_bonus: a recorded route only counts if the current
    # query's terms cover at least this fraction of the route's own need
    # vocabulary (not just share one word with it) -- see the function's
    # docstring for the "both mention 'quantum'" false-positive this guards
    # against. The bonus itself is capped and scales with how well the past
    # route worked (route.weight) plus how many terms overlap.
    min_memory_route_overlap_ratio: float = 1 / 3
    memory_route_bonus_cap: int = 12
    memory_route_weight_multiplier: int = 4

    # _territory_hint_score: a coordinator-suggested territory hint that
    # fully matches a worker's own terms/territory vocabulary is worth much
    # more than a partial match, which scales with how many hint terms
    # actually landed.
    territory_hint_full_match_bonus: int = 14
    territory_hint_partial_base_bonus: int = 6
    territory_hint_partial_per_overlap_bonus: int = 2

    # Coarse semantic routing signal from the worker-card embedding index:
    # dense_scores are 0..1 cosine similarities, so this is the max push a
    # worker with zero lexical overlap can still get purely from semantic
    # closeness.
    dense_routing_weight: int = 10

    # _initial_workers: the runner-up worker is treated as "genuinely
    # close" to the top pick (both recruited up front, not just the top
    # one) only if its score is at least this fraction of the leader's.
    initial_worker_closeness_ratio: float = 0.9

    # LLM-driven worker selection (reasoner.select_workers): the lexical +
    # dense score above still decides *which workers even get considered*,
    # capped to this many top candidates per round, before an LLM makes the
    # actual recruitment judgment among them. This exists purely for cost
    # control (bounding prompt size / call cost on a large evolved worker
    # population, e.g. 30+ workers on qibo after several evolve cycles) --
    # it is not meant to pre-decide relevance, so it is deliberately
    # generous rather than tight.
    llm_routing_candidate_pool_size: int = 12

    # LLM-driven final evidence selection (reasoner.select_evidence,
    # LocalCoordinator.ask): replaces the old fixed evidence[:12] cut before
    # synthesis. The score-ranked pool is still capped here purely for
    # prompt-size/cost control, not as a relevance decision -- the LLM
    # judges what to keep among the pool, up to llm_evidence_keep_limit.
    # Per-worker collection no longer does its own relevance filtering (see
    # AutonomousWorker's evidence_limit, which is now a generous safety cap,
    # not a decision point), so this single final judgment is deliberately
    # the one place worth spending on, sized larger than the old 12 to give
    # the model room to keep a genuinely multi-part answer's worth of
    # evidence instead of being forced to drop something to make room.
    llm_evidence_pool_size: int = 40
    llm_evidence_keep_limit: int = 16


@dataclass(frozen=True)
class EvolutionConfig:
    """Default thresholds for evolve_workers() (ant.evolution) -- when a
    worker specializes, when a recurring coalition is born as a bridge
    worker, and when two workers merge.

    These are exposed as evolve_workers()'s own keyword arguments (this
    class only supplies their *defaults*), since they are the one part of
    this file already designed to be swept per-experiment.
    """

    min_coalition_count: int = 2
    merge_overlap_threshold: float = 0.9
    min_specialization_routes: int = 4
    min_specialization_group_routes: int = 2

    # How many separate *tasks* the same (strategy, worker-set) collaboration
    # episode must recur in before evolve_workers even asks an
    # EvolutionReasoner what to do about it (see
    # ColonyMemoryStore.aggregate_episodes / EvolutionReasoner.
    # decide_episode_action). Same role as min_coalition_count but for the
    # richer per-need-strategy-outcome signal, not raw worker co-occurrence.
    min_episode_count: int = 2

    # A worker counts as "already working well enough to leave alone" once
    # it has at least this many recorded routes, and at least
    # `healthy_route_ratio` of them are flagged is_high_quality (derived
    # from correctness+completeness -- see RouteQualityConfig -- not
    # relevance/clarity, so a worker that finds something relevant-looking
    # without actually answering correctly/completely still counts as
    # struggling). specialize/birth/merge all skip touching a worker (or,
    # for merge, either side of a candidate pair) that already clears this
    # bar: added after a real regression -- merging two already-adequate
    # bridge workers into one test-file-heavy worker measurably degraded a
    # question that a pre-merge generation had answered well (see qibo's
    # gen2->gen3 comparison on dc7063c4...), which this gate is designed to
    # prevent by refusing to touch a worker with a track record of good
    # answers just because it also happens to satisfy a structural
    # threshold (overlap ratio, route count, recurring coalition count).
    min_routes_for_health_check: int = 3
    healthy_route_ratio: float = 0.7

    # specialize skips a worker once this fraction of its *typed* struggling
    # routes (need_type is populated -- see MemoryRoute, colony.py's
    # record_task_memory) are need_type="negative_presence": the repo
    # genuinely doesn't contain the answer, confirmed directly on both qibo
    # (a question about a tool that isn't part of that codebase) and seaborn
    # (a doc-build-performance question with no matching implementation).
    # Reorganizing territory boundaries cannot fix "the information was
    # never here" -- this exists so specialize doesn't spend an evolution
    # cycle on a failure mode it has no power over.
    negative_presence_gate_ratio: float = 0.5


@dataclass(frozen=True)
class RouteQualityConfig:
    """Thresholds used by the eval runner (ant.evaluation.runner) to decide
    which answers are trustworthy enough to feed back into colony memory as
    high-quality routes, and how much weight a route carries once recorded.
    """

    high_quality_correctness_min: int = 8
    high_quality_completeness_min: int = 8
    route_weight_divisor: int = 4
    default_route_weight: float = 2.0


@dataclass(frozen=True)
class DenseRetrievalConfig:
    """Parameters for symbol-level dense/embedding retrieval
    (ant.retrieval.dense)."""

    # A symbol's embedded text is truncated to at most this many lines from
    # its start, regardless of the symbol's actual length -- bounds both
    # embedding cost and how much a long function's unrelated tail can
    # dilute its own vector. Long functions whose semantically
    # distinguishing logic sits past this line are under-represented; this
    # is the sharpest-edged constant in this file and the first candidate
    # for a sensitivity pass.
    symbol_snippet_max_lines: int = 20
    embed_batch_size: int = 256


@dataclass(frozen=True)
class ScoringConfig:
    evidence: EvidenceScoringConfig = field(default_factory=EvidenceScoringConfig)
    routing: WorkerRoutingConfig = field(default_factory=WorkerRoutingConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    route_quality: RouteQualityConfig = field(default_factory=RouteQualityConfig)
    dense: DenseRetrievalConfig = field(default_factory=DenseRetrievalConfig)


# The shared default every call site reads from unless an experiment
# explicitly constructs and passes its own ScoringConfig (or one of its
# sub-configs). A single module-level instance, not a fresh one per call,
# so every part of the pipeline agrees on "the current values" within one
# process -- exactly the property that made the pre-refactor scattered
# literals hard to reason about.
DEFAULT_SCORING_CONFIG = ScoringConfig()
