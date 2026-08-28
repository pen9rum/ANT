from pathlib import Path

from ant.domain import CodeSymbol, Territory, WorkerCard
from ant.evolution import evolve_workers
from ant.memory import CoalitionRecord, ColonyMemoryStore, IndexStore, MemoryRoute
from ant.memory.colony import CollaborationEpisode


class _AlwaysVetoReasoner:
    """Stub EvolutionReasoner that vetoes every specialize/merge candidate --
    proves evolve_workers actually consults a supplied reasoner and honors a
    "no" (not just that it runs without crashing), without needing a real
    model call.
    """

    def should_specialize(self, *, worker_id, worker_summary, candidate_groups, route_summaries):
        return False

    def should_merge(self, *, worker_a_id, worker_a_summary, worker_b_id, worker_b_summary):
        return False

    def decide_episode_action(
        self, *, strategy, need_terms, occurrences, successes, total_evidence_gain, workers
    ):
        raise AssertionError("this test does not exercise decide_episode_action()")

    def summarize_routing(self, *, card):
        return f"stub routing summary for {card.id}"


def test_evolve_workers_births_bridge_from_recurring_coalition(tmp_path: Path) -> None:
    index_path = tmp_path / ".ant"
    workers = [
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["a.py"]),
        WorkerCard(id="worker-b", territory_id="b", name="b", root="b", files=["b.py"]),
    ]
    territories = [
        Territory(id="a", root="a", files=["a.py"]),
        Territory(id="b", root="b", files=["b.py"]),
    ]
    IndexStore(index_path).save(territories, workers)
    memory = ColonyMemoryStore(index_path)
    memory.record_coalition(
        CoalitionRecord(
            worker_ids=["worker-a", "worker-b"],
            question="q1",
            evidence_count=2,
            unresolved_need_count=0,
        )
    )
    memory.record_coalition(
        CoalitionRecord(
            worker_ids=["worker-a", "worker-b"],
            question="q2",
            evidence_count=2,
            unresolved_need_count=0,
        )
    )

    result = evolve_workers(index_path, min_coalition_count=2)

    assert result.events[0].kind == "birth"
    stored_workers = IndexStore(index_path).load_workers()
    assert any(worker.id.startswith("worker-bridge") for worker in stored_workers)
    # No reasoner supplied -- routing_summary must still be non-empty (the
    # zero-cost template fallback), not left at the WorkerCard default "".
    bridge = next(worker for worker in stored_workers if worker.id.startswith("worker-bridge"))
    assert bridge.routing_summary


def test_recurring_coalition_birth_uses_the_reasoners_routing_summary_when_supplied(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / ".ant"
    workers = [
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["a.py"]),
        WorkerCard(id="worker-b", territory_id="b", name="b", root="b", files=["b.py"]),
    ]
    territories = [
        Territory(id="a", root="a", files=["a.py"]),
        Territory(id="b", root="b", files=["b.py"]),
    ]
    IndexStore(index_path).save(territories, workers)
    memory = ColonyMemoryStore(index_path)
    for question in ("q1", "q2"):
        memory.record_coalition(
            CoalitionRecord(
                worker_ids=["worker-a", "worker-b"],
                question=question,
                evidence_count=2,
                unresolved_need_count=0,
            )
        )

    result = evolve_workers(index_path, reasoner=_AlwaysVetoReasoner(), min_coalition_count=2)

    assert result.events[0].kind == "birth"
    bridge = next(
        worker
        for worker in IndexStore(index_path).load_workers()
        if worker.id.startswith("worker-bridge")
    )
    # A reasoner was supplied, so routing_summary must come from
    # reasoner.summarize_routing() -- distinguishable from the zero-cost
    # template fallback the no-reasoner test above gets instead.
    assert bridge.routing_summary == f"stub routing summary for {bridge.id}"


def test_evolve_workers_retires_empty_and_merges_overlap(tmp_path: Path) -> None:
    index_path = tmp_path / ".ant"
    workers = [
        WorkerCard(id="worker-empty", territory_id="empty", name="empty", root="", files=[]),
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["shared.py"]),
        WorkerCard(id="worker-b", territory_id="b", name="b", root="b", files=["shared.py"]),
    ]
    territories = [
        Territory(id=worker.territory_id, root=worker.root, files=worker.files)
        for worker in workers
    ]
    IndexStore(index_path).save(territories, workers)

    result = evolve_workers(index_path, min_coalition_count=99, merge_overlap=0.9)

    assert {event.kind for event in result.events} == {"retire", "merge"}
    merged = next(
        worker
        for worker in IndexStore(index_path).load_workers()
        if worker.id.startswith("worker-merge")
    )
    assert merged.routing_summary


def test_evolve_workers_does_not_merge_when_the_reasoner_vetoes_it(tmp_path: Path) -> None:
    # Same structurally-mergeable fixture as the test above (full file
    # overlap >= merge_overlap), but with an EvolutionReasoner supplied that
    # says no: the merge must not happen even though the old purely-
    # structural gate would have allowed it. Retire (unrelated to the
    # reasoner) still fires normally.
    index_path = tmp_path / ".ant"
    workers = [
        WorkerCard(id="worker-empty", territory_id="empty", name="empty", root="", files=[]),
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["shared.py"]),
        WorkerCard(id="worker-b", territory_id="b", name="b", root="b", files=["shared.py"]),
    ]
    territories = [
        Territory(id=worker.territory_id, root=worker.root, files=worker.files)
        for worker in workers
    ]
    IndexStore(index_path).save(territories, workers)

    result = evolve_workers(
        index_path,
        min_coalition_count=99,
        merge_overlap=0.9,
        reasoner=_AlwaysVetoReasoner(),
    )

    assert {event.kind for event in result.events} == {"retire"}
    stored_worker_ids = {worker.id for worker in IndexStore(index_path).load_workers()}
    assert {"worker-a", "worker-b"} <= stored_worker_ids


def test_evolve_workers_specializes_worker_overloaded_with_diverse_needs(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / ".ant"
    worker = WorkerCard(
        id="worker-mixed",
        territory_id="mixed",
        name="mixed worker",
        root="pkg",
        searchable_terms=["auth", "billing", "service"],
        files=["pkg/auth/service.py", "pkg/billing/service.py"],
        symbols=[
            CodeSymbol(name="AuthService", kind="class", path="pkg/auth/service.py", line=1),
            CodeSymbol(name="BillingService", kind="class", path="pkg/billing/service.py", line=1),
        ],
    )
    territories = [Territory(id="mixed", root="pkg", files=worker.files)]
    IndexStore(index_path).save(territories, [worker])
    memory = ColonyMemoryStore(index_path)
    # is_high_quality=False: this worker's answers have not actually been
    # good, not just topically diverse -- the health gate (added after a
    # real merge-caused regression) exempts a worker with a track record of
    # high-quality answers from specialize/birth/merge, so a test meant to
    # exercise "overloaded worker gets split" must show it struggling, not
    # succeeding, or the gate now correctly leaves it alone.
    for _ in range(2):
        memory.save_route(
            MemoryRoute(
                need_terms=["auth"], worker_ids=["worker-mixed"], weight=2.0, is_high_quality=False
            )
        )
    for _ in range(2):
        memory.save_route(
            MemoryRoute(
                need_terms=["billing"],
                worker_ids=["worker-mixed"],
                weight=2.0,
                is_high_quality=False,
            )
        )

    result = evolve_workers(
        index_path,
        min_coalition_count=99,
        retire_empty=False,
        merge_overlap=0.99,
        min_specialization_routes=4,
        min_specialization_group_routes=2,
    )

    specialize_events = [event for event in result.events if event.kind == "specialize"]
    assert {event.worker_id for event in specialize_events} == {
        "worker-pkg-auth",
        "worker-pkg-billing",
    }
    assert all(event.source_worker_ids == ["worker-mixed"] for event in specialize_events)

    stored_workers = {worker.id: worker for worker in IndexStore(index_path).load_workers()}
    assert "worker-mixed" not in stored_workers
    auth_worker = stored_workers["worker-pkg-auth"]
    billing_worker = stored_workers["worker-pkg-billing"]
    assert auth_worker.files == ["pkg/auth/service.py"]
    assert [symbol.name for symbol in auth_worker.symbols] == ["AuthService"]
    assert billing_worker.files == ["pkg/billing/service.py"]
    assert [symbol.name for symbol in billing_worker.symbols] == ["BillingService"]

    # The old worker id is now stale: any memory that pointed at it should not
    # be served until it is revalidated (and it will be discarded then, since
    # worker-mixed no longer exists in the colony).
    assert memory.matching_routes(["auth"]) == []
    current_worker_ids = {worker.id for worker in IndexStore(index_path).load_workers()}
    outcome = memory.revalidate_stale(current_worker_ids)
    assert outcome["discarded"] == 4


def test_evolve_workers_generates_real_child_cards_when_repo_root_is_supplied(
    tmp_path: Path,
) -> None:
    # When repo_root is supplied, a specialized child worker's card should
    # be derived from actually reading its files (same machinery as initial
    # card generation), not the old static "Specialized from X after N
    # needs" template with terms scraped only from already-known
    # symbol/path metadata. Assert on a term that only exists in the file
    # body text (not in any path or symbol name already on the parent
    # worker) to prove the file was actually read, not just labeled.
    repo_root = tmp_path / "repo"
    (repo_root / "pkg" / "auth").mkdir(parents=True)
    (repo_root / "pkg" / "billing").mkdir(parents=True)
    (repo_root / "pkg" / "auth" / "service.py").write_text(
        "class AuthService:\n"
        "    def login(self, credentials):\n"
        "        return validate_oauth_token(credentials)\n",
        encoding="utf-8",
    )
    (repo_root / "pkg" / "billing" / "service.py").write_text(
        "class BillingService:\n"
        "    def charge(self, invoice):\n"
        "        return process_stripe_payment(invoice)\n",
        encoding="utf-8",
    )
    index_path = tmp_path / ".ant"
    worker = WorkerCard(
        id="worker-mixed",
        territory_id="mixed",
        name="mixed worker",
        root="pkg",
        searchable_terms=["auth", "billing", "service"],
        files=["pkg/auth/service.py", "pkg/billing/service.py"],
        symbols=[
            CodeSymbol(name="AuthService", kind="class", path="pkg/auth/service.py", line=1),
            CodeSymbol(name="BillingService", kind="class", path="pkg/billing/service.py", line=1),
        ],
    )
    territories = [Territory(id="mixed", root="pkg", files=worker.files)]
    IndexStore(index_path).save(territories, [worker])
    memory = ColonyMemoryStore(index_path)
    for _ in range(2):
        memory.save_route(
            MemoryRoute(
                need_terms=["auth"], worker_ids=["worker-mixed"], weight=2.0, is_high_quality=False
            )
        )
    for _ in range(2):
        memory.save_route(
            MemoryRoute(
                need_terms=["billing"],
                worker_ids=["worker-mixed"],
                weight=2.0,
                is_high_quality=False,
            )
        )

    evolve_workers(
        index_path,
        repo_root=repo_root,
        min_coalition_count=99,
        retire_empty=False,
        merge_overlap=0.99,
        min_specialization_routes=4,
        min_specialization_group_routes=2,
    )

    stored_workers = {worker.id: worker for worker in IndexStore(index_path).load_workers()}
    auth_worker = stored_workers["worker-pkg-auth"]
    billing_worker = stored_workers["worker-pkg-billing"]
    # The proof this went through build_worker_cards's real term-frequency
    # extraction, not the old fallback (which only derives terms from
    # already-known symbol names and path segments -- neither list contains
    # these function names anywhere): both terms only exist in the file
    # bodies themselves.
    assert "validate_oauth_token" in auth_worker.searchable_terms
    assert "process_stripe_payment" in billing_worker.searchable_terms


def test_evolve_workers_does_not_specialize_when_the_reasoner_vetoes_it(tmp_path: Path) -> None:
    # Same structurally-overloaded fixture as the diverse-needs test above
    # (2 distinct subgroups, each past min_specialization_group_routes),
    # but with an EvolutionReasoner supplied that says no: the split must
    # not happen even though the old purely-structural gate would have
    # allowed it.
    index_path = tmp_path / ".ant"
    worker = WorkerCard(
        id="worker-mixed",
        territory_id="mixed",
        name="mixed worker",
        root="pkg",
        searchable_terms=["auth", "billing", "service"],
        files=["pkg/auth/service.py", "pkg/billing/service.py"],
        symbols=[
            CodeSymbol(name="AuthService", kind="class", path="pkg/auth/service.py", line=1),
            CodeSymbol(name="BillingService", kind="class", path="pkg/billing/service.py", line=1),
        ],
    )
    territories = [Territory(id="mixed", root="pkg", files=worker.files)]
    IndexStore(index_path).save(territories, [worker])
    memory = ColonyMemoryStore(index_path)
    for _ in range(2):
        memory.save_route(
            MemoryRoute(
                need_terms=["auth"], worker_ids=["worker-mixed"], weight=2.0, is_high_quality=False
            )
        )
    for _ in range(2):
        memory.save_route(
            MemoryRoute(
                need_terms=["billing"],
                worker_ids=["worker-mixed"],
                weight=2.0,
                is_high_quality=False,
            )
        )

    result = evolve_workers(
        index_path,
        min_coalition_count=99,
        retire_empty=False,
        merge_overlap=0.99,
        min_specialization_routes=4,
        min_specialization_group_routes=2,
        reasoner=_AlwaysVetoReasoner(),
    )

    assert not [event for event in result.events if event.kind == "specialize"]
    stored_worker_ids = {worker.id for worker in IndexStore(index_path).load_workers()}
    assert stored_worker_ids == {"worker-mixed"}


def test_evolve_workers_does_not_specialize_a_worker_whose_gaps_are_mostly_absence(
    tmp_path: Path,
) -> None:
    # Regression test: a worker whose accumulated routes look identical to
    # the diverse-and-overloaded case above by every structural signal
    # (route count, subdirectory group count, route-per-group count) must
    # NOT specialize if those routes are mostly need_type="negative_presence"
    # -- confirmed directly on real questions (a qibo question about a tool
    # not in that codebase; a seaborn doc-build-performance question with no
    # matching implementation) that reorganizing territory boundaries cannot
    # fix "the repo doesn't contain this," so specializing here would spend
    # an evolution cycle on a failure mode this mechanism has no power over.
    index_path = tmp_path / ".ant"
    worker = WorkerCard(
        id="worker-mixed",
        territory_id="mixed",
        name="mixed worker",
        root="pkg",
        searchable_terms=["auth", "billing", "service"],
        files=["pkg/auth/service.py", "pkg/billing/service.py"],
        symbols=[
            CodeSymbol(name="AuthService", kind="class", path="pkg/auth/service.py", line=1),
            CodeSymbol(name="BillingService", kind="class", path="pkg/billing/service.py", line=1),
        ],
    )
    territories = [Territory(id="mixed", root="pkg", files=worker.files)]
    IndexStore(index_path).save(territories, [worker])
    memory = ColonyMemoryStore(index_path)
    for _ in range(2):
        memory.save_route(
            MemoryRoute(
                need_terms=["auth"],
                worker_ids=["worker-mixed"],
                weight=2.0,
                is_high_quality=False,
                need_type="negative_presence",
            )
        )
    for _ in range(2):
        memory.save_route(
            MemoryRoute(
                need_terms=["billing"],
                worker_ids=["worker-mixed"],
                weight=2.0,
                is_high_quality=False,
                need_type="negative_presence",
            )
        )

    result = evolve_workers(
        index_path,
        min_coalition_count=99,
        retire_empty=False,
        merge_overlap=0.99,
        min_specialization_routes=4,
        min_specialization_group_routes=2,
    )

    assert not [event for event in result.events if event.kind == "specialize"]
    stored_workers = {worker.id for worker in IndexStore(index_path).load_workers()}
    assert stored_workers == {"worker-mixed"}


def test_evolve_workers_does_not_merge_a_worker_with_a_high_quality_track_record(
    tmp_path: Path,
) -> None:
    # Regression test for a real observed failure: merging a worker that was
    # already answering well into an overlapping-but-unrelated worker
    # measurably degraded a question a pre-merge generation had answered
    # correctly (qibo gen2->gen3, a "why does the system partition gates..."
    # question that lost its source-file evidence to a test-file-heavy
    # merge partner). File overlap alone must not be sufficient grounds to
    # touch a worker with a demonstrated track record of good answers.
    index_path = tmp_path / ".ant"
    workers = [
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["shared.py"]),
        WorkerCard(id="worker-b", territory_id="b", name="b", root="b", files=["shared.py"]),
    ]
    territories = [
        Territory(id=worker.territory_id, root=worker.root, files=worker.files)
        for worker in workers
    ]
    IndexStore(index_path).save(territories, workers)
    memory = ColonyMemoryStore(index_path)
    # worker-a has a strong track record; worker-b has none yet.
    for _ in range(3):
        memory.save_route(
            MemoryRoute(need_terms=["shared"], worker_ids=["worker-a"], is_high_quality=True)
        )

    result = evolve_workers(index_path, min_coalition_count=99, merge_overlap=0.9)

    assert "merge" not in {event.kind for event in result.events}
    stored_workers = {worker.id for worker in IndexStore(index_path).load_workers()}
    assert stored_workers == {"worker-a", "worker-b"}


def test_evolve_workers_merges_two_workers_with_no_or_poor_track_record(tmp_path: Path) -> None:
    # Counterpart to the above: with no quality evidence to protect either
    # side (the common case for a freshly-created colony), overlap-based
    # merge still fires exactly as before this gate existed.
    index_path = tmp_path / ".ant"
    workers = [
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["shared.py"]),
        WorkerCard(id="worker-b", territory_id="b", name="b", root="b", files=["shared.py"]),
    ]
    territories = [
        Territory(id=worker.territory_id, root=worker.root, files=worker.files)
        for worker in workers
    ]
    IndexStore(index_path).save(territories, workers)

    result = evolve_workers(index_path, min_coalition_count=99, merge_overlap=0.9)

    assert {event.kind for event in result.events} == {"merge"}


def test_evolve_workers_does_not_specialize_without_enough_route_diversity(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / ".ant"
    worker = WorkerCard(
        id="worker-mixed",
        territory_id="mixed",
        name="mixed worker",
        root="pkg",
        files=["pkg/auth/service.py", "pkg/billing/service.py"],
    )
    territories = [Territory(id="mixed", root="pkg", files=worker.files)]
    IndexStore(index_path).save(territories, [worker])
    memory = ColonyMemoryStore(index_path)
    for _ in range(4):
        memory.save_route(
            MemoryRoute(need_terms=["auth"], worker_ids=["worker-mixed"], weight=2.0)
        )

    result = evolve_workers(
        index_path,
        min_coalition_count=99,
        retire_empty=False,
        merge_overlap=0.99,
        min_specialization_routes=4,
        min_specialization_group_routes=2,
    )

    assert not [event for event in result.events if event.kind == "specialize"]
    stored_workers = {worker.id for worker in IndexStore(index_path).load_workers()}
    assert stored_workers == {"worker-mixed"}


class _BirthBridgeOnRecurringPatternReasoner(_AlwaysVetoReasoner):
    """Stub EvolutionReasoner whose decide_episode_action always says
    "birth_bridge" -- proves evolve_workers actually consults the reasoner
    over aggregated episodes (not just raw coalition counts) and acts on
    its verdict, without needing a real model call.
    """

    def decide_episode_action(
        self, *, strategy, need_terms, occurrences, successes, total_evidence_gain, workers
    ):
        return "birth_bridge"


class _NoChangeOnEpisodesReasoner(_AlwaysVetoReasoner):
    """Stub EvolutionReasoner whose decide_episode_action always says
    "no_change" -- unlike _AlwaysVetoReasoner (which raises if
    decide_episode_action is ever called, for tests that assert it's never
    reached), this one lets the call happen and proves evolve_workers
    honors an explicit "no" the same way it already does for
    should_specialize/should_merge.
    """

    def decide_episode_action(
        self, *, strategy, need_terms, occurrences, successes, total_evidence_gain, workers
    ):
        return "no_change"


def test_evolve_workers_births_bridge_from_a_recurring_successful_episode_pattern(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / ".ant"
    workers = [
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["a.py"]),
        WorkerCard(id="worker-b", territory_id="b", name="b", root="b", files=["b.py"]),
    ]
    territories = [
        Territory(id=worker.territory_id, root=worker.root, files=worker.files)
        for worker in workers
    ]
    IndexStore(index_path).save(territories, workers)
    memory = ColonyMemoryStore(index_path)
    for need in ("proxy validation boundary task one", "proxy validation boundary task two"):
        memory.record_episode(
            CollaborationEpisode(
                need=need,
                workers=["worker-a", "worker-b"],
                strategy="temporary_bridge",
                outcome="progress",
                evidence_gain=3,
            )
        )

    result = evolve_workers(
        index_path,
        reasoner=_BirthBridgeOnRecurringPatternReasoner(),
        min_coalition_count=99,
        retire_empty=False,
        merge_overlap=0.99,
        min_episode_count=2,
    )

    birth_events = [event for event in result.events if event.kind == "birth"]
    assert len(birth_events) == 1
    assert sorted(birth_events[0].source_worker_ids) == ["worker-a", "worker-b"]
    stored_workers = {worker.id for worker in IndexStore(index_path).load_workers()}
    assert "worker-bridge-a-b" in stored_workers


def test_episode_birth_bridge_on_a_single_source_worker_strengthens_instead(
    tmp_path: Path,
) -> None:
    # Regression test for a real bug found on a live qibo run: the reasoner
    # can call decide_episode_action's verdict "birth_bridge" for an
    # aggregate whose strategy is "normal" (not an actual coalition) and
    # whose workers list has exactly one entry -- a "bridge" born from a
    # single source is just a same-files clone of that worker under a new
    # id, not a cross-territory specialist. Confirmed directly: this
    # produced 3 such duplicates in one real evolve call. The single-worker
    # case this pattern is real evidence for ("this worker does well") is
    # exactly what strengthen_route already expresses without growing the
    # worker pool.
    index_path = tmp_path / ".ant"
    worker = WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["a.py"])
    territories = [Territory(id="a", root="a", files=["a.py"])]
    IndexStore(index_path).save(territories, [worker])
    memory = ColonyMemoryStore(index_path)
    for need in ("task one", "task two"):
        memory.record_episode(
            CollaborationEpisode(
                need=need, workers=["worker-a"], strategy="normal",
                outcome="progress", evidence_gain=3,
            )
        )

    result = evolve_workers(
        index_path,
        reasoner=_BirthBridgeOnRecurringPatternReasoner(),
        min_coalition_count=99,
        retire_empty=False,
        merge_overlap=0.99,
        min_episode_count=2,
    )

    assert not [event for event in result.events if event.kind == "birth"]
    strengthen_events = [event for event in result.events if event.kind == "strengthen_route"]
    assert len(strengthen_events) == 1
    assert strengthen_events[0].source_worker_ids == ["worker-a"]
    stored_worker_ids = {worker.id for worker in IndexStore(index_path).load_workers()}
    assert stored_worker_ids == {"worker-a"}


def test_episode_birth_bridge_with_a_low_success_ratio_is_skipped(tmp_path: Path) -> None:
    # Regression test for a real bug: an aggregate that recurred often but
    # mostly DIDN'T work (13/35 successes = 37% on a live qibo run) still
    # got birth_bridge'd, because the occurrence-count gate
    # (min_episode_count) only checks how often a pattern recurred, not
    # whether it actually succeeded when it did -- and the reasoner's own
    # judgment isn't a reliable enough backstop on its own (it got this one
    # wrong on the real trace this test reproduces).
    index_path = tmp_path / ".ant"
    workers = [
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["a.py"]),
        WorkerCard(id="worker-b", territory_id="b", name="b", root="b", files=["b.py"]),
    ]
    territories = [
        Territory(id=worker.territory_id, root=worker.root, files=worker.files)
        for worker in workers
    ]
    IndexStore(index_path).save(territories, workers)
    memory = ColonyMemoryStore(index_path)
    for index in range(3):
        memory.record_episode(
            CollaborationEpisode(
                need=f"task {index}",
                workers=["worker-a", "worker-b"],
                strategy="temporary_bridge",
                outcome="progress" if index == 0 else "no_progress",
                evidence_gain=3 if index == 0 else 0,
            )
        )

    result = evolve_workers(
        index_path,
        reasoner=_BirthBridgeOnRecurringPatternReasoner(),
        min_coalition_count=99,
        retire_empty=False,
        merge_overlap=0.99,
        min_episode_count=2,
    )

    assert not result.events
    stored_worker_ids = {worker.id for worker in IndexStore(index_path).load_workers()}
    assert stored_worker_ids == {"worker-a", "worker-b"}


def test_recurring_coalition_birth_skips_a_near_duplicate_of_an_existing_bridge(
    tmp_path: Path,
) -> None:
    # Regression test for a real bug found on a live qibo run: a recurring
    # coalition between an existing bridge (worker-bridge-a-b, files a.py +
    # b.py) and one of that bridge's own source workers (worker-b, files
    # b.py) produced a new "bridge of a bridge" whose file set was
    # identical to the existing bridge's -- pure redundancy that a merge
    # pass immediately had to collapse back down in the same evolve call,
    # leaving a worker id that concatenated both. The resulting file set
    # must be checked against every current worker, not just the two
    # immediate sources, so this is caught regardless of which source
    # carried the redundancy.
    index_path = tmp_path / ".ant"
    workers = [
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["a.py"]),
        WorkerCard(id="worker-b", territory_id="b", name="b", root="b", files=["b.py"]),
        WorkerCard(
            id="worker-bridge-a-b",
            territory_id="bridge-a-b",
            name="a + b bridge",
            root="",
            files=["a.py", "b.py"],
        ),
    ]
    territories = [
        Territory(id=worker.territory_id, root=worker.root, files=worker.files)
        for worker in workers
    ]
    IndexStore(index_path).save(territories, workers)
    memory = ColonyMemoryStore(index_path)
    for question in ("q1", "q2", "q3"):
        memory.record_coalition(
            CoalitionRecord(
                worker_ids=["worker-bridge-a-b", "worker-b"],
                question=question,
                evidence_count=2,
                unresolved_need_count=0,
            )
        )

    result = evolve_workers(index_path, min_coalition_count=2, merge_overlap=0.9)

    assert not [event for event in result.events if event.kind == "birth"]
    stored_worker_ids = {worker.id for worker in IndexStore(index_path).load_workers()}
    assert stored_worker_ids == {"worker-a", "worker-b", "worker-bridge-a-b"}


class _RedundantWithWorkerCReasoner(_AlwaysVetoReasoner):
    """should_merge says yes specifically for worker-c -- proves the
    semantic-redundancy check is doing real LLM-judged work, not just
    vetoing everything the way _AlwaysVetoReasoner's own should_merge
    (always False) would."""

    def should_merge(self, *, worker_a_id, worker_a_summary, worker_b_id, worker_b_summary):
        return worker_b_id == "worker-c"


def test_recurring_coalition_birth_skips_a_worker_semantically_redundant_with_existing(
    tmp_path: Path,
) -> None:
    # Regression test for a real bug found on a live qibo run: even after
    # the file-overlap near-duplicate check above, a birthed bridge's
    # routing_summary could still read as the same specialty as an
    # existing sibling worker's despite a genuinely distinct file set
    # (confirmed: "Models and abstractions spanning qibo core modules" vs
    # "src/qibo/models algorithms and circuit helpers") -- the Orchestrator
    # then keeps selecting both instead of one, inflating coalitions
    # without adding real coverage. worker-c here shares only one file
    # with the candidate (overlap ratio 1/3, well under the 0.9 file-
    # overlap threshold), so only the semantic check -- not
    # _overlaps_existing_worker -- can catch this.
    index_path = tmp_path / ".ant"
    workers = [
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["a.py"]),
        WorkerCard(id="worker-b", territory_id="b", name="b", root="b", files=["b.py"]),
        WorkerCard(
            id="worker-c", territory_id="c", name="c", root="c", files=["b.py", "c.py"]
        ),
    ]
    territories = [
        Territory(id=worker.territory_id, root=worker.root, files=worker.files)
        for worker in workers
    ]
    IndexStore(index_path).save(territories, workers)
    memory = ColonyMemoryStore(index_path)
    for question in ("q1", "q2"):
        memory.record_coalition(
            CoalitionRecord(
                worker_ids=["worker-a", "worker-b"],
                question=question,
                evidence_count=2,
                unresolved_need_count=0,
            )
        )

    result = evolve_workers(
        index_path,
        reasoner=_RedundantWithWorkerCReasoner(),
        min_coalition_count=2,
        merge_overlap=0.9,
    )

    assert not [event for event in result.events if event.kind == "birth"]
    stored_worker_ids = {worker.id for worker in IndexStore(index_path).load_workers()}
    assert stored_worker_ids == {"worker-a", "worker-b", "worker-c"}


class _BirthBridgeButRedundantReasoner(_AlwaysVetoReasoner):
    def decide_episode_action(
        self, *, strategy, need_terms, occurrences, successes, total_evidence_gain, workers
    ):
        return "birth_bridge"

    def should_merge(self, *, worker_a_id, worker_a_summary, worker_b_id, worker_b_summary):
        return worker_b_id == "worker-c"


def test_episode_birth_bridge_downgrades_to_strengthen_route_when_semantically_redundant(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / ".ant"
    workers = [
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["a.py"]),
        WorkerCard(id="worker-b", territory_id="b", name="b", root="b", files=["b.py"]),
        WorkerCard(
            id="worker-c", territory_id="c", name="c", root="c", files=["b.py", "c.py"]
        ),
    ]
    territories = [
        Territory(id=worker.territory_id, root=worker.root, files=worker.files)
        for worker in workers
    ]
    IndexStore(index_path).save(territories, workers)
    memory = ColonyMemoryStore(index_path)
    for need in ("task one", "task two"):
        memory.record_episode(
            CollaborationEpisode(
                need=need,
                workers=["worker-a", "worker-b"],
                strategy="temporary_bridge",
                outcome="progress",
                evidence_gain=3,
            )
        )

    result = evolve_workers(
        index_path,
        reasoner=_BirthBridgeButRedundantReasoner(),
        min_coalition_count=99,
        retire_empty=False,
        merge_overlap=0.99,
        min_episode_count=2,
    )

    assert not [event for event in result.events if event.kind == "birth"]
    strengthen_events = [event for event in result.events if event.kind == "strengthen_route"]
    assert len(strengthen_events) == 1
    assert strengthen_events[0].worker_id == "worker-c"
    stored_worker_ids = {worker.id for worker in IndexStore(index_path).load_workers()}
    assert stored_worker_ids == {"worker-a", "worker-b", "worker-c"}


def test_evolve_workers_leaves_episodes_alone_when_reasoner_says_no_change(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / ".ant"
    workers = [
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["a.py"]),
        WorkerCard(id="worker-b", territory_id="b", name="b", root="b", files=["b.py"]),
    ]
    territories = [
        Territory(id=worker.territory_id, root=worker.root, files=worker.files)
        for worker in workers
    ]
    IndexStore(index_path).save(territories, workers)
    memory = ColonyMemoryStore(index_path)
    for need in ("proxy validation boundary task one", "proxy validation boundary task two"):
        memory.record_episode(
            CollaborationEpisode(
                need=need,
                workers=["worker-a", "worker-b"],
                strategy="temporary_bridge",
                outcome="progress",
                evidence_gain=3,
            )
        )

    result = evolve_workers(
        index_path,
        reasoner=_NoChangeOnEpisodesReasoner(),
        min_coalition_count=99,
        retire_empty=False,
        merge_overlap=0.99,
        min_episode_count=2,
    )

    assert not [
        event for event in result.events if event.kind in ("birth", "merge", "strengthen_route")
    ]
