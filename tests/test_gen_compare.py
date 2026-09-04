import json
from pathlib import Path

from ant.domain import EvidenceState, WorkerCard
from ant.evaluation.datasets import EvalExample
from ant.evaluation.gen_compare import _run_fast_generation, run_gen_compare
from ant.evaluation.metrics import EvalScore
from ant.evolution import EvolutionEvent, EvolutionResult


def test_run_gen_compare_passes_evolve_workers_the_resolved_repo_not_the_raw_parent_dir(
    tmp_path: Path, monkeypatch
) -> None:
    # Regression test: run_gen_compare's own `repo_root` parameter, for an
    # "owner/repo"-formatted dataset, must be the checkout's PARENT
    # directory -- run_batch's own per-example _resolve_repo (repo_root /
    # basename fallback) needs that to land on the actual repo, not a
    # nested repo_root/repo_root/<name> double-up. evolve_workers, by
    # contrast, reads a worker's own files straight off `repo_root / file`
    # (both via _child_worker's real-card rebuild and _semantic_groups'
    # live retrieval) -- it needs the actual repo checkout directly, one
    # level below that parent. Passing the same raw repo_root straight
    # through to both silently pointed evolve_workers' file reads at the
    # wrong directory whenever specialization ever fired -- confirmed live
    # against this project's own repos/ layout before this fix.
    repo_root = tmp_path / "repos"
    actual_repo = repo_root / "myrepo"
    actual_repo.mkdir(parents=True)

    index_path = tmp_path / ".ant"
    # evolve_workers is stubbed below so nothing else creates this on disk,
    # but run_gen_compare's own post-evolve index snapshot still does a
    # real shutil.copytree from it.
    (index_path / "myrepo").mkdir(parents=True)
    run_dir = tmp_path / "run"

    captured: dict[str, object] = {}

    def fake_run_batch(**kwargs):
        return []

    def fake_evolve_workers(index_path, repo_root=None, reasoner=None, generation=0):
        captured["repo_root"] = repo_root
        return EvolutionResult(events=[], worker_count=0)

    class _StubIndexStore:
        def __init__(self, path):
            pass

        def load_workers(self):
            return []

    monkeypatch.setattr("ant.evaluation.gen_compare.run_batch", fake_run_batch)
    monkeypatch.setattr("ant.evaluation.gen_compare.evolve_workers", fake_evolve_workers)
    monkeypatch.setattr("ant.evaluation.gen_compare.IndexStore", _StubIndexStore)
    monkeypatch.setattr("ant.evaluation.gen_compare.OpenAIProvider", lambda: object())

    examples = [
        EvalExample(id="q1", question="q", answer="a", repo="someorg/myrepo"),
    ]

    run_gen_compare(
        examples=examples,
        repo_root=repo_root,
        index_path=index_path,
        run_dir=run_dir,
        run_gen0=False,
        slow_generations=1,
        fast_generations=0,
    )

    assert captured["repo_root"] == actual_repo.resolve()


def test_run_gen_compare_passes_evolve_workers_the_same_index_run_batch_actually_populates(
    tmp_path: Path, monkeypatch
) -> None:
    # Regression test for a real, previously-silent bug: run_batch's own
    # _run_example resolves its REAL per-repo index to
    # index_path / _repo_basename(example.repo) whenever a dataset row's
    # repo isn't "." (every "owner/repo"-formatted dataset row this project
    # uses) -- the raw index_path itself is never read or written by any
    # actual question in that case. evolve_workers, called with that same
    # raw index_path, was therefore reading/mutating a permanently-empty
    # ColonyMemoryStore instead of the one run_batch had just spent an
    # entire gen0 stage populating -- confirmed live (pennylane: 45 real
    # recorded routes at the resolved path, 0 at the raw one), meaning
    # specialize/birth/merge silently found nothing to act on for every
    # such run before this fix.
    index_path = tmp_path / ".ant"
    resolved_index_path = index_path / "myrepo"
    resolved_index_path.mkdir(parents=True)

    captured: dict[str, object] = {}

    def fake_run_batch(**kwargs):
        return []

    def fake_evolve_workers(index_path, repo_root=None, reasoner=None, generation=0):
        captured["index_path"] = index_path
        return EvolutionResult(events=[], worker_count=0)

    class _StubIndexStore:
        def __init__(self, path):
            pass

        def load_workers(self):
            return []

    monkeypatch.setattr("ant.evaluation.gen_compare.run_batch", fake_run_batch)
    monkeypatch.setattr("ant.evaluation.gen_compare.evolve_workers", fake_evolve_workers)
    monkeypatch.setattr("ant.evaluation.gen_compare.IndexStore", _StubIndexStore)
    monkeypatch.setattr("ant.evaluation.gen_compare.OpenAIProvider", lambda: object())

    examples = [EvalExample(id="q1", question="q", answer="a", repo="someorg/myrepo")]

    run_gen_compare(
        examples=examples,
        repo_root=tmp_path / "repos",
        index_path=index_path,
        run_dir=tmp_path / "run",
        run_gen0=False,
        slow_generations=1,
        fast_generations=0,
    )

    assert captured["index_path"] == resolved_index_path


def test_generation_snapshot_reports_population_lifecycle_and_route_consolidation_stats(
    tmp_path: Path, monkeypatch
) -> None:
    # Phase 8 instrumentation for the multi-generation organizational
    # evolution redesign: a generation's own snapshot must let a later
    # audit answer "base vs overlay population, lifecycle distribution,
    # what structural actions actually happened, and is route consolidation
    # actually preventing unbounded cardinality growth" without re-deriving
    # any of it from raw trace files.
    run_dir = tmp_path / "run"
    index_path = tmp_path / ".ant"
    (index_path / "myrepo").mkdir(parents=True)
    repo_root = tmp_path / "repos"
    (repo_root / "myrepo").mkdir(parents=True)

    current_workers = [
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["a.py"]),
        WorkerCard(id="worker-b", territory_id="b", name="b", root="b", files=["b.py"]),
        WorkerCard(
            id="worker-bridge-a-b",
            territory_id="bridge-a-b",
            name="bridge",
            root="",
            files=["a.py", "b.py"],
            parent_worker_ids=["worker-a", "worker-b"],
            structural_action="birth",
            generation_created=1,
            lifecycle_state="probationary",
        ),
        WorkerCard(
            id="worker-persistent-child",
            territory_id="child",
            name="child",
            root="a/child",
            files=["a/child.py"],
            parent_worker_ids=["worker-a"],
            structural_action="specialize",
            generation_created=1,
            lifecycle_state="persistent",
        ),
    ]
    real_events = [
        EvolutionEvent(kind="birth", worker_id="worker-bridge-a-b", reason="r"),
        EvolutionEvent(kind="specialize", worker_id="worker-persistent-child", reason="r"),
        EvolutionEvent(kind="promote", worker_id="worker-persistent-child", reason="r"),
        EvolutionEvent(kind="strengthen_route", worker_id="worker-a", reason="r"),
        EvolutionEvent(kind="strengthen_route", worker_id="worker-b", reason="r"),
    ]

    class _StubIndexStore:
        def __init__(self, path):
            pass

        def load_workers(self):
            return current_workers

    def fake_run_batch(**kwargs):
        return []

    def fake_evolve_workers(index_path, repo_root=None, reasoner=None, generation=0):
        return EvolutionResult(events=real_events, worker_count=len(current_workers))

    def fake_load_or_rebuild_scores(**kwargs):
        return {}

    monkeypatch.setattr("ant.evaluation.gen_compare.IndexStore", _StubIndexStore)
    monkeypatch.setattr("ant.evaluation.gen_compare.run_batch", fake_run_batch)
    monkeypatch.setattr("ant.evaluation.gen_compare.evolve_workers", fake_evolve_workers)
    monkeypatch.setattr("ant.evaluation.gen_compare.OpenAIProvider", lambda: object())
    monkeypatch.setattr(
        "ant.evaluation.gen_compare._load_or_rebuild_scores", fake_load_or_rebuild_scores
    )

    examples = [EvalExample(id="q1", question="q", answer="a", repo="someorg/myrepo")]
    result = run_gen_compare(
        examples=examples,
        repo_root=repo_root,
        index_path=index_path,
        run_dir=run_dir,
        run_gen0=False,
        slow_generations=1,
        fast_generations=0,
    )

    snapshot = result.generation_snapshots[1]
    assert snapshot.overlay_worker_ids == ["worker-bridge-a-b", "worker-persistent-child"]
    assert snapshot.lifecycle_counts == {"base": 2, "probationary": 1, "persistent": 1}
    assert snapshot.structural_action_counts == {
        "birth": 1,
        "specialize": 1,
        "promote": 1,
        "strengthen_route": 2,
    }
    # No routes were ever saved against this generation's (real, empty)
    # ColonyMemoryStore in this test -- raw_route_proposals and
    # total_route_count both correctly read 0 rather than crashing.
    assert snapshot.raw_route_proposals == 0
    assert snapshot.total_route_count == 0


def test_a_fast_only_followup_does_not_lose_the_earlier_slow_gen1_generation_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    # Regression test for a real bug found live on qibo: this project's own
    # documented two-call pattern for "owner/repo"-formatted datasets runs
    # slow-gen1 in one `gen-compare` call, then a fast pass in a SEPARATE
    # later call against the same run_dir. The evolution telemetry used to
    # live in flat evolution_events/slow_worker_count fields that were only
    # ever set on the call that actually ran slow-gen1 -- a later
    # fast-only call re-initialized them to []/default and its own summary
    # write silently discarded the first call's real, non-empty telemetry.
    # Now each generation persists its own slow-gen{k}-evolution.json, and
    # a later call that keeps that generation in its reported range (even
    # without recomputing it) reloads that file instead of re-initializing.
    run_dir = tmp_path / "run"
    index_path = tmp_path / ".ant"
    (index_path / "myrepo").mkdir(parents=True)
    repo_root = tmp_path / "repos"
    (repo_root / "myrepo").mkdir(parents=True)

    real_events = [
        EvolutionEvent(
            kind="strengthen_route",
            worker_id="worker-a",
            reason="Episode-driven.",
            source_worker_ids=["worker-a"],
        )
    ]

    class _StubIndexStore:
        def __init__(self, path):
            pass

        def load_workers(self):
            return []

    def fake_run_batch(**kwargs):
        return []

    def fake_evolve_workers(index_path, repo_root=None, reasoner=None, generation=0):
        return EvolutionResult(events=real_events, worker_count=7)

    def fake_load_or_rebuild_scores(**kwargs):
        return {}

    examples = [EvalExample(id="q1", question="q", answer="a", repo="someorg/myrepo")]

    monkeypatch.setattr("ant.evaluation.gen_compare.IndexStore", _StubIndexStore)
    monkeypatch.setattr("ant.evaluation.gen_compare.run_batch", fake_run_batch)
    monkeypatch.setattr("ant.evaluation.gen_compare.evolve_workers", fake_evolve_workers)
    monkeypatch.setattr("ant.evaluation.gen_compare.OpenAIProvider", lambda: object())
    monkeypatch.setattr(
        "ant.evaluation.gen_compare._load_or_rebuild_scores", fake_load_or_rebuild_scores
    )

    slow_result = run_gen_compare(
        examples=examples,
        repo_root=repo_root,
        index_path=index_path,
        run_dir=run_dir,
        run_gen0=False,
        slow_generations=1,
        fast_generations=0,
    )
    assert [e.kind for e in slow_result.generation_snapshots[1].events] == ["strengthen_route"]
    assert slow_result.generation_snapshots[1].worker_count == 7

    def fake_run_fast_generation(**kwargs):
        return {}

    monkeypatch.setattr(
        "ant.evaluation.gen_compare._run_fast_generation", fake_run_fast_generation
    )
    # run_gen_compare's own pre-flight check for a fast pass requires that
    # generation's own trajectory to already exist on disk under run_dir --
    # create a placeholder, since this test's gen0 stage never actually ran.
    (run_dir / "gen0-q1.json").write_text(
        EvidenceState(question="q", answer="a").model_dump_json(), encoding="utf-8"
    )

    fast_only_result = run_gen_compare(
        examples=examples,
        repo_root=repo_root / "myrepo",
        index_path=index_path,
        run_dir=run_dir,
        run_gen0=False,
        # Keep generation 1 in the reported range (so its real snapshot is
        # read back) without recomputing it (start_generation=2 keeps it
        # below the "actually run evolve_workers" threshold).
        slow_generations=1,
        start_generation=2,
        # fast-gen0 only -- anchored to gen0's own trajectory.
        fast_generations=1,
    )

    assert [e.kind for e in fast_only_result.generation_snapshots[1].events] == [
        "strengthen_route"
    ]
    assert fast_only_result.generation_snapshots[1].events[0].source_worker_ids == ["worker-a"]
    assert fast_only_result.generation_snapshots[1].worker_count == 7


def _write_gen0_fixture(run_dir: Path, example_id: str, answer: str, score: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    state = EvidenceState(question="q", answer=answer)
    (run_dir / f"gen0-{example_id}.json").write_text(
        json.dumps(state.model_dump(), indent=2), encoding="utf-8"
    )
    row = {
        "example_id": example_id,
        "question": "q",
        "prediction": answer,
        "score": score,
    }
    (run_dir / "gen0-results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_run_fast_generation_reuses_the_anchors_score_when_the_monotonic_gate_leaves_it_untouched(
    tmp_path: Path, monkeypatch
) -> None:
    # Regression test for a real finding on a fresh yt-dlp run: 6/10
    # questions hit the monotonic gate (nothing grounded this retry, so
    # LocalCoordinator.ask leaves prior_state.answer completely
    # untouched -- see that method's own docstring), and re-judging the
    # SAME text a second time produced a DIFFERENT score up to +-4 points
    # apart -- f1 (a deterministic metric) was identical both times, only
    # the LLM judge's own rubric scores moved. That is pure judge-sampling
    # noise on a case where nothing about the actual answer changed at
    # all. Reusing the anchor generation's own already-judged score
    # removes it, at zero extra judge cost.
    run_dir = tmp_path / "run"
    gen0_score = {
        "exact_match": False,
        "contains_answer": False,
        "f1": 0.42,
        "evidence_count": 3,
        "unresolved_need_count": 1,
    }
    _write_gen0_fixture(run_dir, "q1", "gen0's own verbatim answer", gen0_score)

    class _StubIndexStore:
        def __init__(self, path):
            pass

        def load_workers(self):
            return []

    class _StubCoordinator:
        def __init__(self, *args, **kwargs):
            pass

        def retry_from_trajectory(self, prior_state, fast_reasoner, max_rounds):
            # The monotonic gate: answer identical to prior_state's own.
            return EvidenceState(question="q", answer=prior_state.answer)

    judge_calls = []

    def fake_judge_answer(**kwargs):
        judge_calls.append(kwargs)
        raise AssertionError("judge_answer must not be called when reusing the anchor's score")

    monkeypatch.setattr("ant.evaluation.gen_compare.IndexStore", _StubIndexStore)
    monkeypatch.setattr("ant.evaluation.gen_compare.LocalCoordinator", _StubCoordinator)
    monkeypatch.setattr("ant.evaluation.gen_compare.OpenAIProvider", lambda: object())
    monkeypatch.setattr("ant.evaluation.gen_compare.judge_answer", fake_judge_answer)

    examples = [EvalExample(id="q1", question="q", answer="expected", repo="someorg/myrepo")]

    scores = _run_fast_generation(
        examples=examples,
        repo_root=tmp_path / "repos",
        anchor_index=tmp_path / ".ant",
        run_dir=run_dir,
        trace_prefix="gen0-",
        out_prefix="fast-gen0-",
        max_rounds=1,
        judge="openai",
    )

    assert judge_calls == []
    assert scores["q1"] == EvalScore.model_validate(gen0_score)


def test_run_fast_generation_still_judges_when_the_answer_actually_changed(
    tmp_path: Path, monkeypatch
) -> None:
    # The reuse optimization must not become a blanket skip -- a retry
    # that actually produced different text still needs a real judge call.
    run_dir = tmp_path / "run"
    _write_gen0_fixture(
        run_dir,
        "q1",
        "gen0's own verbatim answer",
        {
            "exact_match": False,
            "contains_answer": False,
            "f1": 0.42,
            "evidence_count": 3,
            "unresolved_need_count": 1,
        },
    )

    class _StubIndexStore:
        def __init__(self, path):
            pass

        def load_workers(self):
            return []

    class _StubCoordinator:
        def __init__(self, *args, **kwargs):
            pass

        def retry_from_trajectory(self, prior_state, fast_reasoner, max_rounds):
            return EvidenceState(question="q", answer="a genuinely different retry answer")

    judge_calls = []

    def fake_judge_answer(**kwargs):
        judge_calls.append(kwargs)
        return EvalScore(
            exact_match=False, contains_answer=False, f1=0.9,
            evidence_count=0, unresolved_need_count=0,
        )

    monkeypatch.setattr("ant.evaluation.gen_compare.IndexStore", _StubIndexStore)
    monkeypatch.setattr("ant.evaluation.gen_compare.LocalCoordinator", _StubCoordinator)
    monkeypatch.setattr("ant.evaluation.gen_compare.OpenAIProvider", lambda: object())
    monkeypatch.setattr("ant.evaluation.gen_compare.judge_answer", fake_judge_answer)

    examples = [EvalExample(id="q1", question="q", answer="expected", repo="someorg/myrepo")]

    scores = _run_fast_generation(
        examples=examples,
        repo_root=tmp_path / "repos",
        anchor_index=tmp_path / ".ant",
        run_dir=run_dir,
        trace_prefix="gen0-",
        out_prefix="fast-gen0-",
        max_rounds=1,
        judge="openai",
    )

    assert len(judge_calls) == 1
    assert scores["q1"].f1 == 0.9


def test_slow_generations_are_cumulative_evolve_workers_runs_once_per_generation(
    tmp_path: Path, monkeypatch
) -> None:
    # The core new invariant: --slow-generations N evolves the SAME,
    # ever-accumulating index N times in a row (slow-gen2 evolves whatever
    # slow-gen1's own run left behind, not a fresh copy of gen0's), each
    # producing its own named results/evolution-metadata/index-snapshot,
    # never overwriting an earlier generation's.
    repo_root = tmp_path / "repos"
    (repo_root / "myrepo").mkdir(parents=True)
    index_path = tmp_path / ".ant"
    (index_path / "myrepo").mkdir(parents=True)
    run_dir = tmp_path / "run"

    evolve_calls: list[int] = []
    run_batch_calls: list[str] = []

    class _StubIndexStore:
        def __init__(self, path):
            pass

        def load_workers(self):
            return []

    def fake_run_batch(**kwargs):
        run_batch_calls.append(kwargs["state_dump_prefix"])
        return []

    def fake_evolve_workers(index_path, repo_root=None, reasoner=None, generation=0):
        evolve_calls.append(len(evolve_calls) + 1)
        return EvolutionResult(events=[], worker_count=len(evolve_calls))

    monkeypatch.setattr("ant.evaluation.gen_compare.run_batch", fake_run_batch)
    monkeypatch.setattr("ant.evaluation.gen_compare.evolve_workers", fake_evolve_workers)
    monkeypatch.setattr("ant.evaluation.gen_compare.IndexStore", _StubIndexStore)
    monkeypatch.setattr("ant.evaluation.gen_compare.OpenAIProvider", lambda: object())

    examples = [EvalExample(id="q1", question="q", answer="a", repo="someorg/myrepo")]

    result = run_gen_compare(
        examples=examples,
        repo_root=repo_root,
        index_path=index_path,
        run_dir=run_dir,
        run_gen0=False,
        slow_generations=3,
        fast_generations=0,
    )

    assert evolve_calls == [1, 2, 3]
    assert run_batch_calls == ["slow-gen1-", "slow-gen2-", "slow-gen3-"]
    assert set(result.slow_generations) == {1, 2, 3}
    assert set(result.generation_snapshots) == {1, 2, 3}
    assert [
        result.generation_snapshots[g].worker_count for g in (1, 2, 3)
    ] == [1, 2, 3]
    for generation in (1, 2, 3):
        assert (run_dir / f"slow-gen{generation}-evolution.json").exists()
        assert (run_dir / f"_gen{generation}_index_snapshot").exists()


def test_start_generation_resumes_without_recomputing_earlier_generations(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = tmp_path / "repos"
    (repo_root / "myrepo").mkdir(parents=True)
    index_path = tmp_path / ".ant"
    (index_path / "myrepo").mkdir(parents=True)
    run_dir = tmp_path / "run"

    evolve_calls: list[int] = []

    class _StubIndexStore:
        def __init__(self, path):
            pass

        def load_workers(self):
            return []

    def fake_run_batch(**kwargs):
        return []

    def fake_evolve_workers(index_path, repo_root=None, reasoner=None, generation=0):
        evolve_calls.append(len(evolve_calls) + 1)
        return EvolutionResult(events=[], worker_count=len(evolve_calls))

    monkeypatch.setattr("ant.evaluation.gen_compare.run_batch", fake_run_batch)
    monkeypatch.setattr("ant.evaluation.gen_compare.evolve_workers", fake_evolve_workers)
    monkeypatch.setattr("ant.evaluation.gen_compare.IndexStore", _StubIndexStore)
    monkeypatch.setattr("ant.evaluation.gen_compare.OpenAIProvider", lambda: object())

    examples = [EvalExample(id="q1", question="q", answer="a", repo="someorg/myrepo")]

    # First call reaches generation 2.
    run_gen_compare(
        examples=examples,
        repo_root=repo_root,
        index_path=index_path,
        run_dir=run_dir,
        run_gen0=False,
        slow_generations=2,
        fast_generations=0,
    )
    assert evolve_calls == [1, 2]

    # A second call asking for generation 3 with start_generation=3 must
    # NOT re-run evolve_workers for generations 1 or 2 -- only generation 3
    # is new work -- yet must still report 1 and 2 in its own summary.
    result = run_gen_compare(
        examples=examples,
        repo_root=repo_root,
        index_path=index_path,
        run_dir=run_dir,
        run_gen0=False,
        slow_generations=3,
        start_generation=3,
        fast_generations=0,
    )

    assert evolve_calls == [1, 2, 3]
    assert set(result.generation_snapshots) == {1, 2, 3}
