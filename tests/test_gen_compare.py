import json
from pathlib import Path

from ant.domain import EvidenceState
from ant.evaluation.datasets import EvalExample
from ant.evaluation.gen_compare import _run_fast_gen1, run_gen_compare
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
    run_dir = tmp_path / "run"

    captured: dict[str, object] = {}

    def fake_run_batch(**kwargs):
        return []

    def fake_evolve_workers(index_path, repo_root=None, reasoner=None):
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
        run_slow_gen1=True,
        run_fast_gen1=False,
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

    captured: dict[str, object] = {}

    def fake_run_batch(**kwargs):
        return []

    def fake_evolve_workers(index_path, repo_root=None, reasoner=None):
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
        run_slow_gen1=True,
        run_fast_gen1=False,
    )

    assert captured["index_path"] == resolved_index_path


def test_fast_gen1_only_followup_does_not_overwrite_the_earlier_slow_gen1_evolution_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    # Regression test for a real bug found live on qibo: this project's own
    # documented two-call pattern for "owner/repo"-formatted datasets runs
    # slow-gen1 in one `gen-compare` call, then fast-gen1 in a SEPARATE
    # later call against the same run_dir. evolution_events/slow_worker_count
    # (unlike gen0/slow-gen1/fast-gen1's own score dicts, which always
    # reload via _load_or_rebuild_scores regardless of which stage ran this
    # call) were only ever set `if run_slow_gen1:` -- the later fast-gen1-
    # only call re-initialized them to []/gen0_worker_count and its own
    # summary write silently discarded the first call's real, non-empty
    # evolution telemetry. Confirmed live: evolve_workers really did fire 9
    # strengthen_route events and record them in its own return value, but
    # the run_dir's on-disk summary.json ended up reporting evolution_events:
    # [] because the fast-gen1-only call ran after it.
    run_dir = tmp_path / "run"
    index_path = tmp_path / ".ant"
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

    def fake_evolve_workers(index_path, repo_root=None, reasoner=None):
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
        run_slow_gen1=True,
        run_fast_gen1=False,
    )
    assert [e.kind for e in slow_result.evolution_events] == ["strengthen_route"]
    assert slow_result.slow_worker_count == 7

    def fake_run_fast_gen1(**kwargs):
        return {}

    monkeypatch.setattr("ant.evaluation.gen_compare._run_fast_gen1", fake_run_fast_gen1)
    # run_gen_compare's own pre-flight check for run_fast_gen1 requires each
    # example's gen0 trace to already exist on disk under run_dir -- create
    # a placeholder, since this test's gen0 stage never actually ran.
    (run_dir / "gen0-q1.json").write_text(
        EvidenceState(question="q", answer="a").model_dump_json(), encoding="utf-8"
    )

    fast_only_result = run_gen_compare(
        examples=examples,
        repo_root=repo_root / "myrepo",
        index_path=index_path,
        run_dir=run_dir,
        run_gen0=False,
        run_slow_gen1=False,
        run_fast_gen1=True,
    )

    assert [e.kind for e in fast_only_result.evolution_events] == ["strengthen_route"]
    assert fast_only_result.evolution_events[0].source_worker_ids == ["worker-a"]
    assert fast_only_result.slow_worker_count == 7


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


def test_run_fast_gen1_reuses_gen0_score_when_the_monotonic_gate_leaves_the_answer_untouched(
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
    # all. Reusing gen0's own already-judged score removes it, at zero
    # extra judge cost.
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
        raise AssertionError("judge_answer must not be called when reusing gen0's score")

    monkeypatch.setattr("ant.evaluation.gen_compare.IndexStore", _StubIndexStore)
    monkeypatch.setattr("ant.evaluation.gen_compare.LocalCoordinator", _StubCoordinator)
    monkeypatch.setattr("ant.evaluation.gen_compare.OpenAIProvider", lambda: object())
    monkeypatch.setattr("ant.evaluation.gen_compare.judge_answer", fake_judge_answer)

    examples = [EvalExample(id="q1", question="q", answer="expected", repo="someorg/myrepo")]

    scores = _run_fast_gen1(
        examples=examples,
        repo_root=tmp_path / "repos",
        gen0_index=tmp_path / ".ant",
        run_dir=run_dir,
        max_rounds=1,
        judge="openai",
    )

    assert judge_calls == []
    assert scores["q1"] == EvalScore.model_validate(gen0_score)


def test_run_fast_gen1_still_judges_when_the_answer_actually_changed(
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

    scores = _run_fast_gen1(
        examples=examples,
        repo_root=tmp_path / "repos",
        gen0_index=tmp_path / ".ant",
        run_dir=run_dir,
        max_rounds=1,
        judge="openai",
    )

    assert len(judge_calls) == 1
    assert scores["q1"].f1 == 0.9
