import hashlib
import json
from pathlib import Path

from ant.domain import EvidenceState
from ant.environment import RepoEnvironment
from ant.evaluation import EvalExample, build_report, load_examples, run_batch, run_gen_compare
from ant.evaluation.datasets import _dataset_alias
from ant.evaluation.judge import JUDGE_PROMPT, judge_answer
from ant.evaluation.metrics import EvalScore
from ant.evolution import EvolutionResult
from ant.indexing import build_worker_cards, discover_territories
from ant.memory import ColonyMemoryStore, IndexStore

# Pinned hash of the user-specified evaluator prompt (see the "LOCKED
# PROMPT" comment above JUDGE_PROMPT in judge.py). If this test fails, the
# prompt text was changed -- get explicit confirmation from the user before
# updating both JUDGE_PROMPT and this hash together.
_LOCKED_JUDGE_PROMPT_SHA256 = "b5356a26ddc57dab13e69fe8b52bd1ab7252c95065a20128a3064f00d9df114d"


def test_judge_prompt_is_pinned_and_unmodified() -> None:
    digest = hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest()
    assert digest == _LOCKED_JUDGE_PROMPT_SHA256, (
        "JUDGE_PROMPT text changed -- this prompt is locked per explicit user "
        "instruction. If the change was intentional, update "
        "_LOCKED_JUDGE_PROMPT_SHA256 in this test to match."
    )


def test_load_jsonl_examples(tmp_path: Path) -> None:
    path = tmp_path / "examples.jsonl"
    path.write_text(
        json.dumps({"id": "q1", "question": "Where is auth?", "answer": "auth.py"}) + "\n",
        encoding="utf-8",
    )

    examples = load_examples(str(path))

    assert examples[0].id == "q1"
    assert examples[0].question == "Where is auth?"
    assert _dataset_alias("swe-qa-pro") == "TIGER-Lab/SWE-QA-Pro-Bench"


def test_run_batch_writes_results(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text("def authenticate_user():\n    return True\n", encoding="utf-8")
    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    workers = build_worker_cards(environment.root, territories)
    index_path = tmp_path / ".ant"
    IndexStore(index_path).save(territories, workers)

    results = run_batch(
        examples=[EvalExample(id="q1", question="authenticate", answer="auth.py")],
        repo_root=repo,
        index_path=index_path,
        out_path=tmp_path / "results.jsonl",
    )

    assert len(results) == 1
    assert results[0].score.evidence_count > 0
    assert (tmp_path / "results.jsonl").exists()
    report = build_report(tmp_path / "results.jsonl")
    assert report.count == 1
    routes = ColonyMemoryStore(index_path).matching_routes(["authenticate"])
    assert routes
    assert routes[0].worker_ids == ["worker-root"]


def test_run_batch_isolates_a_crashing_example_and_continues(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text("def authenticate_user():\n    return True\n", encoding="utf-8")
    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    workers = build_worker_cards(environment.root, territories)
    index_path = tmp_path / ".ant"
    IndexStore(index_path).save(territories, workers)

    class CrashingCoordinator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def ask(self, question, max_rounds=2):
            raise RuntimeError("simulated model failure")

    monkeypatch.setattr("ant.evaluation.runner.LocalCoordinator", CrashingCoordinator)

    results = run_batch(
        examples=[
            EvalExample(id="bad-1", question="boom one", answer=""),
            EvalExample(id="bad-2", question="boom two", answer=""),
        ],
        repo_root=repo,
        index_path=index_path,
        out_path=tmp_path / "results.jsonl",
    )

    # Both examples must be present -- a crash on the first must not abort
    # the loop and silently drop the second.
    assert [result.example_id for result in results] == ["bad-1", "bad-2"]
    assert all(result.status.startswith("error: RuntimeError") for result in results)
    assert all(result.score.evidence_count == 0 for result in results)
    written = (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(written) == 2


def test_run_batch_skips_missing_swe_repo(tmp_path: Path) -> None:
    results = run_batch(
        examples=[EvalExample(id="q1", question="q", repo="owner/missing")],
        repo_root=tmp_path / "repos",
        index_path=tmp_path / ".ant",
        out_path=tmp_path / "results.jsonl",
    )

    assert results[0].status == "skipped_missing_repo"


def test_openai_judge_uses_official_sweqa_evaluator_model(monkeypatch) -> None:
    captured = {}

    class FakeProvider:
        def __init__(self, model=None, reasoning_effort=None):
            captured["model"] = model
            captured["reasoning_effort"] = reasoning_effort

        def responses_json(self, prompt, max_output_tokens=512):
            captured["prompt"] = prompt
            captured["max_output_tokens"] = max_output_tokens

            class Result:
                text = json.dumps(
                    {
                        "correctness": 6,
                        "completeness": 5,
                        "relevance": 7,
                        "clarity": 8,
                        "reasoning": 4,
                    }
                )

            return Result()

        def drain_usage(self):
            from ant.domain import TokenUsage

            return TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)

    monkeypatch.setattr("ant.evaluation.judge.OpenAIProvider", FakeProvider)

    score = judge_answer(
        question="q",
        prediction="candidate",
        expected="reference",
        evidence_count=1,
        unresolved_need_count=0,
        judge="openai",
    )

    assert captured["model"] == "gpt-5-2025-08-07"
    assert captured["reasoning_effort"] == "low"
    assert captured["max_output_tokens"] == 1024
    assert isinstance(score, EvalScore)
    assert score.correctness == 6
    # Regression test: the judge's own OpenAIProvider is local to
    # judge_answer() and was never drained anywhere -- its cost simply
    # vanished. score.usage must carry it out.
    assert score.usage.input_tokens == 10


def test_run_batch_dumps_each_examples_full_evidence_state_when_asked(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text("def authenticate_user():\n    return True\n", encoding="utf-8")
    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    workers = build_worker_cards(environment.root, territories)
    index_path = tmp_path / ".ant"
    IndexStore(index_path).save(territories, workers)
    dump_dir = tmp_path / "dumps"

    run_batch(
        examples=[EvalExample(id="q1", question="authenticate", answer="auth.py")],
        repo_root=repo,
        index_path=index_path,
        out_path=tmp_path / "results.jsonl",
        state_dump_dir=dump_dir,
        state_dump_prefix="gen0-",
    )

    dumped = EvidenceState.model_validate_json(
        (dump_dir / "gen0-q1.json").read_text(encoding="utf-8")
    )
    assert dumped.question == "authenticate"
    assert dumped.has_evidence()


def test_run_batch_dumps_nothing_when_state_dump_dir_is_not_given(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text("def authenticate_user():\n    return True\n", encoding="utf-8")
    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    workers = build_worker_cards(environment.root, territories)
    index_path = tmp_path / ".ant"
    IndexStore(index_path).save(territories, workers)

    run_batch(
        examples=[EvalExample(id="q1", question="authenticate", answer="auth.py")],
        repo_root=repo,
        index_path=index_path,
        out_path=tmp_path / "results.jsonl",
    )

    # No dump directory materializes anywhere under tmp_path -- the plain
    # eval CLI path (no state_dump_dir) must be unaffected.
    assert not any(path.name.startswith("gen0-") for path in tmp_path.rglob("*.json"))


def test_run_gen_compare_freezes_the_gen0_worker_snapshot_before_evolve_mutates_it(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text("def authenticate_user():\n    return True\n", encoding="utf-8")
    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    workers = build_worker_cards(environment.root, territories)
    index_path = tmp_path / ".ant"
    IndexStore(index_path).save(territories, workers)
    gen0_worker_count = len(IndexStore(index_path).load_workers())
    run_dir = tmp_path / "run"

    calls: list[str] = []

    def fake_run_batch(
        *,
        examples,
        repo_root,
        index_path,
        out_path,
        max_rounds,
        synthesize,
        judge,
        state_dump_dir=None,
        state_dump_prefix="",
    ):
        calls.append(f"run_batch:{state_dump_prefix or 'gen0-'}")
        results = []
        for example in examples:
            score = EvalScore(
                exact_match=False, contains_answer=False, evidence_count=1, unresolved_need_count=0
            )
            results.append(
                {
                    "example_id": example.id,
                    "question": example.question,
                    "score": score.model_dump(),
                }
            )
            if state_dump_dir is not None:
                state = EvidenceState(question=example.question, answer="an answer")
                (state_dump_dir / f"{state_dump_prefix}{example.id}.json").write_text(
                    state.model_dump_json(), encoding="utf-8"
                )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row) + "\n")
        return []

    def fake_evolve_workers(index_path, repo_root=None, reasoner=None):
        calls.append("evolve_workers")
        # Simulate evolution actually mutating the live index -- if the
        # fast-gen1 stage's snapshot were taken AFTER this, it would see
        # this extra worker instead of gen0's original set.
        current = IndexStore(index_path).load_workers()
        extra = current[0].model_copy(update={"id": f"{current[0].id}-evolved"})
        IndexStore(index_path).save(territories, [*current, extra])
        return EvolutionResult(events=[], worker_count=len(current) + 1)

    class FakeCoordinator:
        def __init__(self, repo_root, workers, synthesizer=None, index_path=None):
            self.workers = workers

        def retry_from_trajectory(self, prior_state, fast_reasoner, max_rounds):
            calls.append("retry_from_trajectory")
            # The whole point of the pre-evolve snapshot: fast-gen1 must
            # see gen0's own worker count, never the post-evolve one.
            assert len(self.workers) == gen0_worker_count
            return EvidenceState(question=prior_state.question, answer="fast answer")

    monkeypatch.setattr("ant.evaluation.gen_compare.run_batch", fake_run_batch)
    monkeypatch.setattr("ant.evaluation.gen_compare.evolve_workers", fake_evolve_workers)
    monkeypatch.setattr("ant.evaluation.gen_compare.LocalCoordinator", FakeCoordinator)
    monkeypatch.setattr("ant.evaluation.gen_compare.OpenAIProvider", lambda: object())

    result = run_gen_compare(
        examples=[EvalExample(id="q1", question="authenticate", answer="auth.py")],
        repo_root=repo,
        index_path=index_path,
        run_dir=run_dir,
    )

    assert calls == [
        "run_batch:gen0-",
        "evolve_workers",
        "run_batch:slow-gen1-",
        "retry_from_trajectory",
    ]
    assert result.gen0_worker_count == gen0_worker_count
    assert result.slow_worker_count == gen0_worker_count + 1
    assert "q1" in result.fast_gen1
    assert (run_dir / "fast-gen1-q1.json").exists()
    fast_state = EvidenceState.model_validate_json(
        (run_dir / "fast-gen1-q1.json").read_text(encoding="utf-8")
    )
    assert fast_state.answer == "fast answer"


def test_run_gen_compare_skips_slow_and_fast_stages_when_asked(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    workers = build_worker_cards(environment.root, territories)
    index_path = tmp_path / ".ant"
    IndexStore(index_path).save(territories, workers)

    calls: list[str] = []

    def fake_run_batch(
        *,
        examples,
        repo_root,
        index_path,
        out_path,
        max_rounds,
        synthesize,
        judge,
        state_dump_dir=None,
        state_dump_prefix="",
    ):
        calls.append(f"run_batch:{state_dump_prefix or 'gen0-'}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for example in examples:
                if state_dump_dir is not None:
                    state = EvidenceState(question=example.question, answer="an answer")
                    (state_dump_dir / f"{state_dump_prefix}{example.id}.json").write_text(
                        state.model_dump_json(), encoding="utf-8"
                    )
                handle.write(
                    json.dumps(
                        {
                            "example_id": example.id,
                            "question": example.question,
                            "score": EvalScore(
                                exact_match=False,
                                contains_answer=False,
                                evidence_count=0,
                                unresolved_need_count=0,
                            ).model_dump(),
                        }
                    )
                    + "\n"
                )
        return []

    def fail_evolve_workers(*args, **kwargs):
        raise AssertionError("evolve_workers must not run when skip_slow_gen1 is set")

    class FailCoordinator:
        def __init__(self, *args, **kwargs):
            raise AssertionError("LocalCoordinator must not run when skip_fast_gen1 is set")

    monkeypatch.setattr("ant.evaluation.gen_compare.run_batch", fake_run_batch)
    monkeypatch.setattr("ant.evaluation.gen_compare.evolve_workers", fail_evolve_workers)
    monkeypatch.setattr("ant.evaluation.gen_compare.LocalCoordinator", FailCoordinator)
    monkeypatch.setattr("ant.evaluation.gen_compare.OpenAIProvider", lambda: object())

    result = run_gen_compare(
        examples=[EvalExample(id="q1", question="authenticate", answer="auth.py")],
        repo_root=repo,
        index_path=index_path,
        run_dir=tmp_path / "run",
        run_slow_gen1=False,
        run_fast_gen1=False,
    )

    assert calls == ["run_batch:gen0-"]
    assert result.slow_gen1 == {}
    assert result.fast_gen1 == {}


def test_run_gen_compare_rebuilds_a_missing_results_jsonl_from_saved_traces_instead_of_erasing_it(
    tmp_path: Path, monkeypatch
) -> None:
    # Regression test: a run_dir written by an ad-hoc pre-run_gen_compare
    # script saves each stage's EvidenceState JSON but never a
    # *-results.jsonl. A fast-gen1-only re-run against it used to silently
    # report gen0/slow-gen1 as {} (the real, already-computed numbers
    # discarded) instead of rebuilding them from the still-intact trace
    # files -- confirmed live on two real run dirs before this fix existed.
    repo = tmp_path / "repo"
    repo.mkdir()
    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    workers = build_worker_cards(environment.root, territories)
    index_path = tmp_path / ".ant"
    IndexStore(index_path).save(territories, workers)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    example = EvalExample(id="q1", question="authenticate", answer="auth.py")
    gen0_state = EvidenceState(question=example.question, answer="gen0 answer")
    (run_dir / "gen0-q1.json").write_text(gen0_state.model_dump_json(), encoding="utf-8")
    # No gen0-results.jsonl written -- the exact legacy shape that triggered
    # the bug.

    judge_calls: list[str] = []

    def fake_judge_answer(
        *, question, prediction, expected, evidence_count, unresolved_need_count, judge, idf=None
    ):
        judge_calls.append(prediction)
        return EvalScore(
            exact_match=False,
            contains_answer=False,
            evidence_count=evidence_count,
            unresolved_need_count=unresolved_need_count,
            correctness=7,
        )

    monkeypatch.setattr("ant.evaluation.gen_compare.judge_answer", fake_judge_answer)
    monkeypatch.setattr(
        "ant.evaluation.gen_compare.run_batch",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    monkeypatch.setattr(
        "ant.evaluation.gen_compare.evolve_workers",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    result = run_gen_compare(
        examples=[example],
        repo_root=repo,
        index_path=index_path,
        run_dir=run_dir,
        run_gen0=False,
        run_slow_gen1=False,
        run_fast_gen1=False,
    )

    assert judge_calls == ["gen0 answer"]
    assert result.gen0["q1"].correctness == 7
    assert result.slow_gen1 == {}
    # The rebuild is persisted -- a second call must not re-judge.
    assert (run_dir / "gen0-results.jsonl").exists()
    judge_calls.clear()
    run_gen_compare(
        examples=[example],
        repo_root=repo,
        index_path=index_path,
        run_dir=run_dir,
        run_gen0=False,
        run_slow_gen1=False,
        run_fast_gen1=False,
    )
    assert judge_calls == []


def test_run_gen_compare_fast_gen1_only_reuses_saved_gen0_traces_and_prior_stage_scores(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    workers = build_worker_cards(environment.root, territories)
    index_path = tmp_path / ".ant"
    IndexStore(index_path).save(territories, workers)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # A gen0 (and slow-gen1) stage already ran against this run_dir in an
    # earlier call -- exactly the "fast-gen1 only" scenario: only
    # fast-gen1's own code changed, gen0/slow-gen1 numbers are unaffected
    # and must not be recomputed, only carried through into the summary.
    example = EvalExample(id="q1", question="authenticate", answer="auth.py")
    gen0_state = EvidenceState(question=example.question, answer="gen0 answer")
    (run_dir / "gen0-q1.json").write_text(gen0_state.model_dump_json(), encoding="utf-8")
    prior_score = EvalScore(
        exact_match=False, contains_answer=False, evidence_count=1, unresolved_need_count=0
    )
    prior_row = json.dumps(
        {"example_id": "q1", "question": example.question, "score": prior_score.model_dump()}
    )
    (run_dir / "gen0-results.jsonl").write_text(prior_row + "\n", encoding="utf-8")
    (run_dir / "slow-gen1-results.jsonl").write_text(prior_row + "\n", encoding="utf-8")

    def fail_run_batch(**kwargs):
        raise AssertionError("run_batch must not run for either gen0 or slow-gen1")

    def fail_evolve_workers(*args, **kwargs):
        raise AssertionError("evolve_workers must not run when slow-gen1 is skipped")

    class FakeCoordinator:
        def __init__(self, repo_root, workers, synthesizer=None, index_path=None):
            self.index_path = index_path

        def retry_from_trajectory(self, prior_state, fast_reasoner, max_rounds):
            # No _gen0_index_snapshot exists in this run_dir (it predates
            # the snapshot mechanism) -- must fall back to index_path
            # as-is rather than crash.
            assert self.index_path == index_path
            return EvidenceState(question=prior_state.question, answer="fast answer")

    monkeypatch.setattr("ant.evaluation.gen_compare.run_batch", fail_run_batch)
    monkeypatch.setattr("ant.evaluation.gen_compare.evolve_workers", fail_evolve_workers)
    monkeypatch.setattr("ant.evaluation.gen_compare.LocalCoordinator", FakeCoordinator)
    monkeypatch.setattr("ant.evaluation.gen_compare.OpenAIProvider", lambda: object())

    result = run_gen_compare(
        examples=[example],
        repo_root=repo,
        index_path=index_path,
        run_dir=run_dir,
        run_gen0=False,
        run_slow_gen1=False,
        run_fast_gen1=True,
    )

    assert result.gen0["q1"] == prior_score
    assert result.slow_gen1["q1"] == prior_score
    assert result.fast_gen1["q1"].evidence_count == 0
    fast_state = EvidenceState.model_validate_json(
        (run_dir / "fast-gen1-q1.json").read_text(encoding="utf-8")
    )
    assert fast_state.answer == "fast answer"


def test_run_gen_compare_fast_gen1_only_raises_clearly_when_a_gen0_trace_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    workers = build_worker_cards(environment.root, territories)
    index_path = tmp_path / ".ant"
    IndexStore(index_path).save(territories, workers)
    run_dir = tmp_path / "run"
    run_dir.mkdir()  # no gen0-q1.json saved here

    monkeypatch.setattr(
        "ant.evaluation.gen_compare.run_batch",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    monkeypatch.setattr(
        "ant.evaluation.gen_compare.evolve_workers",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    try:
        run_gen_compare(
            examples=[EvalExample(id="q1", question="authenticate", answer="auth.py")],
            repo_root=repo,
            index_path=index_path,
            run_dir=run_dir,
            run_gen0=False,
            run_slow_gen1=False,
            run_fast_gen1=True,
        )
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as exc:
        assert "q1" in str(exc)
