import hashlib
import json
from pathlib import Path

from ant.environment import RepoEnvironment
from ant.evaluation import EvalExample, build_report, load_examples, run_batch
from ant.evaluation.datasets import _dataset_alias
from ant.evaluation.judge import JUDGE_PROMPT, judge_answer
from ant.evaluation.metrics import EvalScore
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
