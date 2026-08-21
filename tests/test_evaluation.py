import json
from pathlib import Path

from ant.environment import RepoEnvironment
from ant.evaluation import EvalExample, build_report, load_examples, run_batch
from ant.evaluation.datasets import _dataset_alias
from ant.indexing import build_worker_cards, discover_territories
from ant.memory import IndexStore


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
