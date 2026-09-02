from pathlib import Path

from ant.evaluation.datasets import EvalExample
from ant.evaluation.gen_compare import run_gen_compare
from ant.evolution import EvolutionResult


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
