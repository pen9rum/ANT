from __future__ import annotations

from pathlib import Path

import pytest

from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.sampling import (
    apply_sample_manifest,
    build_sample_manifest,
    load_sample_manifest,
    save_sample_manifest,
    stratified_sample,
)


def _examples(repo_counts: dict[str, int]) -> list[TaskExample]:
    out = []
    for repo, count in repo_counts.items():
        for i in range(count):
            out.append(
                TaskExample(
                    benchmark="x",
                    task_id=f"{repo}-{i:03d}",
                    question=f"question {i} about {repo}",
                    reference="",
                    metadata={"repo": repo},
                )
            )
    return out


def test_stratified_sample_is_deterministic_for_the_same_seed() -> None:
    examples = _examples({"repoA": 10, "repoB": 10})
    first = stratified_sample(examples, per_stratum=3, seed=42)
    second = stratified_sample(examples, per_stratum=3, seed=42)
    assert [e.task_id for e in first] == [e.task_id for e in second]


def test_stratified_sample_differs_across_seeds() -> None:
    examples = _examples({"repoA": 20})
    first = stratified_sample(examples, per_stratum=5, seed=1)
    second = stratified_sample(examples, per_stratum=5, seed=2)
    assert [e.task_id for e in first] != [e.task_id for e in second]


def test_stratified_sample_covers_every_stratum_up_to_per_stratum() -> None:
    examples = _examples({"repoA": 5, "repoB": 5, "repoC": 5})
    sampled = stratified_sample(examples, per_stratum=2, seed=7)
    repos = {e.metadata["repo"] for e in sampled}
    assert repos == {"repoA", "repoB", "repoC"}
    assert len(sampled) == 6


def test_stratified_sample_does_not_pad_a_small_stratum() -> None:
    examples = _examples({"repoA": 1, "repoB": 10})
    sampled = stratified_sample(examples, per_stratum=5, seed=3)
    by_repo = {"repoA": 0, "repoB": 0}
    for e in sampled:
        by_repo[e.metadata["repo"]] += 1
    assert by_repo == {"repoA": 1, "repoB": 5}


def test_stratified_sample_does_not_depend_on_input_order() -> None:
    examples = _examples({"repoA": 8, "repoB": 8})
    reversed_examples = list(reversed(examples))
    a = stratified_sample(examples, per_stratum=3, seed=11)
    b = stratified_sample(reversed_examples, per_stratum=3, seed=11)
    assert {e.task_id for e in a} == {e.task_id for e in b}


def test_manifest_round_trips_through_disk(tmp_path: Path) -> None:
    examples = _examples({"repoA": 6, "repoB": 6})
    manifest = build_sample_manifest(examples, benchmark="demo", per_stratum=2, seed=5)
    out_path = tmp_path / "manifest.json"
    save_sample_manifest(manifest, out_path)
    loaded = load_sample_manifest(out_path)
    assert loaded.task_ids == manifest.task_ids
    assert loaded.strata == {"repoA": 6, "repoB": 6}
    assert loaded.total_examples_considered == 12


def test_apply_sample_manifest_returns_exactly_the_saved_task_ids_in_saved_order() -> None:
    examples = _examples({"repoA": 6, "repoB": 6})
    manifest = build_sample_manifest(examples, benchmark="demo", per_stratum=2, seed=5)
    # Shuffle the live examples to simulate a re-fetch returning a different order.
    reshuffled = list(reversed(examples))
    applied = apply_sample_manifest(reshuffled, manifest)
    assert [e.task_id for e in applied] == manifest.task_ids


def test_apply_sample_manifest_raises_on_missing_task_id_instead_of_silently_resampling() -> None:
    examples = _examples({"repoA": 6, "repoB": 6})
    manifest = build_sample_manifest(examples, benchmark="demo", per_stratum=2, seed=5)
    reduced = [e for e in examples if e.task_id != manifest.task_ids[0]]
    with pytest.raises(KeyError):
        apply_sample_manifest(reduced, manifest)


def test_sampling_never_reads_any_score_or_metric_field() -> None:
    # A pre-specification mechanism must not be able to condition on
    # observed performance -- TaskExample itself carries no score field at
    # all, so this is enforced structurally, but assert it explicitly: the
    # sampler's own logic touches only task_id and the stratum key.
    import inspect

    from ant.evaluation_suite import sampling as sampling_module

    source = inspect.getsource(sampling_module.stratified_sample)
    assert "score" not in source.lower()
    assert "metric" not in source.lower()
