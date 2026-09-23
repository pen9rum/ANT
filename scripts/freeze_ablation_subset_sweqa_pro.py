"""Freeze a 40/80 SWE-QA-Pro subset for the adaptive-coordination ablations.

Selection is metadata-only stratified random sampling, fixed before any
ablation variant is run: stratify by repository (the primary stratum, since
the 80-question manifest is exactly 8 repos x 10 questions each -> a fixed
5 per repo), uniform random within each repo using a fixed seed. No model
output or performance measurement is read or used to pick questions.
`qa_type` is used only afterward, to report how closely the sampled
distribution tracks the full 80's distribution -- never to hand-curate
"diverse" questions.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = (
    REPO_ROOT / "third_party" / "manifests" / "sweqa_pro" / "sample_manifest_sweqa_pro_80.json"
)
TASK_METADATA = (
    REPO_ROOT / "third_party" / "manifests" / "sweqa_pro" / "task_metadata_sweqa_pro_80.json"
)
OUT_PATH = (
    REPO_ROOT
    / "third_party"
    / "manifests"
    / "sweqa_pro"
    / "ablation_subset_sweqa_pro_40.json"
)
SEED = 42
QUESTIONS_PER_REPO_SAMPLED = 5


def main() -> None:
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    task_metadata = json.loads(TASK_METADATA.read_text(encoding="utf-8"))
    all_task_ids = source["task_ids"]
    assert len(all_task_ids) == 80, f"expected 80 source task_ids, got {len(all_task_ids)}"

    by_repo: dict[str, list[str]] = {}
    for task_id in all_task_ids:
        repo = task_metadata[task_id]["repo"]
        by_repo.setdefault(repo, []).append(task_id)
    assert len(by_repo) == 8, f"expected 8 repos, got {len(by_repo)}"
    for repo, ids in by_repo.items():
        assert len(ids) == 10, f"expected 10 questions for {repo}, got {len(ids)}"

    selected_by_repo: dict[str, list[str]] = {}
    for repo in sorted(by_repo):
        ids_sorted = sorted(by_repo[repo])
        rng = random.Random(SEED)
        selected_by_repo[repo] = sorted(rng.sample(ids_sorted, QUESTIONS_PER_REPO_SAMPLED))

    selected_task_ids = sorted(
        task_id for ids in selected_by_repo.values() for task_id in ids
    )
    assert len(selected_task_ids) == 40, len(selected_task_ids)
    assert len(set(selected_task_ids)) == 40, "duplicate task_ids in selection"

    def qa_class_distribution(task_ids: list[str]) -> dict[str, float]:
        counts = Counter(task_metadata[tid]["qa_type"]["class_name"] for tid in task_ids)
        n = len(task_ids)
        return {cls: round(count / n, 4) for cls, count in sorted(counts.items())}

    full_dist = qa_class_distribution(all_task_ids)
    sampled_dist = qa_class_distribution(selected_task_ids)

    payload = {
        "name": "sweqa_pro_ablation_subset_40",
        "status": "frozen_final",
        "frozen_final": True,
        "benchmark": "sweqa_pro",
        "source_manifest": str(SOURCE_MANIFEST.relative_to(REPO_ROOT)),
        "selection_method": (
            "Metadata-only stratified random sampling, fixed before running any "
            "ablation variant. Primary stratum: repository (8 repos x 10 questions "
            "each in the source 80-question manifest, proportional to the original "
            "distribution -> a fixed 5 questions sampled per repo). Within each "
            "repo, uniform random sampling (random.Random(seed).sample) over the "
            "repo's 10 task_ids, sorted before sampling for determinism. No model "
            "output or performance measurement was read or used to select "
            "questions. qa_type distribution below is reported only as a post-hoc "
            "check that the sampled subset roughly tracks the full 80's "
            "distribution, not as a selection criterion."
        ),
        "seed": SEED,
        "questions_per_repository_sampled": QUESTIONS_PER_REPO_SAMPLED,
        "total_task_ids": 40,
        "total_repositories": 8,
        "selected_task_ids_by_repo": selected_by_repo,
        "task_ids": selected_task_ids,
        "qa_type_class_distribution_full_80": full_dist,
        "qa_type_class_distribution_sampled_40": sampled_dist,
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Selected {len(selected_task_ids)} task_ids across {len(selected_by_repo)} repos.")
    print("\nqa_type class_name distribution -- full 80 vs sampled 40:")
    all_classes = sorted(set(full_dist) | set(sampled_dist))
    for cls in all_classes:
        print(f"  {cls:<35} full={full_dist.get(cls, 0.0):.4f}  sampled={sampled_dist.get(cls, 0.0):.4f}")


if __name__ == "__main__":
    main()
