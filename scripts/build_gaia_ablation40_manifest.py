"""Freezes a 40-question stratified subset of GAIA-Text-103 for the
adaptive-coordination ablations (Static ANTMAN, Graph-free Adaptive, w/o
Need Revision, w/o Adaptive Rerouting, w/o Recovery -- Table 5/10 of the
paper). Ablations are expensive (5 variants x 40 questions, each a full
ANTMAN-shaped coordination loop) and are diagnostic, not a leaderboard
number, so they run on a subset rather than the full 103 -- same
cost/scope tradeoff the paper itself makes for its 512K-controlled and
RepoProbe-Python ablations (Table 5: "matched execution budgets", not the
full benchmark).

SAMPLING METHOD (disclosed, reproducible, not hand-picked):
  1. Start from the frozen 103-task GAIA-Text-103 manifest (gaia_text_103_
     manifest.json) -- never re-select from the raw 165-row live split.
  2. Group its 103 task_ids by GAIA's own `Level` field (1/2/3), read from
     the live dataset (requires HF_TOKEN; this script refuses to run
     against the synthetic fixtures).
  3. Allocate 40 slots across the three levels proportionally to the
     103-set's own level distribution (39/52/12) via the largest-remainder
     method -- not equal thirds, not rounded down -- so the subset's
     difficulty mix stays close to the full set's (see level_distribution
     in the output manifest: alloc 15/20/5 vs the full set's 39/52/12,
     i.e. 37.5/50.0/12.5% vs 37.9/50.5/11.7%).
  4. Within each level, sample without replacement using
     `random.Random(42).sample(sorted(task_ids_at_that_level), k)` --
     seeded and applied to a SORTED pool so the result does not depend on
     dict/set iteration order, which Python does not guarantee stable
     across processes for arbitrary hash seeds.

Usage: python scripts/build_gaia_ablation40_manifest.py
"""

from __future__ import annotations

import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ant.benchmarks.gaia import GaiaAdapter  # noqa: E402

SOURCE_MANIFEST = REPO_ROOT / "third_party" / "manifests" / "gaia" / "gaia_text_103_manifest.json"
OUT_MANIFEST = (
    REPO_ROOT / "third_party" / "manifests" / "gaia" / "gaia_ablation40_manifest.json"
)
SEED = 42
TARGET = 40


def _largest_remainder_alloc(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    raw = {k: v / total * target for k, v in counts.items()}
    floors = {k: int(raw[k]) for k in raw}
    remainder = target - sum(floors.values())
    order = sorted(raw.keys(), key=lambda k: raw[k] - floors[k], reverse=True)
    alloc = dict(floors)
    for k in order[:remainder]:
        alloc[k] += 1
    assert sum(alloc.values()) == target
    return alloc


def main() -> None:
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    assert source["status"] == "frozen"
    assert source["count"] == 103
    wanted = set(source["task_ids"])
    assert len(wanted) == 103

    adapter = GaiaAdapter(source="live")
    examples = adapter.load_examples()
    if adapter.resolved_source() != "live":
        raise SystemExit(
            f"GaiaAdapter resolved to source={adapter.resolved_source()!r}, not 'live'. "
            "This subset must be drawn against real GAIA Level labels, not synthetic "
            "fixtures -- set HF_TOKEN and retry."
        )
    by_id = {e.task_id: e for e in examples}
    missing = wanted - set(by_id)
    if missing:
        raise SystemExit(
            f"{len(missing)} task_ids from the 103-manifest not found live: {sorted(missing)}"
        )

    by_level: dict[str, list[str]] = {"1": [], "2": [], "3": []}
    for task_id in wanted:
        level = by_id[task_id].metadata["level"]
        by_level[level].append(task_id)
    for level in by_level:
        by_level[level].sort()

    full_counts = {level: len(ids) for level, ids in by_level.items()}
    assert full_counts == source["level_distribution"], (full_counts, source["level_distribution"])

    alloc = _largest_remainder_alloc(full_counts, TARGET)

    rng = random.Random(SEED)
    chosen: list[str] = []
    for level in ("1", "2", "3"):
        chosen.extend(rng.sample(by_level[level], alloc[level]))
    chosen.sort()
    assert len(chosen) == TARGET
    assert len(set(chosen)) == TARGET
    assert set(chosen) <= wanted

    manifest = {
        "name": "GAIA-Text-103-Ablation40",
        "role": (
            "40-question stratified subset of GAIA-Text-103, used ONLY for the "
            "adaptive-coordination ablations (Static ANTMAN, Graph-free Adaptive, "
            "w/o Need Revision, w/o Adaptive Rerouting, w/o Recovery -- paper Table "
            "5/10). Never used for the 7-method main bake-off, which runs the full "
            "103-task gaia_text_103_manifest.json."
        ),
        "benchmark": "gaia",
        "status": "frozen",
        "derived_from": {
            "manifest": "third_party/manifests/gaia/gaia_text_103_manifest.json",
            "manifest_count": 103,
            "level_distribution": full_counts,
        },
        "sampling_method": {
            "allocation": (
                "Largest-remainder proportional allocation of 40 slots across "
                "Levels 1/2/3 by the 103-set's own level distribution (not equal "
                "thirds, not floor-rounded)."
            ),
            "selection": (
                "random.Random(seed).sample(sorted(task_ids_at_level), k) per level, "
                "on a sorted pool for reproducibility independent of hash/iteration order."
            ),
            "seed": SEED,
            "target_count": TARGET,
        },
        "level_distribution": alloc,
        "level_distribution_pct": {
            level: round(alloc[level] / TARGET * 100, 1) for level in alloc
        },
        "source_level_distribution_pct": {
            level: round(full_counts[level] / 103 * 100, 1) for level in full_counts
        },
        "count": TARGET,
        "task_ids": chosen,
        "created_at": datetime.now(UTC).isoformat(),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MANIFEST} ({TARGET} task_ids, alloc={alloc})")


if __name__ == "__main__":
    main()
