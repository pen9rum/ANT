"""Fixed-concurrency, per-repo parallel fast-gen1 rerun.

`ant gen-compare --no-gen0 --no-slow-gen1 --fast-gen1` processes one
repo's whole dataset sequentially in a single process, and its own
`fast-gen1-results.jsonl` write is `open(..., "w")` for the *entire*
examples loop -- one call, one full truncate-and-rewrite. Two such calls
against the same `--run-dir` at once silently clobber each other (last
process to finish wins; the other's rows vanish with no error), so
naive parallelism -- just launching several `gen-compare` calls against
one run-dir -- is not safe. This script shards one repo's dataset into
`--shards` (fixed at 3 by default) contiguous slices, runs each slice
as its own `gen-compare` subprocess against its own throwaway run-dir,
then merges the per-question trace files and results rows back into the
canonical run-dir afterward -- so results.jsonl is only ever written by
one process at a time, from data that already finished.

Each shard needs two things copied out of the canonical run-dir before
it can run `--no-gen0`: this repo's `_gen0_index_snapshot` (fast-gen1
reads this frozen index, not the live one, whenever it exists -- see
gen_compare.run_gen_compare's own docstring) and the `gen0-<id>.json`
trace for every question in that shard (`_run_fast_gen1` reads it
directly from `run_dir`). Both are copied into each shard's own
throwaway dir so `--no-gen0 --no-slow-gen1 --fast-gen1` there behaves
identically to a plain (slower, sequential) rerun against the real
run-dir would have.

Usage:
    .venv/Scripts/python scripts/parallel_fast_gen1_rerun.py \\
        --repo qibo \\
        --dataset output/qibo_all_10.jsonl \\
        --run-dir output/runs/qibo-consolidation-gen0-gen1 \\
        --index .ant/sweqa/qibo \\
        [--shards 3] [--judge openai] [--max-rounds 6] [--keep-shard-dirs]

Concurrency is intentionally fixed at 3 by default (not scaled to
dataset size): this repo's own OpenAI usage tier is unknown from here,
and each shard's own fast-gen1 rounds already fire several LLM calls
concurrently-in-spirit (planning, worker actions, resolution/upgrade
checks, the judge) -- 3 parallel `gen-compare` processes was the
conservative starting point agreed on, to watch for 429s before trying
to push higher. Works unchanged for a 10-, 20-, or 30-question dataset:
sharding is by count, not by any fixed schedule.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_examples(dataset_path: Path) -> list[dict]:
    examples = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _shard(examples: list[dict], shards: int) -> list[list[dict]]:
    # Contiguous slices, sized as evenly as possible (e.g. 10 into 3 ->
    # 4/3/3) -- which question lands in which shard doesn't matter, only
    # that every id ends up in exactly one shard and the merge step can
    # restore the original dataset order afterward.
    buckets: list[list[dict]] = [[] for _ in range(shards)]
    for index, example in enumerate(examples):
        buckets[index % shards].append(example)
    return [bucket for bucket in buckets if bucket]


def _prepare_shard_dir(
    *, canonical_run_dir: Path, shard_dir: Path, shard_examples: list[dict]
) -> Path:
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)

    snapshot_src = canonical_run_dir / "_gen0_index_snapshot"
    if snapshot_src.exists():
        shutil.copytree(snapshot_src, shard_dir / "_gen0_index_snapshot")

    missing_gen0 = []
    for example in shard_examples:
        example_id = example["id"]
        src = canonical_run_dir / f"gen0-{example_id}.json"
        if not src.exists():
            missing_gen0.append(example_id)
            continue
        shutil.copy2(src, shard_dir / f"gen0-{example_id}.json")
    if missing_gen0:
        raise SystemExit(
            f"canonical run-dir {canonical_run_dir} is missing gen0 traces for: "
            f"{missing_gen0} -- fast-gen1 needs an existing gen0 run to retry from"
        )

    dataset_path = shard_dir / "dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for example in shard_examples:
            handle.write(json.dumps(example) + "\n")
    return dataset_path


def _launch_shard(
    *,
    dataset_path: Path,
    repo_name: str,
    index_path: Path,
    shard_dir: Path,
    judge: str,
    max_rounds: int | None,
    log_path: Path,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "ant.cli",
        "gen-compare",
        str(dataset_path),
        "--repo",
        f"repos/{repo_name}",
        "--index",
        str(index_path),
        "--run-dir",
        str(shard_dir),
        "--no-gen0",
        "--no-slow-gen1",
        "--fast-gen1",
        "--judge",
        judge,
    ]
    if max_rounds is not None:
        cmd.extend(["--max-rounds", str(max_rounds)])
    log_handle = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        cmd, cwd=REPO_ROOT, stdout=log_handle, stderr=subprocess.STDOUT
    )


def _backup(path: Path, tag: str) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = path.with_name(f"{path.name}.{tag}-backup-{stamp}")
    shutil.copy2(path, backup_path)


def _merge(
    *, canonical_run_dir: Path, shard_dirs: list[Path], original_order: list[str]
) -> None:
    """Merges this run's shard results into the canonical
    fast-gen1-results.jsonl -- non-destructively. `original_order` is
    only this run's OWN dataset (e.g. "the remaining 9" of a repo's 10),
    so a naive overwrite-with-just-this-run's-rows would silently drop
    any question the canonical file already had that wasn't part of
    THIS run (confirmed live: a prior single-question deep-dive rerun's
    own row vanished from a 9-question "rest of the repo" rerun the
    first time this ran, recoverable only because of the backup below).
    Existing rows for ids NOT in this run are preserved as-is; rows for
    ids that WERE in this run are overwritten with the fresh result;
    brand-new ids (a dataset that grew) are appended after the existing
    order.
    """
    out_path = canonical_run_dir / "fast-gen1-results.jsonl"
    _backup(out_path, "pre-parallel-merge")

    existing_order: list[str] = []
    rows_by_id: dict[str, dict] = {}
    if out_path.exists():
        with out_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows_by_id[row["example_id"]] = row
                existing_order.append(row["example_id"])

    for shard_dir in shard_dirs:
        results_path = shard_dir / "fast-gen1-results.jsonl"
        if not results_path.exists():
            continue
        with results_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows_by_id[row["example_id"]] = row

        for trace_path in shard_dir.glob("fast-gen1-*.json"):
            dest = canonical_run_dir / trace_path.name
            _backup(dest, "pre-parallel-merge")
            shutil.copy2(trace_path, dest)

    final_order = list(existing_order)
    for example_id in original_order:
        if example_id not in final_order:
            final_order.append(example_id)

    missing = [example_id for example_id in original_order if example_id not in rows_by_id]
    if missing:
        print(
            f"WARNING: {len(missing)} question(s) produced no result row and are "
            f"NOT in the merged file: {missing}",
            file=sys.stderr,
        )

    with out_path.open("w", encoding="utf-8") as handle:
        for example_id in final_order:
            if example_id in rows_by_id:
                handle.write(json.dumps(rows_by_id[example_id], ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="short repo name, e.g. qibo")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path, dest="run_dir")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--judge", default="openai")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument(
        "--keep-shard-dirs",
        action="store_true",
        help="don't delete the per-shard throwaway run-dirs after a successful merge",
    )
    args = parser.parse_args()

    canonical_run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir
    dataset_path = args.dataset if args.dataset.is_absolute() else REPO_ROOT / args.dataset
    index_path = args.index if args.index.is_absolute() else REPO_ROOT / args.index

    examples = _load_examples(dataset_path)
    original_order = [example["id"] for example in examples]
    shards = _shard(examples, args.shards)
    print(f"{len(examples)} question(s) split into {len(shards)} shard(s): "
          f"{[len(s) for s in shards]}")

    shard_dirs: list[Path] = []
    procs: list[tuple[subprocess.Popen, Path, Path]] = []
    for index, shard_examples in enumerate(shards):
        shard_dir = canonical_run_dir.parent / f"{canonical_run_dir.name}-parallel-shard{index}"
        shard_dataset = _prepare_shard_dir(
            canonical_run_dir=canonical_run_dir,
            shard_dir=shard_dir,
            shard_examples=shard_examples,
        )
        shard_dirs.append(shard_dir)
        log_path = shard_dir / "gen-compare.log"
        proc = _launch_shard(
            dataset_path=shard_dataset,
            repo_name=args.repo,
            index_path=index_path,
            shard_dir=shard_dir,
            judge=args.judge,
            max_rounds=args.max_rounds,
            log_path=log_path,
        )
        procs.append((proc, shard_dir, log_path))
        print(f"shard {index}: launched pid={proc.pid}, "
              f"{len(shard_examples)} question(s), log={log_path}")

    start = time.monotonic()
    failed = []
    for index, (proc, _shard_dir, log_path) in enumerate(procs):
        returncode = proc.wait()
        elapsed = time.monotonic() - start
        print(f"shard {index}: exit code {returncode} (at {elapsed:.0f}s)")
        if returncode != 0:
            failed.append((index, log_path))

    if failed:
        print(
            "One or more shards failed -- NOT merging, so a good shard's results "
            "aren't paired with a failed shard's incomplete/missing ones.",
            file=sys.stderr,
        )
        for index, log_path in failed:
            print(f"  shard {index} failed, see {log_path}", file=sys.stderr)
        raise SystemExit(1)

    _merge(
        canonical_run_dir=canonical_run_dir,
        shard_dirs=shard_dirs,
        original_order=original_order,
    )
    print(f"merged {len(original_order)} question(s) into {canonical_run_dir}")

    if not args.keep_shard_dirs:
        for shard_dir in shard_dirs:
            shutil.rmtree(shard_dir, ignore_errors=True)
        print("cleaned up shard dirs")


if __name__ == "__main__":
    main()
