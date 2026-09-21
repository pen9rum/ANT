"""Formal runner for the adaptive-coordination ablation table.

Each invocation runs one or all five paper ablations against one frozen
benchmark set.  Results are isolated by benchmark/profile and resumable via
the evaluation-suite JSONL convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ant.agents.ablation_agents import (  # noqa: E402
    AblationAntAgent,
    AblationAntDocumentAgent,
)
from ant.benchmarks.niah_plus_adapter import NiahPlusAdapter, _to_task_example  # noqa: E402
from ant.benchmarks.repoprobe import RepoProbeAdapter  # noqa: E402
from ant.benchmarks.sweqa_pro import SweQaProAdapter  # noqa: E402
from ant.coordinator.ablations import PROFILES  # noqa: E402
from ant.evaluation_suite.niah_plus import build_multi_needle_instance  # noqa: E402
from ant.evaluation_suite.runner import run_suite  # noqa: E402

_ABLATION_KEYS = (
    "static",
    "graph_free_adaptive",
    "no_need_revision",
    "no_adaptive_rerouting",
    "no_recovery",
)
_MANIFESTS = {
    "repoprobe": REPO_ROOT
    / "third_party"
    / "manifests"
    / "repoprobe"
    / "sample_manifest_repoprobe_python_full.json",
    "sweqa_pro": REPO_ROOT
    / "third_party"
    / "manifests"
    / "sweqa_pro"
    / "sample_manifest_sweqa_pro_80.json",
    "niah_512k": REPO_ROOT
    / "third_party"
    / "manifests"
    / "long_context"
    / "multineedle_scaling_manifest_512k.json",
}


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("frozen_final") and payload.get("status") not in {
        "frozen_final",
        "frozen_full",
    }:
        raise ValueError(f"Refusing non-frozen ablation manifest: {path}")
    return payload


def _task_ids(payload: dict) -> list[str]:
    task_ids = payload.get("task_ids")
    if isinstance(task_ids, list):
        return [str(task_id) for task_id in task_ids]
    by_repo = payload.get("task_ids_by_repo")
    if isinstance(by_repo, dict):
        return [str(task_id) for repo_ids in by_repo.values() for task_id in repo_ids]
    raise ValueError("Frozen repository manifest does not contain task_ids")


def _select_by_manifest(examples: list, payload: dict) -> list:
    by_id = {example.task_id: example for example in examples}
    selected_ids = _task_ids(payload)
    missing = [task_id for task_id in selected_ids if task_id not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} frozen task_id(s) unavailable: {missing[:10]}")
    return [by_id[task_id] for task_id in selected_ids]


def _load_niah_512k(payload: dict) -> list:
    conditions = payload.get("conditions")
    if not isinstance(conditions, list):
        raise ValueError("512K manifest does not contain conditions")
    examples = []
    for condition in conditions:
        instance = build_multi_needle_instance(
            context_length_tokens=int(condition["context_length_tokens"]),
            position=condition["position"],
            question_index=int(condition["question_index"]),
        )
        example = _to_task_example(instance)
        if example.task_id != condition["task_id"] or example.question != condition["question"]:
            raise ValueError(f"Frozen 512K condition drifted: {condition['task_id']}")
        if json.loads(example.reference) != condition["gold_answers"]:
            raise ValueError(f"Frozen 512K reference drifted: {condition['task_id']}")
        examples.append(example)
    return examples


def _load_track(benchmark: str):
    manifest_path = _MANIFESTS[benchmark]
    payload = _load_manifest(manifest_path)
    if benchmark == "repoprobe":
        adapter = RepoProbeAdapter()
        examples = _select_by_manifest(adapter.load_examples(), payload)
        agent_cls = AblationAntAgent
    elif benchmark == "sweqa_pro":
        adapter = SweQaProAdapter()
        examples = _select_by_manifest(adapter.load_examples(), payload)
        agent_cls = AblationAntAgent
    else:
        adapter = NiahPlusAdapter()
        examples = _load_niah_512k(payload)
        agent_cls = AblationAntDocumentAgent

    expected = payload.get(
        "total_questions", payload.get("total_task_ids", payload.get("total_conditions"))
    )
    if expected is None or len(examples) != int(expected):
        raise ValueError(f"Manifest count mismatch: expected {expected}, loaded {len(examples)}")
    return adapter, agent_cls, examples, manifest_path


def _write_run_config(out_dir: Path, *, benchmark: str, profile: str, manifest_path: Path) -> None:
    manifest_bytes = manifest_path.read_bytes()
    config = {
        "benchmark": benchmark,
        "coordination_profile": profile,
        "coordination_profile_label": PROFILES[profile].display_name,
        "source_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    path = out_dir / "ablation_config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
        raise ValueError(
            f"Existing output directory has a different ablation configuration: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(_MANIFESTS), required=True)
    parser.add_argument("--profile", choices=[*_ABLATION_KEYS, "all"], default="all")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "output" / "runs" / "adaptive-ablations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the frozen manifest and print the planned calls without inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    adapter, agent_cls, examples, manifest_path = _load_track(args.benchmark)
    profiles = _ABLATION_KEYS if args.profile == "all" else (args.profile,)
    print(
        f"Validated {args.benchmark}: {len(examples)} frozen examples from "
        f"{manifest_path.relative_to(REPO_ROOT)}."
    )
    for profile in profiles:
        agent = agent_cls(
            coordination_profile=profile,
            model=args.model,
            max_rounds=args.max_rounds,
        )
        out_dir = args.output_root / args.benchmark / profile
        _write_run_config(
            out_dir,
            benchmark=args.benchmark,
            profile=profile,
            manifest_path=manifest_path,
        )
        print(f"{profile}: {len(examples)} examples -> {out_dir}")
        if args.dry_run:
            continue
        run_suite(
            benchmark=adapter,
            agent=agent,
            examples=examples,
            out_path=out_dir / "results.jsonl",
            trajectory_dump_dir=out_dir / "trajectories",
            resume=True,
        )


if __name__ == "__main__":
    main()
