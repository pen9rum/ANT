"""DeltaAI (GH200) environment smoke test for ANT.

Checks, in order, and prints a PASS/FAIL/SKIP line for each:
  1. host/arch/python sanity
  2. GPU visible via nvidia-smi
  3. torch sees CUDA (ad hoc check -- torch is NOT a declared ANT dependency;
     nothing in src/ant imports it today)
  4. `ant` core package imports
  5. dense retrieval (fastembed/ONNX, CPU-only by design -- see
     ant.retrieval.dense.DenseEmbedder) embeds real text using the cached
     bge-small model
  6. a full LocalCoordinator.ask() pass over a tiny slice of ANT's own repo,
     with --synthesize none, so it needs no OPENAI_API_KEY
  7. OpenAI connectivity, only if OPENAI_API_KEY is actually set -- skipped
     (not failed) otherwise, since no key has been provisioned on DeltaAI yet

Exits non-zero if any non-skipped check fails.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
results: list[dict[str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append({"check": name, "status": status, "detail": detail})
    print(f"[{status}] {name}: {detail}")


def check_host() -> None:
    record(
        "host",
        "PASS",
        f"{platform.node()} {platform.machine()} python={platform.python_version()}",
    )


def check_gpu_visible() -> None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        record("nvidia-smi", "FAIL", "nvidia-smi not on PATH")
        return
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        record("nvidia-smi", "PASS", out.stdout.strip().replace("\n", " | "))
    except Exception as exc:  # noqa: BLE001
        record("nvidia-smi", "FAIL", repr(exc))


def check_torch_cuda() -> None:
    try:
        import torch
    except ImportError:
        record("torch-cuda", "SKIP", "torch not installed (not an ANT dependency)")
        return
    available = torch.cuda.is_available()
    if not available:
        record("torch-cuda", "FAIL", "torch installed but torch.cuda.is_available() is False")
        return
    name = torch.cuda.get_device_name(0)
    record("torch-cuda", "PASS", f"torch {torch.__version__} sees: {name}")


def check_ant_imports() -> None:
    try:
        from ant.coordinator import LocalCoordinator  # noqa: F401
        from ant.domain import EvidenceState  # noqa: F401
        from ant.indexing import discover_territories  # noqa: F401
        from ant.memory import IndexStore  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        record("ant-imports", "FAIL", repr(exc))
        raise
    record("ant-imports", "PASS", "core package imports resolved")


def check_dense_retrieval() -> None:
    try:
        from ant.retrieval.dense import DenseEmbedder
    except Exception as exc:  # noqa: BLE001
        record("dense-retrieval", "FAIL", f"import error: {exc!r}")
        return
    try:
        embedder = DenseEmbedder()
        vectors = embedder.embed(["def add(a, b): return a + b", "class Worker: pass"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 384
        record("dense-retrieval", "PASS", f"embedded 2 chunks, dim={len(vectors[0])} (CPU/ONNX)")
    except Exception as exc:  # noqa: BLE001
        record("dense-retrieval", "FAIL", repr(exc))


def check_ant_end_to_end() -> None:
    from ant.environment import RepoEnvironment
    from ant.generation import generate_worker_cards
    from ant.indexing import discover_territories
    from ant.memory import IndexStore
    from ant.coordinator import LocalCoordinator

    target_repo = REPO_ROOT / "src" / "ant" / "domain"
    try:
        with tempfile.TemporaryDirectory(prefix="ant-smoke-") as tmp:
            index_path = Path(tmp) / ".ant"
            environment = RepoEnvironment(target_repo)
            territories = discover_territories(environment)
            workers = generate_worker_cards(environment.root, territories, generator=None)
            store = IndexStore(index_path)
            store.save(territories, workers)

            coordinator = LocalCoordinator(
                target_repo.resolve(),
                store.load_workers(),
                synthesizer=None,
                index_path=index_path,
            )
            state = coordinator.ask(
                "What fields does the WorkerCard model have?",
                max_rounds=2,
            )
            n_evidence = len(state.evidence)
            record(
                "ant-end-to-end",
                "PASS",
                f"{len(territories)} territories, {len(workers)} workers, "
                f"{n_evidence} evidence items gathered (synthesize=none)",
            )
    except Exception as exc:  # noqa: BLE001
        record("ant-end-to-end", "FAIL", repr(exc))


def check_openai() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        record("openai-api", "SKIP", "OPENAI_API_KEY not set in this environment")
        return
    try:
        from ant.providers import OpenAIProvider

        provider = OpenAIProvider()
        reply = provider.smoke_test()
        record("openai-api", "PASS", str(reply)[:200])
    except Exception as exc:  # noqa: BLE001
        record("openai-api", "FAIL", repr(exc))


def main() -> int:
    check_host()
    check_gpu_visible()
    check_torch_cuda()
    check_ant_imports()
    check_dense_retrieval()
    check_ant_end_to_end()
    check_openai()

    print("\n--- summary ---")
    print(json.dumps(results, indent=2))

    failed = [r for r in results if r["status"] == "FAIL"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
