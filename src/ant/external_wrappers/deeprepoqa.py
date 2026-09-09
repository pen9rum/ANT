from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.usage import UsageStats

# Pinned per third_party/manifests/deeprepoqa/manifest.json.
DEEPREPOQA_REPO_URL = "https://github.com/peng-weihan/DeepRepoQA"


class DeepRepoQANotCheckedOut(RuntimeError):
    pass


class DeepRepoQAAgent:
    """External-baseline wrapper around DeepRepoQA (arXiv:2608.24221) --
    MCTS/UCT search over a Perception/Planning/Execution/Evaluation agent
    loop (see third_party/manifests/deeprepoqa/manifest.json for the full
    architecture writeup). NOT vendored per Phase J -- shells out to a
    pinned external checkout.

    Fairness note baked into this wrapper's own design, per Phase I's
    explicit instruction: DeepRepoQA's own SemanticSearch tool uses the
    proprietary `voyage-code-3` embedding model. This wrapper never
    silently substitutes a different embedder to "even the playing field"
    -- `requires_voyage_api_key` is checked explicitly, and any run
    without a configured Voyage key fails loudly rather than degrading to
    a different (uncomparable) retrieval channel. If a future run
    deliberately substitutes ANT's own dense retriever instead, that must
    be labeled SEMI-CONTROLLED in the fairness report, never CONTROLLED.

    NOT yet exercised against a real checkout in this session.
    """

    name = "deeprepoqa"

    def __init__(
        self,
        checkout_root: Path | None = None,
        model: str = "gpt-4.1",
        max_iterations: int = 15,
        max_expand: int = 3,
    ) -> None:
        self.checkout_root = checkout_root or Path("third_party/checkouts/deeprepoqa")
        self.model = model
        self.max_iterations = max_iterations
        self.max_expand = max_expand

    def _require_checkout(self) -> Path:
        if not (self.checkout_root / "environment.yml").exists():
            raise DeepRepoQANotCheckedOut(
                f"DeepRepoQA is not checked out at {self.checkout_root}. To run this "
                f"baseline: git clone {DEEPREPOQA_REPO_URL} {self.checkout_root}, verify the "
                "latest commit SHA and record it in third_party/manifests/deeprepoqa/"
                "manifest.json (not pinned yet -- the manifest explicitly flags this), then "
                "`conda env create -f environment.yml` (or translate to your own env "
                "manager) in an ISOLATED environment, not ANT's own .venv."
            )
        return self.checkout_root

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        checkout_root = self._require_checkout()
        if not os.environ.get("VOYAGE_API_KEY"):
            raise RuntimeError(
                "DeepRepoQA's own SemanticSearch tool requires a Voyage API key "
                "(voyage-code-3 embeddings) -- VOYAGE_API_KEY is not set. Per this "
                "wrapper's own fairness design (see class docstring), this raises rather "
                "than silently falling back to a different embedder, which would make any "
                "resulting comparison SEMI-CONTROLLED at best without disclosure."
            )
        # example.py / example_batch.py in the official repo demonstrate the
        # invocation shape; a real run would build a per-question config
        # pointing repo_root=environment_root, model=self.model,
        # max_iterations=self.max_iterations, max_expand=self.max_expand and
        # invoke it via subprocess exactly as example.py itself does. Not
        # yet wired to a live checkout in this session -- this method is a
        # structurally complete but unexercised design, matching the other
        # external wrappers.
        result = subprocess.run(
            [
                "python",
                "example.py",
                "--repo-root",
                str(environment_root),
                "--question",
                example.question,
                "--model",
                self.model,
                "--max-iterations",
                str(self.max_iterations),
                "--max-expand",
                str(self.max_expand),
                "--output",
                "_ant_smoke_output.json",
            ],
            cwd=checkout_root,
            capture_output=True,
            text=True,
            timeout=3600,  # MCTS search is meaningfully slower than a single ReAct pass
        )
        output_file = checkout_root / "_ant_smoke_output.json"
        if result.returncode != 0 or not output_file.exists():
            return AgentResult(
                benchmark=example.benchmark,
                task_id=example.task_id,
                method=self.name,
                final_answer="",
                termination_reason=f"subprocess_failed: {result.stderr[-500:]}",
            )
        payload = json.loads(output_file.read_text(encoding="utf-8"))
        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=payload.get("answer", ""),
            trajectory=payload.get("mcts_tree_log", []),
            usage=UsageStats(),  # populated from real output once run live
            termination_reason="completed",
            metadata={
                "raw_output": payload,
                "uses_proprietary_embedding": "voyage-code-3",
                "fairness_classification": "SEMI-CONTROLLED",
                "generation_model": self.model,
            },
        )
