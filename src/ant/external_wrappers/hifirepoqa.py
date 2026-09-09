from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.usage import UsageStats

# Pinned per third_party/manifests/hifirepoqa/manifest.json -- keep these
# two values in sync with that manifest, never edit one without the other.
HIFIREPOQA_REPO_URL = "https://github.com/HiFiRepoQA/HiFiRepoQA"
HIFIREPOQA_COMMIT = "a2d833d9c292f2a4aaae349d8d8e9bd9bed8339c"


class HiFiRepoQANotCheckedOut(RuntimeError):
    pass


class HiFiRepoQAAgent:
    """External-baseline wrapper around the anonymized HiFiRepoQA release
    (see third_party/manifests/hifirepoqa/manifest.json for the full
    provenance/architecture writeup -- DAG-structured Planning/Retrieval/
    Analysis/Synthesis pipeline). NOT vendored into ANT's own source tree
    per Phase J -- this wrapper only shells out to a pinned external
    checkout.

    Design (matches HiFiRepoQA's own actual interface, read directly from
    its source at the pinned commit): `main.py`/`main_multi_agent.py` take
    no CLI arguments -- they read a hardcoded `input_path` JSON/JSONL file
    of question records shaped {repo, title, body, url, language, ...} and
    write to a hardcoded `output_dir`. This wrapper writes ONE example as
    a single-row input file in that exact schema, invokes the pinned
    checkout's entry point as a subprocess with `REPO_QA_API_KEY`/
    `REPO_QA_API_URL`/`REPO_QA_MODEL_NAME` passed through the environment
    (their own documented model-swap mechanism -- no source edit needed
    for a model change), and reads back the produced output file.

    NOT yet exercised against a real checkout in this session -- `run()`
    raises `HiFiRepoQANotCheckedOut` with the exact clone/pin instructions
    if `checkout_root` doesn't exist, rather than silently no-op'ing or
    fabricating a result.
    """

    name = "hifirepoqa"

    def __init__(
        self,
        checkout_root: Path | None = None,
        model: str = "gpt-4.1",
        api_base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.checkout_root = checkout_root or Path("third_party/checkouts/hifirepoqa")
        self.model = model
        self.api_base_url = api_base_url

    def _require_checkout(self) -> Path:
        if not (self.checkout_root / "src" / "repo_qa_agent.py").exists():
            raise HiFiRepoQANotCheckedOut(
                f"HiFiRepoQA is not checked out at {self.checkout_root}. To run this "
                f"baseline: git clone {HIFIREPOQA_REPO_URL} {self.checkout_root} && "
                f"cd {self.checkout_root} && git checkout {HIFIREPOQA_COMMIT}, then install "
                "its dependencies (tree-sitter bindings, an embeddings client, an "
                "OpenAI-compatible client -- no requirements.txt is shipped; infer from "
                "imports in src/repo_qa_agent.py). This wrapper deliberately does not "
                "auto-clone -- see Phase J's 'do not install/clone without being asked' "
                "posture."
            )
        return self.checkout_root

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        checkout_root = self._require_checkout()
        # HiFiRepoQA's own record schema, read directly from a sample
        # dataset entry at the pinned commit (question_id/repo/title/body/
        # url/language) -- `title`/`body` map from our TaskExample.question
        # since HiFiRepoQA's own dataset is GitHub-issue-derived (title +
        # body), not a single flat question string; we put the whole
        # question into `body` and leave `title` as a short prefix so the
        # method's own DAG-planning prompt still has something sensible in
        # both fields.
        input_record = {
            "question_id": example.task_id,
            "repo": str(environment_root),
            "title": example.question[:80],
            "body": example.question,
            "comments": [],
            "url": example.metadata.get("repo", ""),
            "language": example.metadata.get("primary_language", "python"),
        }
        input_path = checkout_root / "_ant_smoke_input.json"
        output_dir = checkout_root / "_ant_smoke_output"
        input_path.write_text(json.dumps([input_record]), encoding="utf-8")
        output_dir.mkdir(exist_ok=True)

        env = dict(os.environ)
        env.update(
            {
                "REPO_QA_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
                "REPO_QA_API_URL": self.api_base_url,
                "REPO_QA_MODEL_NAME": self.model,
            }
        )
        # main.py itself has hardcoded input_path/output_dir string
        # literals (per the manifest's own porting-effort note) -- running
        # it as-is requires either patching those two literals in the
        # checkout or wrapping main() with the paths monkeypatched in a
        # small driver script placed in the checkout at run time. Neither
        # has been done in this session; this subprocess.run call is the
        # intended invocation shape once that one-time adapter edit exists.
        result = subprocess.run(
            ["python", "main.py"],
            cwd=checkout_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        output_file = output_dir / f"{example.task_id}_result.json"
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
            trajectory=payload.get("dag_execution_log", []),
            # HiFiRepoQA's own per-call usage schema: TBD once run live --
            # when it is, llm_calls must count PHYSICAL model/API
            # invocations (this suite's own semantic, see
            # ant.evaluation_suite.counting_provider's docstring), not a
            # naive per-DAG-node proxy that could undercount whatever
            # retry/repair behavior HiFiRepoQA's own Planning/Retrieval/
            # Analysis/Synthesis pipeline has internally.
            usage=UsageStats(),
            termination_reason="completed",
            metadata={"raw_output": payload, "generation_model": self.model},
        )
