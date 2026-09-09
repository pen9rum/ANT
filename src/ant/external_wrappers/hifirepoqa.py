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
    Analysis/Synthesis pipeline, RepoQAMultiAgent class). NOT vendored into
    ANT's own source tree per Phase J -- this wrapper only shells out to a
    pinned external checkout's own isolated venv.

    Verified live in this session (real repo clone to the exact pinned
    commit, real GPT-4.1 chat-completion calls succeeding, real function-
    summarization progress observed) -- NOT just designed. Getting there
    required THREE small, disclosed local modifications to the pinned
    checkout itself (never to ANT's own source; see the manifest's
    `local_modifications` for the exact diffs and why each was necessary
    just to make the released code importable/functional at all, not a
    method-logic change):
    1. config.py's DEFAULT_CONFIG['api_provider'] was 'xxx', not a key in
       API_CONFIGS (only 'gpt-4o' exists) -- a module-level `Config.
       get_config()` call at import time meant `import config` (and thus
       `python main.py`) crashed unconditionally, before any environment
       variable override could run. Changed the default to 'gpt-4o'.
    2. repo_qa_agent.py's get_embedding() posted to a literal, never-filled
       placeholder ("https://xxx.com/v1/embeddings", model "qwen-embedding",
       no auth header) -- as released this call can never succeed against
       any real service. Added REPO_QA_EMBEDDING_URL/_MODEL/_API_KEY env
       vars, mirroring exactly how the chat-completion trio already works,
       falling back to the original placeholder unchanged when unset.
    3. clone_repository() always shallow-clones the CURRENT default-branch
       HEAD with no way to pin a commit -- a benchmark question written
       against a fixed commit_id could silently be answered against a
       different repo state than every other system in this suite. Added
       an optional `commit` parameter (full, non-shallow clone + checkout
       when given) threaded from `issue_data['commit_id']`. Verified live:
       the resulting checkout's `git rev-parse HEAD` matched the pinned
       commit_id byte-for-byte.

    A driver script (`_ant_driver.py`, written into the checkout root,
    NOT part of the official release) calls `RepoQAMultiAgent.
    process_issue()` directly instead of going through main.py's own
    hardcoded input_path/output_dir string literals -- reads one JSON
    record from stdin, writes the result dict to stdout as JSON. Config
    resolution itself (Config.get_config() + load_env_config()) is used
    completely unmodified.

    Known, disclosed, NOT-fixed-in-this-pass limitation: HiFiRepoQA's own
    knowledge-base-building step summarizes EVERY function and class in
    the ENTIRE target repository via a separate LLM call each, with no
    rate-limit backoff in the released code -- measured directly against
    qibo (a mid-sized ~400-file-eligible Python repo): 1483 functions
    found, sustained 429 "Too Many Requests" from OpenAI's standard rate
    limits, on pace for multiple hours for a first-time build of even one
    repo. This is a genuine, disclosed, METHOD-INHERENT cost (the DAG/
    Planning/Retrieval/Analysis/Synthesis architecture this class exists
    to preserve faithfully includes this whole-repo pre-indexing step) --
    not something this wrapper works around by limiting scope, which would
    be exactly the "erase method-specific components" this suite's own
    fairness rules forbid. The function/class-summary cache IS keyed by
    repo_name and persists incrementally (a second question against an
    ALREADY-indexed repo reuses it and is far cheaper) -- but `run()`'s
    default `timeout` here is unlikely to be sufficient for a repo's own
    FIRST question. Budgeting a realistic timeout (and possibly
    pre-warming each target repo's knowledge base in advance, outside any
    per-question timing) is a decision for whoever launches a real batch,
    not something silently assumed here.
    """

    name = "hifirepoqa"

    def __init__(
        self,
        checkout_root: Path | None = None,
        model: str = "gpt-4.1",
        api_base_url: str = "https://api.openai.com/v1/chat/completions",
        embedding_model: str = "text-embedding-3-small",
        embedding_url: str = "https://api.openai.com/v1/embeddings",
        timeout_seconds: int = 1800,
    ) -> None:
        self.checkout_root = checkout_root or Path("third_party/checkouts/hifirepoqa")
        self.venv_python = self.checkout_root / ".venv" / "Scripts" / "python.exe"
        self.model = model
        self.api_base_url = api_base_url
        self.embedding_model = embedding_model
        self.embedding_url = embedding_url
        self.timeout_seconds = timeout_seconds

    def _require_checkout(self) -> Path:
        if not (self.checkout_root / "src" / "repo_qa_agent.py").exists():
            raise HiFiRepoQANotCheckedOut(
                f"HiFiRepoQA is not checked out at {self.checkout_root}. To run this "
                f"baseline: git clone {HIFIREPOQA_REPO_URL} {self.checkout_root} && "
                f"cd {self.checkout_root} && git checkout {HIFIREPOQA_COMMIT}, then apply the "
                "three local modifications documented in this class's own docstring and in "
                "third_party/manifests/hifirepoqa/manifest.json, then create an ISOLATED venv "
                "at .venv (python -m venv .venv) and `pip install requests tqdm` (tree_sitter_"
                "languages is optional -- the code falls back to Python's own ast module when "
                "it's unavailable, which was the case in this session's own environment: no "
                "matching wheel was found)."
            )
        if not self.venv_python.exists():
            raise HiFiRepoQANotCheckedOut(
                f"Checkout found at {self.checkout_root} but its own isolated venv does not "
                f"exist at {self.venv_python}. Create it and `pip install requests tqdm` before "
                "running this baseline."
            )
        if not (self.checkout_root / "_ant_driver.py").exists():
            raise HiFiRepoQANotCheckedOut(
                f"{self.checkout_root}/_ant_driver.py is missing -- this is the small, "
                "not-part-of-the-official-release driver script this wrapper depends on (calls "
                "RepoQAMultiAgent.process_issue() directly; see this class's own docstring)."
            )
        return self.checkout_root

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        checkout_root = self._require_checkout()
        # HiFiRepoQA's own record schema, read directly from a sample
        # dataset entry at the pinned commit (question_id/repo/title/body/
        # comments/url/language) plus commit_id (this wrapper's own
        # addition, consumed by the commit-pinning local modification
        # above) -- `title`/`body` map from our TaskExample.question since
        # HiFiRepoQA's own dataset is GitHub-issue-derived (title + body),
        # not a single flat question string.
        input_record = {
            "question_id": example.task_id,
            "repo": example.metadata.get("repo", ""),
            "title": example.question[:80],
            "body": example.question,
            "comments": [],
            "url": f"https://github.com/{example.metadata.get('repo', '')}",
            "commit_id": example.metadata.get("commit_id", ""),
            "language": example.metadata.get("primary_language", "python"),
        }

        env = dict(os.environ)
        env.update(
            {
                "REPO_QA_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
                "REPO_QA_API_URL": self.api_base_url,
                "REPO_QA_MODEL_NAME": self.model,
                "REPO_QA_EMBEDDING_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
                "REPO_QA_EMBEDDING_URL": self.embedding_url,
                "REPO_QA_EMBEDDING_MODEL": self.embedding_model,
            }
        )
        result = subprocess.run(
            [str(self.venv_python), "_ant_driver.py"],
            cwd=checkout_root,
            input=json.dumps(input_record),
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return AgentResult(
                benchmark=example.benchmark,
                task_id=example.task_id,
                method=self.name,
                final_answer="",
                termination_reason=f"subprocess_failed: {result.stderr[-500:]}",
            )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        if payload.get("error"):
            return AgentResult(
                benchmark=example.benchmark,
                task_id=example.task_id,
                method=self.name,
                final_answer=payload.get("repo_qa_answer", {}).get("answer", ""),
                termination_reason=f"agent_error: {payload['error'][:300]}",
                metadata={"raw_output": payload, "generation_model": self.model},
            )
        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=payload.get("repo_qa_answer", {}).get("answer", ""),
            trajectory=payload.get("chain_results", []),
            # HiFiRepoQA's own code makes raw requests.post calls with no
            # usage/cost tracking of its own (unlike ANT's own
            # CountingOpenAIProvider) -- TBD once a full run's own
            # call/token accounting is built; llm_calls must count
            # PHYSICAL model/API invocations (this suite's own semantic,
            # see ant.evaluation_suite.counting_provider's docstring), not
            # a naive per-DAG-node proxy that could undercount whatever
            # retry/repair behavior this pipeline has internally.
            usage=UsageStats(),
            termination_reason="completed",
            metadata={"raw_output": payload, "generation_model": self.model},
        )
