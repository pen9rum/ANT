from __future__ import annotations

import subprocess
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.usage import UsageStats

REPOPROBE_REPO_URL = "https://github.com/Tencent-Hunyuan/RepoProbe"
REPOPROBE_COMMIT = "5ac4212"  # kept in sync with benchmarks/repoprobe.py's own pin


class RepoProbeScaffoldNotCheckedOut(RuntimeError):
    pass


class RepoProbeReferenceScaffoldAgent:
    """Wraps RepoProbe's own released, SWAPPABLE example agent
    (`agent_configs/example_agent.py`), run through the benchmark's own
    Docker-isolated harness -- NOT the paper's Claude Code headline result.

    IMPORTANT distinction, per Phase G's own explicit instruction: this
    class produces a NEW, our-own GPT-4.1 reproduction using RepoProbe's
    swappable reference scaffold and Docker environment -- it is NOT the
    paper's own published Claude-Code-based number. Any report using this
    class's output must label it 'RepoProbe reference scaffold (GPT-4.1,
    our reproduction)', kept in a visibly separate row from 'RepoProbe
    official (Claude Code, paper-reported)'.

    Per `agent_configs/agent.json` (read directly from the pinned commit):
    the container's own command is
    `python3 /app/example_agent.py --prompt-file /tmp/prompt/prompt.txt`,
    with `OPENAI_API_KEY`/`OPENAI_BASE_URL` passed through
    (`passthrough_env`) and `REPOPROBE_MAX_TURNS=30` set in the container
    env -- this wrapper drives that same container directly via `docker
    run`, not a reimplementation of example_agent.py's own logic.

    NOT yet exercised against a real checkout/built image in this session.
    """

    name = "repoprobe_reference_scaffold"

    def __init__(
        self,
        checkout_root: Path | None = None,
        image_tag: str = "repoprobe-agent:pinned",
        max_turns: int = 30,
        model: str = "gpt-4.1",
    ) -> None:
        self.checkout_root = checkout_root or Path("third_party/checkouts/repoprobe")
        self.image_tag = image_tag
        self.max_turns = max_turns
        # The container's own example_agent.py reads its model choice from
        # whatever OPENAI_API_KEY/OPENAI_BASE_URL's account default is, not
        # a param this wrapper controls directly -- `model` here records
        # the model this run is INTENDED to use (per this class's own
        # GPT-4.1-reproduction docstring above) for run-metadata purposes,
        # not something the docker invocation currently threads through.
        self.model = model

    def _require_checkout_and_image(self) -> None:
        if not (self.checkout_root / "agent_configs" / "example_agent.py").exists():
            raise RepoProbeScaffoldNotCheckedOut(
                f"RepoProbe is not checked out at {self.checkout_root}. To run this "
                f"baseline: git clone {REPOPROBE_REPO_URL} {self.checkout_root} && "
                f"cd {self.checkout_root} && git checkout {REPOPROBE_COMMIT}, then "
                f"`cd agent_configs && bash build_base_image.sh` to produce {self.image_tag} "
                "(or whatever tag that script actually assigns -- verify and update this "
                "wrapper's own image_tag default to match)."
            )
        image_check = subprocess.run(
            ["docker", "image", "inspect", self.image_tag],
            capture_output=True,
            text=True,
        )
        if image_check.returncode != 0:
            raise RepoProbeScaffoldNotCheckedOut(
                f"Docker image {self.image_tag!r} not found -- build it first (see "
                f"{self.checkout_root}/agent_configs/build_base_image.sh)."
            )

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        self._require_checkout_and_image()
        prompt_dir = self.checkout_root / "_ant_smoke_prompt"
        prompt_dir.mkdir(exist_ok=True)
        (prompt_dir / "prompt.txt").write_text(example.question, encoding="utf-8")

        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-e",
                "OPENAI_API_KEY",
                "-e",
                f"REPOPROBE_MAX_TURNS={self.max_turns}",
                "-v",
                f"{environment_root}:/app/repo:ro",
                "-v",
                f"{prompt_dir}:/tmp/prompt:ro",
                self.image_tag,
            ],
            capture_output=True,
            text=True,
            timeout=self.max_turns * 60 + 300,
        )
        if result.returncode != 0:
            return AgentResult(
                benchmark=example.benchmark,
                task_id=example.task_id,
                method=self.name,
                final_answer="",
                termination_reason=f"docker_run_failed: {result.stderr[-500:]}",
            )
        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=result.stdout.strip(),
            usage=UsageStats(),  # populated from container logs once run live
            termination_reason="completed",
            metadata={
                "note": (
                    "GPT-4.1 reproduction via RepoProbe's own swappable scaffold, "
                    "NOT the paper's Claude Code number"
                ),
                "generation_model": self.model,
            },
        )
