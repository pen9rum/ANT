"""OWL (camel-ai's Workforce) over GAIA -- `name = "owl_gaia"`. An
additional agentic baseline alongside Matched ReAct/S2G-RAG: a
hierarchical multi-agent framework (Planner + Coordinator + Worker nodes,
with AUTO_DECOMPOSE runtime worker creation) from Hendryx et al., 2025
(arXiv:2505.23885), compared here on the identical GAIA substrate every
other method in this suite uses.

WHY A SUBPROCESS: camel-ai (OWL's package) pins dependencies -- notably
`mcp` -- that conflict with this project's main `.venv` (confirmed live:
installing it there downgraded pydantic/pydantic-core/tiktoken and broke
325 unrelated tests). It lives in an isolated `.venv-owl` instead, and
this class drives it as a subprocess, communicating only through
`gaia_owl_bridge.GaiaOwlBridge` -- see that module's own docstring for the
full fairness argument. `owl_worker_subprocess.py` is the actual driver
that runs under `.venv-owl`'s interpreter; nothing in this file imports
`camel` directly.

OUTPUT-FORMAT HANDLING: same principle as `ant_gaia.reformat_to_gaia_template`
-- OWL's own question is never perturbed with GAIA's `FINAL ANSWER:`
template instructions (which could bias its own task decomposition);
instead its raw answer is restated, content unchanged, in one separate,
clearly-labeled call once OWL itself is done.

GOLD-LEAKAGE NOTE: this file and the bridge read only `example.question`
and task-observable metadata (`file_name`, `has_attachment`) -- never
`example.reference` or `example.metadata["_audit_only"]`.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from ant.agents.base import AgentResult
from ant.agents.gaia_shared import build_registry, reformat_to_gaia_template
from ant.benchmarks.base import TaskExample
from ant.domain.models import TokenUsage
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.gaia_owl_bridge import GaiaOwlBridge
from ant.evaluation_suite.usage import UsageStats
from ant.providers.pricing import estimate_cost_usd

REPO_ROOT = Path(__file__).resolve().parents[3]
OWL_VENV_PYTHON = REPO_ROOT / ".venv-owl" / "bin" / "python"
OWL_SUBPROCESS_SCRIPT = REPO_ROOT / "scripts" / "owl_worker_subprocess.py"

_RESULT_PREFIX = "OWL_RESULT_JSON:"
DEFAULT_TIMEOUT_SECONDS = 1800.0


class OwlSubprocessError(RuntimeError):
    """The OWL subprocess exited without producing a parseable result --
    stderr is included verbatim so a failed run's cause is visible in the
    resulting error row rather than a bare non-zero-exit-code message."""


class OwlGaiaAgent:
    name = "owl_gaia"

    def __init__(
        self,
        model: str = "openai/gpt-4.1",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        if not OWL_VENV_PYTHON.exists():
            raise RuntimeError(
                f"{OWL_VENV_PYTHON} not found -- run `python3.12 -m venv .venv-owl && "
                f".venv-owl/bin/pip install camel-ai 'mcp<2'` first (see owl_gaia.py's own "
                "docstring for why this lives in its own venv, never the main one)."
            )

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        started = time.time()
        registry = build_registry(example, environment_root)
        provider = CountingOpenAIProvider(model=self.model)

        with GaiaOwlBridge(registry) as bridge:
            env = {
                **os.environ,
                "GAIA_BRIDGE_URL": bridge.base_url,
                "OWL_MODEL_NAME": self.model,
                "OWL_OPENAI_BASE_URL": os.environ.get(
                    "OPENAI_BASE_URL", "https://api.openai.com/v1"
                ),
                "OWL_OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
                "GAIA_QUESTION": example.question,
                "GAIA_HAS_ATTACHMENT": "1" if example.metadata.get("has_attachment") else "0",
            }
            try:
                proc = subprocess.run(
                    [str(OWL_VENV_PYTHON), str(OWL_SUBPROCESS_SCRIPT)],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise OwlSubprocessError(
                    f"OWL subprocess exceeded {self.timeout_seconds}s timeout"
                ) from exc

        result_line = next(
            (line for line in proc.stdout.splitlines() if line.startswith(_RESULT_PREFIX)),
            None,
        )
        if result_line is None:
            raise OwlSubprocessError(
                f"OWL subprocess produced no result (exit={proc.returncode}). "
                f"stderr:\n{proc.stderr[-4000:]}"
            )
        owl_result = json.loads(result_line[len(_RESULT_PREFIX) :])

        raw_answer = owl_result.get("final_answer", "")
        final_answer = reformat_to_gaia_template(provider, example.question, raw_answer)

        reformat_usage = provider.drain_usage()
        reformat_calls = provider.drain_call_count()

        owl_usage = TokenUsage(
            input_tokens=owl_result.get("input_tokens", 0),
            output_tokens=owl_result.get("output_tokens", 0),
            total_tokens=owl_result.get("total_tokens", 0),
        )
        owl_cost = estimate_cost_usd(self.model, owl_usage)

        elapsed = time.time() - started
        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=final_answer,
            trajectory=[],
            evidence=[],
            usage=UsageStats(
                llm_calls=owl_result.get("llm_calls", 0) + reformat_calls,
                tool_calls=registry.tool_call_count(),
                input_tokens=owl_result.get("input_tokens", 0) + reformat_usage.input_tokens,
                output_tokens=owl_result.get("output_tokens", 0) + reformat_usage.output_tokens,
                total_tokens=owl_result.get("total_tokens", 0) + reformat_usage.total_tokens,
                estimated_cost_usd=round(owl_cost + reformat_usage.estimated_cost_usd, 8),
                wall_clock_seconds=elapsed,
                unique_files_inspected=len(
                    {c["arguments"].get("url") for c in registry.log_as_dicts() if c["tool"] == "open_url"}
                ),
            ),
            termination_reason="owl_workforce_completed",
            metadata={
                "generation_model": self.model,
                "gaia_call_log": registry.log_as_dicts(),
                "raw_answer_before_reformat": raw_answer,
                "owl_subprocess_exit_code": proc.returncode,
                "owl_venv": str(OWL_VENV_PYTHON),
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(OwlGaiaAgent())
