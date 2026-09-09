from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.usage import UsageStats

SWEQA_PRO_REPO_URL = "https://github.com/TIGER-AI-Lab/SWE-QA-Pro"
SWEQA_PRO_COMMIT = "93ac6a4"  # kept in sync with benchmarks/sweqa_pro.py's own pin

# Driver script run inside the pinned checkout's OWN isolated venv (needs
# langgraph/langchain-core/rich -- not ANT's own dependencies) -- imports
# ToolCallingAgent directly (in-process within that venv, not re-shelled
# per call) rather than going through their own CLI entry point, since the
# class itself (eval/sweqapro/agent.py, read in full at the pinned commit)
# is a clean, self-contained, provider-agnostic unit once `llm`/
# `llm_no_tools` are built via their own `sweqapro.registry.build_llm`.
_DRIVER_SCRIPT = """
import json, sys
sys.path.insert(0, "eval")
from sweqapro.agent import ToolCallingAgent
from sweqapro.registry import build_llm

question, repo_path, model_key, max_iterations = json.loads(sys.stdin.read())
llm, llm_no_tools, provider = build_llm(model_key)
agent = ToolCallingAgent(
    llm, llm_no_tools, provider, model_key,
    max_iterations=max_iterations, quiet=True,
)
result = agent.query(question, repo_path)
print(json.dumps(result))
"""


class SweQaProNativeAgentNotCheckedOut(RuntimeError):
    pass


class SweQaProNativeAgent:
    """Wraps the OFFICIAL SWE-QA-Pro `ToolCallingAgent`
    (eval/sweqapro/agent.py at the pinned commit -- full source read
    directly, not reimplemented: LangGraph state machine, 3 tools
    (semantic_search/view_codebase/execute_readonly_command), <finish>
    block extraction, force-finish on max-iterations/context-limit,
    degenerate-output detection, retry-with-sliding-history-window). This
    is the benchmark-native reference baseline (per Phase G), explicitly
    distinct from ANT's own Matched ReAct controlled baseline -- do not
    conflate the two in any report.

    `model_key` must be a key in the pinned checkout's own
    eval/configs/models.yaml -- 'gpt-4.1' is one of the officially
    supported entries (model_id: gpt-4.1-2025-04-14), preferred here per
    Phase G's "prefer GPT-4.1 where the official implementation supports
    it".

    NOT yet exercised against a real checkout in this session -- `run()`
    raises `SweQaProNativeAgentNotCheckedOut` with exact setup
    instructions if the checkout/its own isolated venv aren't present.
    """

    name = "sweqa_pro_native_agent"

    def __init__(
        self,
        checkout_root: Path | None = None,
        venv_python: Path | None = None,
        model_key: str = "gpt-4.1",
        max_iterations: int = 25,
    ) -> None:
        self.checkout_root = checkout_root or Path("third_party/checkouts/sweqa_pro")
        self.venv_python = venv_python or (self.checkout_root / ".venv" / "Scripts" / "python.exe")
        self.model_key = model_key
        self.max_iterations = max_iterations

    def _require_checkout(self) -> None:
        if not (self.checkout_root / "eval" / "sweqapro" / "agent.py").exists():
            raise SweQaProNativeAgentNotCheckedOut(
                f"SWE-QA-Pro is not checked out at {self.checkout_root}. To run this "
                f"baseline: git clone {SWEQA_PRO_REPO_URL} {self.checkout_root} && "
                f"cd {self.checkout_root} && git checkout {SWEQA_PRO_COMMIT}, then create an "
                f"ISOLATED venv at {self.venv_python.parent} (python -m venv .venv) and "
                "install their eval/ dependencies (langgraph, langchain-core, rich, plus "
                "provider SDKs) into it -- never into ANT's own .venv (see Phase 13's "
                "dependency-isolation strategy)."
            )
        if not self.venv_python.exists():
            raise SweQaProNativeAgentNotCheckedOut(
                f"Checkout found at {self.checkout_root} but its own isolated venv does not "
                f"exist at {self.venv_python}. Create it and install eval/'s own dependencies "
                "before running this baseline."
            )

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        self._require_checkout()
        payload = json.dumps(
            [example.question, str(environment_root), self.model_key, self.max_iterations]
        )
        result = subprocess.run(
            [str(self.venv_python), "-c", _DRIVER_SCRIPT],
            cwd=self.checkout_root,
            input=payload,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode != 0:
            return AgentResult(
                benchmark=example.benchmark,
                task_id=example.task_id,
                method=self.name,
                final_answer="",
                termination_reason=f"subprocess_failed: {result.stderr[-500:]}",
            )
        raw = json.loads(result.stdout.strip().splitlines()[-1])
        token_usage = raw.get("token_usage", {})
        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=raw.get("answer", ""),
            trajectory=raw.get("trajectory", []),
            usage=UsageStats(
                input_tokens=token_usage.get("prompt_tokens", 0),
                output_tokens=token_usage.get("completion_tokens", 0),
                total_tokens=token_usage.get("total_tokens", 0),
                tool_calls=sum(len(t.get("tool_calls", [])) for t in raw.get("trajectory", [])),
                llm_calls=raw.get("steps_completed", 0),
                wall_clock_seconds=raw.get("total_time", 0.0),
            ),
            termination_reason=raw.get("stop_reason", "unknown"),
            metadata={
                "status": raw.get("status"),
                "raw_output": raw,
                "generation_model": self.model_key,
            },
        )
