from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.usage import UsageStats

SWEQA_PRO_REPO_URL = "https://github.com/TIGER-AI-Lab/SWE-QA-Pro"
SWEQA_PRO_COMMIT = "93ac6a4"  # kept in sync with benchmarks/sweqa_pro.py's own pin


class SweQaProNativeAgentNotCheckedOut(RuntimeError):
    pass


class SweQaProNativeAgent:
    """Wraps the OFFICIAL SWE-QA-Pro `ToolCallingAgent`
    (eval/sweqapro/agent.py at the pinned commit -- full source read
    directly, not reimplemented: LangGraph state machine, 3 tools
    (semantic_search/view_codebase/execute_readonly_command), <finish>
    block extraction, force-finish on max-iterations/context-limit,
    degenerate-output detection, retry-with-sliding-history-window). This
    is the benchmark-native reference baseline, explicitly distinct from
    ANT's own Matched ReAct controlled baseline -- do not conflate the
    two in any report.

    VERIFIED LIVE this session (real repo, real GPT-4.1 calls) via a
    small driver script (`_ant_driver.py`, checkout root, NOT part of the
    official release -- calls `ToolCallingAgent.query()` directly instead
    of going through `eval/scripts/run_agent.py`'s own benchmark-loading/
    resume/threading CLI logic, since we only want one question at a
    time). `registry.resolve()`/`registry.build_llm()` are used
    completely unmodified -- no bug was found in this repo's own code, no
    source patch was needed (unlike HiFiRepoQA).

    Confirmed live, real end-to-end results (qibo and Pillow dev
    questions): status=success, tool usage break down correctly by the 3
    official tools, `stop_reason="Forced after max iterations (25)"` on
    both (used the full budget rather than a natural early stop).
    Physical-LLM-call counting uses a standard LangChain
    `BaseCallbackHandler` (`on_chat_model_start`) attached via
    `llm.with_config(callbacks=[...])` in `_ant_driver.py` -- purely
    observational instrumentation. This caught a real, small accounting
    discrepancy: the official agent's own `steps_completed` field (25 on
    the Pillow question) undercounted the true physical call count (26)
    by one -- the same class of gap ANT's own baselines had before
    `CountingOpenAIProvider` fixed it (see that module's docstring). This
    wrapper reports the physically-counted figure, not `steps_completed`.

    `model_key` must be a key in the pinned checkout's own
    eval/configs/models.yaml -- 'gpt-4.1' is one of the officially
    supported entries (model_id: gpt-4.1-2025-04-14).
    """

    name = "sweqa_pro_native_agent"

    def __init__(
        self,
        checkout_root: Path | None = None,
        venv_python: Path | None = None,
        model_key: str = "gpt-4.1",
        env_path: Path | None = None,
        timeout_seconds: int = 1800,
    ) -> None:
        self.checkout_root = checkout_root or Path("third_party/checkouts/sweqa_pro")
        self.venv_python = venv_python or (self.checkout_root / ".venv" / "Scripts" / "python.exe")
        self.model_key = model_key
        # ant.config.load_dotenv resolves its `path` argument relative to
        # the CURRENT WORKING DIRECTORY, not a fixed project root -- an
        # earlier version of this wrapper's own driver testing hit exactly
        # this (a silently-empty OPENAI_API_KEY when invoked from the
        # checkout's own directory), which was originally misdiagnosed as
        # an organization/project credential-scoping problem before the
        # real cause was found. Always pass an explicit, resolved path.
        self.env_path = env_path or Path(".env").resolve()
        self.timeout_seconds = timeout_seconds

    def _require_checkout(self) -> None:
        if not (self.checkout_root / "eval" / "sweqapro" / "agent.py").exists():
            raise SweQaProNativeAgentNotCheckedOut(
                f"SWE-QA-Pro is not checked out at {self.checkout_root}. To run this "
                f"baseline: git clone {SWEQA_PRO_REPO_URL} {self.checkout_root} && "
                f"cd {self.checkout_root} && git checkout {SWEQA_PRO_COMMIT}, then create an "
                f"ISOLATED venv at {self.venv_python.parent} (python -m venv .venv) and "
                "`pip install langchain langchain-core langchain-openai langgraph openai "
                "datasets huggingface-hub rich tqdm python-dotenv PyYAML requests` (the "
                "openai-provider subset of eval/requirements.txt -- anthropic/gemini/vllm/"
                "transformers/tokenizers deliberately omitted since this suite only drives "
                "the openai provider path) -- never into ANT's own .venv, and never through "
                "`_ant_driver.py`, which must also be created at the checkout root (see this "
                "class's own docstring)."
            )
        if not self.venv_python.exists():
            raise SweQaProNativeAgentNotCheckedOut(
                f"Checkout found at {self.checkout_root} but its own isolated venv does not "
                f"exist at {self.venv_python}. Create it and install eval/'s own dependencies "
                "before running this baseline."
            )
        if not (self.checkout_root / "_ant_driver.py").exists():
            raise SweQaProNativeAgentNotCheckedOut(
                f"{self.checkout_root}/_ant_driver.py is missing -- the small, "
                "not-part-of-the-official-release driver script this wrapper depends on."
            )

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        self._require_checkout()
        import os

        from ant.config import load_dotenv

        env_values = load_dotenv(self.env_path)
        env = dict(os.environ)
        if env_values.get("OPENAI_API_KEY"):
            env["OPENAI_API_KEY"] = env_values["OPENAI_API_KEY"]

        payload = json.dumps(
            {
                "model": self.model_key,
                "question": example.question,
                "repo_path": str(environment_root),
            }
        )
        result = subprocess.run(
            [str(self.venv_python), "_ant_driver.py"],
            cwd=self.checkout_root,
            input=payload,
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=self.timeout_seconds,
        )
        if result.returncode != 0:
            return AgentResult(
                benchmark=example.benchmark,
                task_id=example.task_id,
                method=self.name,
                final_answer="",
                termination_reason=f"subprocess_failed: {result.stderr[-500:]}",
            )
        # _ant_driver.py's own stdout is prefixed by ToolCallingAgent's own
        # rich-console progress panels (real console output, not an error)
        # -- the JSON result is the LAST '{"query": ...' object on stdout.
        marker = '{"query"'
        start = result.stdout.rfind(marker)
        if start == -1:
            return AgentResult(
                benchmark=example.benchmark,
                task_id=example.task_id,
                method=self.name,
                final_answer="",
                termination_reason=f"no_json_output: {result.stdout[-500:]}",
            )
        raw = json.loads(result.stdout[start:])
        token_usage = raw.get("token_usage") or {}
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
                tool_calls=sum((raw.get("tool_usage") or {}).get("counts", {}).values()),
                # Physically counted via a LangChain callback in
                # _ant_driver.py, not the official agent's own
                # steps_completed field -- confirmed live to undercount by
                # 1 on at least one real question (see class docstring).
                llm_calls=raw.get("physical_llm_calls", 0),
                wall_clock_seconds=raw.get("total_time") or 0.0,
            ),
            termination_reason=raw.get("stop_reason", "unknown"),
            metadata={
                "status": raw.get("status"),
                "steps_completed": raw.get("steps_completed"),
                "raw_output": raw,
                "generation_model": self.model_key,
            },
        )
