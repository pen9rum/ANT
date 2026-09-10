"""SWE-QA-Pro official ToolCallingAgent wrapper -- baseline-fidelity audit
tests. The real subprocess/checkout is never invoked here (subprocess.run
is mocked); these tests exercise only this evaluation suite's OWN parsing
of the (real, previously observed) shape of _ant_driver.py's JSON output.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ant.benchmarks.base import TaskExample
from ant.external_wrappers.sweqa_pro_native_agent import SweQaProNativeAgent


def _fake_driver_stdout(**overrides) -> str:
    payload = {
        "query": "Where is X?",
        "code_base_dir": "/repo",
        "answer": "X lives in foo.py.",
        "status": "success",
        "stop_reason": "Natural completion",
        "steps_completed": 12,
        "max_iterations": 25,
        "trajectory": [{"step": 0, "note": "intermediate, must not be scored"}],
        "retry_attempts": 0,
        "token_usage": {"prompt_tokens": 10_000, "completion_tokens": 500, "total_tokens": 10_500},
        "total_time": 12.3,
        "tool_usage": {"counts": {"view_codebase": 2, "semantic_search": 1}, "records": []},
        "provider": "openai",
        "model": "gpt-4.1",
        "physical_llm_calls": 13,  # deliberately != steps_completed, matching the real observed gap
    }
    payload.update(overrides)
    # Real console noise precedes the JSON on stdout, per the class's own docstring.
    return "some rich-console progress panel text\n" + json.dumps(payload)


def _agent_with_checkout_present(tmp_path: Path) -> SweQaProNativeAgent:
    checkout_root = tmp_path / "checkout"
    (checkout_root / "eval" / "sweqapro").mkdir(parents=True)
    (checkout_root / "eval" / "sweqapro" / "agent.py").write_text("", encoding="utf-8")
    (checkout_root / ".venv" / "Scripts").mkdir(parents=True)
    venv_python = checkout_root / ".venv" / "Scripts" / "python.exe"
    venv_python.write_text("", encoding="utf-8")
    (checkout_root / "_ant_driver.py").write_text("", encoding="utf-8")
    return SweQaProNativeAgent(checkout_root=checkout_root, venv_python=venv_python)


def test_final_answer_comes_from_the_answer_key_not_the_trajectory(
    tmp_path: Path, monkeypatch
) -> None:
    """Matches the official submission path exactly: run_agent.py's own
    result_to_save = dict(result); result_to_save.pop("trajectory", None)
    -- proving `trajectory` is explicitly excluded from what gets scored
    upstream too, and confirming this wrapper reads the SAME `answer` key
    ToolCallingAgent.query() itself returns (final_state["final_answer"]),
    never anything else."""
    agent = _agent_with_checkout_present(tmp_path)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=_fake_driver_stdout(), stderr="")

    monkeypatch.setattr("ant.external_wrappers.sweqa_pro_native_agent.subprocess.run", fake_run)

    example = TaskExample(benchmark="sweqa_pro", task_id="q1", question="Where is X?", reference="")
    result = agent.run(example, tmp_path)

    assert result.final_answer == "X lives in foo.py."
    # Trajectory is carried for observability but is never what final_answer
    # reads from -- an intermediate trajectory note must never leak into it.
    assert "intermediate, must not be scored" not in result.final_answer


def test_physical_llm_calls_used_not_steps_completed(tmp_path: Path, monkeypatch) -> None:
    agent = _agent_with_checkout_present(tmp_path)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=_fake_driver_stdout(steps_completed=12, physical_llm_calls=13),
            stderr="",
        )

    monkeypatch.setattr("ant.external_wrappers.sweqa_pro_native_agent.subprocess.run", fake_run)

    example = TaskExample(benchmark="sweqa_pro", task_id="q1", question="Where is X?", reference="")
    result = agent.run(example, tmp_path)

    assert result.usage.llm_calls == 13
    assert result.metadata["steps_completed"] == 12


def test_generation_cost_is_computed_transparently_from_real_token_counts(
    tmp_path: Path, monkeypatch
) -> None:
    """New: cost accounting added without touching agent behavior --
    estimate_cost_usd(model, TokenUsage(...)) is a pure, read-only
    computation over the SAME token_usage the official agent's own
    final_state already accumulates, using this suite's shared pricing
    table (gpt-4.1: $2.00/$8.00 per million input/output tokens)."""
    agent = _agent_with_checkout_present(tmp_path)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=_fake_driver_stdout(
                token_usage={
                    "prompt_tokens": 100_000,
                    "completion_tokens": 5_000,
                    "total_tokens": 105_000,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("ant.external_wrappers.sweqa_pro_native_agent.subprocess.run", fake_run)

    example = TaskExample(benchmark="sweqa_pro", task_id="q1", question="Where is X?", reference="")
    result = agent.run(example, tmp_path)

    expected = round(100_000 / 1_000_000 * 2.00 + 5_000 / 1_000_000 * 8.00, 8)
    assert result.usage.estimated_cost_usd == expected
    assert result.usage.input_tokens == 100_000
    assert result.usage.output_tokens == 5_000
