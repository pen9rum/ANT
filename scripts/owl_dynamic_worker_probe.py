"""Minimal, deterministic probe: directly calls Workforce._create_worker_
node_for_task (the SOLE internal entry point that creates a new worker at
runtime -- confirmed via source inspection to be the only caller of
_create_new_agent) and inspects the resulting worker's actual tool list,
rather than trying to coax the full coordinator LLM into deciding to spawn
one (unreliable, prompt-dependent, and not what we're actually verifying).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

_env_path = Path(__file__).resolve().parents[1] / ".env"
for _line in _env_path.read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _key, _value = _line.split("=", 1)
        os.environ.setdefault(_key.strip(), _value.strip())

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.societies.workforce import Workforce
from camel.tasks import Task
from camel.toolkits import FunctionTool
from camel.types import ModelPlatformType


def fake_search(query: str) -> str:
    """Search the web for the query. Returns a short snippet."""
    return f"[FAKE BRIDGE SEARCH] no-op stub for: {query}"


def fake_open_url(url: str) -> str:
    """Fetch a URL's text content."""
    return f"[FAKE BRIDGE FETCH] no-op stub for: {url}"


async def main() -> None:
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type="openai/gpt-4.1",
        url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )

    bridged_tools = [FunctionTool(fake_search), FunctionTool(fake_open_url)]
    bridged_tool_names = sorted(t.get_function_name() for t in bridged_tools)

    new_worker_template = ChatAgent(
        system_message="placeholder",
        model=model,
        tools=bridged_tools,
    )

    print(f"Template tools: {bridged_tool_names}")

    # --- Case A: new_worker_agent constrained to our 5 (2, here) bridge tools ---
    wf_constrained = Workforce(
        description="probe A",
        new_worker_agent=new_worker_template,
        default_model=model,
    )
    new_agent_a = await wf_constrained._create_new_agent(
        role="Web Researcher", sys_msg="Find facts online."
    )
    tools_a = sorted(new_agent_a.tool_dict.keys())
    print(f"\nCase A (new_worker_agent SET): dynamically-created worker tools = {tools_a}")
    print(f"  Matches bridge-only tool set exactly: {tools_a == bridged_tool_names}")

    # --- Case B: default behavior, no new_worker_agent provided ---
    wf_default = Workforce(description="probe B", default_model=model)
    new_agent_b = await wf_default._create_new_agent(
        role="Web Researcher", sys_msg="Find facts online."
    )
    tools_b = sorted(new_agent_b.tool_dict.keys())
    print(f"\nCase B (new_worker_agent NOT set): dynamically-created worker tools = {tools_b}")
    print(f"  Contains only our bridge tools: {set(tools_b) <= set(bridged_tool_names)}")


if __name__ == "__main__":
    asyncio.run(main())
