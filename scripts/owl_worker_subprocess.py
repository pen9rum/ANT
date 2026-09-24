"""Runs ONE GAIA question through OWL's Workforce (camel-ai). MUST be
invoked with `.venv-owl`'s python, never the main `.venv` -- see
gaia_owl_bridge.py's own docstring for why camel-ai lives in an isolated
venv. This script imports nothing from `ant.*`: it is a standalone driver
that talks to the caller's GaiaToolRegistry only through the HTTP bridge
(GAIA_BRIDGE_URL), which is the ONLY capability this process is given.

FAIRNESS DESIGN (see the ICLR-2027-project convention this substrate
enforces everywhere else): OWL's own native toolkits (SearchToolkit,
CodeExecutionToolkit, ThinkingToolkit, ...) are never imported or attached
to anything here. Both (a) the one explicitly-registered worker and (b)
the `new_worker_agent` template that EVERY dynamically-spawned worker
clones from (confirmed via direct source inspection of
camel.societies.workforce.workforce._create_new_agent: the sole runtime
worker-creation path, and ChatAgent.clone: which faithfully copies the
`tools` list) carry the SAME five bridged tools and nothing else. This was
verified live before this script was written -- see the dynamic-worker
probe run during development.

Reads all configuration from environment variables (never argv, so
secrets never show up in `ps`):
  GAIA_BRIDGE_URL      e.g. http://127.0.0.1:54321
  OWL_MODEL_NAME        e.g. openai/gpt-4.1
  OWL_OPENAI_BASE_URL
  OWL_OPENAI_API_KEY
  GAIA_QUESTION         the raw question text
  GAIA_HAS_ATTACHMENT   "1" or "0" -- purely informational, shapes the
                         worker's system message; the tool set is
                         identical either way.

Prints exactly one line to stdout prefixed with `OWL_RESULT_JSON:`,
containing the final answer and aggregated usage across every agent in
the Workforce (coordinator, task planner, the initial worker, and every
dynamically-created worker) -- collected via each ChatAgent's own
`on_request_usage` hook, the same per-physical-call granularity this
project's own CountingOpenAIProvider uses elsewhere.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.error
import urllib.request

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.societies.workforce import Workforce
from camel.tasks import Task
from camel.toolkits import FunctionTool
from camel.types import ModelPlatformType

def _call_bridge(tool: str, **kwargs: object) -> dict:
    # Read lazily (not at module import time) so this module can be
    # imported for its tool-building functions (e.g. by a smoke test)
    # without GAIA_BRIDGE_URL having to be set yet.
    bridge_url = os.environ["GAIA_BRIDGE_URL"]
    data = json.dumps(kwargs).encode("utf-8")
    req = urllib.request.Request(
        f"{bridge_url}/{tool}", data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return {"error": f"bridge unreachable: {exc!r}"}


def search(query: str, limit: int = 8) -> str:
    """Search the open web. Returns numbered title/url/snippet hits."""
    result = _call_bridge("search", query=query, limit=limit)
    if "error" in result:
        return f"(search error: {result['error']})"
    hits = result["hits"]
    if not hits:
        return "(no results)"
    return "\n".join(f"[{i + 1}] {h['title']} ({h['url']}): {h['snippet']}" for i, h in enumerate(hits))


def open_url(url: str) -> str:
    """Fetch one URL and return its extracted text."""
    result = _call_bridge("open_url", url=url)
    return result.get("text") or f"(fetch error: {result.get('error')})"


def inspect_file(offset: int = 0, limit: int = 20000) -> str:
    """Read this task's attachment as text, starting at `offset`."""
    result = _call_bridge("inspect_file", offset=offset, limit=limit)
    return result.get("text") or f"(inspect_file error: {result.get('error')})"


def inspect_table(max_rows: int = 200) -> str:
    """Read this task's attachment as tabular rows (tab-separated)."""
    result = _call_bridge("inspect_table", max_rows=max_rows)
    return result.get("text") or f"(inspect_table error: {result.get('error')})"


def run_python(code: str) -> str:
    """Run a short Python program in an isolated, network-free sandbox."""
    result = _call_bridge("run_python", code=code)
    return result.get("text") or f"(run_python error: {result.get('error')})"


class UsageTracker:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record(self, payload: dict) -> None:
        self.calls.append(payload)

    def totals(self) -> dict:
        input_tokens = sum(c["request_usage"]["prompt_tokens"] for c in self.calls)
        output_tokens = sum(c["request_usage"]["completion_tokens"] for c in self.calls)
        total_tokens = sum(c["request_usage"]["total_tokens"] for c in self.calls)
        return {
            "llm_calls": len(self.calls),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }


def build_bridged_tools() -> list:
    """The ONLY tools any agent in this Workforce -- initial or
    dynamically created -- is ever given. Factored out so a smoke test can
    assert on this exact list (by function name) rather than re-declaring
    a parallel copy that could silently drift from what production uses."""
    return [
        FunctionTool(search),
        FunctionTool(open_url),
        FunctionTool(inspect_file),
        FunctionTool(inspect_table),
        FunctionTool(run_python),
    ]


def build_workforce(
    model, tracker: UsageTracker, *, has_attachment: bool
) -> tuple[Workforce, ChatAgent]:
    """Builds the exact Workforce production uses -- one explicitly
    registered worker plus a `new_worker_agent` template, both carrying
    `build_bridged_tools()` and nothing else -- and returns it alongside
    the initial worker agent so a caller (main() or a smoke test) can
    inspect either. See this module's own docstring for the fairness
    argument this construction implements."""
    bridged_tools = build_bridged_tools()

    worker_sys_msg = (
        "You are a GAIA task-solving worker. You can search the open web, "
        "fetch a URL's text, and run short Python programs in a sandbox."
        + (
            " This task has an attached file you can read with inspect_file/inspect_table."
            if has_attachment
            else " This task has no attachment."
        )
    )

    initial_worker = ChatAgent(
        system_message=worker_sys_msg,
        model=model,
        tools=bridged_tools,
        on_request_usage=tracker.record,
    )
    # Separate INSTANCE (not the same object as initial_worker) used only as
    # the clone template for runtime-created workers -- clone() copies the
    # tools list by value, so this never needs its own live conversation.
    new_worker_template = ChatAgent(
        system_message="placeholder",  # overwritten per-role by Workforce._create_new_agent
        model=model,
        tools=build_bridged_tools(),
        on_request_usage=tracker.record,
    )
    coordinator_agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Workforce Manager", content=""
        ),
        model=model,
        on_request_usage=tracker.record,
    )
    task_agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(role_name="Task Planner", content=""),
        model=model,
        on_request_usage=tracker.record,
    )

    workforce = Workforce(
        description="GAIA task force",
        coordinator_agent=coordinator_agent,
        task_agent=task_agent,
        new_worker_agent=new_worker_template,
        default_model=model,
    )
    workforce.add_single_agent_worker(
        "General GAIA worker: web search, URL fetch, sandboxed Python, attachment reading",
        worker=initial_worker,
    )
    return workforce, initial_worker


async def main() -> None:
    question = os.environ["GAIA_QUESTION"]
    has_attachment = os.environ.get("GAIA_HAS_ATTACHMENT") == "1"
    model_name = os.environ["OWL_MODEL_NAME"]
    base_url = os.environ["OWL_OPENAI_BASE_URL"]
    api_key = os.environ["OWL_OPENAI_API_KEY"]

    tracker = UsageTracker()

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=model_name,
        url=base_url,
        api_key=api_key,
    )

    workforce, _initial_worker = build_workforce(model, tracker, has_attachment=has_attachment)

    task = Task(content=question, id="gaia-q")
    result_task = await asyncio.wait_for(
        asyncio.to_thread(workforce.process_task, task), timeout=1800
    )
    final_answer = result_task.result or ""

    output = {"final_answer": final_answer, **tracker.totals()}
    print("OWL_RESULT_JSON:" + json.dumps(output))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:  # noqa: BLE001 - reported to the parent process via stderr + exit code
        print(f"OWL_SUBPROCESS_ERROR:{exc!r}", file=sys.stderr)
        raise
