"""Matched ReAct for the Track B (web-navigation) substrate: ANT's primary
causal controlled baseline for WebWalkerQA, mirroring
`ant.agents.matched_react.MatchedReActAgent`'s own role for Track A.

One single agent, one continuous Thought-Action-Observation loop, no Need
Graph, no WorkerCards, no multi-worker coordination, no ANTMAN recovery,
no gold-guided routing, no external search-engine access -- exactly the
negative-space the governing spec requires. The agent has exactly two
actions available: follow one of the CURRENT page's own discovered links,
or finish with an answer -- there is no search() tool, matching
WebWalkerQA's own paper-audited navigation model ("purely click-based, no
search function," `docs/webwalkerqa_ant_mapping.md` section 2).

Uses `ant.evaluation_suite.web_scope.EvalWebEnvironment` unmodified --
`navigate()` is the sole discovery channel (raises on any link not
actually present on the current page), `inspect()` re-reads only
already-visited pages, same-site restriction and the deterministic disk
cache are enforced by `PageCache` underneath, never re-implemented here.
`max_steps` defaults to `DEFAULT_MAX_STEPS` (15, WebWalkerQA's own
paper-stated Explorer cap) -- the SAME default `AntWebAgent` uses, per
this suite's explicit fairness requirement that ReAct and ANTMAN receive
identical runtime navigation primitives/budget.

GOLD-LEAKAGE NOTE: `run()` reads only `example.question` and
`example.metadata["root_url"]` -- never `example.reference` (the gold
answer, read exclusively by `WebWalkerQaAdapter.score()`) and never
`example.metadata["_audit_only"]` (source_websites/golden_path). No
variable in this module is ever assigned from either.
"""

from __future__ import annotations

import time
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.usage import UsageStats
from ant.evaluation_suite.web_fetch import FetchedPage, PageCache, urllib_fetcher
from ant.evaluation_suite.web_scope import EvalWebEnvironment
from ant.providers.openai_provider import _loads_json_object

# WebWalkerQA's own paper-stated Explorer-agent step ceiling
# (`docs/webwalkerqa_ant_mapping.md` section 2) -- kept in sync with
# `ant.agents.webwalkerqa_stubs.DEFAULT_MAX_STEPS` (both must stay equal;
# see that module's own fairness note -- ReAct and ANTMAN share this
# exact budget).
DEFAULT_MAX_STEPS = 15

# Page text is truncated before entering a prompt (same "bounded evidence
# window per step" principle Track A's matched_react.py applies to its own
# tool-result history) -- unlike the OFFICIAL WebWalker method's Critic
# role (a real context-compression mechanism), this baseline deliberately
# keeps only a short PER-STEP SUMMARY in history (not full page text) and
# only the CURRENT page's full (truncated) text in the prompt, so context
# size stays roughly constant across steps rather than accumulating every
# page ever visited.
DEFAULT_MAX_PAGE_CHARS = 3000

_SYSTEM_PROMPT = """You are a single autonomous agent answering a question by navigating \
one website, starting from its root page. You may ONLY follow links that are actually \
shown to you on the CURRENT page -- there is no search function and no way to jump to an \
arbitrary URL. Read the current page's content, then either click one of its links to \
explore further, or finish once you have enough evidence to answer.

At each step, respond with ONLY a JSON object of one of these two shapes:
{{"thought": "<your reasoning>", "action": "navigate", \
"link_index": <integer index of the link to follow>}}
{{"thought": "<your reasoning>", "action": "finish", "answer": "<your complete final answer>"}}

You have a budget of {budget} steps for this question. Use them efficiently; finish as \
soon as you have enough evidence. No explanation outside the JSON object."""

_FORCE_FINISH_PROMPT = """You explored a website to try to answer a question but ran out \
of navigation steps before declaring a final answer.

Question: {question}

Navigation history:
{history}

Last page visited:
{page}

Based on everything you have seen, give your single best final answer now. Respond with \
ONLY the answer text -- no JSON, no explanation prefix."""


def _format_page(page: FetchedPage, max_chars: int) -> str:
    if page.status != "ok":
        return f"[Page inaccessible: {page.error or page.status}]"
    text = page.text[:max_chars]
    links_block = "\n".join(f"[{i}] {url}" for i, url in enumerate(page.links))
    if not links_block:
        links_block = "(no links found on this page)"
    return f"URL: {page.url}\n\nContent:\n{text}\n\nLinks on this page:\n{links_block}"


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none yet)"
    lines = []
    for entry in history:
        if entry["action"] == "navigate" and "to_url" in entry:
            lines.append(
                f"[{entry['step']}] navigated to {entry['to_url']} (status={entry['status']})"
            )
        else:
            lines.append(f"[{entry['step']}] {entry['action']}: {entry.get('error', '')}")
    return "\n".join(lines)


class MatchedReActWebAgent:
    """See this module's own docstring for the full design/fairness note."""

    name = "matched_react_web"

    def __init__(
        self,
        model: str = "gpt-4.1",
        max_steps: int = DEFAULT_MAX_STEPS,
        max_page_chars: int = DEFAULT_MAX_PAGE_CHARS,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.model = model
        self.max_steps = max_steps
        self.max_page_chars = max_page_chars
        self.timeout_seconds = timeout_seconds

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        provider = CountingOpenAIProvider(model=self.model)
        root_url = example.metadata["root_url"]
        cache = PageCache(
            cache_dir=environment_root,
            fetcher=urllib_fetcher(timeout_seconds=self.timeout_seconds),
            root_url=root_url,
        )
        env = EvalWebEnvironment(root_url, cache, max_steps=self.max_steps)

        started = time.time()
        history: list[dict] = []
        final_answer = ""
        termination_reason = "step_budget_exhausted"
        n_inaccessible = 0

        current_page = env.root_page()
        if current_page.status != "ok":
            n_inaccessible += 1

        budget = self.max_steps
        for step in range(budget):
            prompt = (
                _SYSTEM_PROMPT.format(budget=budget)
                + f"\n\nQuestion: {example.question}\n\n"
                + f"Navigation history so far:\n{_format_history(history)}\n\n"
                + f"Current page ({step}/{budget} steps used):\n"
                + f"{_format_page(current_page, self.max_page_chars)}\n\n"
                + "Decide your next action."
            )
            response = provider.responses_json(prompt, max_output_tokens=400)
            decision = _loads_json_object(response.text)
            action = decision.get("action") if isinstance(decision, dict) else None

            if action == "finish":
                answer = decision.get("answer") if isinstance(decision, dict) else None
                if isinstance(answer, str) and answer.strip():
                    final_answer = answer.strip()
                    termination_reason = "agent_declared_finish"
                    break
                history.append(
                    {"step": step, "action": "finish", "error": "empty/malformed answer"}
                )
                continue

            if action == "navigate":
                link_index = decision.get("link_index") if isinstance(decision, dict) else None
                if not isinstance(link_index, int) or not (
                    0 <= link_index < len(current_page.links)
                ):
                    history.append(
                        {
                            "step": step,
                            "action": "navigate",
                            "error": f"invalid link_index {link_index!r}",
                        }
                    )
                    continue
                target_url = current_page.links[link_index]
                try:
                    new_page = env.navigate(current_page, target_url)
                except (ValueError, RuntimeError) as exc:
                    history.append({"step": step, "action": "navigate", "error": str(exc)})
                    continue
                if new_page.status != "ok":
                    n_inaccessible += 1
                history.append(
                    {
                        "step": step,
                        "action": "navigate",
                        "from_url": current_page.url,
                        "to_url": new_page.url,
                        "status": new_page.status,
                    }
                )
                current_page = new_page
                continue

            history.append({"step": step, "action": str(action), "error": "unrecognized action"})

        if not final_answer:
            force_prompt = _FORCE_FINISH_PROMPT.format(
                question=example.question,
                history=_format_history(history),
                page=_format_page(current_page, self.max_page_chars),
            )
            forced = provider.responses_text(force_prompt, max_output_tokens=300)
            final_answer = forced.text.strip()
            termination_reason = "step_budget_exhausted_forced_finish"

        llm_calls = provider.drain_call_count()
        token_usage = provider.drain_usage()
        elapsed = time.time() - started

        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=final_answer,
            trajectory=history,
            evidence=[],
            usage=UsageStats(
                llm_calls=llm_calls,
                tool_calls=env.step_count(),
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                total_tokens=token_usage.total_tokens,
                estimated_cost_usd=token_usage.estimated_cost_usd,
                wall_clock_seconds=elapsed,
                unique_files_inspected=len(env.discovered_pages()),
            ),
            termination_reason=termination_reason,
            metadata={
                "generation_model": self.model,
                "max_steps": self.max_steps,
                "navigation_steps": env.step_count(),
                "pages_visited": len(env.discovered_pages()),
                "n_inaccessible_pages_in_trajectory": n_inaccessible,
                "step_budget_exhausted": termination_reason
                == "step_budget_exhausted_forced_finish",
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(MatchedReActWebAgent())
