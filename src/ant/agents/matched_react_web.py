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

FIDELITY CORRECTIONS (per the post-N=60-run fidelity audit, see
docs/webwalkerqa_matched_react_fidelity_audit.md): the original version
of this file diverged from the official WebWalker paper's own ReAct
formalism (arXiv:2501.07572 §4.1, and the released Explorer code's own
`_run()` context-accumulation) in three confirmed ways, all corrected
here:

1. **Links now carry their own visible anchor text**, not a bare URL --
   see `ant.evaluation_suite.web_fetch.PageLink`. The official environment
   shows `(button_text, url)` pairs (`app.py::extract_links_with_text`);
   showing bare URLs made navigation largely blind and was the primary,
   evidenced driver of the repeated-navigation pathology seen in every
   step-budget-exhaustion trajectory in the original run.
2. **Page content is no longer truncated to an arbitrary character cap.**
   The official paper explicitly chose >=128K-context backbones
   specifically to avoid truncating page content (§5.1) and the released
   code contains no truncation logic at all. GPT-4.1's real ~1M-token
   input window makes the same choice available here. The only remaining
   cap (`MAX_PROMPT_TOKENS` below) is a genuine API-safety backstop
   against actually exceeding the model's real input-token ceiling, not a
   default-path truncation -- see its own docstring.
3. **Full (Thought, Action, Observation) history is retained across
   steps**, matching the official formalism's own
   `ℋₜ=(𝒯₁,𝒜₁,𝒪₁,...,𝒪ₜ₋₁,𝒯ₜ,𝒜ₜ)` (§4.1) and the Explorer's own code
   (`agent.py::_run`, literally appending `thought + action + observation`
   to accumulated context every step, never summarized). The prior
   version kept only a one-line URL+status summary per step, discarding
   all previously-seen page content -- confirmed to correlate with heavy
   repeated navigation (the agent had no memory of what a page actually
   contained, only that it had visited *a* URL).

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

import tiktoken

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

# Same token-counting convention used throughout this project
# (external_wrappers/longagent.py, evaluation_suite/niah_plus.py, ...) --
# a consistent, disclosed ESTIMATE (GPT-4.1 itself may tokenize slightly
# differently), not a claim of byte-exact parity with OpenAI's own
# tokenizer for this specific model.
_ENCODING_NAME = "cl100k_base"

# GPT-4.1's real, verified input-token ceiling is 1,047,576 (confirmed via
# live web search against openai.com's own published limits, not assumed
# from memory). This is set well below that -- genuine safety headroom
# for (a) cl100k_base under/over-counting relative to GPT-4.1's actual
# tokenizer, (b) the fixed prompt scaffolding/question text, and (c) not
# pushing a real API call right up against a hard documented ceiling --
# NOT a default-path truncation. In the audited N=60 corpus this ceiling
# is never approached even once (p99 single-page size was ~36K tokens;
# 15 steps of accumulated history at that rate is far below this cap) --
# it exists only for the genuine long tail (one real page in that corpus
# was over 1 MiB of extracted text on its own).
MAX_PROMPT_TOKENS = 800_000

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

Full navigation history:
{history}

Based on everything you have seen, give your single best final answer now. Respond with \
ONLY the answer text -- no JSON, no explanation prefix."""


def _format_page(page: FetchedPage) -> str:
    if page.status != "ok":
        return f"[Page inaccessible: {page.error or page.status}]"
    links_block = "\n".join(
        f"[{i}] {link.text.strip() or '(no visible text)'} -> {link.url}"
        for i, link in enumerate(page.links)
    )
    if not links_block:
        links_block = "(no links found on this page)"
    return f"URL: {page.url}\n\nContent:\n{page.text}\n\nLinks on this page:\n{links_block}"


def _token_count(text: str, encoding: tiktoken.Encoding) -> int:
    return len(encoding.encode(text, disallowed_special=()))


def _join_history(blocks: list[str]) -> str:
    return "\n\n".join(blocks) if blocks else "(none yet)"


class MatchedReActWebAgent:
    """See this module's own docstring for the full design/fairness/
    fidelity-correction notes."""

    name = "matched_react_web"

    def __init__(
        self,
        model: str = "gpt-4.1",
        max_steps: int = DEFAULT_MAX_STEPS,
        timeout_seconds: float = 10.0,
        max_prompt_tokens: int = MAX_PROMPT_TOKENS,
    ) -> None:
        self.model = model
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.max_prompt_tokens = max_prompt_tokens

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        provider = CountingOpenAIProvider(model=self.model)
        root_url = example.metadata["root_url"]
        cache = PageCache(
            cache_dir=environment_root,
            fetcher=urllib_fetcher(timeout_seconds=self.timeout_seconds),
            root_url=root_url,
        )
        env = EvalWebEnvironment(root_url, cache, max_steps=self.max_steps)
        encoding = tiktoken.get_encoding(_ENCODING_NAME)

        started = time.time()
        history: list[
            dict
        ] = []  # short, structured log for AgentResult.trajectory/checkpoint reporting
        history_blocks: list[
            str
        ] = []  # full (Thought, Action, Observation) text -- what the prompt actually shows
        final_answer = ""
        termination_reason = "step_budget_exhausted"
        n_inaccessible = 0
        n_history_blocks_dropped = 0

        current_page = env.root_page()
        if current_page.status != "ok":
            n_inaccessible += 1
        history_blocks.append(f"Initial page (root):\n{_format_page(current_page)}")

        budget = self.max_steps
        for step in range(budget):
            prompt = (
                _SYSTEM_PROMPT.format(budget=budget)
                + f"\n\nQuestion: {example.question}\n\n"
                + f"Full navigation history so far:\n{_join_history(history_blocks)}\n\n"
                + f"({step}/{budget} steps used.) Decide your next action."
            )
            while (
                _token_count(prompt, encoding) > self.max_prompt_tokens and len(history_blocks) > 1
            ):
                # Genuine API-safety fallback (see MAX_PROMPT_TOKENS's own
                # docstring) -- drop the OLDEST retained block first, never
                # the current/most-recent page, and never below one block
                # (there is always at least the initial root page).
                history_blocks.pop(0)
                n_history_blocks_dropped += 1
                prompt = (
                    _SYSTEM_PROMPT.format(budget=budget)
                    + f"\n\nQuestion: {example.question}\n\n"
                    + f"Full navigation history so far:\n{_join_history(history_blocks)}\n\n"
                    + f"({step}/{budget} steps used.) Decide your next action."
                )

            response = provider.responses_json(prompt, max_output_tokens=400)
            decision = _loads_json_object(response.text)
            action = decision.get("action") if isinstance(decision, dict) else None
            thought = decision.get("thought") if isinstance(decision, dict) else None
            thought_text = (
                thought.strip() if isinstance(thought, str) and thought.strip() else "(none given)"
            )

            if action == "finish":
                answer = decision.get("answer") if isinstance(decision, dict) else None
                if isinstance(answer, str) and answer.strip():
                    final_answer = answer.strip()
                    termination_reason = "agent_declared_finish"
                    break
                history.append(
                    {"step": step, "action": "finish", "error": "empty/malformed answer"}
                )
                history_blocks.append(
                    f"Step {step}:\nThought: {thought_text}\n"
                    "Action: finish (malformed -- no answer text)\n"
                    "Observation: [action rejected -- an answer must be provided to finish]"
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
                    history_blocks.append(
                        f"Step {step}:\nThought: {thought_text}\n"
                        f"Action: navigate(link_index={link_index!r})\n"
                        "Observation: [invalid link_index -- no navigation occurred; "
                        f"valid indices for the current page are 0-{len(current_page.links) - 1}]"
                    )
                    continue
                link = current_page.links[link_index]
                try:
                    new_page = env.navigate(current_page, link.url)
                except (ValueError, RuntimeError) as exc:
                    history.append({"step": step, "action": "navigate", "error": str(exc)})
                    history_blocks.append(
                        f"Step {step}:\nThought: {thought_text}\n"
                        f'Action: navigate to "{link.text}" ({link.url})\n'
                        f"Observation: [navigation error: {exc}]"
                    )
                    continue
                if new_page.status != "ok":
                    n_inaccessible += 1
                history.append(
                    {
                        "step": step,
                        "action": "navigate",
                        "from_url": current_page.url,
                        "to_url": new_page.url,
                        "link_text": link.text,
                        "status": new_page.status,
                    }
                )
                history_blocks.append(
                    f"Step {step}:\nThought: {thought_text}\n"
                    f'Action: navigate to "{link.text}" ({link.url})\n'
                    f"Observation:\n{_format_page(new_page)}"
                )
                current_page = new_page
                continue

            history.append({"step": step, "action": str(action), "error": "unrecognized action"})
            history_blocks.append(
                f"Step {step}:\nThought: {thought_text}\nAction: {action!r} (unrecognized)\n"
                'Observation: [action not recognized -- must be "navigate" or "finish"]'
            )

        if not final_answer:
            force_prompt = _FORCE_FINISH_PROMPT.format(
                question=example.question, history=_join_history(history_blocks)
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
                "n_history_blocks_dropped": n_history_blocks_dropped,
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(MatchedReActWebAgent())
