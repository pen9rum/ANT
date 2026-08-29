from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import request

from openai import OpenAI

from ant.config import load_dotenv
from ant.domain import (
    AbsenceProof,
    CodeSymbol,
    Evidence,
    FrontierResult,
    NeedGraph,
    NeedNode,
    NeedResolution,
    NodeExecutionTrace,
    PlanningRound,
    RepairAction,
    RepairPlan,
    RoundPlan,
    TaskTrajectoryPackage,
    TokenUsage,
    UnresolvedNeed,
    WorkerCard,
    WorkerObservation,
)
from ant.indexing.cards import template_routing_summary
from ant.providers.pricing import estimate_cost_usd
from ant.tools.symbol_index import build_symbol_index

# 512/768 was cutting detailed technical answers off mid-sentence (e.g. a
# call-path walk-through truncated inside a code block). 8192 gives ample
# room for a full multi-part answer without meaningfully raising baseline
# cost -- this is a ceiling, not a floor, so short answers still cost the
# same.
SYNTHESIS_MAX_OUTPUT_TOKENS = 8192

# 30s was tight enough to occasionally time out on an otherwise-successful
# reasoning-effort call (observed directly: two separate questions across a
# 20-question batch each lost their entire answer to "TimeoutError: The read
# operation timed out", not to any actual API error) -- a single slow
# response killed that example outright with no retry. 120s gives a
# reasoning-effort call realistic headroom.
#
# Note this is enforced as a genuine wall-clock deadline (see
# _call_with_hard_timeout), not just urlopen's own socket-level read
# timeout: observed directly that a stuck request can hold a live,
# Established TCP connection open for 30+ minutes without urlopen's
# timeout=N ever firing -- consistent with the server (or an intermediate
# proxy) periodically sending something on the wire to keep the connection
# alive during a long reasoning pass, which resets a read timeout on every
# partial receive without ever completing the response. A hard deadline
# from a separate thread has no such loophole.
REQUEST_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    model: str
    reasoning_effort: str | None = None
    organization: str | None = None
    project: str | None = None


@dataclass(frozen=True)
class ResponseResult:
    text: str
    usage: TokenUsage
    raw: dict


class OpenAIProvider:
    """Thin wrapper so orchestration code does not depend directly on the SDK."""

    def __init__(self, model: str | None = None, reasoning_effort: str | None = None) -> None:
        env_values = load_dotenv()
        # OPENAI_API_KEY/OPENAI_ORG_ID/OPENAI_PROJECT_ID are a matched set
        # that must come from the same source together -- prefer .env's
        # own values for all three when .env defines them, rather than
        # os.getenv (which, now that load_dotenv defaults to override=
        # False, could source one of the three from .env and another from
        # an unrelated, stale OS-level env var). Confirmed directly: a
        # stale OS-level OPENAI_API_KEY paired with .env's own
        # OPENAI_ORG_ID -- a key and an organization that don't belong
        # together -- failed every request with 401
        # "mismatched_organization" (see load_dotenv's docstring).
        # ANT_MODEL is deliberately NOT part of this trio: a caller setting
        # os.environ["ANT_MODEL"] before construction is a real, intended
        # override this project's scripts rely on, not a stale leftover.
        self.settings = OpenAISettings(
            api_key=env_values.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
            model=model or os.getenv("ANT_MODEL", "gpt-5.4-nano"),
            reasoning_effort=reasoning_effort,
            organization=env_values.get("OPENAI_ORG_ID") or os.getenv("OPENAI_ORG_ID"),
            project=env_values.get("OPENAI_PROJECT_ID") or os.getenv("OPENAI_PROJECT_ID"),
        )
        self.model = self.settings.model
        self.reasoning_effort = self.settings.reasoning_effort
        self._usage = TokenUsage()

    def is_configured(self) -> bool:
        return bool(self.settings.api_key)

    def require_configured(self) -> None:
        if not self.is_configured():
            msg = "OPENAI_API_KEY is not set."
            raise RuntimeError(msg)

    def client(self) -> OpenAI:
        self.require_configured()
        return OpenAI(
            api_key=self.settings.api_key,
            organization=self.settings.organization,
            project=self.settings.project,
        )

    def smoke_test(self, prompt: str = "Reply exactly: OK") -> str:
        return self.responses_text(prompt, max_output_tokens=16).text

    def responses_text(self, prompt: str, max_output_tokens: int = 512) -> ResponseResult:
        start = time.perf_counter()
        if self.settings.organization and self.settings.project:
            result = self._responses_create_raw(prompt, max_output_tokens=max_output_tokens)
            return self._record_result(_with_latency_and_cost(result, self.model, start))
        request_kwargs = self._responses_kwargs(prompt, max_output_tokens)
        response = self.client().responses.create(**request_kwargs)
        raw = response.model_dump()
        return self._record_result(
            _with_latency_and_cost(
                ResponseResult(
                    text=response.output_text,
                    usage=_extract_usage(raw),
                    raw=raw,
                ),
                self.model,
                start,
            )
        )

    def drain_usage(self) -> TokenUsage:
        usage = self._usage
        self._usage = TokenUsage()
        return usage

    def _record_result(self, result: ResponseResult) -> ResponseResult:
        self._usage = TokenUsage(
            input_tokens=self._usage.input_tokens + result.usage.input_tokens,
            output_tokens=self._usage.output_tokens + result.usage.output_tokens,
            total_tokens=self._usage.total_tokens + result.usage.total_tokens,
            latency_ms=self._usage.latency_ms + result.usage.latency_ms,
            estimated_cost_usd=round(
                self._usage.estimated_cost_usd + result.usage.estimated_cost_usd,
                8,
            ),
        )
        return result

    def responses_json(self, prompt: str, max_output_tokens: int = 512) -> ResponseResult:
        result = self.responses_text(prompt, max_output_tokens=max_output_tokens)
        if _is_json_object(result.text):
            return result
        repair_prompt = (
            "Convert the following response into one valid JSON object. "
            "Return only JSON, no markdown.\n"
            f"Response:\n{result.text}"
        )
        repaired = self.responses_text(repair_prompt, max_output_tokens=max_output_tokens)
        if _is_json_object(repaired.text):
            return repaired
        # The repair pass can itself come back malformed (most often the
        # model's output was cut off by max_output_tokens). Every caller of
        # this method does an unguarded json.loads on the result, so
        # returning unparseable text here would crash the caller -- and for
        # a batch eval, that kills every remaining example, not just this
        # one. An empty object is a safe degradation: callers already treat
        # "no fields present" as "nothing found this round".
        return ResponseResult(text="{}", usage=repaired.usage, raw=repaired.raw)

    def _responses_create_raw(self, prompt: str, max_output_tokens: int) -> ResponseResult:
        payload = json.dumps(self._responses_kwargs(prompt, max_output_tokens)).encode("utf-8")
        api_request = request.Request(
            "https://api.openai.com/v1/responses",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "OpenAI-Organization": self.settings.organization or "",
                "OpenAI-Project": self.settings.project or "",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        def _do_request() -> dict:
            with request.urlopen(api_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return json.loads(response.read().decode("utf-8"))

        try:
            data = _call_with_hard_timeout(_do_request, REQUEST_TIMEOUT_SECONDS)
        except TimeoutError:
            # One retry for a genuinely transient stall; a second
            # consecutive timeout is left to propagate rather than retried
            # again, so a real outage still surfaces as a failure instead of
            # silently stalling this example even longer.
            data = _call_with_hard_timeout(_do_request, REQUEST_TIMEOUT_SECONDS)
        return ResponseResult(text=_extract_output_text(data), usage=_extract_usage(data), raw=data)

    def _responses_kwargs(self, prompt: str, max_output_tokens: int) -> dict:
        kwargs: dict = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        return kwargs

    def generate_card(self, *, repo_root: str, territory_root: str, files: list[str]) -> WorkerCard:
        file_sample = "\n".join(files[:80])
        prompt = (
            "Create a concise JSON worker card for a codebase territory.\n"
            f"Repository: {repo_root}\n"
            f"Territory root: {territory_root or 'repository root'}\n"
            f"Files:\n{file_sample}\n"
            "Return JSON with keys: name, responsibilities, searchable_terms.\n"
        )
        result = self.responses_json(prompt, max_output_tokens=512)
        data = _loads_json_object(result.text)
        root = territory_root
        territory_id = root or "root"
        symbols = _owned_symbols(Path(repo_root), files)
        card = WorkerCard(
            id=f"worker-{territory_id}",
            territory_id=territory_id,
            name=str(data.get("name") or f"{root or 'root'} worker"),
            root=root,
            responsibilities=[str(item) for item in data.get("responsibilities", [])][:8],
            searchable_terms=[str(item).lower() for item in data.get("searchable_terms", [])][:24],
            files=files,
            symbols=symbols,
        )
        return card.model_copy(update={"routing_summary": self.summarize_routing(card=card)})

    def summarize_routing(self, *, card: WorkerCard) -> str:
        responsibilities = "; ".join(card.responsibilities[:4]) or "(no description)"
        terms = ", ".join(card.searchable_terms[:16])
        prompt = (
            "Summarize this codebase worker's territory for a routing index "
            "that an orchestrator reads every round for every worker in the "
            "colony -- it must stay short. Return exactly three short "
            "pieces of information joined with ' | ':\n"
            "territory: <what part of the codebase this worker owns>\n"
            "capability: <what it can actually find/do there>\n"
            "typical needs: <the kinds of questions/gaps this worker resolves>\n"
            f"Worker id: {card.id}\n"
            f"Responsibilities: {responsibilities}\n"
            f"Terms: {terms}\n"
            "Return JSON with key summary: a single string with all three "
            "pieces, e.g. "
            '"territory: ... | capability: ... | typical needs: ...".'
        )
        result = self.responses_json(prompt, max_output_tokens=200)
        data = _loads_json_object(result.text)
        summary = str(data.get("summary") or "").strip()
        # Malformed/empty response: degrade to the same zero-cost template
        # used when no LLM is available at all, rather than leaving
        # routing_summary blank (which would make this worker invisible in
        # the Orchestrator's routing-relevant context every round).
        return summary or template_routing_summary(card)

    def observe(
        self,
        *,
        question: str,
        worker_id: str,
        territory_id: str,
        evidence: list[Evidence],
    ) -> WorkerObservation:
        evidence_text = "\n".join(
            f"[{index}] {item.path}:{item.line_start}-{item.line_end}\n{item.quote[:1200]}"
            for index, item in enumerate(evidence[:8])
        )
        prompt = (
            "Identify at most one grounded semantic knowledge gap for a code navigation task. "
            "A gap is missing information needed to answer the original question after considering "
            "the evidence; it is not a tool failure, lack of confidence, or budget condition. "
            "Return an empty unresolved_needs list if the evidence already supports an answer.\n"
            f"Question: {question}\n"
            f"Worker: {worker_id}\n"
            f"Territory: {territory_id}\n"
            f"Evidence:\n{evidence_text}\n"
            "Return JSON with key unresolved_needs as a list of objects with "
            "known (list of grounded claims), missing (specific missing relation), "
            "need_type (one of subclass_lookup, call_path, implementation_location, "
            "behavior_flow, negative_presence, data_flow, unknown), scope "
            "(local, cross_territory, or unknown), evidence_ids (list of evidence indices), "
            "relevant_symbols, suggested_terms, and suggested_territories. "
            "description should equal missing.\n"
        )
        result = self.responses_json(prompt, max_output_tokens=512)
        data = _loads_json_object(result.text)
        needs = data.get("unresolved_needs", [])
        unresolved_needs = [
            UnresolvedNeed(
                description=str(item.get("description", "")),
                kind=str(item.get("kind", "missing_detail")),
                need_type=str(item.get("need_type", "unknown")),
                known=[str(claim) for claim in item.get("known", [])][:4],
                missing=str(item.get("missing", item.get("description", ""))),
                scope=str(item.get("scope", "unknown")),
                source_worker_id=worker_id,
                evidence_ids=[str(value) for value in item.get("evidence_ids", [])][:8],
                relevant_symbols=[str(symbol) for symbol in item.get("relevant_symbols", [])],
                suggested_terms=[str(term) for term in item.get("suggested_terms", [])],
                suggested_territories=[str(term) for term in item.get("suggested_territories", [])],
            )
            for item in needs
            if isinstance(item, dict)
        ]
        return WorkerObservation(
            worker_id=worker_id,
            territory_id=territory_id,
            unresolved_needs=unresolved_needs,
        )

    def select_lookups(
        self,
        *,
        need: str,
        evidence: list[Evidence],
        candidates: list[str],
    ) -> list[str]:
        # Replaces a purely mechanical filter (previously: keep a token if
        # it's capitalized or underscored) with real judgment. That
        # heuristic had no way to tell a real code symbol from an ordinary
        # capitalized word in free-text -- confirmed directly: a need
        # mentioning "Information about the documentation build system"
        # got "Information" treated as a lookup-worthy symbol name, wasting
        # a tool call and, combined with several other misfires, starving
        # the round of budget for the lookup that would have actually
        # helped. This costs one extra LLM call per worker per round (not
        # per candidate/tool call) to keep the increase bounded.
        if not candidates:
            return []
        evidence_text = "\n".join(
            f"[{index}] {item.path}:{item.line_start}-{item.line_end}\n{item.quote[:600]}"
            for index, item in enumerate(evidence[:6])
        )
        prompt = (
            "You are deciding which candidate strings are worth a follow-up "
            "code lookup (definition, callers, or usages) -- not answering "
            "the question yet. Some candidates are real code symbols "
            "(classes, functions, constants); others are ordinary words "
            "pulled from free text that only happen to look like a symbol "
            "(e.g. capitalized, or an underscore-joined phrase). Keep only "
            "candidates that plausibly name an actual code identifier "
            "relevant to the need below, ordered by how likely a lookup on "
            "them is to help. Drop anything that reads like a normal English "
            "word or sentence fragment.\n"
            f"Need: {need}\n"
            f"Evidence so far:\n{evidence_text}\n"
            f"Candidates: {candidates}\n"
            "Return JSON with key selected: an ordered list of the chosen "
            "candidate strings, copied exactly from Candidates -- no new "
            "names, no explanations."
        )
        result = self.responses_json(prompt, max_output_tokens=256)
        raw = _extract_selected_list(result.text)
        if raw is None:
            # Malformed/empty model response: degrade to the old unfiltered
            # behavior rather than silently exploring nothing this round.
            return candidates
        candidate_set = set(candidates)
        return [item for item in raw if isinstance(item, str) and item in candidate_set]

    def select_evidence(
        self,
        *,
        question: str,
        evidence: list[Evidence],
        limit: int,
    ) -> tuple[list[str], list[str]]:
        # Replaces the final fixed score-based evidence[:12] cut with real
        # judgment. Per-worker collection no longer does its own filtering
        # pass (AutonomousWorker's own evidence_limit is now a generous
        # safety cap, not a relevance decision) -- this is the one place
        # that decides what the synthesizer actually sees, so it is the one
        # place worth spending an LLM call on rather than a fixed top-N
        # slice by score alone. Also asks in the same call which kept items
        # have a quote too narrow to answer from and should be reopened to
        # their full source region first (Shared Evidence State: "evidence
        # compression is reversible") -- this used to only exist for the
        # coalition cross-check branch, never for the common single-worker
        # path or the final synthesis gate.
        if not evidence:
            return [], []
        evidence_lines = [
            f"[{index}] {item.path}:{item.line_start}-{item.line_end} "
            f"(worker={item.worker_id or 'unknown'})\n{item.quote[:900]}"
            for index, item in enumerate(evidence)
        ]
        prompt = (
            "You are choosing which pieces of grounded evidence are actually "
            "needed to answer the question below -- not answering it yet. "
            "Keep everything that contributes a distinct fact, location, or "
            "step toward a complete, correct answer; drop only evidence that "
            "is redundant with something already kept or genuinely "
            "irrelevant to the question. Do not drop something just to keep "
            "the list short -- an incomplete answer is a worse outcome than "
            f"a longer one, up to a hard cap of {limit} items.\n"
            "First identify the concrete technical concepts, symbols, and "
            "terms the question is actually asking about (e.g. a question "
            "about reproducibility across backends is asking about specific "
            "things like seeding/sampling functions, not the general "
            "execution flow that happens to surround them). Order `selected` "
            "by how directly each item addresses one of those concrete "
            "concepts, most-directly-relevant first -- not by how "
            "prominent, well-documented, or narratively convenient the item "
            "reads on its own. An item that names or implements one of the "
            "question's own concrete concepts outranks a generic, "
            "well-written item that only touches the surrounding area.\n"
            "Separately, for each kept item whose shown quote is too narrow "
            "to actually answer from (e.g. it shows a signature or "
            "docstring but cuts off before the logic that matters), flag it "
            "for expansion so its full surrounding region gets reopened "
            "before the answer is written.\n"
            f"Question: {question}\n"
            f"Evidence:\n{chr(10).join(evidence_lines)}\n"
            f"Return JSON with two keys: selected (an ordered list of up to "
            f"{limit} evidence indices, as strings, most important first) and "
            "expand (a list, possibly empty, of indices from within selected "
            "whose region should be reopened) -- indices copied exactly from "
            "the [N] markers above, no new indices, no explanations."
        )
        result = self.responses_json(prompt, max_output_tokens=768)
        valid_indices = {str(index) for index in range(len(evidence))}
        selected_raw = _extract_list_field(result.text, "selected")
        if selected_raw is None:
            # Malformed/empty model response: degrade to the pool's
            # existing score order rather than silently handing the
            # synthesizer nothing.
            return [str(index) for index in range(len(evidence))][:limit], []
        selected = [
            item for item in selected_raw if isinstance(item, str) and item in valid_indices
        ][:limit]
        expand_raw = _extract_list_field(result.text, "expand") or []
        selected_set = set(selected)
        expand = [
            item for item in expand_raw if isinstance(item, str) and item in selected_set
        ]
        return selected, expand

    def plan_worker_actions(
        self,
        *,
        need: str,
        evidence: list[Evidence],
        candidate_symbols: list[str],
        available_tools: list[str],
        hints: list[str],
        max_actions: int,
    ) -> list[tuple[str, str]]:
        # Replaces the old fixed tool sequence (always run every tool
        # against every candidate symbol, in a hardcoded order, until a
        # call-count budget ran out) with one planning call: the reasoner
        # decides which (tool, symbol) lookups are actually worth running
        # and when it already has enough, instead of exhausting the budget
        # by rote regardless of whether earlier steps already answered the
        # need. max_actions is a hard ceiling (see WorkerRunConfig), not a
        # target -- an empty plan is a legitimate answer ("already enough").
        if not candidate_symbols or max_actions <= 0:
            return []
        evidence_text = "\n".join(
            f"[{index}] {item.path}:{item.line_start}-{item.line_end}\n{item.quote[:600]}"
            for index, item in enumerate(evidence[:8])
        )
        tool_descriptions = {
            "navigate": "jump to a symbol's own definition/implementation block",
            "references": "find local references/call sites for a symbol",
            "callers": "find blocks that call/invoke a symbol",
            "callees": "find helpers a symbol's implementation calls into",
            "assignments": "trace local data-flow assignments/uses of a symbol",
            "imports": "resolve how a symbol is imported, useful before tracing call/data flow",
            "subclasses": "find class definitions that inherit from a symbol",
        }
        tools_text = "\n".join(
            f"- {tool}: {tool_descriptions.get(tool, '')}" for tool in available_tools
        )
        hints_text = "".join(f"Hint: {hint}\n" for hint in hints)
        prompt = (
            "You are planning follow-up code lookups (not answering yet) for the "
            "need below, given the evidence already found. Pick specific (tool, "
            "symbol) pairs -- do not reflexively run every tool on every symbol; "
            "skip a lookup if the evidence already answers what it would tell "
            "you, and stop planning once the need is answerable. Return an empty "
            "list if the evidence already suffices.\n"
            f"Need: {need}\n"
            f"{hints_text}"
            f"Evidence so far:\n{evidence_text}\n"
            f"Available tools:\n{tools_text}\n"
            f"Candidate symbols: {candidate_symbols}\n"
            f"Return JSON with key actions: an ordered list (most useful first, "
            f"at most {max_actions} items) of objects with keys tool and symbol, "
            "both copied exactly from the lists above -- no new tools, no new "
            "symbols, no explanations."
        )
        result = self.responses_json(prompt, max_output_tokens=768)
        raw = _extract_list_field(result.text, "actions")
        if raw is None:
            # Malformed/empty model response: degrade to the old default
            # order (every tool against every symbol) rather than silently
            # running nothing this round.
            default_tools = [
                tool
                for tool in ("navigate", "references", "callers", "callees", "assignments")
                if tool in available_tools
            ]
            return [
                (tool, symbol) for symbol in candidate_symbols for tool in default_tools
            ][:max_actions]
        tool_set = set(available_tools)
        symbol_set = set(candidate_symbols)
        plan: list[tuple[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            tool = item.get("tool")
            symbol = item.get("symbol")
            if tool in tool_set and symbol in symbol_set:
                plan.append((tool, symbol))
        return plan[:max_actions]

    def should_continue_recruiting(
        self,
        *,
        question: str,
        need: UnresolvedNeed,
        evidence: list[Evidence],
        rounds_completed: int,
    ) -> bool:
        # Replaces the old fixed max_rounds=2 cap -- which stopped
        # recruiting after exactly N rounds regardless of whether the need
        # was actually resolved or the colony genuinely had no better
        # worker left to try -- with a per-round judgment call. max_rounds
        # is still a hard safety ceiling (see LocalCoordinator.ask), this
        # is what decides whether it's worth spending another round *before*
        # hitting that ceiling.
        evidence_text = "\n".join(
            f"[{index}] {item.path}:{item.line_start}-{item.line_end} "
            f"(worker={item.worker_id or 'unknown'})\n{item.quote[:600]}"
            for index, item in enumerate(evidence[:10])
        )
        prompt = (
            "A code-navigation task has an unresolved information need after "
            f"{rounds_completed} round(s) of recruiting workers. Decide whether "
            "recruiting one more worker for this specific need is worth it, or "
            "whether the evidence gathered so far already suggests no worker in "
            "this codebase will resolve it (e.g. the need concerns something "
            "that plausibly does not exist here, or every plausible territory "
            "has already been tried without progress).\n"
            f"Original question: {question}\n"
            f"Unresolved need: {need.missing or need.description} "
            f"(need_type={need.need_type}, scope={need.scope})\n"
            f"Evidence gathered so far:\n{evidence_text or '(none)'}\n"
            "Return JSON with key continue: true to recruit another worker for "
            "this need, false to stop here."
        )
        result = self.responses_json(prompt, max_output_tokens=128)
        data = _loads_json_object(result.text)
        value = data.get("continue")
        if isinstance(value, bool):
            return value
        # Malformed/empty response: degrade to the old unconditional
        # behavior (keep going until the caller's max_rounds ceiling)
        # rather than silently cutting recruitment short on a parse error.
        return True

    def check_need_resolution(
        self,
        *,
        need: UnresolvedNeed,
        new_evidence: list[Evidence],
        question: str,
    ) -> NeedResolution:
        # Generalizes the old heuristic closure (_CLOSABLE_BY_EVIDENCE in
        # coordinator/local.py), which only understood 3 of ~6 need_types --
        # subclass_lookup/implementation_location/source_test_coalition --
        # by pattern-matching a `class X`/`def X` quote. Every other
        # need_type (data_flow, call_path, behavior_flow, negative_presence,
        # unknown -- the majority) was never auto-closed at all, so it just
        # got re-raised, cosmetically reworded, round after round. This
        # judges every need_type the same way: did the evidence gathered
        # *since* this need was raised (not the whole accumulated pool --
        # that would mark a need "resolved" by evidence a much earlier round
        # already had and had already failed to resolve it with) actually
        # satisfy it, partially narrow it down, or leave it no better off.
        if not new_evidence:
            return NeedResolution(status="unresolved")
        evidence_text = "\n".join(
            f"[{index}] {item.path}:{item.line_start}-{item.line_end}\n{item.quote[:900]}"
            for index, item in enumerate(new_evidence[:8])
        )
        prompt = (
            "Judge whether the NEW evidence below actually resolves the "
            "unresolved need, given the original question. Three outcomes:\n"
            "- resolved: the new evidence directly answers what was missing.\n"
            "- partial: the new evidence narrows the gap (e.g. rules out one "
            "possibility, points at the right file/module without the exact "
            "symbol, or answers half of a two-part need) but does not fully "
            "answer it -- in this case also produce a refined need that is "
            "MORE SPECIFIC than the original given what was just learned, not "
            "a restatement of it.\n"
            "- unresolved: the new evidence does not meaningfully advance this "
            "need at all.\n"
            f"Original question: {question}\n"
            f"Unresolved need: {need.missing or need.description} "
            f"(need_type={need.need_type}, scope={need.scope})\n"
            f"New evidence since this need was raised:\n{evidence_text}\n"
            "Return JSON with key status (one of resolved, partial, "
            "unresolved) and, only when status is partial, key refined_need: "
            "an object with description, missing, need_type, scope, "
            "suggested_terms (list), and suggested_territories (list), "
            "using the same conventions as the original need but sharpened by "
            "what the new evidence showed."
        )
        result = self.responses_json(prompt, max_output_tokens=512)
        data = _loads_json_object(result.text)
        status = data.get("status")
        if status not in ("resolved", "partial", "unresolved"):
            # Malformed/empty response: degrade to the conservative default
            # (treat as still open) rather than guessing a need is resolved
            # on a parse error.
            return NeedResolution(status="unresolved")
        if status != "partial":
            return NeedResolution(status=status)
        raw_refined = data.get("refined_need")
        if not isinstance(raw_refined, dict):
            # "partial" without a usable refined need degrades to
            # "unresolved" rather than silently dropping the original need
            # with nothing to replace it.
            return NeedResolution(status="unresolved")
        refined_need = UnresolvedNeed(
            description=str(raw_refined.get("description") or need.description),
            kind=need.kind,
            need_type=str(raw_refined.get("need_type") or need.need_type),
            missing=str(raw_refined.get("missing") or raw_refined.get("description") or ""),
            scope=str(raw_refined.get("scope") or need.scope),
            source_worker_id=need.source_worker_id,
            suggested_terms=[str(term) for term in raw_refined.get("suggested_terms", [])],
            suggested_territories=[
                str(territory) for territory in raw_refined.get("suggested_territories", [])
            ],
            relevant_symbols=need.relevant_symbols,
        )
        return NeedResolution(status="partial", refined_need=refined_need)

    def plan_round(
        self,
        *,
        question: str,
        graph: NeedGraph,
        resolution_results: dict[str, NeedResolution],
        evidence: list[Evidence],
        workers: list[WorkerCard],
        memory_hints: dict[str, str],
        frontier: FrontierResult,
        observed_needs: list[UnresolvedNeed],
        incomplete_parents: list[str],
        cross_repo_experience: list[str],
        validation_feedback: str = "",
        repair_guidance: str = "",
        stuck_tried_workers: dict[str, list[str]] | None = None,
        worker_relevance_rank: dict[str, int] | None = None,
    ) -> RoundPlan:
        # The single per-round Orchestrator call: replaces select_workers/
        # decide_local_action and the hand-coded escalation ladder
        # together (see WorkerReasoner.plan_round's docstring). Shows
        # every worker's routing_summary with no relevance-based
        # prefiltering (WorkerCard.routing_summary exists precisely to
        # make that affordable) and every piece of accumulated evidence
        # with no pool cap (same principle as _select_evidence -- verified
        # empirically that a score-ranked cap discards real evidence on
        # wide-scope questions), each individually truncated for prompt
        # size instead.
        #
        # worker_lines also shows a handful of each worker's own
        # searchable_terms, not just routing_summary. Confirmed missing on
        # a real qibo trace: a worker's searchable_terms contained "bloch",
        # "sphere", "paint_world_map" -- an exact lexical hit for a
        # question about Bloch sphere visualization -- but its
        # routing_summary (an LLM-compressed natural-language sentence)
        # read as "clarifies classifier concepts, usage examples, and test
        # or algorithm gaps", with zero surface overlap with the question.
        # The Orchestrator never assigned that worker directly across 5 of
        # 6 rounds; only global_fallback's repo-wide search (a last-resort
        # recovery tactic, not normal routing) found it, on the final
        # round. routing_summary's own compression is exactly what drops
        # the literal terms an Orchestrator's own lexical judgment could
        # otherwise catch.
        graph_lines = [_node_prompt_line(node) for node in graph.nodes.values()]
        resolution_lines = [
            f"[{need_id}] status={resolution.status}"
            + (
                f" refined_need={resolution.refined_need.description}"
                if resolution.refined_need
                else ""
            )
            for need_id, resolution in resolution_results.items()
        ]
        evidence_lines = [
            f"[{index}] {item.path}:{item.line_start}-{item.line_end} "
            f"(worker={item.worker_id or 'unknown'})\n{item.quote[:600]}"
            for index, item in enumerate(evidence)
        ]
        worker_relevance_rank = worker_relevance_rank or {}
        # Ranked workers first (best retrieval match first), unranked ones
        # keep their original relative order after -- every worker is still
        # listed, nothing excluded, only reordered/annotated. See
        # ant.coordinator.worker_retrieval.rank_workers: this rank comes
        # from BM25 + exact-symbol + dense retrieval over WorkerCard.symbols
        # (the AST's real, non-truncated definition list), not the lossy
        # round-robin-truncated searchable_terms this prompt used to rely
        # on alone -- confirmed on a real qibo worker that a defining class
        # (FALQON) sat at position 15 of a 48-term list, past the [:12]
        # slice below, invisible to every prior version of this prompt.
        ordered_workers = sorted(
            workers,
            key=lambda worker: worker_relevance_rank.get(worker.id, len(workers) + 1),
        )
        worker_lines = [
            f"- {worker.id}"
            + (
                f" (retrieval rank {worker_relevance_rank[worker.id]}/{len(workers)})"
                if worker.id in worker_relevance_rank
                else ""
            )
            + f": {worker.routing_summary or '(no routing summary)'}"
            + (
                f"\n  terms: {', '.join(worker.searchable_terms[:12])}"
                if worker.searchable_terms
                else ""
            )
            + (f"\n  memory: {memory_hints[worker.id]}" if worker.id in memory_hints else "")
            for worker in ordered_workers
        ]
        stuck_tried_workers = stuck_tried_workers or {}
        stuck_lines = [
            f"- subgraph {index}: {', '.join(group)}"
            + "".join(
                f"\n    {need_id} already tried with no progress: "
                f"{', '.join(stuck_tried_workers[need_id])}"
                for need_id in group
                if stuck_tried_workers.get(need_id)
            )
            for index, group in enumerate(frontier.stuck_subgraphs)
        ]
        observed_need_lines = [
            f"[{index}] {need.missing or need.description} (need_type={need.need_type}, "
            f"scope={need.scope})"
            for index, need in enumerate(observed_needs)
        ]
        experience_lines = [f"- {experience}" for experience in cross_repo_experience]
        feedback_block = (
            f"\nYour previous graph_updates for this round were rejected: "
            f"{validation_feedback}\nRevise and try again.\n"
            if validation_feedback
            else ""
        )
        repair_block = f"\n{repair_guidance}\n" if repair_guidance else ""
        prompt = (
            "You are the Orchestrator for one round of a codebase-QA "
            "task's Need Graph. You decide exactly three things, and "
            "nothing else -- never resolution/execution/progress status, "
            "those are computed elsewhere:\n"
            "1. graph_updates: create/decompose need nodes and edit their "
            "depends_on/related_to/children edges. A node's need_id, once "
            "it exists, is permanent -- never reuse an existing id to mean "
            "a different underlying gap; when a node is too coarse, add "
            "children under it instead of replacing it.\n"
            "2. assignments: which worker id(s) handle each ready-frontier "
            "need_id this round. One worker id is a plain follow-up/"
            "handoff; more than one is a coalition -- same kind of entry "
            "either way.\n"
            "3. special_tactics: ONLY for a need_id inside one of the "
            "listed stuck subgraphs, and only if its recovery plan needs "
            "one of exactly two special mechanisms: temporary_bridge "
            "(spin up an ephemeral worker spanning the tried territories) "
            "or global_fallback (an unscoped repository-wide search). "
            "Every other kind of recovery (reassign to a different "
            "worker, redecompose the need, form a coalition) is just an "
            "ordinary graph_updates/assignments entry, no special tactic "
            "needed. A stuck subgraph line below may list, per need_id, "
            "which workers were already tried on it with no progress -- "
            "assigning the exact same worker(s) again is expected to be a "
            "deliberate choice (e.g. that worker now has a narrower, "
            "different sub-need), not the default; prefer a different "
            "worker, a coalition, or one of the two special tactics "
            "instead unless you have a specific reason to repeat.\n"
            "4. For each item in 'Observed needs' below: decide whether to "
            "create a new graph node from it (via graph_updates), fold its "
            "content into an existing node you're editing, or leave it "
            "alone -- then list its index in resolved_observed_need_indices "
            "if you did anything with it (created a node, merged it, or "
            "deliberately decided to discard it); leave its index out to "
            "keep it pending for a future round.\n"
            "5. For each id in 'Parents needing more decomposition' below: "
            "its children are all resolved but its own closure check says "
            "the decomposition still doesn't cover its original scope -- "
            "add more children under it via graph_updates (or otherwise "
            "revise it), it is not directly assignable.\n"
            f"{feedback_block}"
            f"{repair_block}"
            f"Question: {question}\n"
            f"Current graph:\n{chr(10).join(graph_lines) or '(empty)'}\n"
            f"This round's resolution results:\n"
            f"{chr(10).join(resolution_lines) or '(none yet)'}\n"
            f"Ready frontier (assignable this round): {', '.join(frontier.ready) or '(none)'}\n"
            f"Blocked: {', '.join(frontier.blocked) or '(none)'}\n"
            f"Stuck subgraphs needing a recovery plan:\n"
            f"{chr(10).join(stuck_lines) or '(none)'}\n"
            f"Parents needing more decomposition: {', '.join(incomplete_parents) or '(none)'}\n"
            f"Observed needs (act on these, see instruction 4):\n"
            f"{chr(10).join(observed_need_lines) or '(none)'}\n"
            f"Workers:\n{chr(10).join(worker_lines)}\n"
            f"Evidence:\n{chr(10).join(evidence_lines) or '(none yet)'}\n"
            "Patterns from OTHER repos' past tasks (reference only -- judge "
            "for yourself whether any of this actually applies here, it is "
            "not a rule you must follow):\n"
            f"{chr(10).join(experience_lines) or '(none)'}\n"
            "Return JSON with keys graph_updates (a list of objects with "
            "need_id, need, depends_on, related_to, children -- omit any "
            "you're not changing; children lists other need_ids already "
            "present in this same graph_updates list or the current "
            "graph), assignments (an object mapping need_id to a list of "
            "worker ids, ready-frontier or stuck-subgraph-member need_ids "
            "only), special_tactics (an object mapping need_id to exactly "
            "\"temporary_bridge\" or \"global_fallback\", only for the two "
            "special cases above), and resolved_observed_need_indices (a "
            "list of indices, as strings, from Observed needs that you "
            "acted on this round)."
        )
        result = self.responses_json(prompt, max_output_tokens=2048)
        data = _loads_json_object(result.text)
        return _parse_round_plan(data, graph=graph, workers=workers)

    def propose_repair(self, *, package: TaskTrajectoryPackage) -> RepairPlan:
        # Task-conditioned ("fast") evolution's one reasoning call --
        # judges a single finished task's own trajectory, never a
        # reference answer or judge score (package carries neither). See
        # FastEvolutionReasoner's docstring for how this differs from
        # EvolutionReasoner.decide_episode_action (that one only acts on
        # patterns aggregated across many tasks and mutates the persistent
        # colony; this is ephemeral and task-scoped).
        node_lines = []
        for node in package.stuck_nodes:
            node_lines.append(
                f"[{node.need_id}] {node.need} (resolution={node.resolution}"
                + (", ABANDONED" if node.is_abandoned else "")
                + (f", stuck_episode={node.stuck_episode_id}" if node.stuck_episode_id else "")
                + ")\n"
                f"  depends_on={node.depends_on} children={node.children}\n"
                f"  missing: {node.missing or '(unspecified)'}\n"
                f"  suggested_terms: {', '.join(node.suggested_terms) or '(none)'}\n"
                f"  suggested_territories: {', '.join(node.suggested_territories) or '(none)'}\n"
                f"  tried workers: {', '.join(node.tried_worker_ids) or '(none)'}\n"
                f"  tried special tactics: {', '.join(node.tried_special_tactics) or '(none)'}\n"
                f"  no-progress executions: {node.no_progress_execution_count}\n"
                f"  evidence already gathered: "
                + ("; ".join(node.evidence_claims[:8]) or "(none)")
            )
        decomposition_lines = []
        for round_index, delta in enumerate(package.graph_decomposition_log):
            if not (
                delta.created_nodes
                or delta.dependency_changes
                or delta.created_children
                or delta.closure_results
            ):
                continue
            decomposition_lines.append(
                f"round {round_index}: created={delta.created_nodes} "
                f"dependency_changes={delta.dependency_changes} "
                f"new_children={delta.created_children} closed={delta.closure_results}"
            )
        prompt = (
            "A prior attempt at answering this codebase question got "
            "stuck or was abandoned on some of its sub-needs. Propose an "
            "ephemeral, task-local repair plan for a SECOND attempt at "
            "this exact question -- this is not colony reorganization, "
            "nothing you propose is persistent or applies to any other "
            "question.\n"
            "The candidate answer produced by the prior attempt is given "
            "below as CONTEXT ONLY -- it is not a correct-answer key, "
            "there is no reference answer or score available to you, and "
            "you must not judge whether it was right. Reason only from "
            "what the trajectory below shows was tried and what it "
            "actually found.\n"
            "Every action you propose actually happens, not just a "
            "suggestion the retry may or may not follow: "
            "change_dependency/redecompose/merge_needs directly edit the "
            "retry's starting graph before it begins, and "
            "reuse_assignment/replace_assignment/form_local_bridge/"
            "force_global_search are force-executed exactly once at the "
            "retry's very first round before the Orchestrator regains its "
            "normal freedom -- so only propose an action where the "
            "trajectory below actually supports it, not as a guess.\n"
            f"Question: {package.question}\n"
            f"Prior attempt's answer (context only, not supervision): "
            f"{package.prior_answer or '(none produced)'}\n"
            f"Stuck/unresolved/abandoned needs:\n"
            f"{chr(10).join(node_lines) or '(none -- nothing was stuck)'}\n"
            f"How the graph was decomposed across rounds:\n"
            f"{chr(10).join(decomposition_lines) or '(no structural changes)'}\n"
            "Return JSON with key actions: a list of objects, each with "
            "kind (one of reuse_assignment, replace_assignment, "
            "merge_needs, redecompose, change_dependency, "
            "form_local_bridge, force_global_search), need_id (which "
            "stuck need this targets), and as applicable: worker_ids "
            "(list of worker ids, REQUIRED for reuse_assignment/"
            "replace_assignment/form_local_bridge or the action is "
            "dropped -- for replace_assignment, suggest a DIFFERENT "
            "worker than 'tried workers' already tried and failed; for "
            "form_local_bridge, 2+ worker ids to run together as a "
            "one-off coalition), merge_with (list of other need_ids to "
            "fold into need_id as the same underlying gap, REQUIRED for "
            "merge_needs or the action is dropped -- this permanently "
            "removes those need_ids from the retry's graph and redirects "
            "their dependents to need_id, so only propose it when you are "
            "confident they are genuinely the same gap, not merely "
            "related), new_depends_on (list of need_ids, for "
            "change_dependency -- only if the current dependency is "
            "genuinely wrong, e.g. blocking on something already resolved "
            "elsewhere or an unnecessary edge), and rationale (one "
            "sentence, referencing what the trajectory actually showed). "
            "Return an empty actions list if carrying forward the prior "
            "state and retrying as-is is already the right call -- do not "
            "invent an action just to have one."
        )
        result = self.responses_json(prompt, max_output_tokens=1536)
        data = _loads_json_object(result.text)
        return _parse_repair_plan(data)

    def summarize_task_experience(
        self,
        *,
        question: str,
        rounds: list[PlanningRound],
        unresolved_needs: list[UnresolvedNeed],
        evidence_count: int,
    ) -> str:
        # Written once per finished task for GlobalMemoryStore -- must
        # abstract away anything repo-specific (worker ids, file/symbol
        # names, this repo's own vocabulary) since the whole point is a
        # pattern transferable to a completely different codebase.
        def _execution_line(trace: NodeExecutionTrace) -> str:
            strategy = trace.special_tactic or ("coalition" if trace.coalition_formed else "normal")
            return (
                f"- need type/shape: {trace.need[:120] or '(unnamed)'}; strategy={strategy}; "
                f"resolution={trace.resolution}; evidence_gain={trace.evidence_gain}; "
                f"need_reduction={trace.need_reduction}"
            )

        execution_lines = [
            _execution_line(trace)
            for round_state in rounds
            for trace in round_state.node_executions
        ]
        prompt = (
            "Write a short, repo-agnostic case study of how this finished "
            "codebase-QA task went, for a cross-repo memory that other "
            "repos' tasks will later search by semantic similarity. "
            "Describe the SHAPE of what happened -- what kind of need "
            "showed up (not the literal question), whether/where it got "
            "stuck, which strategy (coalition, temporary_bridge, "
            "global_fallback, redecomposition, plain handoff) actually "
            "resolved it and why, or why nothing did. Do NOT mention this "
            "repo's own name, any worker id, or any file/symbol/vocabulary "
            "specific to this codebase -- write it so it reads as a "
            "transferable pattern, not a summary of this one task. If "
            f"there is nothing collaboration-shaped worth remembering (a "
            "single-round lookup with no stuck/recovery/coalition shape), "
            "return an empty summary instead of forcing one.\n"
            f"Question shape: {question}\n"
            f"Rounds: {len(rounds)}, evidence found: {evidence_count}, "
            f"still unresolved at the end: {len(unresolved_needs)}\n"
            f"Executions:\n{chr(10).join(execution_lines) or '(none)'}\n"
            "Return JSON with key summary: a string (2-5 sentences), or an "
            "empty string if nothing is worth recording."
        )
        result = self.responses_json(prompt, max_output_tokens=400)
        data = _loads_json_object(result.text)
        return str(data.get("summary") or "").strip()

    def should_specialize(
        self,
        *,
        worker_id: str,
        worker_summary: str,
        candidate_groups: dict[str, list[str]],
        route_summaries: list[str],
    ) -> bool:
        # evolve_workers previously decided this purely by counting routes
        # per subdirectory (>= min_group_routes in >= 2 groups => split),
        # with zero judgment about whether the resulting groups are actually
        # distinct specialties or just an arbitrary directory split of one
        # coherent topic. Route counts still decide *which* worker is even a
        # candidate (cost control -- this call is per candidate worker per
        # evolve cycle, not per question); this is the judgment on top.
        groups_text = "\n".join(
            f"- {group}: representative terms = {', '.join(terms) or '(none)'}"
            for group, terms in candidate_groups.items()
        )
        routes_text = "\n".join(f"- {summary}" for summary in route_summaries)
        prompt = (
            "A worker in a code-navigation colony is a candidate for being "
            "split into separate specialized workers, one per subgroup below, "
            "because recurring questions have clustered on each. Judge "
            "whether these subgroups actually represent distinct specialties "
            "worth separate ownership, or whether they are really the same "
            "underlying topic split by an arbitrary directory boundary (in "
            "which case splitting would just fragment one coherent area into "
            "pieces that all still need each other).\n"
            f"Worker: {worker_id} -- {worker_summary}\n"
            f"Candidate subgroups:\n{groups_text}\n"
            f"Why each subgroup is a candidate:\n{routes_text}\n"
            "Return JSON with key specialize: true if these are genuinely "
            "distinct specialties worth splitting into separate workers, "
            "false if they belong together as one worker."
        )
        result = self.responses_json(prompt, max_output_tokens=128)
        data = _loads_json_object(result.text)
        value = data.get("specialize")
        if isinstance(value, bool):
            return value
        # Malformed/empty response: degrade to the old unconditional
        # behavior (split whenever the structural route-count gate passes)
        # rather than silently blocking every specialize on a parse error.
        return True

    def should_merge(
        self,
        *,
        worker_a_id: str,
        worker_a_summary: str,
        worker_b_id: str,
        worker_b_summary: str,
    ) -> bool:
        # Same rationale as should_specialize: the old gate was a pure file-
        # overlap-ratio threshold (>= merge_overlap_threshold => merge),
        # which cannot tell "these are the same specialty duplicated across
        # two workers" apart from "these happen to touch a lot of shared
        # files but are conceptually distinct" (e.g. a bridge worker and a
        # test-file-heavy worker that both reference the same source files
        # for different reasons).
        prompt = (
            "Two workers in a code-navigation colony have highly overlapping "
            "file ownership and are candidates for merging into one. Judge "
            "whether they actually represent the same underlying specialty "
            "(merging would consolidate duplicated ownership into one clearer "
            "worker), or whether they are conceptually distinct despite the "
            "file overlap (merging would dilute one or both into a less "
            "focused worker).\n"
            f"Worker A: {worker_a_id} -- {worker_a_summary}\n"
            f"Worker B: {worker_b_id} -- {worker_b_summary}\n"
            "Return JSON with key merge: true if these should be merged into "
            "one worker, false if they should stay separate."
        )
        result = self.responses_json(prompt, max_output_tokens=128)
        data = _loads_json_object(result.text)
        value = data.get("merge")
        if isinstance(value, bool):
            return value
        # Malformed/empty response: degrade to the old unconditional
        # behavior (merge whenever the structural overlap gate passes)
        # rather than silently blocking every merge on a parse error.
        return True

    def decide_episode_action(
        self,
        *,
        strategy: str,
        need_terms: list[str],
        occurrences: int,
        successes: int,
        total_evidence_gain: int,
        workers: list[str],
    ) -> str:
        # evolve_workers' existing birth/merge signals (recurring_coalitions)
        # only see raw worker co-occurrence -- they cannot tell "these two
        # workers happened to get selected together" apart from "a specific
        # temporary adaptation (e.g. a bridge worker) kept actually solving
        # this kind of need across separate tasks". This is the richer
        # signal: which strategy worked, how often, and with how much real
        # evidence gain, aggregated across tasks by ColonyMemoryStore.
        # aggregate_episodes -- not a single task's outcome, which
        # persistent reorganization must not react to on its own.
        prompt = (
            "A recurring collaboration pattern was observed across multiple "
            "separate tasks in a code-navigation colony's memory. Decide what "
            "the colony's persistent organization should do about it:\n"
            "- no_change: not compelling enough evidence yet, or the pattern "
            "does not warrant a structural change.\n"
            "- strengthen_route: reinforce routing so future similar needs "
            "reach these workers faster, without creating a new worker.\n"
            "- birth_bridge: create a permanent worker dedicated to this "
            "recurring interface between the involved workers' territories.\n"
            "- merge: the involved workers should be combined into one.\n"
            f"Strategy that kept being used: {strategy}\n"
            f"Need vocabulary in common across occurrences: {', '.join(need_terms)}\n"
            f"Workers involved: {', '.join(workers)}\n"
            f"Occurrences across separate tasks: {occurrences}\n"
            f"Of which successful (found genuinely new evidence): {successes}\n"
            f"Total evidence items gained across all occurrences: {total_evidence_gain}\n"
            "Return JSON with key action: one of no_change, strengthen_route, "
            "birth_bridge, merge."
        )
        result = self.responses_json(prompt, max_output_tokens=128)
        data = _loads_json_object(result.text)
        action = data.get("action")
        if action in ("no_change", "strengthen_route", "birth_bridge", "merge"):
            return action
        # Malformed/empty response: degrade to the conservative default --
        # do nothing -- rather than guessing a structural change on a parse
        # error. Unlike should_specialize/should_merge (which gate a
        # structural change a cheaper signal already proposed), there is no
        # existing structural trigger here to fall back to.
        return "no_change"

    def synthesize(
        self,
        *,
        question: str,
        evidence: list[Evidence],
        absence_proofs: list[AbsenceProof] | None = None,
    ) -> str:
        # No evidence[:12] cut here: _select_evidence already did the one
        # relevance judgment call that decides what the synthesizer should
        # see (see its docstring -- zero-truncation is deliberate there).
        # Re-slicing to 12 on top of that undid the point of that call and
        # was observed to drop real mapping-table rows (e.g. qibo's gate-
        # symbol labels dict) that the reasoner had already kept on
        # purpose.
        evidence_text = "\n".join(_format_evidence_block(item) for item in evidence)
        completeness_text = _completeness_notes(absence_proofs)
        prompt = (
            "Answer the codebase question using the evidence below.\n"
            f"{_SYNTHESIS_PRINCIPLES}"
            f"{_completeness_instruction(completeness_text)}"
            f"Question: {question}\n"
            f"Evidence:\n{evidence_text}\n"
            f"{_completeness_section(completeness_text)}"
        )
        return self.responses_text(prompt, max_output_tokens=SYNTHESIS_MAX_OUTPUT_TOKENS).text

    def synthesize_coalition(
        self,
        *,
        question: str,
        worker_ids: list[str],
        evidence: list[Evidence],
        absence_proofs: list[AbsenceProof] | None = None,
    ) -> str:
        evidence_text = "\n".join(_format_evidence_block(item) for item in evidence)
        completeness_text = _completeness_notes(absence_proofs)
        prompt = (
            "A temporary worker coalition is jointly answering a repository question.\n"
            f"Workers: {', '.join(worker_ids)}\n"
            "Cross-check evidence across territories and name conflicts or missing links.\n"
            f"{_SYNTHESIS_PRINCIPLES}"
            f"{_completeness_instruction(completeness_text)}"
            f"Question: {question}\n"
            f"Evidence:\n{evidence_text}\n"
            f"{_completeness_section(completeness_text)}"
        )
        return self.responses_text(prompt, max_output_tokens=SYNTHESIS_MAX_OUTPUT_TOKENS).text


# Shared epistemic standard for both synthesize() and synthesize_coalition():
# aligns the synthesizer's claim discipline with the official judge's own
# "do not infer or assume missing information" rule instead of leaving that
# only enforced after the fact at scoring time. The architecture-overclaim
# line exists because example-level/utility-level evidence was observed
# being narrated as a formal "module"/"subsystem" the repo never actually
# has (qibo Bloch-sphere visualization: no visualization module exists,
# only ad-hoc example code, but the answer described it as one). The
# integration line exists for the opposite failure, seen once the evidence
# pool itself was fixed (qibo statistical-sampling-architecture: abstract
# Backend interface + GlobalBackend singleton + per-backend implementations
# were all present in evidence, but the answer described each in isolation
# instead of stating the interface->implementation->singleton relationship
# the combined evidence actually supports) -- overclaiming and
# under-synthesizing are both instances of the same rule (claim only what
# the evidence supports, but claim all of what it supports), not opposing
# pulls to balance.
_SYNTHESIS_PRINCIPLES = (
    "Use only the provided evidence.\n"
    "Before composing the answer, identify the concrete technical aspects asked by the "
    "question and ensure every aspect supported by the provided evidence is explicitly "
    "covered. Do not let the ordering of evidence determine the answer structure -- an item "
    "that appears late in the list but directly names or implements one of the question's "
    "own concrete concepts (e.g. a specific function the question is effectively asking "
    "about) must still be covered, even if earlier evidence already reads as a complete "
    "narrative on its own.\n"
    "Cover every supported aspect of the question with concrete implementation details.\n"
    "Prefer exact files, functions, classes, symbols, mappings, and call relationships.\n"
    "Do not infer architectural structure that the evidence does not explicitly establish. "
    "In particular, do not describe example-level, utility-level, or scattered code as a "
    "formal module/subsystem/component unless the evidence directly supports that claim.\n"
    "Integrate related evidence across files before answering. When multiple snippets "
    "jointly establish an architectural relationship, inheritance path, shared instance, "
    "call chain, or division of responsibility, explicitly state that relationship if it "
    "is supported by the combined evidence -- do not require a single snippet to state the "
    "entire relationship on its own, and do not describe related pieces of evidence in "
    "isolation from each other when the evidence itself connects them. A relationship must "
    "still be evidenced by something concrete in the given snippets (e.g. a class "
    "definition, an inheritance declaration, an import, a call site, a constructor) -- not "
    "asserted just because two pieces of evidence are topically related or seem like they "
    "would plausibly connect.\n"
    "Distinguish direct evidence from inference. Match the scope of each claim to the scope "
    "of the evidence -- if the evidence only supports a local implementation detail, make "
    "only that local claim rather than generalizing it.\n"
    "If evidence is incomplete, state the missing link rather than filling it in. Claims of "
    "absence or completeness are allowed only when backed by an exhaustive absence proof "
    "below.\n"
)


def _format_evidence_block(item: Evidence) -> str:
    line = f"- {item.path}:{item.line_start}-{item.line_end}\n{item.quote}"
    return f"{line}\n  ({item.reason})" if item.reason else line


def _completeness_notes(absence_proofs: list[AbsenceProof] | None) -> str:
    if not absence_proofs:
        return ""
    lines = []
    for proof in absence_proofs:
        if not proof.exhaustive:
            continue
        symbols = ", ".join(proof.relevant_symbols) or "the requested symbol"
        tools = ", ".join(proof.tools) or "available tools"
        lines.append(
            f"- Exhaustive search for {symbols}: searched {len(proof.searched_paths)} "
            f"indexed files across {len(proof.searched_territories)} territories using "
            f"{tools}; result: {proof.conclusion}."
        )
    return "\n".join(lines)


def _completeness_instruction(completeness_text: str) -> str:
    if not completeness_text:
        return ""
    return (
        "The completeness notes below describe exhaustive searches already "
        "performed across the whole indexed repository, not just the evidence "
        "shown. You may state something is absent or that a list is complete "
        "only when a note explicitly confirms it; otherwise say what more "
        "evidence would be needed rather than guessing.\n"
    )


def _completeness_section(completeness_text: str) -> str:
    return f"Completeness notes:\n{completeness_text}\n" if completeness_text else ""


def _call_with_hard_timeout(fn, timeout_seconds: float):
    """Run fn() and enforce a genuine wall-clock deadline, not just
    urlopen's own socket-level read timeout (see REQUEST_TIMEOUT_SECONDS's
    comment for why that alone was observed to not be sufficient).

    fn runs in a daemon thread; if it hasn't finished within
    timeout_seconds this raises TimeoutError and stops waiting on it. The
    thread itself is not killed -- Python has no safe way to do that -- but
    daemon=True means it can't block process exit, and whatever it
    eventually returns or raises is simply discarded once abandoned.
    """
    result: list = []
    error: list[BaseException] = []

    def _run() -> None:
        try:
            result.append(fn())
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread below
            error.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise TimeoutError(f"Request did not complete within {timeout_seconds}s (hard deadline)")
    if error:
        raise error[0]
    return result[0]


def _extract_output_text(data: dict) -> str:
    parts: list[str] = []
    for output in data.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts)


def _extract_usage(data: dict) -> TokenUsage:
    usage = data.get("usage") or {}
    return TokenUsage(
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
    )


def _with_latency_and_cost(result: ResponseResult, model: str, start: float) -> ResponseResult:
    usage = result.usage.model_copy(
        update={
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "estimated_cost_usd": estimate_cost_usd(model, result.usage),
        }
    )
    return ResponseResult(text=result.text, usage=usage, raw=result.raw)


def _is_json_object(text: str) -> bool:
    try:
        _loads_json_object(text)
    except json.JSONDecodeError:
        return False
    return True


def _loads_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _extract_list_field(text: str, key: str) -> list | None:
    """Parses a `{key: [...]}` response, tolerating a model that ignores
    the requested object wrapper and returns the bare JSON array instead --
    confirmed to happen in practice (a live seaborn eval run crashed two
    examples outright with `AttributeError: 'list' object has no attribute
    'get'` because `_loads_json_object` only looks for `{`/`}` and silently
    returns whatever `json.loads` parses when neither is present, i.e. a
    list here).
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            parsed = _loads_json_object(text)
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        value = parsed.get(key)
        return value if isinstance(value, list) else None
    return None


def _extract_selected_list(text: str) -> list | None:
    return _extract_list_field(text, "selected")


def _node_prompt_line(node: NeedNode) -> str:
    return (
        f"[{node.need_id}] {node.need} "
        f"(resolution={node.resolution}, execution={node.execution}, "
        f"progress={node.progress}, depends_on={node.depends_on}, "
        f"related_to={node.related_to}, children={node.children})"
    )


def _parse_round_plan(
    data: dict,
    *,
    graph: NeedGraph,
    workers: list[WorkerCard],
) -> RoundPlan:
    """Builds a RoundPlan from plan_round's raw JSON response, tolerant of
    a malformed/partial reply: any individual graph_updates entry, or the
    assignments/special_tactics maps, that doesn't parse cleanly is simply
    dropped rather than failing the whole round -- a partially-usable plan
    (e.g. valid assignments but a malformed graph edit) is better than
    discarding an entire round's real judgment over one bad field.
    """
    valid_worker_ids = {worker.id for worker in workers}
    known_node_ids = set(graph.nodes)

    graph_updates: dict[str, NeedNode] = {}
    raw_updates = data.get("graph_updates")
    if isinstance(raw_updates, list):
        for raw_node in raw_updates:
            node = _parse_graph_update(raw_node)
            if node is not None:
                graph_updates[node.need_id] = node
                known_node_ids.add(node.need_id)

    assignments: dict[str, list[str]] = {}
    raw_assignments = data.get("assignments")
    if isinstance(raw_assignments, dict):
        for need_id, raw_worker_ids in raw_assignments.items():
            if not isinstance(need_id, str) or not isinstance(raw_worker_ids, list):
                continue
            picked = [
                worker_id
                for worker_id in raw_worker_ids
                if isinstance(worker_id, str) and worker_id in valid_worker_ids
            ]
            if picked:
                assignments[need_id] = picked

    special_tactics: dict[str, str] = {}
    raw_tactics = data.get("special_tactics")
    if isinstance(raw_tactics, dict):
        for need_id, tactic in raw_tactics.items():
            if (
                isinstance(need_id, str)
                and tactic in ("temporary_bridge", "global_fallback")
            ):
                special_tactics[need_id] = tactic

    raw_resolved_indices = data.get("resolved_observed_need_indices")
    resolved_observed_need_indices = (
        [str(item) for item in raw_resolved_indices if isinstance(item, (str, int))]
        if isinstance(raw_resolved_indices, list)
        else []
    )

    return RoundPlan(
        graph_updates=graph_updates,
        assignments=assignments,
        special_tactics=special_tactics,
        resolved_observed_need_indices=resolved_observed_need_indices,
    )


_REPAIR_ACTION_KINDS = {
    "reuse_assignment",
    "replace_assignment",
    "merge_needs",
    "redecompose",
    "change_dependency",
    "form_local_bridge",
    "force_global_search",
}


def _parse_repair_plan(data: dict) -> RepairPlan:
    """Same tolerant-parsing shape as _parse_round_plan: an individual
    malformed action is dropped, not the whole plan. A completely
    malformed/empty response degrades to RepairPlan(actions=[]) -- a valid
    "just retry with carried-forward state" verdict, not an error (see
    propose_repair's docstring)."""
    actions: list[RepairAction] = []
    raw_actions = data.get("actions")
    if not isinstance(raw_actions, list):
        return RepairPlan(actions=actions)
    for raw_action in raw_actions:
        if not isinstance(raw_action, dict):
            continue
        kind = raw_action.get("kind")
        need_id = raw_action.get("need_id")
        if kind not in _REPAIR_ACTION_KINDS or not isinstance(need_id, str) or not need_id:
            continue
        raw_worker_ids = raw_action.get("worker_ids")
        raw_merge_with = raw_action.get("merge_with")
        raw_new_depends_on = raw_action.get("new_depends_on")
        actions.append(
            RepairAction(
                kind=kind,
                need_id=need_id,
                worker_ids=(
                    [item for item in raw_worker_ids if isinstance(item, str)]
                    if isinstance(raw_worker_ids, list)
                    else []
                ),
                merge_with=(
                    [item for item in raw_merge_with if isinstance(item, str)]
                    if isinstance(raw_merge_with, list)
                    else []
                ),
                new_depends_on=(
                    [item for item in raw_new_depends_on if isinstance(item, str)]
                    if isinstance(raw_new_depends_on, list)
                    else None
                ),
                rationale=str(raw_action.get("rationale") or ""),
            )
        )
    return RepairPlan(actions=actions)


def _parse_graph_update(raw_node: object) -> NeedNode | None:
    if not isinstance(raw_node, dict):
        return None
    need_id = raw_node.get("need_id")
    need_text = raw_node.get("need")
    if not isinstance(need_id, str) or not need_id:
        return None
    if not isinstance(need_text, str) or not need_text:
        return None

    def _str_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)]

    return NeedNode(
        need_id=need_id,
        need=need_text,
        depends_on=_str_list(raw_node.get("depends_on")),
        related_to=_str_list(raw_node.get("related_to")),
        children=_str_list(raw_node.get("children")),
        detail=UnresolvedNeed(
            description=need_text,
            need_type=str(raw_node.get("need_type") or "unknown"),
            scope=str(raw_node.get("scope") or "unknown"),
            suggested_terms=_str_list(raw_node.get("suggested_terms")),
            suggested_territories=_str_list(raw_node.get("suggested_territories")),
        ),
    )


def _owned_symbols(repo_root: Path, files: list[str], limit: int = 160) -> list[CodeSymbol]:
    index = build_symbol_index(repo_root, files)
    return [
        CodeSymbol(
            name=definition.name,
            kind=definition.kind,
            path=definition.path,
            line=definition.line,
            qualname=definition.qualname,
            bases=list(definition.bases),
        )
        for definition in sorted(index.definitions, key=lambda item: (item.path, item.line))[:limit]
    ]
