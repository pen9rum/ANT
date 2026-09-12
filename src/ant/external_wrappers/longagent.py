from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path
from typing import Any

import tiktoken

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.domain import TokenUsage
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.repo_scope import EvalRepoEnvironment
from ant.evaluation_suite.usage import UsageStats
from ant.providers.openai_provider import _loads_json_object

# See docs/longagent_fidelity_audit.md for the full audit this adapter is
# built from. Summary of the two facts that most shape this file:
#
# 1. No reference implementation exists, official or unofficial (the only
#    author-linked repo is a benchmark dataset with zero inference code;
#    the one third-party repo found is an empty package-template stub --
#    every .py file in it is 0 bytes). This is reconstructed directly from
#    the paper's own methodology section and appendix prompt tables, and
#    must always be labeled IMPLEMENTATION_LABEL below, never "exact
#    reproduction".
# 2. The paper's own leader action space is named {NEW_STATE, CONFLICT,
#    ANSWER} (Section 2.3) -- there is no action literally called "QUERY"
#    in the paper. This file uses the paper's own names; "query" in any
#    external brief maps onto NEW_STATE here.

PAPER_CITATION = (
    "Zhao, Zu, Xu, Lu, He, Ding, Gui, Zhang, Huang. \"LONGAGENT: Achieving "
    "Question Answering for 128k-Token-Long Documents through Multi-Agent "
    "Collaboration.\" EMNLP 2024 Main, pp. 16310-16324. "
    "https://aclanthology.org/2024.emnlp-main.912/ "
    "(same work as arXiv:2402.11550v2, earlier title "
    "\"LongAgent: Scaling Language Models to 128k Context through "
    "Multi-Agent Collaboration\")."
)
IMPLEMENTATION_LABEL = "paper-faithful LongAgent adaptation"
UPSTREAM_COMMIT = None  # no reference implementation exists -- see fidelity audit section 2
UPSTREAM_REPO_NOTE = (
    "No official or unofficial runnable LongAgent implementation exists. The only "
    "author-linked repo (zuucan/NeedleInAHaystack-PLUS) is the paper's own benchmark "
    "dataset, not inference code. The one third-party repo found under the paper's "
    "name (vyomakesh09/longagent) is an empty kyegomez/Python-Package-Template stub "
    "(every .py file is 0 bytes, confirmed via the GitHub API tree listing) -- not usable "
    "for anything. See docs/longagent_fidelity_audit.md section 2."
)

# Paper's own chunk-size ablation (Section 4.2) sweeps {500, 1000, 1500,
# 2000} tokens with no single stated default for the main results; 2000 is
# the largest value tested and the one the paper itself reports as best for
# hallucination mitigation. See fidelity audit section 3.2/4.
DEFAULT_CHUNK_SIZE_TOKENS = 2000

# The paper states no maximum round count anywhere. Running genuinely
# unbounded is not viable for an automated harness (an unconverged leader
# would hang a task forever). This bound is an explicit, disclosed
# ENGINEERING safety net, not a value taken from the paper -- see fidelity
# audit section 4, point 3. It caps every leader decision point (whether
# the leader chooses NEW_STATE, CONFLICT, or ANSWER), not just NEW_STATE
# rounds, since an unconverging CONFLICT loop would be just as unbounded.
DEFAULT_MAX_LEADER_DECISIONS = 8

# The paper's own architecture allows independent members to be processed
# in parallel within a round (each member reads only its own chunk and
# responds to the same broadcast instruction with no cross-member
# dependency) -- see docs/longagent_fidelity_audit.md. This is a pure
# EXECUTION optimization, chosen once and disclosed, not a value tuned
# against benchmark score: it bounds concurrent outbound HTTP connections
# to the model API for stability under real rate limits, nothing else. It
# never changes member prompt/context/outputs, leader inputs/policy,
# member count, or round semantics -- see _run_members_concurrently's own
# docstring for why. Logged every run via metadata["member_concurrency"].
DEFAULT_MEMBER_CONCURRENCY = 4

_ENCODING_NAME = "cl100k_base"

_FILE_MARKER_FMT = "===== FILE: {path} =====\n"

_MEMBER_PROMPT = (
    "You are one member of a team of {n_members} agents collaboratively answering a "
    "question about a long document (a software repository's serialized source files). "
    "You have access to ONLY your own chunk (chunk {member_id} of {n_members}); other "
    "members hold the other chunks and you cannot see them.\n\n"
    "# Your document chunk:\n{chunk_text}\n\n"
    "# Instruction from your team leader:\n{instruction}\n\n"
    "Respond to the leader's instruction using ONLY the content of your own chunk above. "
    "If your chunk does not contain information relevant to the instruction, say so "
    "plainly rather than guessing or inventing an answer.\n\n"
    "Respond with a single JSON object of exactly this form, nothing else:\n"
    '{{"type": "response", "content": "<your answer, referencing only your own chunk>"}}'
)

_LEADER_INITIAL_PROMPT = (
    "You are the leader of a team of {n_members} member agents. Each member has read a "
    "different, non-overlapping chunk of a long document (a software repository's "
    "serialized source files) and cannot see any other member's chunk. Your job is to "
    "answer the question below by directing your members and reasoning over their "
    "responses across possibly several rounds.\n\n"
    "# Question:\n{question}\n\n"
    "This is the FIRST round: no member has responded yet. Break the question down and "
    "issue your first instruction, to be broadcast identically to every member, telling "
    "them what to look for in their own chunk.\n\n"
    "Respond with a single JSON object of exactly this form, nothing else:\n"
    '{{"type": "new_state", "content": "<the instruction to broadcast to every member>"}}'
)

_LEADER_DECISION_PROMPT = (
    "You are the leader of a team of {n_members} member agents. Each member has read a "
    "different, non-overlapping chunk of a long document (a software repository's "
    "serialized source files) and cannot see any other member's chunk.\n\n"
    "# Question:\n{question}\n\n"
    "# Discussion history so far (each round's broadcast instruction and every member's "
    "response to it; member IDs are stable across rounds):\n{history_text}\n\n"
    "Decide exactly ONE of these three actions:\n"
    '1. "conflict" -- if two or more members give conflicting answers about the same '
    "fact and you suspect at least one is hallucinating (rather than one having no "
    "relevant information at all), list the exact member IDs (integers) whose answers "
    "conflict, so they can share chunks pairwise and re-answer.\n"
    '2. "new_state" -- if you do not yet have enough information to answer, give a new '
    "instruction to broadcast to ALL members for the next round.\n"
    '3. "answer" -- if you now have enough information, give your final answer to the '
    "question.\n\n"
    "Respond with a single JSON object of exactly this form, nothing else. Only include "
    '"conflict_member_ids" when type is "conflict"; omit it otherwise:\n'
    '{{"type": "conflict" | "new_state" | "answer", "content": "<new instruction, OR your '
    'final answer, OR a short note on the suspected conflict>", "conflict_member_ids": '
    "[<int>, ...]}}"
)

_LEADER_FORCED_ANSWER_PROMPT = (
    "You are the leader of a team of {n_members} member agents, as before.\n\n"
    "# Question:\n{question}\n\n"
    "# Discussion history so far:\n{history_text}\n\n"
    "Your round budget is exhausted. You must answer NOW, conclusively, using only what "
    "you have already gathered above -- do not request another round.\n\n"
    "Respond with a single JSON object of exactly this form, nothing else:\n"
    '{{"type": "answer", "content": "<your final answer>"}}'
)

_CONFLICT_MEMBER_PROMPT = (
    "You are member {member_id} of a team of {n_members} agents. Earlier you were asked: "
    '"{instruction}" and answered based only on your own chunk. The team leader has '
    "flagged your response as potentially conflicting with another member's response, "
    "and is now sharing that other member's chunk with you so you can check for the "
    "correct information.\n\n"
    "# Your own original chunk:\n{own_chunk}\n\n"
    "# Additional chunk shared by the leader for conflict resolution:\n{shared_chunk}\n\n"
    "# Original instruction:\n{instruction}\n\n"
    "Re-answer the instruction now using BOTH chunks above. If the additional chunk "
    "contains information showing your original answer was wrong, correct it; if your "
    "original answer already looks right, restate it.\n\n"
    "Respond with a single JSON object of exactly this form, nothing else:\n"
    '{{"type": "response", "content": "<your corrected or reconfirmed answer>"}}'
)


def serialize_repository(environment_root: Path) -> tuple[str, int]:
    """Deterministically serializes the SAME eligible-file universe the
    rest of this evaluation suite uses (`EvalRepoEnvironment.iter_files()`
    -- not a bespoke walk, not a .py-only filter) into one long document,
    in stable sorted-relative-path order, with an explicit file-boundary
    marker before each file's VERBATIM text (no summarization, no
    relevance filtering, no reordering by any heuristic -- see fidelity
    audit section 6). Returns (serialized_text, eligible_file_count).
    """
    environment = EvalRepoEnvironment(environment_root)
    files = sorted(environment.iter_files(), key=lambda p: str(p.relative_to(environment.root)))
    parts: list[str] = []
    for path in files:
        rel = str(path.relative_to(environment.root))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        parts.append(_FILE_MARKER_FMT.format(path=rel) + text)
    return "\n".join(parts), len(files)


def chunk_document(serialized_text: str, chunk_size_tokens: int) -> list[str]:
    """Fixed-size token-count chunking, no overlap, no boundary-awareness
    beyond the raw token count -- exactly as faithful to "chunks of
    predefined size" (Section 2.1) as a repository-QA setting permits.
    Adding sentence/file-boundary-aware splitting would be inventing
    repository-specific intelligence the paper's own protocol does not
    have; deliberately not done here (fidelity audit section 4, point 2).
    """
    encoding = tiktoken.get_encoding(_ENCODING_NAME)
    tokens = encoding.encode(serialized_text, disallowed_special=())
    if not tokens:
        return [""]
    return [
        encoding.decode(tokens[i : i + chunk_size_tokens])
        for i in range(0, len(tokens), chunk_size_tokens)
    ]


def _parse_or_empty(text: str) -> dict[str, Any]:
    return _loads_json_object(text) if _safe_is_json(text) else {}


def _merge_usage(total: TokenUsage, addition: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=total.input_tokens + addition.input_tokens,
        output_tokens=total.output_tokens + addition.output_tokens,
        total_tokens=total.total_tokens + addition.total_tokens,
        latency_ms=total.latency_ms + addition.latency_ms,
        estimated_cost_usd=total.estimated_cost_usd + addition.estimated_cost_usd,
    )


def _run_members_concurrently(
    tasks: list[tuple[int, str]],
    model: str,
    max_output_tokens: int,
    concurrency: int,
) -> tuple[dict[int, str], int, TokenUsage, list[dict]]:
    """Runs a batch of independent (member_id, prompt) member calls under a
    bounded thread pool -- the paper's own architecture permits parallel
    member processing within a round (see DEFAULT_MEMBER_CONCURRENCY's own
    comment); this is the only place that concurrency is actually
    exercised. One fresh `CountingOpenAIProvider` per call, never a single
    instance shared across threads: usage/call-count accounting is only
    ever mutated in the single-threaded submitting code below, after each
    future resolves, so there is no concurrent-write race on any shared
    counter (the underlying provider's own usage bookkeeping is NOT
    thread-safe, and this file does not modify that frozen core code to
    make it so).

    `responses` is keyed by member_id, exactly as the pre-concurrency
    sequential version keyed it -- so every downstream consumer
    (_format_history, the leader's next prompt, conflict resolution) sees
    an IDENTICAL result regardless of the physical order calls actually
    completed in. Member prompt/context, leader inputs, leader policy,
    member count, and round semantics are therefore all unaffected by this
    function; only wall-clock execution order changes.
    """
    responses: dict[int, str] = {}
    total_calls = 0
    usage = TokenUsage()
    retry_logs: list[dict] = []

    def _call(member_id: int, prompt: str) -> tuple[int, str, int, TokenUsage, list[dict]]:
        member_provider = CountingOpenAIProvider(model=model)
        result = member_provider.responses_json(prompt, max_output_tokens=max_output_tokens)
        parsed = _parse_or_empty(result.text)
        content = str(parsed.get("content") or result.text)
        return (
            member_id,
            content,
            member_provider.drain_call_count(),
            member_provider.drain_usage(),
            member_provider.drain_retry_log(),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(_call, member_id, prompt) for member_id, prompt in tasks]
        for future in concurrent.futures.as_completed(futures):
            member_id, content, calls, member_usage, member_retry_log = future.result()
            responses[member_id] = content
            total_calls += calls
            usage = _merge_usage(usage, member_usage)
            retry_logs.extend(member_retry_log)
    return responses, total_calls, usage, retry_logs


def _format_history(history: list[dict[str, Any]]) -> str:
    lines = []
    for entry in history:
        lines.append(f"-- Round {entry['round']} instruction: {entry['instruction']}")
        for member_id in sorted(entry["responses"]):
            lines.append(f"   Member {member_id}: {entry['responses'][member_id]}")
    return "\n".join(lines) if lines else "(no rounds yet)"


class LongAgentAdapter:
    """Repository-adapted LongAgent: static partition (deterministic
    fixed-size chunking, one chunk per member, no retrieval, no relevance
    filtering) + leader/member multi-agent coordination, reconstructed
    directly from the EMNLP 2024 paper's own protocol since no reference
    implementation exists (see docs/longagent_fidelity_audit.md). Serves
    as the "static partition" baseline contrasted against ANT's own
    runtime-adaptive coordination -- it must remain recognizably
    LongAgent, not gradually acquire ANT-specific mechanisms.

    Model: GPT-4.1 for both leader and members (a disclosed model
    substitution for the paper's own training-free GPT-3.5 leader+member
    configuration, which the paper itself reports as a MAIN result, not a
    hypothetical this adapter invents -- see fidelity audit section 5).
    Never claims to reproduce either of the paper's own two headline
    configurations (fine-tuned LLaMA-2-7B, or GPT-3.5).
    """

    name = "longagent"

    def __init__(
        self,
        model: str = "gpt-4.1",
        chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
        max_leader_decisions: int = DEFAULT_MAX_LEADER_DECISIONS,
        member_concurrency: int = DEFAULT_MEMBER_CONCURRENCY,
    ) -> None:
        self.model = model
        self.chunk_size_tokens = chunk_size_tokens
        self.max_leader_decisions = max_leader_decisions
        # Execution-only knob -- see DEFAULT_MEMBER_CONCURRENCY's own
        # comment and _run_members_concurrently's docstring. Never varied
        # per example within a run; fixed once at construction time and
        # logged into every result's metadata.
        self.member_concurrency = member_concurrency

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        provider = CountingOpenAIProvider(model=self.model)
        started = time.time()

        serialized_text, eligible_file_count = serialize_repository(environment_root)
        chunks = chunk_document(serialized_text, self.chunk_size_tokens)
        n_members = len(chunks)

        trajectory: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        member_calls = 0
        leader_calls = 0
        new_state_rounds = 0
        conflict_events = 0
        query_used = False
        conflict_used = False
        final_answer = ""
        termination_reason = "unknown"
        member_physical_calls = 0
        member_usage_total = TokenUsage()
        member_retry_logs: list[dict] = []

        def _broadcast(instruction: str) -> dict[int, str]:
            nonlocal member_calls, member_physical_calls, member_usage_total, member_retry_logs
            tasks = [
                (
                    member_id,
                    _MEMBER_PROMPT.format(
                        n_members=n_members,
                        member_id=member_id,
                        chunk_text=chunk_text,
                        instruction=instruction,
                    ),
                )
                for member_id, chunk_text in enumerate(chunks)
            ]
            responses, calls, usage, retry_log = _run_members_concurrently(
                tasks, self.model, max_output_tokens=400, concurrency=self.member_concurrency
            )
            member_calls += len(tasks)
            member_physical_calls += calls
            member_usage_total = _merge_usage(member_usage_total, usage)
            member_retry_logs.extend(retry_log)
            return responses

        for decision_index in range(self.max_leader_decisions):
            if not history:
                leader_prompt = _LEADER_INITIAL_PROMPT.format(
                    n_members=n_members, question=example.question
                )
            else:
                leader_prompt = _LEADER_DECISION_PROMPT.format(
                    n_members=n_members,
                    question=example.question,
                    history_text=_format_history(history),
                )
            leader_result = provider.responses_json(leader_prompt, max_output_tokens=700)
            leader_calls += 1
            decision = _parse_or_empty(leader_result.text)
            action = decision.get("type")
            trajectory.append({"step": decision_index, "role": "leader", "decision": decision})

            if action == "answer":
                final_answer = str(decision.get("content") or "")
                termination_reason = "leader_answer"
                break
            elif action == "conflict":
                conflict_used = True
                conflict_events += 1
                raw_ids = decision.get("conflict_member_ids") or []
                member_ids = sorted(
                    {i for i in raw_ids if isinstance(i, int) and 0 <= i < n_members}
                )
                if len(member_ids) < 2 or not history:
                    # Malformed/degenerate conflict decision (parser
                    # defect guard, not a quality judgment) -- treat as a
                    # no-op new_state-less round and let the leader
                    # re-decide next iteration with the same history.
                    trajectory.append(
                        {
                            "step": decision_index,
                            "role": "conflict",
                            "note": "degenerate, skipped",
                            "raw_ids": raw_ids,
                        }
                    )
                    continue
                last_round = history[-1]
                shared_text = "\n---\n".join(chunks[i] for i in member_ids)
                conflict_tasks = [
                    (
                        member_id,
                        _CONFLICT_MEMBER_PROMPT.format(
                            member_id=member_id,
                            n_members=n_members,
                            instruction=last_round["instruction"],
                            own_chunk=chunks[member_id],
                            shared_chunk=shared_text,
                        ),
                    )
                    for member_id in member_ids
                ]
                conflict_responses, calls, usage, retry_log = _run_members_concurrently(
                    conflict_tasks,
                    self.model,
                    max_output_tokens=400,
                    concurrency=self.member_concurrency,
                )
                member_calls += len(conflict_tasks)
                member_physical_calls += calls
                member_usage_total = _merge_usage(member_usage_total, usage)
                member_retry_logs.extend(retry_log)
                last_round["responses"].update(conflict_responses)
                trajectory.append(
                    {"step": decision_index, "role": "conflict", "member_ids": member_ids}
                )
                continue
            else:  # "new_state" (also the required action on the first, history-empty call)
                query_used = True
                instruction = str(decision.get("content") or example.question)
                new_state_rounds += 1
                responses = _broadcast(instruction)
                history.append(
                    {"round": new_state_rounds, "instruction": instruction, "responses": responses}
                )
                trajectory.append(
                    {
                        "step": decision_index,
                        "role": "members",
                        "round": new_state_rounds,
                        "responses": responses,
                    }
                )
        else:
            termination_reason = "round_budget_exhausted"
            forced_prompt = _LEADER_FORCED_ANSWER_PROMPT.format(
                n_members=n_members,
                question=example.question,
                history_text=_format_history(history),
            )
            forced_result = provider.responses_json(forced_prompt, max_output_tokens=700)
            leader_calls += 1
            forced_decision = _parse_or_empty(forced_result.text)
            final_answer = str(forced_decision.get("content") or "")
            trajectory.append(
                {
                    "step": self.max_leader_decisions,
                    "role": "leader",
                    "decision": forced_decision,
                    "forced": True,
                }
            )

        # `drain_*` calls empty the provider's internal counters, so each
        # must be captured exactly once into a local before use. Leader
        # calls all go through the single shared `provider` (never
        # dispatched concurrently -- the leader is a strictly sequential
        # decision loop); member calls each used their own short-lived
        # provider instance (see _run_members_concurrently) and are merged
        # in here.
        physical_llm_calls = provider.drain_call_count() + member_physical_calls
        token_usage = _merge_usage(provider.drain_usage(), member_usage_total)
        retry_logs = provider.drain_retry_log() + member_retry_logs
        elapsed = time.time() - started

        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=final_answer,
            trajectory=trajectory,
            usage=UsageStats(
                llm_calls=physical_llm_calls,
                tool_calls=0,
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                total_tokens=token_usage.total_tokens,
                estimated_cost_usd=token_usage.estimated_cost_usd,
                wall_clock_seconds=elapsed,
                unique_files_inspected=eligible_file_count,
            ),
            termination_reason=termination_reason,
            metadata={
                "generation_model": self.model,
                "implementation_label": IMPLEMENTATION_LABEL,
                "upstream_commit": UPSTREAM_COMMIT,
                "upstream_repo_note": UPSTREAM_REPO_NOTE,
                "chunk_size_tokens": self.chunk_size_tokens,
                "member_concurrency": self.member_concurrency,
                "n_chunks": n_members,
                "n_members": n_members,
                "leader_rounds": new_state_rounds,
                # Logical decision-point counts (one per responses_json
                # call site in this file's own loop) -- distinct from
                # `usage.llm_calls` above, which is the PHYSICAL call count
                # including any internal JSON-repair passes
                # (CountingOpenAIProvider/responses_json's own retry, see
                # module docstring reference to "schema/JSON repair
                # requires extra physical calls, count them too").
                "leader_calls": leader_calls,
                "member_calls": member_calls,
                "total_logical_llm_calls": leader_calls + member_calls,
                "total_physical_llm_calls": physical_llm_calls,
                "new_state_used": query_used,
                "conflict_used": conflict_used,
                "conflict_events": conflict_events,
                "retry_total_physical_attempts": sum(e["attempt_count"] for e in retry_logs),
                "retry_total_retries": sum(e["retry_count"] for e in retry_logs),
            },
        )


def _safe_is_json(text: str) -> bool:
    try:
        _loads_json_object(text)
    except Exception:
        return False
    return True


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(LongAgentAdapter())
