"""Cross-repo experience memory (Phase 8).

Deliberately separate from ColonyMemoryStore, which is scoped to one
repo's own index_path. This store lives at a single fixed location shared
across every repo ANT is ever run against, and holds nothing but
repo-agnostic verbal case studies -- "this kind of need got stuck this
way, this recovery worked, here's why" -- not repo-specific worker
identities, routes, or coalitions. There is deliberately no formal
worker-role taxonomy or coordination-prior table here: retrieval-by-
semantic-similarity, feeding the Orchestrator's planning prompt as
reference text it can take or leave, IS the cross-repo learning
mechanism -- there is no separate structural mutation step the way
evolve_workers has for repo-local memory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from ant.retrieval.dense import DenseEmbedder, EmbeddingEntry, EmbeddingIndex, get_shared_embedder

if TYPE_CHECKING:
    from ant.domain import EvidenceState
    from ant.providers import WorkerReasoner

EXPERIENCE_KEY = "experiences"


class TaskExperience(BaseModel):
    """One repo-agnostic verbal case study of how a finished task went.

    `repo` is provenance only (which repo this was recorded from, useful
    for debugging/auditing) -- retrieval never filters or weights by it,
    since the entire point is surfacing patterns learned on OTHER repos.
    """

    summary: str
    repo: str = ""


def default_global_memory_path() -> Path:
    override = os.getenv("ANT_GLOBAL_MEMORY_PATH")
    if override:
        return Path(override)
    return Path.home() / ".ant" / "global_memory"


class GlobalMemoryStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_global_memory_path()

    def record_experience(self, experience: TaskExperience, embedder: DenseEmbedder) -> None:
        if not experience.summary.strip():
            return
        vector = _embed_and_normalize(embedder, experience.summary)
        entry = EmbeddingEntry(
            path=experience.repo or "unknown",
            line_start=0,
            line_end=0,
            quote=experience.summary[:4000],
        )
        existing = EmbeddingIndex.load(self.path, EXPERIENCE_KEY)
        if existing is None or not existing.entries:
            updated = EmbeddingIndex(entries=[entry], vectors=vector.reshape(1, -1))
        else:
            updated = EmbeddingIndex(
                entries=[*existing.entries, entry],
                vectors=np.concatenate([existing.vectors, vector.reshape(1, -1)], axis=0),
            )
        updated.save(self.path, EXPERIENCE_KEY)

    def retrieve_similar(
        self,
        query: str,
        embedder: DenseEmbedder,
        limit: int = 5,
        exclude_repo: str = "",
    ) -> list[str]:
        index = EmbeddingIndex.load(self.path, EXPERIENCE_KEY)
        if index is None or not index.entries:
            return []
        vector = _embed_and_normalize(embedder, query)
        # exclude_repo: confirmed live that without this, a repo's own
        # gen0 attempt on THIS SAME question -- recorded minutes earlier
        # by record_global_experience_safe, in the same run -- is
        # embedding-similarity's own top match for slow-gen1's retry of
        # that identical question. That is the opposite of "cross-repo":
        # the Orchestrator gets handed a summary describing its own prior
        # self's struggle ("this needed coalition/temporary_bridge") as
        # if it were outside precedent, priming it to reach for the same
        # recovery tactics again rather than exploring a fresh escalation
        # path gen0 itself may only have found in a later round. `repo` is
        # the same provenance string record_global_experience_safe was
        # called with, so an omitted/empty exclude_repo (e.g. no `repo`
        # available) degrades to today's exact behavior -- no entries
        # excluded -- rather than ever silently under-filling `limit`.
        paths = None
        if exclude_repo:
            paths = {entry.path for entry in index.entries if entry.path != exclude_repo}
            if not paths:
                return []
        hits = index.search(vector.tolist(), limit=limit, paths=paths)
        return [entry.quote for _, entry in hits]


def retrieve_cross_repo_experience_safe(
    global_memory: GlobalMemoryStore, query: str, limit: int = 5, exclude_repo: str = ""
) -> list[str]:
    """Same graceful-degradation shape as the rest of dense retrieval in
    this codebase: no embedder available (the optional 'dense' extra isn't
    installed) means cross-repo experience is simply absent this run, not
    an error -- callers get [] and the Orchestrator prompt shows "(none)".

    `exclude_repo`: pass the same repo-provenance string this call's own
    later record_global_experience_safe will use, so a gen0/slow-gen1 pair
    for the same repo never has the later stage handed the earlier one's
    own just-recorded experience as if it were outside precedent -- see
    GlobalMemoryStore.retrieve_similar's own docstring for why this
    mattered in practice.
    """
    embedder = get_shared_embedder()
    if embedder is None:
        return []
    return global_memory.retrieve_similar(query, embedder, limit=limit, exclude_repo=exclude_repo)


def record_global_experience_safe(
    global_memory: GlobalMemoryStore,
    reasoner: WorkerReasoner,
    question: str,
    state: EvidenceState,
    repo: str = "",
) -> None:
    """Same graceful-degradation wrapper as retrieve_cross_repo_experience_safe,
    for the recording side."""
    embedder = get_shared_embedder()
    if embedder is None:
        return
    record_global_experience(global_memory, reasoner, embedder, question, state, repo)


def record_global_experience(
    global_memory: GlobalMemoryStore,
    reasoner: WorkerReasoner,
    embedder: DenseEmbedder,
    question: str,
    state: EvidenceState,
    repo: str = "",
) -> None:
    """Runs WorkerReasoner.summarize_task_experience() once, after a task
    finishes, and records the result (if any) into `global_memory`. Must
    run alongside record_task_memory (repo-local), not instead of it --
    see cli.py's `ask` command and evaluation/runner.py's _run_example for
    the two call sites. A reasoner that judges nothing worth remembering
    returns "" from summarize_task_experience, and this is a no-op.
    """
    summary = reasoner.summarize_task_experience(
        question=question,
        rounds=state.rounds,
        unresolved_needs=state.unresolved_needs,
        evidence_count=len(state.evidence),
    )
    if not summary:
        return
    global_memory.record_experience(TaskExperience(summary=summary, repo=repo), embedder)


def _embed_and_normalize(embedder: DenseEmbedder, text: str) -> np.ndarray:
    [vector] = embedder.embed([text])
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm > 0:
        array = array / norm
    return array
