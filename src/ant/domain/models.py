from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Territory(BaseModel):
    id: str
    root: str
    files: list[str] = Field(default_factory=list)
    summary: str = ""


class WorkerCard(BaseModel):
    id: str
    territory_id: str
    name: str
    root: str
    responsibilities: list[str] = Field(default_factory=list)
    searchable_terms: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    estimated_cost_usd: float = 0.0


class Evidence(BaseModel):
    path: str
    line_start: int
    line_end: int
    quote: str
    reason: str


class WorkerObservation(BaseModel):
    worker_id: str
    territory_id: str
    evidence: list[Evidence] = Field(default_factory=list)
    unresolved_needs: list[UnresolvedNeed] = Field(default_factory=list)


class UnresolvedNeed(BaseModel):
    description: str
    suggested_terms: list[str] = Field(default_factory=list)
    suggested_territories: list[str] = Field(default_factory=list)


class RecruitmentRound(BaseModel):
    round_index: int
    query: str
    selected_worker_ids: list[str] = Field(default_factory=list)
    rationale: str
    observations: list[WorkerObservation] = Field(default_factory=list)


class EvidenceState(BaseModel):
    question: str
    answer: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    unresolved_needs: list[UnresolvedNeed] = Field(default_factory=list)
    rounds: list[RecruitmentRound] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)

    def has_evidence(self) -> bool:
        return bool(self.evidence)


def as_posix(path: Path) -> str:
    return path.as_posix()
