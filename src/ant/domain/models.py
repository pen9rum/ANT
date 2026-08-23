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
    symbols: list[CodeSymbol] = Field(default_factory=list)


class CodeSymbol(BaseModel):
    name: str
    kind: str
    path: str
    line: int
    qualname: str = ""
    bases: list[str] = Field(default_factory=list)


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
    claim: str = ""
    worker_id: str = ""
    symbols: list[str] = Field(default_factory=list)
    dense_score: float = 0.0


class WorkerAction(BaseModel):
    tool: str
    query: str
    result_count: int = 0
    rationale: str = ""


class ExecutionDiagnostic(BaseModel):
    kind: str
    message: str
    tool: str = ""
    suggested_terms: list[str] = Field(default_factory=list)


class WorkerObservation(BaseModel):
    worker_id: str
    territory_id: str
    evidence: list[Evidence] = Field(default_factory=list)
    unresolved_needs: list[UnresolvedNeed] = Field(default_factory=list)
    diagnostics: list[ExecutionDiagnostic] = Field(default_factory=list)
    actions: list[WorkerAction] = Field(default_factory=list)
    stop_reason: str = ""


class UnresolvedNeed(BaseModel):
    description: str
    kind: str = "missing_detail"
    need_type: str = "unknown"
    known: list[str] = Field(default_factory=list)
    missing: str = ""
    scope: str = "unknown"
    source_worker_id: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    suggested_terms: list[str] = Field(default_factory=list)
    suggested_territories: list[str] = Field(default_factory=list)
    relevant_symbols: list[str] = Field(default_factory=list)


class WorkerRoutingScore(BaseModel):
    worker_id: str
    territory_id: str
    final_score: int
    query_hits: list[str] = Field(default_factory=list)
    suggested_term_hits: list[str] = Field(default_factory=list)
    relevant_symbol_hits: list[str] = Field(default_factory=list)
    territory_hint_score: int = 0
    source_worker_bonus: int = 0
    source_path_bonus: int = 0
    memory_route_bonus: int = 0
    dense_routing_bonus: int = 0
    test_path_penalty: int = 0
    seen_worker_penalty: int = 0


class RecruitmentRound(BaseModel):
    round_index: int
    query: str
    input_need: str = ""
    candidate_worker_ids: list[str] = Field(default_factory=list)
    routing_scores: list[WorkerRoutingScore] = Field(default_factory=list)
    selected_worker_ids: list[str] = Field(default_factory=list)
    rationale: str
    selection_reason: str = ""
    coalition_formed: bool = False
    coalition_reason: str = ""
    observations: list[WorkerObservation] = Field(default_factory=list)


class EvidenceState(BaseModel):
    question: str
    answer: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    unresolved_needs: list[UnresolvedNeed] = Field(default_factory=list)
    rounds: list[RecruitmentRound] = Field(default_factory=list)
    absence_proofs: list[AbsenceProof] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)

    def has_evidence(self) -> bool:
        return bool(self.evidence)


class AbsenceProof(BaseModel):
    query: str
    relevant_symbols: list[str] = Field(default_factory=list)
    searched_worker_ids: list[str] = Field(default_factory=list)
    searched_territories: list[str] = Field(default_factory=list)
    searched_paths: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    exhaustive: bool = False
    conclusion: str = "inconclusive"


def as_posix(path: Path) -> str:
    return path.as_posix()
