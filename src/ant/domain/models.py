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


class Evidence(BaseModel):
    path: str
    line_start: int
    line_end: int
    quote: str
    reason: str


class UnresolvedNeed(BaseModel):
    description: str
    suggested_terms: list[str] = Field(default_factory=list)
    suggested_territories: list[str] = Field(default_factory=list)


class EvidenceState(BaseModel):
    question: str
    evidence: list[Evidence] = Field(default_factory=list)
    unresolved_needs: list[UnresolvedNeed] = Field(default_factory=list)

    def has_evidence(self) -> bool:
        return bool(self.evidence)


def as_posix(path: Path) -> str:
    return path.as_posix()
