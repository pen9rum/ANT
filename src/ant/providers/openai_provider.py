from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import request

from openai import OpenAI

from ant.config import load_dotenv
from ant.domain import (
    CodeSymbol,
    Evidence,
    TokenUsage,
    UnresolvedNeed,
    WorkerCard,
    WorkerObservation,
)
from ant.providers.pricing import estimate_cost_usd
from ant.tools.symbol_index import build_symbol_index


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
        load_dotenv()
        self.settings = OpenAISettings(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=model or os.getenv("ANT_MODEL", "gpt-5.4-nano"),
            reasoning_effort=reasoning_effort,
            organization=os.getenv("OPENAI_ORG_ID"),
            project=os.getenv("OPENAI_PROJECT_ID"),
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
        try:
            _loads_json_object(result.text)
            return result
        except json.JSONDecodeError:
            repair_prompt = (
                "Convert the following response into one valid JSON object. "
                "Return only JSON, no markdown.\n"
                f"Response:\n{result.text}"
            )
            return self.responses_text(repair_prompt, max_output_tokens=max_output_tokens)

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
        with request.urlopen(api_request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
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
        return WorkerCard(
            id=f"worker-{territory_id}",
            territory_id=territory_id,
            name=str(data.get("name") or f"{root or 'root'} worker"),
            root=root,
            responsibilities=[str(item) for item in data.get("responsibilities", [])][:8],
            searchable_terms=[str(item).lower() for item in data.get("searchable_terms", [])][:24],
            files=files,
            symbols=symbols,
        )

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

    def synthesize(self, *, question: str, evidence: list[Evidence]) -> str:
        evidence_text = "\n".join(
            f"- {item.path}:{item.line_start} {item.quote}" for item in evidence[:12]
        )
        prompt = (
            "Answer the codebase question using only the evidence below. "
            "If evidence is insufficient, say what is missing.\n"
            f"Question: {question}\n"
            f"Evidence:\n{evidence_text}\n"
        )
        return self.responses_text(prompt, max_output_tokens=512).text

    def synthesize_coalition(
        self,
        *,
        question: str,
        worker_ids: list[str],
        evidence: list[Evidence],
    ) -> str:
        evidence_text = "\n".join(
            f"- {item.path}:{item.line_start}-{item.line_end}\n{item.quote}"
            for item in evidence[:12]
        )
        prompt = (
            "A temporary worker coalition is jointly answering a repository question.\n"
            f"Workers: {', '.join(worker_ids)}\n"
            "Cross-check evidence across territories, name conflicts or missing links, "
            "and answer only what is supported.\n"
            f"Question: {question}\n"
            f"Evidence:\n{evidence_text}\n"
        )
        return self.responses_text(prompt, max_output_tokens=768).text


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
