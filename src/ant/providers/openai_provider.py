from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from urllib import request

from openai import OpenAI

from ant.config import load_dotenv
from ant.domain import Evidence, TokenUsage, UnresolvedNeed, WorkerCard, WorkerObservation
from ant.providers.pricing import estimate_cost_usd


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    model: str
    organization: str | None = None
    project: str | None = None


@dataclass(frozen=True)
class ResponseResult:
    text: str
    usage: TokenUsage
    raw: dict


class OpenAIProvider:
    """Thin wrapper so orchestration code does not depend directly on the SDK."""

    def __init__(self, model: str | None = None) -> None:
        load_dotenv()
        self.settings = OpenAISettings(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=model or os.getenv("ANT_MODEL", "gpt-5.4-nano"),
            organization=os.getenv("OPENAI_ORG_ID"),
            project=os.getenv("OPENAI_PROJECT_ID"),
        )
        self.model = self.settings.model
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
        response = self.client().responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=max_output_tokens,
        )
        raw = response.model_dump()
        return ResponseResult(
            text=response.output_text,
            usage=_extract_usage(raw),
            raw=raw,
        )
        return self._record_result(_with_latency_and_cost(result, self.model, start))

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
        payload = json.dumps(
            {
                "model": self.model,
                "input": prompt,
                "max_output_tokens": max_output_tokens,
            }
        ).encode("utf-8")
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
        return WorkerCard(
            id=f"worker-{territory_id}",
            territory_id=territory_id,
            name=str(data.get("name") or f"{root or 'root'} worker"),
            root=root,
            responsibilities=[str(item) for item in data.get("responsibilities", [])][:8],
            searchable_terms=[str(item).lower() for item in data.get("searchable_terms", [])][:24],
            files=files,
        )

    def observe(
        self,
        *,
        question: str,
        worker_id: str,
        territory_id: str,
        evidence_count: int,
    ) -> WorkerObservation:
        prompt = (
            "Return a JSON worker observation for a code navigation task.\n"
            f"Question: {question}\n"
            f"Worker: {worker_id}\n"
            f"Territory: {territory_id}\n"
            f"Evidence count: {evidence_count}\n"
            "Return JSON with key unresolved_needs as a list of objects with "
            "description, suggested_terms, suggested_territories.\n"
        )
        result = self.responses_json(prompt, max_output_tokens=512)
        data = _loads_json_object(result.text)
        needs = data.get("unresolved_needs", [])
        unresolved_needs = [
            UnresolvedNeed(
                description=str(item.get("description", "")),
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
