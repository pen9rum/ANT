from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib import request

from openai import OpenAI

from ant.config import load_dotenv


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    model: str
    organization: str | None = None
    project: str | None = None


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
        if self.settings.organization and self.settings.project:
            return self._responses_create_raw(prompt)
        response = self.client().responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=16,
        )
        return response.output_text

    def _responses_create_raw(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "input": prompt,
                "max_output_tokens": 16,
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
        return _extract_output_text(data)


def _extract_output_text(data: dict) -> str:
    parts: list[str] = []
    for output in data.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts)
