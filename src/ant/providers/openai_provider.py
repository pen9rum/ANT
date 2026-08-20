from __future__ import annotations

import os


class OpenAIProvider:
    """Thin wrapper so orchestration code does not depend directly on the SDK."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("ANT_MODEL", "gpt-5")

    def is_configured(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def require_configured(self) -> None:
        if not self.is_configured():
            msg = "OPENAI_API_KEY is not set."
            raise RuntimeError(msg)
