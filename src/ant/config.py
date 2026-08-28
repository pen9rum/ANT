from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path = Path(".env"), override: bool = False) -> None:
    # override=False: .env fills in variables the process environment
    # doesn't already have, never clobbers ones a caller explicitly set --
    # the standard dotenv contract. This used to default to True, called
    # unconditionally from OpenAIProvider.__init__, which meant .env's
    # ANT_MODEL silently re-applied itself over any os.environ["ANT_MODEL"]
    # a caller had already set before constructing a provider: confirmed
    # this was the reason every real run in this project used gpt-4.1 (the
    # .env value) instead of the gpt-5.4-mini several scripts explicitly
    # set beforehand -- the setting never took effect.
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if override or key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip('"').strip("'")
