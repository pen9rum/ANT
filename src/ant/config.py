from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path = Path(".env"), override: bool = True) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if override or key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip('"').strip("'")
