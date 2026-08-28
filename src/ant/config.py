from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path = Path(".env"), override: bool = False) -> dict[str, str]:
    # override=False: .env fills in variables the process environment
    # doesn't already have, never clobbers ones a caller explicitly set --
    # the standard dotenv contract. This used to default to True, called
    # unconditionally from OpenAIProvider.__init__, which meant .env's
    # ANT_MODEL silently re-applied itself over any os.environ["ANT_MODEL"]
    # a caller had already set before constructing a provider: confirmed
    # this was the reason every real run in this project used gpt-4.1 (the
    # .env value) instead of the gpt-5.4-mini several scripts explicitly
    # set beforehand -- the setting never took effect.
    #
    # Returns every key=value pair actually present in the file, regardless
    # of whether override applied it to os.environ -- OpenAIProvider needs
    # this: OPENAI_API_KEY/OPENAI_ORG_ID/OPENAI_PROJECT_ID are a matched
    # set that must come from the same source together, and override=False
    # alone can silently split that set across two sources. Confirmed
    # directly: a stale OPENAI_API_KEY already set at the OS level (from
    # some unrelated, unremembered earlier context) took precedence over
    # .env's own (self-consistent, correct) key once override became
    # False, while OPENAI_ORG_ID/OPENAI_PROJECT_ID -- not set at the OS
    # level -- still came from .env, pairing a key with an organization it
    # does not belong to and failing every request with a
    # "mismatched_organization" 401. The credential trio needs to be
    # sourced atomically from .env when .env defines it, not filled in
    # var-by-var against whatever the ambient environment happens to hold.
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return values
