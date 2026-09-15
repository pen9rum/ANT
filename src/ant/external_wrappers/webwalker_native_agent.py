"""Interface STUB for the OFFICIAL WebWalker Explorer+Critic baseline --
NOT yet checked out or runnable. See `docs/web_track_baselines.md`'s own
"## WebWalker" section for the (not-yet-executed) setup plan: clone
`github.com/Alibaba-NLP/DeepResearch` (`WebAgent/WebWalker/` subpath --
the original `Alibaba-NLP/WebWalker` URL 301-redirects here, verified),
isolate its Crawl4AI/Qwen-Agent/LangChain stack in a fresh venv, run
`crawl4ai-setup`, verify one real question end-to-end before any
benchmark-scale attempt.

Unlike `sweqa_pro_native_agent.py` (whose official repo IS already
checked out, pinned, and verified live), no commit has been pinned here
yet and `third_party/checkouts/webwalkerqa/` does not exist -- this class
exists only to fix the CONSTRUCTOR/CONFIG shape ahead of that setup work,
per the governing spec's Step 7 ("Create compatible interfaces/config
stubs ... Do not launch any inference").
"""

from __future__ import annotations

from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample

WEBWALKER_REPO_URL = "https://github.com/Alibaba-NLP/DeepResearch"
WEBWALKER_SUBPATH = "WebAgent/WebWalker"
# No commit pinned yet -- see this module's own docstring. Filled in only
# once the setup plan in docs/web_track_baselines.md is actually executed.
WEBWALKER_COMMIT: str | None = None


class WebWalkerNativeAgentNotCheckedOut(RuntimeError):
    pass


class WebWalkerNativeAgent:
    """Would wrap the official Explorer+Critic loop (see
    `docs/webwalkerqa_ant_mapping.md` section 5 for the audited
    architecture) once checked out and isolated. `model` is accepted now
    so the constructor shape matches every other method in this suite,
    but GPT-4.1 substitution for this specific baseline is, per
    `docs/web_track_baselines.md`, "architecturally plausible but NOT
    paper-verified" -- must be confirmed live before being treated as a
    validated configuration, not assumed here.
    """

    name = "webwalker_native_agent"

    def __init__(self, model: str = "gpt-4.1") -> None:
        self.model = model

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        raise WebWalkerNativeAgentNotCheckedOut(
            f"The official WebWalker repo ({WEBWALKER_REPO_URL}/{WEBWALKER_SUBPATH}) has "
            "not been checked out or pinned yet -- see docs/web_track_baselines.md's "
            "'## WebWalker' setup plan (not executed). This class only fixes the "
            "constructor/config interface shape ahead of that work; per the governing "
            "spec's Step 7, no inference is launched from this environment-validation pass."
        )
