"""Evaluation-only wrappers running ANTMAN-over-GAIA under one declared
adaptive-coordination ablation profile (paper Table 5/10) -- the GAIA
counterpart of `ant.agents.ablation_agents.AblationAntAgent`.

No change to `ant_gaia.AntGaiaAgent` or `gaia_tools.GaiaCoordinatorSearchTool`
was needed: `coordination_profile` is read by `LocalCoordinator.ask()` from
a ContextVar when its own `coordination_profile` parameter is left unset
(see `ant.coordinator.ablations.active_coordination_profile`), so scoping
it around an unmodified `AntGaiaAgent.run()` call is sufficient -- exactly
the same seam the repo-QA ablation wrappers use.

WHY GAIA'S WORKER-REASONING FIX GENERALIZES FOR FREE: `AutonomousWorker`
is constructed identically (`reasoner=self.worker_reasoner`) by
`LocalCoordinator._run_selected_workers` regardless of which profile is
active, INCLUDING inside `_ask_graph_free`'s own code path (Graph-free
Adaptive still calls `_run_selected_workers`, just from a different outer
loop). `GaiaCoordinatorSearchTool.search()`'s own worker-reasoning steps
(query refinement, follow-up fetch selection -- see that class's
docstring) therefore run under every profile exactly as they do for full
ANTMAN, with no GAIA-specific ablation wiring required here.

Ablations run ONLY on the frozen 40-question stratified subset
(`gaia_ablation40_manifest.json`), never the full GAIA-Text-103 -- see
that manifest's own docstring/provenance for why (cost/scope, matching
the paper's own ablation protocol).
"""

from __future__ import annotations

from pathlib import Path

from ant.agents.ant_gaia import AntGaiaAgent
from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.coordinator.ablations import (
    AdaptiveCoordinationProfile,
    resolve_profile,
    use_coordination_profile,
)


def _annotate_profile(result: AgentResult, profile: AdaptiveCoordinationProfile) -> AgentResult:
    return result.model_copy(
        update={
            "metadata": {
                **result.metadata,
                "coordination_profile": profile.key,
                "coordination_profile_label": profile.display_name,
            }
        }
    )


class AblationAntGaiaAgent(AntGaiaAgent):
    """ANTMAN-over-GAIA executed under one declared ablation profile.

    Accepts the same `worker_model`/`worker_base_url` seam as `AntGaiaAgent`
    itself, so an ablation x worker-model cross (e.g. "Static ANTMAN, but
    with the Qwen3-8B worker") is expressible, though the paper's own
    ablation table only reports the full-GPT-4.1 cross -- left available
    rather than blocked, since nothing about the profile mechanism assumes
    a particular worker model.
    """

    def __init__(
        self,
        coordination_profile: str | AdaptiveCoordinationProfile,
        model: str = "gpt-4.1",
        max_rounds: int = 6,
        worker_model: str | None = None,
        worker_base_url: str | None = None,
        worker_max_context_tokens: int | None = None,
    ) -> None:
        self.coordination_profile = resolve_profile(coordination_profile)
        if self.coordination_profile.is_full:
            raise ValueError("AblationAntGaiaAgent requires a non-full profile")
        super().__init__(
            model=model,
            max_rounds=max_rounds,
            worker_model=worker_model,
            worker_base_url=worker_base_url,
            worker_max_context_tokens=worker_max_context_tokens,
        )
        self.name = f"ant_gaia_{self.coordination_profile.key}"

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        with use_coordination_profile(self.coordination_profile):
            result = super().run(example, environment_root)
        return _annotate_profile(result, self.coordination_profile)


from ant.evaluation_suite.registry import register_agent  # noqa: E402

for _profile in (
    "static",
    "graph_free_adaptive",
    "no_need_revision",
    "no_adaptive_rerouting",
    "no_recovery",
):
    register_agent(AblationAntGaiaAgent(_profile))
