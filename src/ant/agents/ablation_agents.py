"""Evaluation-only wrappers for adaptive-coordination ablations.

The wrappers keep the production adapters' constructor contract unchanged.
They scope a coordinator profile around one complete task run, so this module
can coexist with independently evolving worker-model adapter parameters.
"""

from __future__ import annotations

from pathlib import Path

from ant.agents.ant_adapter import AntAgent
from ant.agents.ant_document_adapter import AntDocumentAgent
from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.coordinator.ablations import (
    AdaptiveCoordinationProfile,
    resolve_profile,
    use_coordination_profile,
)


def _annotate_profile(
    result: AgentResult, profile: AdaptiveCoordinationProfile
) -> AgentResult:
    return result.model_copy(
        update={
            "metadata": {
                **result.metadata,
                "coordination_profile": profile.key,
                "coordination_profile_label": profile.display_name,
            }
        }
    )


class AblationAntAgent(AntAgent):
    """Repository adapter executed under one declared ablation profile."""

    def __init__(
        self,
        coordination_profile: str | AdaptiveCoordinationProfile,
        model: str = "gpt-4.1",
        max_rounds: int = 6,
        index_root: Path | None = None,
    ) -> None:
        self.coordination_profile = resolve_profile(coordination_profile)
        if self.coordination_profile.is_full:
            raise ValueError("AblationAntAgent requires a non-full profile")
        super().__init__(model=model, max_rounds=max_rounds, index_root=index_root)
        self.name = f"ant_{self.coordination_profile.key}"

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        with use_coordination_profile(self.coordination_profile):
            result = super().run(example, environment_root)
        return _annotate_profile(result, self.coordination_profile)


class AblationAntDocumentAgent(AntDocumentAgent):
    """Document adapter executed under one declared ablation profile."""

    def __init__(
        self,
        coordination_profile: str | AdaptiveCoordinationProfile,
        model: str = "gpt-4.1",
        max_rounds: int = 6,
        search_top_k: int = 4,
        index_root: Path | None = None,
    ) -> None:
        self.coordination_profile = resolve_profile(coordination_profile)
        if self.coordination_profile.is_full:
            raise ValueError("AblationAntDocumentAgent requires a non-full profile")
        super().__init__(
            model=model,
            max_rounds=max_rounds,
            search_top_k=search_top_k,
            index_root=index_root,
        )
        self.name = f"ant_document_{self.coordination_profile.key}"

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
    register_agent(AblationAntAgent(_profile))
    register_agent(AblationAntDocumentAgent(_profile))
