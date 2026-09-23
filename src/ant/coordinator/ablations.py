"""Runtime profiles for the paper's adaptive-coordination ablations.

Each profile is enforced by :class:`LocalCoordinator`, not merely described
to the orchestrator in prompt text.  Keeping the definitions in one small
module also makes the profile name written to an evaluation manifest an
unambiguous experimental condition.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveCoordinationProfile:
    """The runtime mechanisms available to one ANTMAN evaluation variant."""

    key: str
    display_name: str
    uses_explicit_need_state: bool
    allow_need_revision: bool
    allow_adaptive_rerouting: bool
    allow_recovery: bool

    @property
    def is_full(self) -> bool:
        return (
            self.uses_explicit_need_state
            and
            self.allow_need_revision
            and self.allow_adaptive_rerouting
            and self.allow_recovery
        )


FULL = AdaptiveCoordinationProfile(
    key="full",
    display_name="Full ANTMAN",
    uses_explicit_need_state=True,
    allow_need_revision=True,
    allow_adaptive_rerouting=True,
    allow_recovery=True,
)
STATIC = AdaptiveCoordinationProfile(
    key="static",
    display_name="Static ANTMAN",
    uses_explicit_need_state=True,
    allow_need_revision=False,
    allow_adaptive_rerouting=False,
    allow_recovery=False,
)
NO_NEED_REVISION = AdaptiveCoordinationProfile(
    key="no_need_revision",
    display_name="w/o Need Revision",
    uses_explicit_need_state=True,
    allow_need_revision=False,
    allow_adaptive_rerouting=True,
    allow_recovery=True,
)
NO_ADAPTIVE_REROUTING = AdaptiveCoordinationProfile(
    key="no_adaptive_rerouting",
    display_name="w/o Adaptive Rerouting",
    uses_explicit_need_state=True,
    allow_need_revision=True,
    allow_adaptive_rerouting=False,
    allow_recovery=True,
)
NO_RECOVERY = AdaptiveCoordinationProfile(
    key="no_recovery",
    display_name="w/o Recovery",
    uses_explicit_need_state=True,
    allow_need_revision=True,
    allow_adaptive_rerouting=True,
    allow_recovery=False,
)
GRAPH_FREE_ADAPTIVE = AdaptiveCoordinationProfile(
    key="graph_free_adaptive",
    display_name="Graph-free Adaptive",
    uses_explicit_need_state=False,
    allow_need_revision=False,
    allow_adaptive_rerouting=True,
    allow_recovery=True,
)

PROFILES = {
    profile.key: profile
    for profile in (
        FULL,
        STATIC,
        NO_NEED_REVISION,
        NO_ADAPTIVE_REROUTING,
        NO_RECOVERY,
        GRAPH_FREE_ADAPTIVE,
    )
}


_ACTIVE_PROFILE: ContextVar[AdaptiveCoordinationProfile] = ContextVar(
    "active_adaptive_coordination_profile",
    default=FULL,
)


def resolve_profile(
    profile: str | AdaptiveCoordinationProfile = FULL,
) -> AdaptiveCoordinationProfile:
    """Return a declared profile, rejecting spelling drift before a paid run."""

    if isinstance(profile, AdaptiveCoordinationProfile):
        return profile
    try:
        return PROFILES[profile]
    except KeyError:
        raise ValueError(
            f"Unknown adaptive-coordination profile {profile!r}; "
            f"expected one of {sorted(PROFILES)}."
        ) from None


def active_coordination_profile() -> AdaptiveCoordinationProfile:
    """Return the profile scoped to the currently executing evaluation."""

    return _ACTIVE_PROFILE.get()


@contextmanager
def use_coordination_profile(
    profile: str | AdaptiveCoordinationProfile,
) -> Iterator[AdaptiveCoordinationProfile]:
    """Scope an ablation profile without changing shared adapter APIs."""

    resolved = resolve_profile(profile)
    token = _ACTIVE_PROFILE.set(resolved)
    try:
        yield resolved
    finally:
        _ACTIVE_PROFILE.reset(token)
