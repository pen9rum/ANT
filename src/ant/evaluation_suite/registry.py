from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ant.agents.base import AgentAdapter
    from ant.benchmarks.base import BenchmarkAdapter

# Deliberately plain dicts, not a plugin/entry-point system -- the whole
# suite is a handful of adapters known ahead of time, not a third-party
# extension surface. Each concrete adapter module registers itself at
# import time (see e.g. benchmarks/sweqa_pro.py's own bottom-of-module
# `register_benchmark(...)` call), so importing an adapter module is what
# makes it discoverable, matching Python's own ordinary import semantics
# instead of a separate manual wiring step that can silently drift out of
# sync with what actually exists.
_BENCHMARKS: dict[str, BenchmarkAdapter] = {}
_AGENTS: dict[str, AgentAdapter] = {}


def register_benchmark(adapter: BenchmarkAdapter) -> None:
    _BENCHMARKS[adapter.name] = adapter


def register_agent(adapter: AgentAdapter) -> None:
    _AGENTS[adapter.name] = adapter


def get_benchmark(name: str) -> BenchmarkAdapter:
    try:
        return _BENCHMARKS[name]
    except KeyError:
        raise KeyError(
            f"Unknown benchmark {name!r} -- registered: {sorted(_BENCHMARKS)}. "
            "Did you forget to import its adapter module first?"
        ) from None


def get_agent(name: str) -> AgentAdapter:
    try:
        return _AGENTS[name]
    except KeyError:
        raise KeyError(
            f"Unknown agent {name!r} -- registered: {sorted(_AGENTS)}. "
            "Did you forget to import its adapter module first?"
        ) from None


def registered_benchmarks() -> list[str]:
    return sorted(_BENCHMARKS)


def registered_agents() -> list[str]:
    return sorted(_AGENTS)
