"""Tests for the shared GAIA tool registry.

The load-bearing tests here are the FAIRNESS ones: that a
Matched-ReAct-shaped agent and an ANTMAN-shaped agent genuinely share one
capability surface, and that neither can reach a capability the other
cannot. Everything is exercised against synthetic fixtures with stub
backends -- no network, no LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ant.agents.gaia_tools import (
    GAIA_TOOL_NAMES,
    GaiaCoordinatorSearchTool,
    GaiaToolRegistry,
    SearchHit,
    ToolUnavailableError,
    build_gaia_search_tool_factory,
    serialize_call_log,
)
from ant.evaluation_suite.gaia_fixtures import materialize_all
from ant.evaluation_suite.gaia_scope import GaiaEnvironment, UnsupportedModalityError

FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "manifests"
    / "gaia"
    / "synthetic_fixtures.json"
)


class StubSearchBackend:
    """Deterministic stand-in for a live search engine."""

    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, limit: int):
        self.queries.append((query, limit))
        return [
            SearchHit(
                title=f"Result {i}",
                url=f"https://example.invalid/{i}",
                snippet=f"snippet {i}",
            )
            for i in range(min(limit, 3))
        ]


class StubFetchBackend:
    def fetch(self, url: str) -> str:
        return f"page text for {url}"


@pytest.fixture
def registry(tmp_path: Path) -> GaiaToolRegistry:
    materialize_all(tmp_path, FIXTURES)
    return GaiaToolRegistry(
        environment=GaiaEnvironment(tmp_path, "field_notes.txt"),
        search_backend=StubSearchBackend(),
        fetch_backend=StubFetchBackend(),
    )


# --------------------------------------------------------------------
# Tool surface
# --------------------------------------------------------------------


def test_declared_tool_surface_is_the_five_primitives(registry: GaiaToolRegistry):
    assert registry.available_tools() == GAIA_TOOL_NAMES
    assert set(GAIA_TOOL_NAMES) == {
        "search",
        "open_url",
        "inspect_file",
        "inspect_table",
        "run_python",
    }


def test_tool_surface_is_ordered_for_prompt_stability(registry: GaiaToolRegistry):
    """A set would make any prompt rendered from this non-deterministic."""
    assert isinstance(registry.available_tools(), tuple)
    assert registry.available_tools() == registry.available_tools()


def test_every_declared_tool_is_dispatchable_by_name(registry: GaiaToolRegistry):
    for name in registry.available_tools():
        assert callable(getattr(registry, name))


def test_invoke_rejects_an_unknown_tool(registry: GaiaToolRegistry):
    with pytest.raises(ToolUnavailableError, match="Unknown GAIA tool"):
        registry.invoke("read_email", url="x")


# --------------------------------------------------------------------
# Missing backends fail loudly rather than returning empty
# --------------------------------------------------------------------


def test_search_without_a_backend_raises_instead_of_returning_empty(tmp_path: Path):
    """An empty result set and "no search engine is wired up" are
    different facts; conflating them fakes an accuracy number."""
    bare = GaiaToolRegistry(environment=GaiaEnvironment(tmp_path, None))
    with pytest.raises(ToolUnavailableError, match="configuration error"):
        bare.search("anything")


def test_open_url_without_a_backend_raises(tmp_path: Path):
    bare = GaiaToolRegistry(environment=GaiaEnvironment(tmp_path, None))
    with pytest.raises(ToolUnavailableError):
        bare.open_url("https://example.invalid/")


def test_a_missing_backend_still_records_the_attempt(tmp_path: Path):
    bare = GaiaToolRegistry(environment=GaiaEnvironment(tmp_path, None))
    with pytest.raises(ToolUnavailableError):
        bare.search("q")
    log = bare.log_as_dicts()
    assert len(log) == 1
    assert log[0]["ok"] is False
    assert log[0]["tool"] == "search"


# --------------------------------------------------------------------
# REQUIRED: tool calls are logged deterministically
# --------------------------------------------------------------------


def test_tool_calls_are_logged_in_order_with_monotonic_sequence(registry: GaiaToolRegistry):
    registry.search("alpha", limit=2)
    registry.open_url("https://example.invalid/x")
    registry.inspect_file()
    log = registry.log_as_dicts()
    assert [entry["tool"] for entry in log] == ["search", "open_url", "inspect_file"]
    assert [entry["seq"] for entry in log] == [0, 1, 2]
    assert all(entry["ok"] for entry in log)


def test_call_log_records_arguments(registry: GaiaToolRegistry):
    registry.search("alpha", limit=2)
    assert registry.log_as_dicts()[0]["arguments"] == {"limit": 2, "query": "alpha"}


def test_call_log_is_byte_identical_across_identical_runs(tmp_path: Path):
    """Determinism is the whole point of the log: a diff must mean a real
    behavioural change, never timestamp jitter."""

    def run() -> str:
        root = tmp_path / "run"
        root.mkdir(exist_ok=True)
        materialize_all(root, FIXTURES)
        reg = GaiaToolRegistry(
            environment=GaiaEnvironment(root, "quarterly_widgets.csv"),
            search_backend=StubSearchBackend(),
            fetch_backend=StubFetchBackend(),
        )
        reg.search("widgets", limit=2)
        reg.inspect_table(max_rows=3)
        reg.run_python("2 + 2")
        return serialize_call_log(reg)

    assert run() == run()


def test_call_log_contains_no_timestamp_or_duration_fields(registry: GaiaToolRegistry):
    registry.run_python("1 + 1")
    entry = registry.log_as_dicts()[0]
    assert set(entry) == {"seq", "tool", "arguments", "ok", "summary", "error"}


def test_a_failed_tool_call_is_logged_before_the_exception_propagates(tmp_path: Path):
    materialize_all(tmp_path, FIXTURES)
    reg = GaiaToolRegistry(environment=GaiaEnvironment(tmp_path, "diagram.png"))
    with pytest.raises(UnsupportedModalityError):
        reg.inspect_file()
    log = reg.log_as_dicts()
    assert log[0]["ok"] is False
    assert "image" in log[0]["error"]


def test_tool_call_count_tracks_the_log(registry: GaiaToolRegistry):
    assert registry.tool_call_count() == 0
    registry.run_python("3 * 3")
    assert registry.tool_call_count() == 1


# --------------------------------------------------------------------
# run_python() -- the sandboxed execution primitive
#
# The sandbox's OWN isolation guarantees are tested in
# tests/test_gaia_sandbox.py. What is tested HERE is the registry's
# contract over it: that failures are reported rather than raised, that
# the call log stays honest, and that the execution budget belongs to
# the registry rather than the caller (a fairness property).
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [("2 + 2", "4"), ("10 / 4", "2.5"), ("2 ** 10", "1024"), ("(3 + 4) * 2", "14")],
)
def test_run_python_evaluates_arithmetic(registry: GaiaToolRegistry, code, expected):
    assert registry.run_python(code) == expected


def test_run_python_executes_real_programs_not_just_expressions(registry: GaiaToolRegistry):
    """This is the capability the restricted-AST `compute()` did not
    have and is the reason it was replaced."""
    output = registry.run_python(
        "import statistics\nvalues = [3, 1, 4, 1, 5, 9, 2, 6]\nstatistics.median(values)"
    )
    assert "3.5" in output


def test_run_python_reports_failure_instead_of_raising(registry: GaiaToolRegistry):
    """An agent must be able to READ its own failure to correct itself;
    raising would turn a debugging loop into a dead end."""
    output = registry.run_python("1 / 0")
    assert "ZeroDivisionError" in output


def test_a_failed_program_is_logged_as_a_failed_call(registry: GaiaToolRegistry):
    """Reported-not-raised must NOT mean recorded-as-success."""
    registry.run_python("1 / 0")
    entry = registry.log_as_dicts()[0]
    assert entry["ok"] is False
    assert entry["error"]


def test_run_python_is_blocked_from_the_repository_and_the_network(
    registry: GaiaToolRegistry,
):
    """A smoke check that the registry really is going through the
    sandbox rather than some in-process shortcut. The exhaustive escape
    battery lives in tests/test_gaia_sandbox.py."""
    assert "SandboxDenied" in registry.run_python("import socket")
    assert "SandboxDenied" in registry.run_python("import os\nos.system('echo hi')")


def test_run_python_cannot_reach_the_tasks_attachment(registry: GaiaToolRegistry):
    """If it could, it would be a way around the modality policy: a
    method could read an image's bytes and gain a perception capability
    the substrate withholds from every method equally."""
    path = registry.environment.attachment_path()
    output = registry.run_python(f"open({str(path)!r}).read()")
    assert "SandboxDenied" in output


def test_the_registry_owns_the_execution_budget_not_the_caller(registry: GaiaToolRegistry):
    """A caller-chosen timeout would mean "ANTMAN got 60s and the
    baseline got 10s" could happen silently. The limits are registry
    fields, and run_python() takes no budget argument."""
    import inspect

    parameters = set(inspect.signature(GaiaToolRegistry.run_python).parameters)
    assert parameters == {"self", "code"}
    assert registry.python_timeout_seconds > 0
    assert registry.python_memory_bytes > 0


def test_both_shapes_share_one_execution_budget(registry: GaiaToolRegistry):
    react, antman = ReActShapedAgent(registry), AntmanShapedAgent(registry)
    assert react.registry.python_timeout_seconds == antman.registry.python_timeout_seconds
    assert react.registry.python_memory_bytes == antman.registry.python_memory_bytes


# --------------------------------------------------------------------
# REQUIRED: both agent shapes instantiate over ONE shared registry
# --------------------------------------------------------------------


class ReActShapedAgent:
    """Minimal stand-in for a Matched-ReAct-style GAIA baseline: drives
    the registry by tool NAME through `invoke()`, exactly as a ReAct loop
    acting on model-chosen tool names would."""

    def __init__(self, registry: GaiaToolRegistry) -> None:
        self.registry = registry

    def capabilities(self) -> tuple[str, ...]:
        return self.registry.available_tools()

    def step(self, tool: str, **kwargs):
        return self.registry.invoke(tool, **kwargs)


class AntmanShapedAgent:
    """Minimal stand-in for the ANTMAN GAIA agent: consumes the SAME
    registry through the duck-typed coordinator seam tool rather than by
    tool name."""

    def __init__(self, registry: GaiaToolRegistry) -> None:
        self.registry = registry
        self.search_tool_factory = build_gaia_search_tool_factory(registry)

    def capabilities(self) -> tuple[str, ...]:
        return self.registry.available_tools()

    def coordinator_tool(self):
        return self.search_tool_factory(None, None)


def test_both_agent_shapes_instantiate_over_the_same_registry(registry: GaiaToolRegistry):
    react = ReActShapedAgent(registry)
    antman = AntmanShapedAgent(registry)
    assert react.registry is antman.registry


def test_both_agent_shapes_report_identical_capabilities(registry: GaiaToolRegistry):
    assert ReActShapedAgent(registry).capabilities() == AntmanShapedAgent(registry).capabilities()


def test_antman_shape_adds_no_backend_of_its_own(registry: GaiaToolRegistry):
    """The fairness invariant, asserted by identity: the coordinator seam
    tool wraps the shared registry and owns nothing extra."""
    tool = AntmanShapedAgent(registry).coordinator_tool()
    assert tool.registry is registry
    assert tool.registry.search_backend is registry.search_backend
    assert tool.registry.environment is registry.environment


def test_both_shapes_share_one_call_log(registry: GaiaToolRegistry):
    """Shared accounting falls out of the shared registry: whichever
    shape makes a call, the same ledger records it."""
    react = ReActShapedAgent(registry)
    antman = AntmanShapedAgent(registry)
    react.step("run_python", code="1 + 1")
    antman.coordinator_tool().search("stations", files=[], limit=2)
    tools_used = [entry["tool"] for entry in registry.log_as_dicts()]
    assert "run_python" in tools_used
    assert "search" in tools_used


def test_antman_shape_cannot_read_a_modality_the_react_shape_cannot(tmp_path: Path):
    """The privileged-capability check: an unsupported attachment must be
    unreadable through BOTH access paths."""
    materialize_all(tmp_path, FIXTURES)
    reg = GaiaToolRegistry(environment=GaiaEnvironment(tmp_path, "briefing.mp3"))
    with pytest.raises(UnsupportedModalityError):
        ReActShapedAgent(reg).step("inspect_file")
    # The coordinator seam must not smuggle the same bytes out by another
    # route -- it returns no attachment evidence rather than content.
    evidence = AntmanShapedAgent(reg).coordinator_tool()._attachment_evidence("survey", 5)
    assert evidence == []


# --------------------------------------------------------------------
# Coordinator seam
# --------------------------------------------------------------------


def test_search_tool_factory_matches_the_coordinator_seam_signature(registry: GaiaToolRegistry):
    """`LocalCoordinator` calls `search_tool_factory(repo_root, index_path)`;
    GAIA has neither, so both are accepted and ignored."""
    factory = build_gaia_search_tool_factory(registry)
    tool = factory(Path("irrelevant"), None)
    assert isinstance(tool, GaiaCoordinatorSearchTool)


def test_seam_tool_exposes_the_methods_the_coordinator_calls(registry: GaiaToolRegistry):
    tool = GaiaCoordinatorSearchTool(registry)
    for method in (
        "search",
        "dense_search",
        "rank_symbols",
        "resolve_symbol",
        "navigate",
        "references",
        "indexed_callers",
        "callers",
        "callees",
        "assignments",
        "imports",
        "subclasses",
        "read_region",
    ):
        assert callable(getattr(tool, method))


def test_seam_search_returns_evidence_from_web_and_attachment(registry: GaiaToolRegistry):
    evidence = GaiaCoordinatorSearchTool(registry).search("Redwood", files=[], limit=8)
    assert any("example.invalid" in item.path for item in evidence)
    assert any("Redwood Bluff" in item.quote for item in evidence)


def test_seam_dense_search_is_an_honest_no_op(registry: GaiaToolRegistry):
    """No embedding index exists for this substrate; fabricating dense
    hits would inject ungrounded evidence into the coordinator."""
    assert GaiaCoordinatorSearchTool(registry).dense_search("x", files=[], limit=4) == []


def test_seam_symbol_methods_return_empty_rather_than_faking_structure(registry: GaiaToolRegistry):
    tool = GaiaCoordinatorSearchTool(registry)
    assert tool.callers("foo", files=[]) == []
    assert tool.subclasses("Foo", files=[]) == []
    assert tool.rank_symbols(["a"], files=[]) == []
