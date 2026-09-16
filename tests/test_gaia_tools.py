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
        "compute",
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
        reg.compute("2 + 2")
        return serialize_call_log(reg)

    assert run() == run()


def test_call_log_contains_no_timestamp_or_duration_fields(registry: GaiaToolRegistry):
    registry.compute("1 + 1")
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
    registry.compute("3 * 3")
    assert registry.tool_call_count() == 1


# --------------------------------------------------------------------
# compute()
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("2 + 2", "4"), ("10 / 4", "2.5"), ("2 ** 10", "1024"), ("(3 + 4) * 2", "14")],
)
def test_compute_evaluates_arithmetic(registry: GaiaToolRegistry, expression, expected):
    assert registry.compute(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd').read()",
        "some_name",
        "[x for x in range(3)]",
        "(1).__class__",
    ],
)
def test_compute_refuses_anything_beyond_arithmetic(registry: GaiaToolRegistry, expression):
    """compute() is explicitly NOT a Python sandbox -- it evaluates a
    restricted grammar so that a model-chosen string can never become an
    arbitrary-code-execution primitive."""
    with pytest.raises((ValueError, SyntaxError)):
        registry.compute(expression)


def test_compute_refuses_an_expression_that_would_hang_the_process(registry: GaiaToolRegistry):
    with pytest.raises(ValueError, match="ceiling"):
        registry.compute("9 ** 999999")


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
    react.step("compute", expression="1 + 1")
    antman.coordinator_tool().search("stations", files=[], limit=2)
    tools_used = [entry["tool"] for entry in registry.log_as_dicts()]
    assert "compute" in tools_used
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
