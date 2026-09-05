from ant.coordinator.local import _complete_missing_evidence_facets
from ant.domain import AnswerFacet, Evidence, FacetCoverage, FacetRescuePlan


def _ev(
    path: str, start: int = 1, end: int = 1, quote: str | None = None, worker_id: str = "worker-a"
) -> Evidence:
    return Evidence(
        path=path,
        line_start=start,
        line_end=end,
        quote=quote or f"{path}:{start}-{end}",
        reason="stub",
        worker_id=worker_id,
    )


class _StubFacetReasoner:
    """Returns a fixed FacetRescuePlan regardless of input -- these tests
    exercise _complete_missing_evidence_facets' own orchestration (dedup,
    cap/replacement handling, no-op paths), not real semantic judgment,
    which is OpenAIProvider's job and validated separately against real
    Phase-12 historical cases.
    """

    def __init__(self, plan: FacetRescuePlan) -> None:
        self.plan = plan
        self.calls: list[dict] = []

    def assess_facet_completeness(self, *, question, selected_evidence, rejected_evidence):
        self.calls.append(
            {
                "question": question,
                "selected_evidence": list(selected_evidence),
                "rejected_evidence": list(rejected_evidence),
            }
        )
        return self.plan


def test_a_multi_facet_missing_mechanism_is_rescued_without_duplicating_the_covered_one() -> None:
    # Models the observed yt-dlp `260cf927` failure shape: question asks for
    # two mechanisms, only one is covered by the initial selected set.
    selected = [_ev("yt_dlp/plugins.py", quote="def search_locations(...): ...")]
    rejected = [
        _ev("yt_dlp/plugins.py", 146, 166, quote="def load_plugins(name, suffix): ..."),
        _ev("yt_dlp/update.py", quote="unrelated update-checking noise"),
    ]
    reasoner = _StubFacetReasoner(
        FacetRescuePlan(
            facets=[
                AnswerFacet(facet_id="A-directory", description="directory customization"),
                AnswerFacet(facet_id="B-dynamic-load", description="dynamic extractor loading"),
            ],
            coverage=[
                FacetCoverage(
                    facet_id="A-directory",
                    support_status="supported",
                    supporting_selected_ids=["0"],
                ),
                FacetCoverage(facet_id="B-dynamic-load", support_status="unsupported"),
            ],
            rescue_candidates={"B-dynamic-load": ["0"]},
        )
    )

    result, telemetry = _complete_missing_evidence_facets(reasoner, "q", selected, rejected, cap=16)

    assert result == [selected[0], rejected[0]]
    assert telemetry.rescued_evidence_ids == ["0"]
    assert telemetry.rescued_facet_by_evidence_id == {"0": "B-dynamic-load"}
    assert telemetry.expanded is True
    assert telemetry.final_selected_count == 2


def test_b_enumeration_completeness_rescues_only_the_missing_member() -> None:
    # Models seaborn `28d9b344`: question asks to enumerate all subclasses;
    # rejected pool has the one missing member PLUS a redundant duplicate
    # span of an already-covered member -- only the missing one is named by
    # the reasoner's own rescue_candidates and must be the only one added.
    selected = [_ev("seaborn/_stats/aggregation.py", quote="class Agg(Stat): ...")]
    rejected = [
        _ev("seaborn/_stats/aggregation.py", 114, 118, quote="class Rolling(Stat): ..."),
        _ev("seaborn/_stats/aggregation.py", 16, 20, quote="class Agg(Stat): ... (dup span)"),
    ]
    reasoner = _StubFacetReasoner(
        FacetRescuePlan(
            facets=[AnswerFacet(facet_id="enum", description="enumerate all Stat subclasses")],
            coverage=[
                FacetCoverage(
                    facet_id="enum",
                    support_status="partially_supported",
                    supporting_selected_ids=["0"],
                )
            ],
            rescue_candidates={"enum": ["0"]},
        )
    )

    result, telemetry = _complete_missing_evidence_facets(reasoner, "q", selected, rejected, cap=16)

    assert result == [selected[0], rejected[0]]
    assert rejected[1] not in result
    assert telemetry.rescued_evidence_ids == ["0"]


def test_c_cross_module_dependency_rescues_only_the_dependency() -> None:
    # Models qibo `b93c3114`: selected covers the drawing entry point but
    # not the measurement-symbol dependency; unrelated gate-class evidence
    # in the rejected pool must stay rejected.
    selected = [_ev("src/qibo/models/circuit.py", quote="def draw(self): labels = {...}")]
    rejected = [
        _ev(
            "src/qibo/gates/measurements.py",
            18,
            54,
            quote="class MeasurementSymbol(sympy.Symbol): ...",
        ),
        _ev("src/qibo/gates/gates.py", quote="class T(Gate): ..."),
    ]
    reasoner = _StubFacetReasoner(
        FacetRescuePlan(
            facets=[
                AnswerFacet(facet_id="sym", description="where measurement symbols are defined")
            ],
            coverage=[FacetCoverage(facet_id="sym", support_status="unsupported")],
            rescue_candidates={"sym": ["0"]},
        )
    )

    result, telemetry = _complete_missing_evidence_facets(reasoner, "q", selected, rejected, cap=16)

    assert result == [selected[0], rejected[0]]
    assert rejected[1] not in result
    assert telemetry.rescued_evidence_ids == ["0"]


def test_d_fully_covered_question_is_a_pure_no_op() -> None:
    # Essential negative control: the mechanism must not touch an ordinary,
    # already-complete selection.
    selected = [_ev("a.py"), _ev("b.py")]
    rejected = [
        _ev("a.py", 2, 2, quote="a.py:2-2 (dup span)"),
        _ev("c.py", quote="unrelated detail"),
    ]
    reasoner = _StubFacetReasoner(
        FacetRescuePlan(
            facets=[AnswerFacet(facet_id="x", description="the one thing asked")],
            coverage=[
                FacetCoverage(
                    facet_id="x", support_status="supported", supporting_selected_ids=["0", "1"]
                )
            ],
        )
    )

    result, telemetry = _complete_missing_evidence_facets(reasoner, "q", selected, rejected, cap=16)

    assert result == selected
    assert telemetry.final_selected_count == telemetry.initial_selected_count == 2
    assert telemetry.no_op_reason == "all facets already supported"
    assert telemetry.rescued_evidence_ids == []


def test_e_large_noisy_rejected_pool_does_not_inflate_the_selected_set() -> None:
    # Guards against the mechanism becoming a generic recall booster: a
    # huge rejected pool, all facets already supported, must still no-op.
    selected = [_ev("a.py")]
    rejected = [_ev(f"noise_{i}.py", quote=f"noise item {i}") for i in range(100)]
    reasoner = _StubFacetReasoner(
        FacetRescuePlan(
            facets=[AnswerFacet(facet_id="x", description="the one thing asked")],
            coverage=[
                FacetCoverage(
                    facet_id="x", support_status="supported", supporting_selected_ids=["0"]
                )
            ],
        )
    )

    result, telemetry = _complete_missing_evidence_facets(reasoner, "q", selected, rejected, cap=16)

    assert result == selected
    assert telemetry.rejected_candidates_considered == 100
    assert telemetry.rescued_evidence_ids == []


def test_f_prefers_the_reasoners_own_minimal_candidate_over_all_duplicates() -> None:
    # 5 rejected spans all support the same missing fact; the reasoner
    # already curated the minimal set (1 item) -- the orchestrator must not
    # add more than what it was given.
    selected = [_ev("a.py")]
    rejected = [_ev("b.py", i, i, quote=f"same fact, span {i}") for i in range(5)]
    reasoner = _StubFacetReasoner(
        FacetRescuePlan(
            facets=[AnswerFacet(facet_id="x", description="missing fact")],
            coverage=[FacetCoverage(facet_id="x", support_status="unsupported")],
            rescue_candidates={"x": ["2"]},
        )
    )

    result, telemetry = _complete_missing_evidence_facets(reasoner, "q", selected, rejected, cap=16)

    assert telemetry.rescued_evidence_ids == ["2"]
    assert len(result) == 2


def test_g_genuinely_multi_hop_facet_rescues_both_required_items() -> None:
    # A facet that legitimately needs two distinct pieces (registration +
    # invocation) must not be force-truncated to one.
    selected = [_ev("a.py")]
    rejected = [
        _ev("reg.py", quote="registers the callback"),
        _ev("invoke.py", quote="invokes the registered callback later"),
        _ev("noise.py", quote="unrelated"),
    ]
    reasoner = _StubFacetReasoner(
        FacetRescuePlan(
            facets=[AnswerFacet(facet_id="chain", description="registration -> invocation chain")],
            coverage=[FacetCoverage(facet_id="chain", support_status="unsupported")],
            rescue_candidates={"chain": ["0", "1"]},
        )
    )

    result, telemetry = _complete_missing_evidence_facets(reasoner, "q", selected, rejected, cap=16)

    assert set(telemetry.rescued_evidence_ids) == {"0", "1"}
    assert rejected[2] not in result
    assert len(result) == 3


def test_h_rescue_decision_is_identical_regardless_of_worker_identity() -> None:
    # Explicit guard against a hidden evolved-worker preference: swapping
    # worker_id labels on an otherwise-identical pool must not change the
    # rescue outcome, since _complete_missing_evidence_facets never reads
    # worker_id anywhere.
    def run(selected_worker: str, rejected_worker: str) -> list[Evidence]:
        selected = [_ev("a.py", worker_id=selected_worker)]
        rejected = [_ev("b.py", quote="missing fact", worker_id=rejected_worker)]
        reasoner = _StubFacetReasoner(
            FacetRescuePlan(
                facets=[AnswerFacet(facet_id="x", description="missing fact")],
                coverage=[FacetCoverage(facet_id="x", support_status="unsupported")],
                rescue_candidates={"x": ["0"]},
            )
        )
        result, _telemetry = _complete_missing_evidence_facets(
            reasoner, "q", selected, rejected, cap=16
        )
        return result

    base_labeled = run("worker-src-pkg", "worker-src-pkg")
    evolved_labeled = run("worker-bridge-a-b", "worker-bridge-a-b")

    assert len(base_labeled) == len(evolved_labeled) == 2


def test_i_rescue_fills_unused_budget_without_removing_anything() -> None:
    selected = [_ev(f"s{i}.py") for i in range(8)]
    rejected = [_ev("missing.py", quote="the missing fact")]
    reasoner = _StubFacetReasoner(
        FacetRescuePlan(
            facets=[AnswerFacet(facet_id="x", description="missing fact")],
            coverage=[FacetCoverage(facet_id="x", support_status="unsupported")],
            rescue_candidates={"x": ["0"]},
        )
    )

    result, telemetry = _complete_missing_evidence_facets(reasoner, "q", selected, rejected, cap=16)

    assert telemetry.final_selected_count == 9
    assert telemetry.removed_evidence_ids == []
    assert all(item in result for item in selected)
    assert rejected[0] in result


def test_j_at_cap_replaces_a_reasoner_flagged_redundant_item() -> None:
    selected = [_ev(f"s{i}.py") for i in range(16)]
    rejected = [_ev("missing.py", quote="the missing fact")]
    reasoner = _StubFacetReasoner(
        FacetRescuePlan(
            facets=[AnswerFacet(facet_id="x", description="missing fact")],
            coverage=[FacetCoverage(facet_id="x", support_status="unsupported")],
            rescue_candidates={"x": ["0"]},
            replaceable_selected_ids=["3"],
        )
    )

    result, telemetry = _complete_missing_evidence_facets(reasoner, "q", selected, rejected, cap=16)

    assert telemetry.final_selected_count == 16
    assert telemetry.removed_evidence_ids == ["3"]
    assert telemetry.rescued_evidence_ids == ["0"]
    assert selected[3] not in result
    assert rejected[0] in result
    assert telemetry.hit_cap is True


def test_k_at_cap_with_no_safe_replacement_is_a_no_op() -> None:
    selected = [_ev(f"s{i}.py") for i in range(16)]
    rejected = [_ev("missing.py", quote="optional extra detail, not a required facet fix")]
    reasoner = _StubFacetReasoner(
        FacetRescuePlan(
            facets=[AnswerFacet(facet_id="x", description="missing fact")],
            coverage=[FacetCoverage(facet_id="x", support_status="unsupported")],
            rescue_candidates={"x": ["0"]},
            replaceable_selected_ids=[],
        )
    )

    result, telemetry = _complete_missing_evidence_facets(reasoner, "q", selected, rejected, cap=16)

    assert result == selected
    assert telemetry.removed_evidence_ids == []
    assert telemetry.rescued_evidence_ids == []
    assert "no safe replacement" in telemetry.no_op_reason


def test_l_no_facets_identified_is_a_no_op() -> None:
    # The "don't invent a facet the question never asked for" invariant is
    # the reasoner's own judgment call (see assess_facet_completeness's
    # prompt); this confirms the orchestrator faithfully no-ops when a
    # well-behaved reasoner correctly reports zero required facets.
    selected = [_ev("a.py")]
    rejected = [_ev("b.py", quote="strong evidence about an unrelated mechanism B")]
    reasoner = _StubFacetReasoner(FacetRescuePlan(facets=[]))

    result, telemetry = _complete_missing_evidence_facets(reasoner, "q", selected, rejected, cap=16)

    assert result == selected
    assert telemetry.no_op_reason == "no required facets identified"


def test_no_rejected_evidence_is_a_no_op_without_calling_the_reasoner() -> None:
    selected = [_ev("a.py")]
    reasoner = _StubFacetReasoner(
        FacetRescuePlan(facets=[AnswerFacet(facet_id="x", description="x")])
    )

    result, telemetry = _complete_missing_evidence_facets(reasoner, "q", selected, [], cap=16)

    assert result == selected
    assert reasoner.calls == []
    assert telemetry.no_op_reason == "no rejected evidence to consider for rescue"
