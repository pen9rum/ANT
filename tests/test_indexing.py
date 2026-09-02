from pathlib import Path

from ant.environment import RepoEnvironment
from ant.indexing import build_worker_cards, discover_territories
from ant.indexing.cards import _owned_symbols, template_routing_summary


def test_discovers_territories_and_worker_cards(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text("class AuthService:\n    pass\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Demo\nAuthentication notes\n", encoding="utf-8")

    repo = RepoEnvironment(tmp_path)
    territories = discover_territories(repo)
    workers = build_worker_cards(repo.root, territories)

    assert {territory.root for territory in territories} == {"", "src"}
    assert len(workers) == len(territories)
    assert any("authservice" in worker.searchable_terms for worker in workers)


def test_build_worker_cards_fills_a_nonempty_routing_summary_with_no_llm(tmp_path: Path) -> None:
    # build_worker_cards is the no-LLM path (used when `ant index` runs
    # without --llm-cards) -- every card it produces must still carry a
    # routing_summary, since the Orchestrator planning call reads only
    # that field, not the full card, for every worker every round.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text("class AuthService:\n    pass\n", encoding="utf-8")

    repo = RepoEnvironment(tmp_path)
    territories = discover_territories(repo)
    workers = build_worker_cards(repo.root, territories)

    for worker in workers:
        assert worker.routing_summary
        assert worker.routing_summary == template_routing_summary(
            worker.model_copy(update={"routing_summary": ""})
        )


def test_flat_container_splits_by_immediate_subdirectory(tmp_path: Path) -> None:
    for demo in ("reuploading_classifier", "3_tangle"):
        path = tmp_path / "examples" / demo
        path.mkdir(parents=True)
        (path / "main.py").write_text("def run():\n    pass\n", encoding="utf-8")
        (path / "helper.py").write_text("def helper():\n    pass\n", encoding="utf-8")
        (path / "README.md").write_text(f"# {demo}\n", encoding="utf-8")

    territories = discover_territories(RepoEnvironment(tmp_path))

    assert {item.root for item in territories} == {
        "examples/reuploading_classifier",
        "examples/3_tangle",
    }


def test_flat_container_file_directly_under_root_stays_grouped(tmp_path: Path) -> None:
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "shared_utils.py").write_text(
        "def shared():\n    pass\n", encoding="utf-8"
    )

    territories = discover_territories(RepoEnvironment(tmp_path))

    assert {item.root for item in territories} == {"examples"}


def test_descends_through_generic_source_container_to_subsystems(tmp_path: Path) -> None:
    for subsystem in ("backends", "models"):
        path = tmp_path / "src" / "package" / subsystem
        path.mkdir(parents=True)
        (path / "core.py").write_text("def core_symbol():\n    pass\n", encoding="utf-8")
        (path / "helpers.py").write_text("def helper_symbol():\n    pass\n", encoding="utf-8")

    territories = discover_territories(RepoEnvironment(tmp_path))

    assert {item.root for item in territories} == {
        "src/package/backends",
        "src/package/models",
    }


def test_worker_card_terms_prioritize_real_definition_symbols(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "algorithms.py").write_text(
        "class QAOA:\n"
        "    pass\n\n"
        "def build_fused_gate():\n"
        "    return QAOA()\n\n"
        "# qaoa qaoa qaoa return import from lexical filler\n",
        encoding="utf-8",
    )
    territories = discover_territories(RepoEnvironment(tmp_path))
    workers = build_worker_cards(tmp_path, territories)

    terms = workers[0].searchable_terms

    assert terms[:2] == ["QAOA", "build_fused_gate"]


def test_worker_card_symbols_cover_multiple_files_before_limit(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    many_definitions = "\n\n".join(
        f"def circuit_helper_{index}():\n    pass" for index in range(80)
    )
    (tmp_path / "src" / "circuit.py").write_text(many_definitions, encoding="utf-8")
    (tmp_path / "src" / "variational.py").write_text(
        "class QAOA:\n    pass\n\nclass FALQON(QAOA):\n    pass\n",
        encoding="utf-8",
    )
    territories = discover_territories(RepoEnvironment(tmp_path))
    workers = build_worker_cards(tmp_path, territories)

    terms = workers[0].searchable_terms

    assert "QAOA" in terms
    assert "FALQON" in terms


def test_worker_card_keeps_class_names_before_function_names(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "circuit.py").write_text(
        "\n\n".join(f"def helper_{index}():\n    pass" for index in range(80)),
        encoding="utf-8",
    )
    (tmp_path / "src" / "variational.py").write_text(
        "class StateEvolution:\n    pass\n\n"
        "class AdiabaticEvolution:\n    pass\n\n"
        "class QAOA:\n    pass\n\n"
        "class FALQON(QAOA):\n    pass\n",
        encoding="utf-8",
    )
    territories = discover_territories(RepoEnvironment(tmp_path))
    workers = build_worker_cards(tmp_path, territories)

    terms = workers[0].searchable_terms

    assert terms.index("QAOA") < terms.index("helper_10")
    assert terms.index("FALQON") < terms.index("helper_10")


def test_worker_card_includes_typed_owned_symbols(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "variational.py").write_text(
        "class QAOA:\n"
        "    pass\n\n"
        "class FALQON(QAOA):\n"
        "    pass\n\n"
        "def optimize():\n"
        "    return FALQON()\n",
        encoding="utf-8",
    )

    territories = discover_territories(RepoEnvironment(tmp_path))
    workers = build_worker_cards(tmp_path, territories)

    symbols = {symbol.name: symbol for symbol in workers[0].symbols}
    assert symbols["QAOA"].kind == "class"
    assert symbols["FALQON"].bases == ["QAOA"]
    assert symbols["optimize"].kind == "function"


def test_owned_symbols_represents_every_file_even_past_the_old_fixed_cap(tmp_path: Path) -> None:
    # Regression test for a real yt-dlp trace: a single 1010-file territory
    # (yt_dlp/extractor/, one flat directory, no subfolders) used to lose
    # every file past whichever one a plain sort-then-slice[:160] landed
    # on -- confirmed live that only the ~23 alphabetically-earliest files
    # ended up represented in WorkerCard.symbols at all, and Teachable/
    # InfoExtractor/Youtube (all sorting well past 'a') were invisible to
    # every symbol-based routing signal as a result. Breadth (every file
    # gets >=1 definition) must never be sacrificed to a fixed count, no
    # matter how many files the territory has.
    (tmp_path / "src").mkdir()
    file_count = 200  # comfortably past the old limit=160 cap
    for index in range(file_count):
        (tmp_path / "src" / f"site_{index:03d}.py").write_text(
            f"class Site{index:03d}IE:\n    pass\n", encoding="utf-8"
        )
    territories = discover_territories(RepoEnvironment(tmp_path))
    workers = build_worker_cards(tmp_path, territories)

    represented_files = {symbol.path for symbol in workers[0].symbols}

    assert len(represented_files) == file_count


def test_owned_symbols_prefers_a_files_top_level_definition_over_an_earlier_nested_one(
    tmp_path: Path,
) -> None:
    # Regression test for yt-dlp's youtube.py: its first-by-line definition
    # was BadgeType, an unrelated top-level helper enum, with YoutubeIE
    # itself defined 1000+ lines later -- and for common.py: its
    # first-by-line definition sharing the filename's own term was
    # extract_common, a function nested 3 levels inside another method,
    # while InfoExtractor (the file's real top-level class) sorts after it
    # by line number alone. A single definition per file (the guaranteed
    # round-0 slot _owned_symbols' docstring describes) must be the file's
    # actual top-level, filename-matching symbol, not just whichever line
    # comes first.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "youtube.py").write_text(
        "class BadgeType:\n"
        "    pass\n\n"
        "class YoutubeIE:\n"
        "    def helper(self):\n"
        "        def nested_common():\n"
        "            pass\n",
        encoding="utf-8",
    )

    symbols = _owned_symbols(tmp_path, ["src/youtube.py"], limit=1)

    assert [symbol.name for symbol in symbols] == ["YoutubeIE"]


def test_worker_card_searchable_terms_span_the_full_file_set_not_just_the_prefix(
    tmp_path: Path,
) -> None:
    # Regression test for yt-dlp's extractor/ territory (1010 files, no
    # subdirectories): searchable_terms is built from _owned_symbols'
    # result, which is itself now breadth-guaranteed -- but confirmed
    # live that a naive concat-then-slice[:48] over hundreds of
    # per-file, alphabetically-ordered entries just kept whichever files
    # sorted first all over again, at this second layer. A file sorting
    # near the end of a large territory must still have a real chance of
    # reaching the final term budget, not just files sorting near the
    # start.
    (tmp_path / "src").mkdir()
    file_count = 200  # comfortably past _top_terms' limit=48
    for index in range(file_count):
        (tmp_path / "src" / f"site_{index:03d}.py").write_text(
            f"class Site{index:03d}IE:\n    pass\n", encoding="utf-8"
        )
    territories = discover_territories(RepoEnvironment(tmp_path))
    workers = build_worker_cards(tmp_path, territories)

    terms = workers[0].searchable_terms
    site_indices = [int(term[4:7]) for term in terms if term.startswith("Site")]

    assert len(terms) <= 48
    # Under the old bug this maxed out around 47 (only the alphabetically-
    # first ~48 files ever got a term at all) -- an even spread across 200
    # files should reach well past the halfway point.
    assert max(site_indices) > 150


def test_worker_card_readme_terms_survive_beyond_the_lexical_file_cap(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    # Uppercase-prefixed filenames sort before "README.md" (ASCII 'A' < 'R'),
    # so with more than 80 of them the README would fall outside the
    # lexical scan's files[:80] cap and its vocabulary would previously be
    # silently invisible to routing -- exactly the qibo "examples" failure.
    for index in range(85):
        (tmp_path / "src" / f"AAA_helper_{index:03d}.py").write_text(
            f"def helper_{index}():\n    return {index}\n", encoding="utf-8"
        )
    (tmp_path / "src" / "README.md").write_text(
        "# Kaleidoscope renderer\nDraws a kaleidoscope pattern for visualization.\n",
        encoding="utf-8",
    )

    territories = discover_territories(RepoEnvironment(tmp_path))
    workers = build_worker_cards(tmp_path, territories)

    worker = next(worker for worker in workers if worker.root == "src")
    assert len(worker.files) > 80
    assert "kaleidoscope" in worker.searchable_terms
    assert "renderer" in worker.searchable_terms


def test_readme_terms_downweight_words_common_across_every_territory(tmp_path: Path) -> None:
    # "quantum" appears in every territory's README (high document frequency)
    # and repeats more often than "bloch" (which appears only once, only
    # here) -- plain term frequency would rank "quantum" first. Document
    # frequency weighting must let the distinctive, rare word win instead.
    for demo, extra in [
        ("alpha", "quantum quantum quantum"),
        ("beta", "quantum quantum quantum"),
        ("gamma", "quantum quantum quantum"),
    ]:
        path = tmp_path / "examples" / demo
        path.mkdir(parents=True)
        (path / "main.py").write_text("def run():\n    pass\n", encoding="utf-8")
        (path / "README.md").write_text(f"# {demo}\n{extra}\n", encoding="utf-8")
    target = tmp_path / "examples" / "delta"
    target.mkdir(parents=True)
    (target / "main.py").write_text("def run():\n    pass\n", encoding="utf-8")
    (target / "README.md").write_text(
        "# delta\nquantum quantum bloch\n", encoding="utf-8"
    )

    territories = discover_territories(RepoEnvironment(tmp_path))
    workers = build_worker_cards(tmp_path, territories)

    worker = next(worker for worker in workers if worker.root == "examples/delta")
    assert worker.searchable_terms.index("bloch") < worker.searchable_terms.index("quantum")


def test_worker_card_readme_summary_becomes_a_responsibility(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "plot.py").write_text("def draw():\n    pass\n", encoding="utf-8")
    (tmp_path / "src" / "README.md").write_text(
        "# Bloch sphere visualizer\n\n"
        "This module renders qubit states on the Bloch sphere using matplotlib.\n",
        encoding="utf-8",
    )

    territories = discover_territories(RepoEnvironment(tmp_path))
    workers = build_worker_cards(tmp_path, territories)

    worker = next(worker for worker in workers if worker.root == "src")
    assert any("Bloch sphere visualizer" in item for item in worker.responsibilities)
    assert any("matplotlib" in item for item in worker.responsibilities)
