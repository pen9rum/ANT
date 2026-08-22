from pathlib import Path

from ant.environment import RepoEnvironment
from ant.indexing import build_worker_cards, discover_territories


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
