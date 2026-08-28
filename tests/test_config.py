import os
from pathlib import Path

from ant.config import load_dotenv


def test_load_dotenv_does_not_clobber_an_already_set_env_var(tmp_path: Path, monkeypatch) -> None:
    # Regression test: load_dotenv used to default to override=True and was
    # called unconditionally from OpenAIProvider.__init__, so .env's
    # ANT_MODEL silently re-applied itself over any os.environ["ANT_MODEL"]
    # a caller had already set -- confirmed this was why every real run in
    # this project used gpt-4.1 (the .env value) instead of the
    # gpt-5.4-mini several scripts explicitly set beforehand.
    monkeypatch.setenv("ANT_MODEL", "sentinel")
    env_path = tmp_path / ".env"
    env_path.write_text("ANT_MODEL=gpt-4.1\n", encoding="utf-8")

    load_dotenv(path=env_path)

    assert os.environ["ANT_MODEL"] == "sentinel"


def test_load_dotenv_fills_in_a_var_that_is_not_already_set(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SOME_NEW_VAR", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("SOME_NEW_VAR=from-dotenv\n", encoding="utf-8")

    load_dotenv(path=env_path)

    assert os.environ["SOME_NEW_VAR"] == "from-dotenv"


def test_load_dotenv_override_true_still_clobbers_when_explicitly_requested(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ANT_MODEL", "sentinel")
    env_path = tmp_path / ".env"
    env_path.write_text("ANT_MODEL=gpt-4.1\n", encoding="utf-8")

    load_dotenv(path=env_path, override=True)

    assert os.environ["ANT_MODEL"] == "gpt-4.1"


def test_load_dotenv_returns_every_parsed_value_regardless_of_override(
    tmp_path: Path, monkeypatch
) -> None:
    # OpenAIProvider needs the raw parsed values, not just their
    # os.environ side effects -- OPENAI_API_KEY/OPENAI_ORG_ID/
    # OPENAI_PROJECT_ID are a matched set that must be sourced from .env
    # together when .env defines them, which os.environ alone can't
    # express once a caller's ambient environment already has one of the
    # three set to something else.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-already-set")
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=sk-from-dotenv\nOPENAI_ORG_ID=org-from-dotenv\n",
        encoding="utf-8",
    )

    values = load_dotenv(path=env_path)

    assert values == {
        "OPENAI_API_KEY": "sk-from-dotenv",
        "OPENAI_ORG_ID": "org-from-dotenv",
    }
    assert os.environ["OPENAI_API_KEY"] == "sk-already-set"
