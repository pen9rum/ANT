import os
from pathlib import Path

from ant.providers import OpenAIProvider
from ant.providers.openai_provider import _extract_output_text, _extract_usage, _loads_json_object


def test_openai_provider_loads_org_and_project_from_dotenv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
    monkeypatch.delenv("OPENAI_PROJECT_ID", raising=False)
    monkeypatch.delenv("ANT_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-test",
                "OPENAI_ORG_ID=org-test",
                "OPENAI_PROJECT_ID=proj_test",
                "ANT_MODEL=gpt-5.4-nano",
            ]
        ),
        encoding="utf-8",
    )

    provider = OpenAIProvider()

    assert provider.is_configured()
    assert provider.settings.organization == "org-test"
    assert provider.settings.project == "proj_test"
    assert provider.model == "gpt-5.4-nano"
    assert os.environ["OPENAI_API_KEY"] == "sk-test"


def test_extract_output_text_from_responses_payload() -> None:
    assert (
        _extract_output_text(
            {
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": "OK"},
                        ]
                    }
                ]
            }
        )
        == "OK"
    )


def test_extract_usage_and_json_object() -> None:
    usage = _extract_usage(
        {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            }
        }
    )

    assert usage.total_tokens == 15
    assert _loads_json_object('```json\n{"ok": true}\n```') == {"ok": True}


def test_responses_kwargs_include_reasoning_effort_only_when_configured(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ANT_MODEL", raising=False)

    provider = OpenAIProvider(model="gpt-5-2025-08-07", reasoning_effort="low")
    default_provider = OpenAIProvider(model="gpt-4.1")

    assert provider._responses_kwargs("judge", 128)["reasoning"] == {"effort": "low"}
    assert "reasoning" not in default_provider._responses_kwargs("synthesize", 128)


def test_responses_text_records_usage_on_sdk_path(monkeypatch) -> None:
    monkeypatch.chdir(Path.cwd() / "tests")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
    monkeypatch.delenv("OPENAI_PROJECT_ID", raising=False)

    class Response:
        output_text = "OK"

        def model_dump(self) -> dict:
            return {"usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}

    class Responses:
        def create(self, **kwargs):
            return Response()

    class Client:
        responses = Responses()

    provider = OpenAIProvider(model="gpt-4.1")
    provider.client = lambda: Client()  # type: ignore[method-assign]

    assert provider.responses_text("hello").text == "OK"
    assert provider.drain_usage().total_tokens == 5


def test_generate_card_includes_typed_symbols(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "models.py").write_text(
        "class QAOA:\n    pass\n\nclass FALQON(QAOA):\n    pass\n",
        encoding="utf-8",
    )

    provider = OpenAIProvider(model="gpt-4.1")
    provider.responses_json = lambda prompt, max_output_tokens=512: type(  # type: ignore[method-assign]
        "Result",
        (),
        {
            "text": (
                '{"name": "models", "responsibilities": ["models"], '
                '"searchable_terms": ["QAOA"]}'
            )
        },
    )()

    card = provider.generate_card(
        repo_root=str(tmp_path),
        territory_root="src",
        files=["src/models.py"],
    )

    symbols = {symbol.name: symbol for symbol in card.symbols}
    assert symbols["FALQON"].bases == ["QAOA"]
