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
