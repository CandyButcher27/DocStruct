import io
import urllib.error

from docstruct.llm.client import _PROVIDERS, _adapt_payload


def _http_error(code, body):
    return urllib.error.HTTPError(
        "https://api.openai.com/v1/chat/completions", code, "Bad Request", {},
        io.BytesIO(body.encode("utf-8")),
    )


def test_adapt_renames_max_tokens_for_reasoning_models():
    payload = {"model": "gpt-5", "max_tokens": 2000, "temperature": 0.0}
    err = _http_error(400, '{"error": {"param": "max_tokens", "message": '
                           '"Unsupported parameter: use \'max_completion_tokens\' instead."}}')
    assert _adapt_payload(payload, err)
    assert payload["max_completion_tokens"] == 2000
    assert "max_tokens" not in payload


def test_adapt_drops_unsupported_temperature():
    payload = {"model": "gpt-5", "temperature": 0.0}
    err = _http_error(400, '{"error": {"param": "temperature", "message": '
                           '"Unsupported value: only the default (1) is supported."}}')
    assert _adapt_payload(payload, err)
    assert "temperature" not in payload


def test_adapt_ignores_non_parameter_failures():
    # A rate limit must fall through to the backoff path, not be silently rewritten.
    payload = {"model": "gpt-4.1", "max_tokens": 100}
    assert not _adapt_payload(payload, _http_error(429, '{"error": {"message": "rate limit"}}'))
    assert not _adapt_payload(payload, _http_error(400, "not json at all"))
    assert payload == {"model": "gpt-4.1", "max_tokens": 100}


def test_openai_provider_defaults_to_a_temperature_respecting_model():
    # gpt-5 rejects temperature=0, and gold generation is run at 0 so the same
    # corpus yields the same questions twice. The default must not be that family.
    assert _PROVIDERS["openai"]["model"] == "gpt-4.1"
    assert _PROVIDERS["openai"]["key_env"] == "OPENAI_API_KEY"
