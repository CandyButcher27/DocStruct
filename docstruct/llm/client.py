"""Minimal OpenAI-compatible chat client (stdlib only).

Talks to an OpenAI-style ``/v1/chat/completions`` endpoint — Ollama cloud
(``https://ollama.com``) by default, GROQ as an alternative. Used ONLY by eval
tooling (Q&A generation, optional LLM-judge); the core pipeline never calls an
LLM. No third-party HTTP dependency.

Credentials come from the environment (loaded from .env): ``OLLAMA_API_KEY`` +
``OLLAMA_BASE_URL`` for the default provider, or ``GROQ_API_KEY`` for provider
``"groq"``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import List, Optional

from docstruct import config
from docstruct.utils.env import load_dotenv

_USER_AGENT = "docstruct-eval/0.4 (+https://github.com/CandyButcher27/DocStruct)"

_PROVIDERS = {
    "ollama": {"base_url": "https://ollama.com", "key_env": "OLLAMA_API_KEY", "base_env": "OLLAMA_BASE_URL"},
    "groq": {"base_url": "https://api.groq.com/openai", "key_env": "GROQ_API_KEY", "base_env": "GROQ_BASE_URL"},
    # Default model is gpt-4.1, not the stronger gpt-5, on purpose: the gpt-5
    # family rejects `temperature` values other than 1, and gold generation is
    # run at temperature 0 so the same corpus yields the same questions twice.
    # Reproducible gold outranks a more capable generator here. Override with
    # --model or DOCSTRUCT_LLM_MODEL if you accept sampled gold.
    "openai": {
        "base_url": "https://api.openai.com",
        "key_env": "OPENAI_API_KEY",
        "base_env": "OPENAI_BASE_URL",
        "model": "gpt-4.1",
    },
}


_MAX_BACKOFF = 300.0


def _backoff_seconds(err: Exception, attempt: int) -> float:
    """How long to wait before retrying.

    Token-per-minute limits are the normal failure mode when generating gold in
    bulk, and providers signal them with a ``Retry-After`` in seconds. GROQ
    reports them as HTTP 413 ("Request too large ... on tokens per minute"), not
    429, so keying off the status code alone is not enough — honour the header
    whenever one is present. Fixed 1.5s steps never clear a 60-second window and
    just burn the retry budget.
    """
    retry_after = getattr(err, "headers", None)
    if retry_after is not None:
        raw = retry_after.get("retry-after")
        if raw:
            try:
                return min(float(raw) + 1.0, _MAX_BACKOFF)
            except ValueError:
                pass
    return 1.5 * (attempt + 1)


def _adapt_payload(payload: dict, err: urllib.error.HTTPError) -> bool:
    """Rewrite a payload an endpoint rejected on parameter grounds. True if changed.

    The OpenAI-compatible surface is not one surface. The gpt-5 / o-series family
    renamed `max_tokens` to `max_completion_tokens` and accepts only the default
    temperature, while gpt-4.1 and every other provider we talk to want the old
    spelling. Keying this off a model-name table means the table is wrong the day
    a new family ships; the error body already says exactly what to change, so
    read it and comply once rather than guessing up front.

    Only ever *removes* or *renames* — it never invents a value, so an adapted
    request asks for the same thing in the dialect the endpoint speaks.
    """
    if err.code != 400:
        return False
    try:
        detail = json.loads(err.read().decode("utf-8")).get("error", {})
    except (ValueError, OSError):
        return False
    param, message = detail.get("param"), detail.get("message", "")

    if param == "max_tokens" and "max_completion_tokens" in message and "max_tokens" in payload:
        payload["max_completion_tokens"] = payload.pop("max_tokens")
        return True
    # Reasoning models reject any explicit temperature. Dropping it is safe for
    # gold generation only because those models are not the default — see the
    # comment on the `openai` provider entry.
    if param == "temperature" and "temperature" in payload:
        payload.pop("temperature")
        return True
    return False


def available(provider: str = "ollama") -> bool:
    load_dotenv()
    spec = _PROVIDERS.get(provider, _PROVIDERS["ollama"])
    return bool(os.environ.get(spec["key_env"]))


class LLMClient:
    """Thin chat-completions client with retry and JSON parsing."""

    def __init__(
        self,
        model: Optional[str] = None,
        provider: str = "ollama",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = config.LLM_TIMEOUT,
    ) -> None:
        load_dotenv()
        spec = _PROVIDERS.get(provider, _PROVIDERS["ollama"])
        self.base_url = (base_url or os.environ.get(spec["base_env"]) or spec["base_url"]).rstrip("/")
        self.api_key = api_key or os.environ.get(spec["key_env"])
        self.model = (
            model or os.environ.get("DOCSTRUCT_LLM_MODEL") or spec.get("model") or config.LLM_MODEL
        )
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(f"No API key for provider '{provider}' ({spec['key_env']} unset)")

    def chat(
        self,
        messages: List[dict],
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        retries: int = 3,
    ) -> str:
        """Return the assistant message content for a chat request."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = max_tokens

        data = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # GROQ sits behind Cloudflare, which rejects urllib's default
            # "Python-urllib/3.x" agent with error 1010 before the request ever
            # reaches the API. Any ordinary agent string gets through.
            "User-Agent": _USER_AGENT,
        }

        last_err: Optional[Exception] = None
        attempt = 0
        # Payload adaptations deliberately do not consume the retry budget: that
        # budget exists for rate limits and network faults, and a gpt-5-family
        # request needs two rewrites before it is even sent in the right dialect,
        # which would leave one attempt for everything that can actually go wrong.
        while attempt < retries:
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as err:
                last_err = err
                if _adapt_payload(payload, err):
                    # A rejected parameter is not a transient failure; retrying the
                    # identical body just spends the budget. Adapt and go again now.
                    data = json.dumps(payload).encode("utf-8")
                    continue
                time.sleep(_backoff_seconds(err, attempt))
                attempt += 1
            except (urllib.error.URLError, KeyError, TimeoutError) as err:
                last_err = err
                time.sleep(_backoff_seconds(err, attempt))
                attempt += 1
        raise RuntimeError(f"LLM request failed after {retries} attempts: {last_err}")

    def chat_json(self, messages: List[dict], **kwargs) -> dict:
        """Chat and parse the response as JSON, tolerating code fences."""
        text = self.chat(messages, json_mode=True, **kwargs).strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        return json.loads(text)
