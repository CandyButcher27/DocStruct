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

_PROVIDERS = {
    "ollama": {"base_url": "https://ollama.com", "key_env": "OLLAMA_API_KEY", "base_env": "OLLAMA_BASE_URL"},
    "groq": {"base_url": "https://api.groq.com/openai", "key_env": "GROQ_API_KEY", "base_env": "GROQ_BASE_URL"},
}


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
        self.model = model or os.environ.get("DOCSTRUCT_LLM_MODEL") or config.LLM_MODEL
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
        }

        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, TimeoutError) as err:
                last_err = err
                time.sleep(1.5 * (attempt + 1))
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
