"""Tiny .env loader (no python-dotenv dependency).

Reads KEY=VALUE lines from a .env file and sets them in ``os.environ`` without
overwriting variables that are already set. Used so eval tooling can reach the
Ollama-cloud / GROQ credentials without a hard dependency.
"""

from __future__ import annotations

import os
from typing import Optional


def load_dotenv(path: Optional[str] = None) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (no overwrite)."""
    candidates = [path] if path else [".env", os.path.join(os.getcwd(), ".env")]
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        with open(candidate, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        return
