"""LumberChunker baseline (Duarte et al., 2024, arXiv:2406.17526).

An LLM reads a window of consecutive paragraphs, at least ``THETA_TOKENS`` long,
and names the first paragraph where the content shifts; the chunk ends just before
it and the next window starts there. Paragraphs are PyMuPDF text blocks. The prompt
follows the paper's. Every answer is cached by prompt hash, one file per PDF, so the three
relevance runs and any rerun reuse the same chunks and cost no further calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import List

from docstruct.eval.adapters.base import ChunkAdapter, EvalChunk

THETA_TOKENS = 550
PROMPT = (
    "You will receive as input an english document with paragraphs identified by "
    "'ID XXXX: <text>'.\n\n"
    "Task: Find the first paragraph (not the first one) where the content clearly "
    "changes compared to the previous paragraphs.\n\n"
    "Output: Return the ID of the paragraph with the content shift as in the "
    "exemplified format: 'Answer: ID XXXX'.\n\n"
    "Additional Considerations: Avoid very long groups of paragraphs. Aim for a good "
    "balance between identifying content shifts and keeping groups manageable.\n\n"
    "Document:\n"
)
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


class LumberChunkerAdapter(ChunkAdapter):
    name = "lumberchunker"

    def __init__(self) -> None:
        self.model = os.environ.get("LUMBER_MODEL", "gemini-3.5-flash-lite")
        self.cache_dir = os.environ.get("LUMBER_CACHE_DIR", ".cache/lumberchunker")
        self.cache_path = ""
        self.calls = 0

    def available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY"))

    def _ask(self, prompt: str) -> str:
        key = hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()
        cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, encoding="utf-8") as f:
                cache = json.load(f)
        if key in cache:
            return cache[key]
        gen = {"temperature": 0}
        if os.environ.get("LUMBER_THINKING_BUDGET"):
            gen["thinkingConfig"] = {"thinkingBudget": int(os.environ["LUMBER_THINKING_BUDGET"])}
        body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                           "generationConfig": gen}).encode()
        req = urllib.request.Request(
            API.format(model=self.model), data=body,
            headers={"Content-Type": "application/json",
                     "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 503) or attempt == 7:
                    raise RuntimeError(f"Gemini HTTP {e.code}: {e.read()[:300]!r}") from e
                time.sleep(min(60, 5 * 2 ** attempt))
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == 7:
                    raise RuntimeError(f"Gemini unreachable: {e}") from e
                time.sleep(min(60, 5 * 2 ** attempt))
        parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        answer = "".join(p.get("text", "") for p in parts)
        self.calls += 1
        cache[key] = answer
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        return answer

    def chunk(self, pdf_path: str) -> List[EvalChunk]:
        import pymupdf

        self.cache_path = os.path.join(self.cache_dir, os.path.basename(pdf_path) + ".json")
        paras = []
        with pymupdf.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                for block in page.get_text("blocks"):
                    text = " ".join(block[4].split())
                    if block[6] == 0 and text:
                        paras.append((text, page_num))

        out: List[EvalChunk] = []
        start = 0
        while start < len(paras):
            end, size = start, 0
            while end < len(paras) and size < THETA_TOKENS:
                size += _tokens(paras[end][0])
                end += 1
            if end - start < 2:
                split = end
            else:
                window = "\n".join(f"ID {i:04d}: {paras[i][0]}" for i in range(start, end))
                ids = re.findall(r"Answer:\s*ID\s*(\d+)", self._ask(PROMPT + window))
                split = int(ids[-1]) if ids else end
                if not start < split <= end:
                    split = end
            group = paras[start:split]
            out.append(EvalChunk(id=f"lc_{len(out)}", text="\n".join(t for t, _ in group),
                                 metadata={"pages": sorted({p for _, p in group})}))
            start = split
        return out
