"""Section-boundary agreement: does a chunker split where the document splits?

The retrieval leaderboard measures whether the right text comes back. It cannot
say whether a chunker respects document structure, which is the actual claim
behind a structure-aware design. Publishers' JATS gives the document's real
section boundaries (`scripts/build_jats_gold.py`), so the claim becomes testable
against every tool, not just ours -- every chunker has boundaries.

**Pk and WindowDiff, not boundary precision/recall.** Precision and recall over
boundaries are not size-neutral: a tool emitting tiny chunks has boundaries
everywhere and collects recall for free, which is the same trap that makes `span`
relevance reward large chunks and `page` reward small ones. Pk (Beeferman, Berger
& Lafferty 1999) and WindowDiff (Pevzner & Hearst 2002) are the standard
segmentation metrics precisely because they penalise over- and under-segmentation
both. Lower is better for both; 0.0 is perfect agreement.

**The spine is whitespace-free characters, and that is not a detail.** Gold text
comes from XML and chunk text from the PDF, so they are never byte-identical, and
the first version of this module compared *word tokens*. It reported a 68.8%
ceiling that was almost entirely an artefact: pdfplumber's default extraction
welds two-column lines together and loses inter-word spaces on some publishers'
typesetting, so a Nature Communications paper reads
`Plasmidsaremostoftensharedamongbac-` spliced with a right-column line, and no
word n-gram can match. Matching on despaced characters instead moved that same
document from 16.1% to 96.7%. `relevance.py` already reasons this way -- word
spacing in a PDF is inferred, not stored -- and this is the same argument applied
to alignment rather than to containment.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from docstruct.eval.relevance import _despaced

# Characters matched per probe. Long enough that a hit is not a coincidence in a
# 60,000-character document, short enough to sit inside one column's run of text
# before a two-column weld splices in a fragment of the other column.
_PROBE = 40
# How many probes to slide through a section before giving up. Sections open with a
# figure reference or a formula often enough that the first probe alone
# under-reports what is genuinely locatable.
_MAX_PROBES = 8


def spine_of(text: str) -> str:
    """The common sequence both segmentations are located on."""
    return _despaced(text)


def _first_hit(spine: str, text: str, start: int = 0) -> Optional[int]:
    """Where `text` begins in `spine` at or after `start`, by sliding probes."""
    d = _despaced(text)
    for i in range(_MAX_PROBES):
        probe = d[i * _PROBE:(i + 1) * _PROBE]
        if len(probe) < _PROBE:
            break
        hit = spine.find(probe, start)
        if hit != -1:
            # subtract the probe's own offset so the boundary is the section's
            # start, not the start of whichever probe happened to match
            return max(hit - i * _PROBE, start)
    return None


def _longest_increasing(values: List[Optional[int]]) -> List[int]:
    """Indices of a longest strictly-increasing subsequence, ignoring None."""
    tails: List[int] = []          # tails[k] = index into `values` of the smallest
    prev: List[Optional[int]] = [None] * len(values)   # tail of an LIS of length k+1
    for i, v in enumerate(values):
        if v is None:
            continue
        lo, hi = 0, len(tails)
        while lo < hi:                       # first tail whose value is >= v
            mid = (lo + hi) // 2
            if values[tails[mid]] < v:
                lo = mid + 1
            else:
                hi = mid
        prev[i] = tails[lo - 1] if lo else None
        if lo == len(tails):
            tails.append(i)
        else:
            tails[lo] = i
    out: List[int] = []
    cur = tails[-1] if tails else None
    while cur is not None:
        out.append(cur)
        cur = prev[cur]
    return out[::-1]


def locate_boundaries(spine: str, section_texts: Sequence[str]) -> List[Optional[int]]:
    """Character offset in `spine` where each section starts, or None if not found.

    Sections are ordered in the document, so their offsets must increase. The
    obvious way to enforce that -- search forward from the previous match -- is
    wrong, and wrong in a way that looks like a corpus limitation: one section that
    matches spuriously far ahead (a phrase recurring in a figure caption, say)
    locks out *every* section after it. Measured on a Frontiers paper, 20 of 22
    "unlocatable" sections were locatable, just walled off by an earlier bad match.

    So: locate every section independently, then keep a longest strictly-increasing
    subsequence of those offsets and re-search the rest between their surviving
    neighbours. The outlier is discarded instead of the remainder of the document.
    """
    first = [_first_hit(spine, t) for t in section_texts]
    keep = set(_longest_increasing(first))

    out: List[Optional[int]] = list(first)
    for i in range(len(out)):
        if i in keep:
            continue
        # bounded by whichever consistent neighbours survived
        lo = max((out[j] for j in range(i) if j in keep and out[j] is not None), default=-1)
        hi = min((out[j] for j in range(i + 1, len(out)) if j in keep and out[j] is not None),
                 default=len(spine))
        hit = _first_hit(spine, section_texts[i], lo + 1)
        out[i] = hit if hit is not None and hit < hi else None
    return out


def boundary_mask(length: int, offsets: Sequence[Optional[int]]) -> List[int]:
    """1 at every position that opens a segment (position 0 is not a boundary)."""
    mask = [0] * length
    for o in offsets:
        if o is not None and 0 < o < length:
            mask[o] = 1
    return mask


def _window(n: int, n_segments: int) -> int:
    """Half the average true segment length, the convention in both papers."""
    return max(2, int(round(n / max(n_segments, 1) / 2)))


def pk_windowdiff(ref: Sequence[int], hyp: Sequence[int], k: Optional[int] = None) -> Dict[str, float]:
    """Pk and WindowDiff between two boundary masks over the same spine.

    Pk asks, for every window, whether the two segmentations agree that the window's
    ends fall in the same segment. WindowDiff asks whether they agree on the
    *number* of boundaries in the window, which is what makes it sensitive to a
    chunker that puts three boundaries where the document has one.
    """
    n = min(len(ref), len(hyp))
    if n < 3:
        return {"pk": float("nan"), "windowdiff": float("nan"), "k": 0, "n": n}
    if k is None:
        k = _window(n, sum(ref) + 1)
    k = max(2, min(k, n - 1))

    # prefix sums so each window is O(1) rather than O(k)
    cr = [0]
    ch = [0]
    for i in range(n):
        cr.append(cr[-1] + ref[i])
        ch.append(ch[-1] + hyp[i])

    pk_err = wd_err = 0
    windows = n - k
    for i in range(windows):
        r = cr[i + k] - cr[i]
        h = ch[i + k] - ch[i]
        pk_err += int((r > 0) != (h > 0))
        wd_err += int(r != h)
    return {"pk": pk_err / windows, "windowdiff": wd_err / windows, "k": k, "n": n}


def straddle_rate(section_offsets: Sequence[Optional[int]],
                  chunk_offsets: Sequence[int], n: int) -> float:
    """Fraction of chunks that cross at least one gold section boundary.

    Reported beside Pk/WindowDiff because it decides whether a per-chunk section
    *label* is even well defined: a chunk spanning three subsections has no single
    correct path, so any label-accuracy metric has to state how much of the corpus
    it quietly excluded. Measured on this corpus, 57.4% of gold sections are
    shorter than MIN_CHUNK_TOKENS, so straddling is the common case by design, not
    an edge case.
    """
    if not chunk_offsets:
        return float("nan")
    bounds = sorted(o for o in section_offsets if o is not None)
    ends = list(chunk_offsets[1:]) + [n]
    crossed = sum(1 for start, end in zip(chunk_offsets, ends)
                  if any(start < b < end for b in bounds))
    return crossed / len(chunk_offsets)
