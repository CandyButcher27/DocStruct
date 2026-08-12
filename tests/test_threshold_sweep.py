"""The offline threshold sweep must agree with the benchmark's own scoring.

The whole point of dumping scores is to avoid re-running retrieval, so if the
re-thresholded arithmetic drifts from `_score`, the sweep silently answers a
different question than the run it claims to re-analyse.
"""

from docstruct import config
from docstruct.eval.benchmark import _score, metrics_from_flags
from docstruct.eval.relevance import get_relevance, get_score

CHUNKS = [
    "the operating margin for fiscal 2022 was 19.5 percent across all segments",
    "unrelated boilerplate about forward looking statements and risk factors",
    "operating margin fiscal 2022 19.5 percent",
]
REGION = "the operating margin for fiscal 2022 was 19.5 percent"


def test_dumped_scores_rethreshold_to_the_same_metrics():
    for mode in ("span", "region"):
        relevant = get_relevance(mode)
        scorer = get_score(mode)
        thr = (config.RELEVANCE_MIN_OVERLAP if mode == "span"
               else config.RELEVANCE_REGION_MIN_OVERLAP)

        online = _score(CHUNKS, REGION, len(CHUNKS), relevant)
        offline = metrics_from_flags([scorer(c, REGION) >= thr for c in CHUNKS])
        assert online == offline, mode


def test_page_mode_has_no_continuous_score():
    # page is boolean by nature; offering a threshold to sweep would be a lie.
    assert get_score("page") is None
