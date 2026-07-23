"""Uncertainty on benchmark metrics: bootstrap CIs and a paired difference test.

A leaderboard of point estimates invites a question it cannot answer: is a gap of
0.05 MRR over 298 questions real, or resampling noise? These are the two things
needed to answer it.

**Bootstrap CI** — resample the per-question scores with replacement and take
percentiles of the resampled means. No normality assumption, which matters because
per-question reciprocal rank is not remotely normal: it is a spike at 0, a spike at
1, and a few discrete values in between.

**Paired bootstrap** — every tool answers the *same* questions, so the comparison
is paired. Resample question *indices* once per replicate and apply the same
indices to both tools, which cancels the (large) between-question variance and
leaves only the difference between tools. This matters: two tools can have heavily
overlapping marginal CIs and still differ significantly on every question, and
"the CIs overlap so it isn't significant" is the standard way to get that wrong.

Everything is seeded. This project's contract is determinism, and a significance
number that changes between runs of the same data is not evidence.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20260723


def bootstrap_ci(
    values: Sequence[float],
    resamples: int = DEFAULT_RESAMPLES,
    ci: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    if arr.size == 1:
        return (round(float(arr[0]), 4), round(float(arr[0]), 4))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(resamples, arr.size))
    means = arr[idx].mean(axis=1)
    tail = (1.0 - ci) / 2.0
    lo, hi = np.percentile(means, [tail * 100.0, (1.0 - tail) * 100.0])
    return (round(float(lo), 4), round(float(hi), 4))


def paired_bootstrap(
    treatment: Sequence[float],
    control: Sequence[float],
    resamples: int = DEFAULT_RESAMPLES,
    ci: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> Dict[str, float]:
    """Paired bootstrap of ``mean(treatment) - mean(control)``.

    ``treatment[i]`` and ``control[i]`` must be the two tools' scores on the *same*
    question. Returns the observed difference, its CI, and a two-sided p-value for
    H0: no difference, computed by re-centring the resampled differences on zero
    and asking how often that null distribution reaches the observed effect.
    """
    a = np.asarray(treatment, dtype=float)
    b = np.asarray(control, dtype=float)
    if a.size != b.size:
        raise ValueError(f"paired inputs must align: {a.size} vs {b.size}")
    if a.size == 0:
        return {"diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "n": 0}

    diff = a - b
    observed = float(diff.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(resamples, diff.size))
    resampled = diff[idx].mean(axis=1)

    tail = (1.0 - ci) / 2.0
    lo, hi = np.percentile(resampled, [tail * 100.0, (1.0 - tail) * 100.0])

    # Null distribution: the same resampling, shifted so its mean is zero.
    null = resampled - observed
    p = float((np.abs(null) >= abs(observed)).mean())
    # A p of exactly 0 is an artefact of finite resampling, not a measurement.
    p = max(p, 1.0 / resamples)

    return {
        "diff": round(observed, 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "p_value": round(p, 5),
        "n": int(diff.size),
    }


def align_per_question(
    reference: List[dict], other: List[dict], metric: str
) -> Tuple[List[float], List[float]]:
    """Line two tools' per-question records up on the questions they share.

    A tool that errored on a document has no records for its questions, so the
    lists are not positionally comparable and pairing on position would silently
    compare different questions. Keyed on ``(doc, question)``; the reference tool's
    order is preserved so results are reproducible.
    """
    other_by_key = {(r["doc"], r["question"]): r for r in other}
    ref_scores: List[float] = []
    other_scores: List[float] = []
    for record in reference:
        match = other_by_key.get((record["doc"], record["question"]))
        if match is None or metric not in record or metric not in match:
            continue
        ref_scores.append(float(record[metric]))
        other_scores.append(float(match[metric]))
    return ref_scores, other_scores


def demo() -> None:
    """Self-check: the tests that would catch this module being wrong."""
    # A constant vector has zero spread, so its CI collapses onto the value.
    lo, hi = bootstrap_ci([0.5] * 50)
    assert lo == hi == 0.5, (lo, hi)

    # The CI of a mean must contain the mean.
    values = [0.0, 1.0, 0.5, 0.25, 1.0, 0.0, 0.33, 1.0, 0.2, 0.5] * 10
    lo, hi = bootstrap_ci(values)
    mean = sum(values) / len(values)
    assert lo < mean < hi, (lo, mean, hi)

    # Identical inputs: no effect, and the test must not claim one.
    same = paired_bootstrap(values, values)
    assert same["diff"] == 0.0 and same["p_value"] > 0.05, same

    # A uniform +0.2 shift on every question is as paired as an effect gets: tiny
    # CI, significant, even though the marginal distributions overlap heavily.
    shifted = [min(1.0, v + 0.2) for v in values]
    lifted = paired_bootstrap(shifted, values)
    assert lifted["diff"] > 0.0 and lifted["p_value"] < 0.05, lifted
    assert lifted["ci_low"] > 0.0, lifted

    # Alignment must key on the question, not on position.
    ref = [{"doc": "a.pdf", "question": "q1", "hyb_rr": 1.0},
           {"doc": "a.pdf", "question": "q2", "hyb_rr": 0.5},
           {"doc": "b.pdf", "question": "q3", "hyb_rr": 0.0}]
    other = [{"doc": "b.pdf", "question": "q3", "hyb_rr": 1.0},
             {"doc": "a.pdf", "question": "q1", "hyb_rr": 0.25}]
    left, right = align_per_question(ref, other, "hyb_rr")
    assert left == [1.0, 0.0] and right == [0.25, 1.0], (left, right)

    # Determinism: same seed, same numbers, or none of this is evidence.
    assert bootstrap_ci(values) == bootstrap_ci(values)
    print("stats self-check passed")


if __name__ == "__main__":
    demo()
