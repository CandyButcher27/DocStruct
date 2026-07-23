import pytest

from docstruct.eval.stats import align_per_question, bootstrap_ci, paired_bootstrap

_SCORES = [0.0, 1.0, 0.5, 0.25, 1.0, 0.0, 0.3333, 1.0, 0.2, 0.5] * 10


def test_ci_of_a_constant_collapses_to_the_constant():
    assert bootstrap_ci([0.5] * 50) == (0.5, 0.5)


def test_ci_brackets_the_mean_and_is_deterministic():
    lo, hi = bootstrap_ci(_SCORES)
    mean = sum(_SCORES) / len(_SCORES)
    assert lo < mean < hi
    assert bootstrap_ci(_SCORES) == (lo, hi)


def test_ci_of_empty_and_single_samples():
    assert bootstrap_ci([]) == (0.0, 0.0)
    assert bootstrap_ci([0.7]) == (0.7, 0.7)


def test_a_wider_sample_gives_a_wider_interval():
    tight = bootstrap_ci([0.5, 0.51, 0.49] * 40)
    wide = bootstrap_ci([0.0, 1.0] * 60)
    assert (wide[1] - wide[0]) > (tight[1] - tight[0])


def test_identical_tools_show_no_effect():
    out = paired_bootstrap(_SCORES, _SCORES)
    assert out["diff"] == 0.0
    assert out["p_value"] > 0.05
    assert out["n"] == len(_SCORES)


def test_a_consistent_per_question_lift_is_significant():
    """The case marginal CIs get wrong: same spread, every question shifted."""
    shifted = [min(1.0, s + 0.2) for s in _SCORES]
    out = paired_bootstrap(shifted, _SCORES)
    assert out["diff"] > 0
    assert out["ci_low"] > 0
    assert out["p_value"] < 0.05


def test_p_value_never_reports_exactly_zero():
    """p = 0 would be an artefact of finite resampling, not a measurement."""
    shifted = [min(1.0, s + 0.5) for s in _SCORES]
    assert paired_bootstrap(shifted, _SCORES, resamples=1000)["p_value"] >= 1 / 1000


def test_misaligned_inputs_raise_rather_than_compare_wrong_pairs():
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1])


def test_alignment_pairs_on_the_question_not_the_position():
    reference = [
        {"doc": "a.pdf", "question": "q1", "hyb_rr": 1.0},
        {"doc": "a.pdf", "question": "q2", "hyb_rr": 0.5},
        {"doc": "b.pdf", "question": "q3", "hyb_rr": 0.0},
    ]
    # Different order, and q2 is missing entirely (the tool errored on it).
    other = [
        {"doc": "b.pdf", "question": "q3", "hyb_rr": 1.0},
        {"doc": "a.pdf", "question": "q1", "hyb_rr": 0.25},
    ]
    left, right = align_per_question(reference, other, "hyb_rr")
    assert left == [1.0, 0.0]
    assert right == [0.25, 1.0]


def test_alignment_skips_records_missing_the_metric():
    reference = [{"doc": "a.pdf", "question": "q1", "hyb_rr": 1.0}]
    other = [{"doc": "a.pdf", "question": "q1"}]
    assert align_per_question(reference, other, "hyb_rr") == ([], [])
