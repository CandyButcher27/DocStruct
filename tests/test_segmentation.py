"""Pk / WindowDiff and the boundary aligner they run on.

The metrics are only meaningful if both segmentations are located on the same
spine, so the aligner is tested as carefully as the arithmetic.
"""

from docstruct.eval.segmentation import (
    _PROBE,
    boundary_mask,
    locate_boundaries,
    pk_windowdiff,
    spine_of,
    straddle_rate,
)

# three 60-character sections, distinct enough that a probe cannot match the wrong one
SECTIONS = [
    "Alpha bravo charlie delta echo foxtrot golf hotel india ju.",
    "Kilo lima mike november oscar papa quebec romeo sierra tan.",
    "Uniform victor whiskey xray yankee zulu nadir apex crux qu.",
]
DOC = spine_of(" ".join(SECTIONS))


def test_identical_segmentations_score_zero():
    mask = boundary_mask(len(DOC), [50, 100])
    out = pk_windowdiff(mask, mask)
    assert out["pk"] == 0.0
    assert out["windowdiff"] == 0.0


def test_disagreement_is_penalised():
    ref = boundary_mask(len(DOC), [50, 100])
    hyp = boundary_mask(len(DOC), [25])
    out = pk_windowdiff(ref, hyp)
    assert out["pk"] > 0.0
    assert out["windowdiff"] > 0.0


def test_windowdiff_punishes_over_segmentation_that_pk_forgives():
    """The reason WindowDiff exists, and the reason this project reports both.

    A tool that puts many boundaries where the document has one still gets a
    window's ends into different segments, so Pk is content. WindowDiff compares
    boundary *counts* and is not.
    """
    ref = boundary_mask(len(DOC), [80])
    spam = boundary_mask(len(DOC), list(range(70, 95, 3)))
    out = pk_windowdiff(ref, spam, k=10)
    assert out["windowdiff"] > out["pk"]


def test_locate_boundaries_finds_sections_in_order():
    offsets = locate_boundaries(DOC, SECTIONS)
    assert offsets[0] == 0
    assert None not in offsets
    assert offsets == sorted(offsets)


def test_alignment_survives_lost_word_spacing():
    """The defect that made the first version of this module report a false ceiling.

    pdfplumber welds two-column lines and drops inter-word spaces on some
    publishers' typesetting, so a word-token spine matched nothing. The despaced
    spine is immune to exactly that damage.
    """
    mangled = spine_of("Alphabravocharliedeltaechofoxtrotgolfhotelindiaju. "
                       + " ".join(SECTIONS[1:]))
    assert locate_boundaries(mangled, SECTIONS)[0] == 0


def test_locate_boundaries_is_forward_only():
    """A repeated probe must resolve to the occurrence after the previous section.

    Running heads and boilerplate repeat verbatim; an unconstrained search matches
    the first occurrence and silently collapses two boundaries into one.
    """
    repeated = "Not applicable to this study in any way whatsoever at all."
    spine = spine_of(" ".join([repeated, SECTIONS[0], repeated]))
    first, _, third = locate_boundaries(spine, [repeated, SECTIONS[0], repeated])
    assert first == 0
    assert third is not None and third > first


def test_section_shorter_than_the_probe_is_unlocatable():
    # Short boilerplate is reported as unlocatable rather than matched somewhere
    # arbitrary, which is why the reachability report counts it apart.
    assert len(spine_of("Not applicable")) < _PROBE
    assert locate_boundaries(DOC, ["Not applicable"]) == [None]


def test_straddle_rate_counts_chunks_crossing_a_gold_boundary():
    # gold splits at 10 and 20. Chunks [0,10) and [10,30): the first sits inside one
    # gold section, the second swallows the boundary at 20.
    assert straddle_rate([0, 10, 20], [0, 10], n=30) == 0.5
    assert straddle_rate([0, 10, 20], [0, 10, 20], n=30) == 0.0
    # a chunk spanning every section crosses both boundaries -- still one straddling
    # chunk, since the rate counts chunks and not crossings
    assert straddle_rate([0, 10, 20], [0], n=30) == 1.0


def test_chunk_boundaries_are_not_forced_into_order():
    """A tool's emission order is not evidence about where its boundaries are.

    Gold sections are ordered and the aligner uses that. Chunks are not: forcing
    them monotone against pdfplumber's reading order threw away 31 of one tool's 33
    located boundaries on a real paper, then scored it as having almost none.
    """
    from docstruct.eval.segmentation import locate_chunk_boundaries

    # emitted back-to-front; every chunk is present in the document
    out = locate_chunk_boundaries(DOC, list(reversed(SECTIONS)))
    assert len(out) == 3
    assert out == sorted(out)
