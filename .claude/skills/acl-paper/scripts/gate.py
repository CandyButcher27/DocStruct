import re, sys, pathlib

CHECKS = [
    ("M1  em-dash",        r"---|\u2014|\u2013"),
    ("M2  antithesis",     r"not (only |just |merely )?\w[\w\s]{0,25}? but\b|,\s*not\s+\w+ed\b|\brather than\b|\bless \w+ than\b"),
    ("M3  editorializing", r"is the point\b|is not real\b|a testament to\b|the kind of \w+ that\b"),
    ("M4  intensifier",    r"\b(in effect|at its core|at its heart|in essence|truly|genuinely|simply|essentially|fundamentally|in fact|indeed|actually|seamless\w*)\b"),
    # "significant"/"significantly" is kept when it is the statistical term, which
    # in this paper always sits beside a p-value, an n, or a named comparison.
    ("M5  banned adj",     r"\b(novel|substantial|impressive|promising|comprehensive|robust|powerful|paradigm|leverages?|leveraging|utiliz\w+)\b|\bsignificant\b(?!ly)(?!.*[pn]\s*=)"),
    ("M6  throat-clear",   r"(?m)^\s*(Moreover|Furthermore|Additionally|Notably|Importantly|Indeed|Ultimately|Crucially|That said|It is worth noting|It should be noted)\b"),
    ("M8  empty setup",    r"What sets \w+ apart|The key insight is|the real \w+ is that"),
    ("M10 hype verb",      r"\b(unlocks?|powers?|promises? to|stands? to|is poised to|opens the door to|has the potential to)\b"),
    ("M12 wordiness",      r"\b(the fact that|in order to|owing to the fact that|in terms of|one of the most)\b"),
    # "rather than" is a construction, caught by M2; only bare "rather" is a qualifier.
    ("M13 weak qualifier", r"\brather\b(?! than)|\b(very|quite|somewhat|fairly|pretty)\b"),
    ("bold list lead-in",  r"\\item\s+\\textbf"),
    ("todo",               r"\\todo\\{"),
    ("unverified",         r"\\unverified\\{"),
]

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "paper/main.tex")
lines = path.read_text(encoding="utf-8").split("\n")
prose = [(i + 1, l) for i, l in enumerate(lines) if not l.lstrip().startswith("%")]

verbose = "-v" in sys.argv
total = 0
for name, pat in CHECKS:
    rx = re.compile(pat, re.I)
    hits = [(n, l) for n, l in prose if rx.search(l)]
    total += len(hits)
    print(f"{name:20s} {len(hits):4d}")
    if verbose:
        for n, l in hits:
            print(f"      {path}:{n}: {l.strip()[:110]}")
print(f"{'TOTAL':20s} {total:4d}")


def _selftest():
    rx = dict(CHECKS)
    assert re.search(rx["M1  em-dash"], "a --- b")
    assert not re.search(rx["M1  em-dash"], "a - b")
    assert re.search(rx["M2  antithesis"], "measured, not asserted")
    assert re.search(rx["M13 weak qualifier"], "a rather large gain")
    assert not re.search(rx["M13 weak qualifier"], "measured rather than asserted")
    assert not re.search(rx["M5  banned adj"], "we report the effect")
    assert not re.search(rx["M5  banned adj"], "not significant at p = 0.12")
    assert re.search(rx["M5  banned adj"], "a novel approach")
    print("selftest ok")


if "--selftest" in sys.argv:
    _selftest()
