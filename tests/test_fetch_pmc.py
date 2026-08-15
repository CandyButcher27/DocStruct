import importlib.util
import os

SPEC = importlib.util.spec_from_file_location(
    "fetch_pmc",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "fetch_pmc.py"),
)
fetch_pmc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch_pmc)


def _entry(name):
    return {"file": f"{name}.pdf", "xml": f"{name}.xml", "pmcid": name}


def _write(d, name, pdf=True, xml=True):
    if pdf:
        (d / f"{name}.pdf").write_bytes(b"%PDF-1.4")
    if xml:
        (d / f"{name}.xml").write_text("<article/>")


def test_entry_with_both_halves_on_disk_is_kept(tmp_path):
    _write(tmp_path, "bmc__PMC1")
    assert fetch_pmc.prune_to_disk([_entry("bmc__PMC1")], str(tmp_path)) == [
        _entry("bmc__PMC1")
    ]


def test_committed_manifest_in_a_fresh_clone_prunes_to_nothing(tmp_path):
    # The bug: a clone with no data/pmc/ trusted the manifest, so every search hit
    # counted toward the per-journal quota without downloading and the corpus stayed
    # empty. Nothing on disk must mean nothing claimed.
    manifest = [_entry(f"bmc__PMC{i}") for i in range(133)]
    assert fetch_pmc.prune_to_disk(manifest, str(tmp_path)) == []


def test_pdf_without_its_xml_is_dropped(tmp_path):
    # A PDF whose XML never arrived has no gold, and --pdfs-dir globbing would still
    # pick it up, silently growing the corpus with unscoreable documents.
    _write(tmp_path, "plos__PMC2", xml=False)
    assert fetch_pmc.prune_to_disk([_entry("plos__PMC2")], str(tmp_path)) == []


def test_xml_without_its_pdf_is_dropped(tmp_path):
    _write(tmp_path, "plos__PMC3", pdf=False)
    assert fetch_pmc.prune_to_disk([_entry("plos__PMC3")], str(tmp_path)) == []


def test_partial_corpus_keeps_only_what_is_present(tmp_path):
    _write(tmp_path, "bmc__PMC1")
    _write(tmp_path, "elife__PMC9")
    manifest = [_entry("bmc__PMC1"), _entry("plos__PMC5"), _entry("elife__PMC9")]
    kept = [m["file"] for m in fetch_pmc.prune_to_disk(manifest, str(tmp_path))]
    assert kept == ["bmc__PMC1.pdf", "elife__PMC9.pdf"]
