import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docstruct.eval.adapters.lumberchunker_adapter import LumberChunkerAdapter  # noqa: E402

DONE_DIR = os.path.join(os.environ.get("LUMBER_CACHE_DIR", ".cache/lumberchunker"), "done")


def run(pdf):
    marker = os.path.join(DONE_DIR, os.path.basename(pdf))
    if os.path.exists(marker):
        return pdf, 0, None
    adapter = LumberChunkerAdapter()
    try:
        adapter.chunk(pdf)
    except Exception as e:  # noqa: BLE001
        return pdf, adapter.calls, f"{type(e).__name__}: {e}"
    open(marker, "w").close()
    return pdf, adapter.calls, None


def main():
    pdfs = sorted(glob.glob(os.path.join(sys.argv[1], "*.pdf")))
    os.makedirs(DONE_DIR, exist_ok=True)
    with ThreadPoolExecutor(int(os.environ.get("LUMBER_THREADS", "8"))) as ex:
        results = list(ex.map(run, pdfs))
    failed = [(p, err) for p, _, err in results if err]
    for p, err in failed:
        print(f"  {os.path.basename(p)}: {err[:200]}")
    print(f"calls this pass: {sum(c for _, c, _ in results)}; "
          f"done {len(pdfs) - len(failed)}/{len(pdfs)}")
    with open(os.environ.get("GITHUB_OUTPUT", os.devnull), "a") as f:
        f.write(f"complete={'true' if not failed else 'false'}\n")
    return 0 if not failed else 3


if __name__ == "__main__":
    sys.exit(main())
