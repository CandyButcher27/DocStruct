"""Generate every figure in paper/ from the committed reports. No hand-drawn numbers.

    python scripts/make_paper_figures.py

Writes paper/figures/*.pdf. Each figure names the report it was built from, so a
figure can never quietly drift from the table beside it.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")

OURS = "#1d4ed8"
OURS_GEO = "#60a5fa"
OTHER = "#a1a1aa"
ACCENT = "#be123c"
INK = "#18181b"

PRETTY = {
    "docstruct": "DocStruct",
    "docstruct_geo": "DocStruct-geo",
    "pymupdf4llm": "pymupdf4llm",
    "unstructured": "unstructured",
    "langchain": "langchain",
    "llamaindex": "llamaindex",
    "llamaindex_semantic": "llamaindex-sem",
}
ORDER = list(PRETTY)

# Hand-placed label offsets (dx, dy, ha). Two clusters in these scatters overlap at
# any automatic placement, and a collided label is worse than a tuned one.
SEC_LABEL = {
    "pymupdf4llm": (-4, 6, "right"),
    "llamaindex_semantic": (2, 9, "left"),
    "docstruct_geo": (-2, -14, "right"),
    "docstruct": (4, 7, "left"),
    "llamaindex": (0, 8, "center"),
    "langchain": (0, 8, "center"),
    "unstructured": (0, 8, "center"),
}
COST_LABEL = {
    "docstruct": (6, 2, "left"),
    "docstruct_geo": (0, -13, "center"),
    "pymupdf4llm": (6, -3, "left"),
    "unstructured": (6, 0, "left"),
    "langchain": (0, 8, "center"),
    "llamaindex": (0, 8, "center"),
    "llamaindex_semantic": (-4, 6, "right"),
}


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#71717a",
        "axes.linewidth": 0.6,
        "xtick.color": "#52525b",
        "ytick.color": "#52525b",
        "figure.dpi": 200,
    })


def colour(tool):
    return OURS if tool == "docstruct" else OURS_GEO if tool == "docstruct_geo" else OTHER


def load(name):
    with open(os.path.join(ROOT, "reports", name), encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- figure 2
def fig_mode_inversion():
    """The central claim: identical chunks, three rules, the ranking flips."""
    modes = ["page", "span", "region"]
    ranks, mrr = {}, {}
    for m in modes:
        d = load(f"ohr_results_{m}.json")
        rows = sorted(d["results"], key=lambda t: -t["mrr"])
        for i, t in enumerate(rows, 1):
            ranks.setdefault(t["name"], {})[m] = i
            mrr.setdefault(t["name"], {})[m] = t["mrr"]

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    xs = range(len(modes))
    for tool in ORDER:
        if tool not in ranks:
            continue
        ys = [ranks[tool][m] for m in modes]
        emph = tool.startswith("docstruct") or tool == "unstructured"
        ax.plot(xs, ys, "-o",
                color=ACCENT if tool == "unstructured" else colour(tool),
                lw=2.2 if emph else 0.7, ms=5 if emph else 2.5,
                zorder=3 if emph else 1, alpha=1.0 if emph else 0.45)
        ax.annotate(PRETTY[tool], (2, ranks[tool]["region"]),
                    xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=6.8,
                    color=ACCENT if tool == "unstructured" else
                    (INK if tool.startswith("docstruct") else "#a1a1aa"),
                    fontweight="bold" if emph else "normal")

    # MRR beside each emphasised marker: the rank axis alone hides how close
    # the span and region races are.
    for tool in ("docstruct", "unstructured"):
        for i, m in enumerate(modes):
            ax.annotate(f"{mrr[tool][m]:.2f}", (i, ranks[tool][m]),
                        xytext=(0, -11), textcoords="offset points",
                        ha="center", fontsize=5.8,
                        color=ACCENT if tool == "unstructured" else OURS)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(modes, fontstyle="italic")
    ax.set_yticks(range(1, 8))
    ax.invert_yaxis()
    ax.set_ylabel("rank of 7 (1 = best)")
    ax.set_xlabel("relevance rule")
    ax.set_xlim(-0.25, 2.95)
    ax.set_ylim(7.6, 0.4)
    ax.grid(axis="y", color="#e4e4e7", lw=0.5, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mode_inversion.pdf"), bbox_inches="tight")
    plt.close(fig)
    return {t: ranks[t] for t in ("docstruct", "unstructured")}


# ---------------------------------------------------------------- figure 3
def fig_threshold_sweep():
    sw = load("ohr_region_threshold_sweep.json")
    table = sw["table"]
    ths = sorted(table["docstruct"], key=float)
    x = [float(t) for t in ths]

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for tool in ORDER:
        if tool not in table:
            continue
        ys = [table[tool][t]["mrr"] for t in ths]
        emph = tool.startswith("docstruct")
        ax.plot(x, ys, "-", color=colour(tool), lw=1.9 if emph else 0.9,
                zorder=3 if emph else 1, alpha=1.0 if emph else 0.7,
                label=PRETTY[tool] if emph else None)

    ax.axvline(0.7, color=ACCENT, lw=0.8, ls=":", zorder=2)
    ax.annotate("0.7 = the value\nwe use", (0.7, 0.93), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="top",
                fontsize=6.5, color=ACCENT)
    ax.set_xlabel(r"relevance threshold (region overlap)")
    ax.set_ylabel("MRR")
    ax.grid(color="#e4e4e7", lw=0.5, zorder=0)
    ax.legend(frameon=False, loc="lower left")
    ax.set_ylim(0.15, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "threshold_sweep.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_size_confound():
    """Why Pk and WindowDiff must be read together."""
    # A dumbbell sorted by chunk count. The earlier scatter put seven pairs of
    # labelled points in one quadrant and could not be read at column width.
    sec = load("section_scores.json")["results"]
    tools = sorted(sec, key=lambda t: sec[t]["mean_chunks"])
    fig, ax = plt.subplots(figsize=(3.3, 2.6))

    for i, tool in enumerate(tools):
        v = sec[tool]
        c = colour(tool)
        ax.plot([v["pk"], v["windowdiff"]], [i, i], color=c, lw=1.6,
                alpha=0.9 if tool.startswith("docstruct") else 0.35, zorder=2)
        ax.scatter(v["pk"], i, s=30, color=c, marker="^", zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.scatter(v["windowdiff"], i, s=30, color=c, marker="o", zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.annotate(f"{v['mean_chunks']:.0f}", (1.005, i),
                    xycoords=("axes fraction", "data"), va="center",
                    fontsize=6.2, color="#71717a")

    ax.set_yticks(range(len(tools)))
    ax.set_yticklabels(
        [PRETTY.get(t, t) for t in tools],
        fontweight="normal")
    for lbl, t in zip(ax.get_yticklabels(), tools):
        if t.startswith("docstruct"):
            lbl.set_fontweight("bold")
            lbl.set_color(INK)
        else:
            lbl.set_color("#71717a")

    ax.annotate("chunks/doc\n(gold $\\approx$25)", (1.005, len(tools) - 0.35),
                xycoords=("axes fraction", "data"), va="bottom",
                fontsize=5.8, color="#a1a1aa")
    ax.set_xlabel("segmentation error (lower is better)")
    ax.grid(axis="x", color="#e4e4e7", lw=0.5, zorder=0)
    ax.set_xlim(0.28, 0.95)
    # One empty row at the bottom so the legend never sits on a dumbbell.
    ax.set_ylim(-1.5, len(tools) - 0.3)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], marker="^", ls="", color="#52525b", ms=5, label="Pk"),
               Line2D([], [], marker="o", ls="", color="#52525b", ms=5, label="WindowDiff")]
    ax.legend(handles=handles, frameon=False, loc="lower center", ncol=2,
              handletextpad=0.2, columnspacing=1.4, borderpad=0.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "size_confound.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig_cost():
    """Accuracy against the token bill."""
    d = load("ohr_results_span.json")
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for t in d["results"]:
        c = colour(t["name"])
        ax.scatter(t["context_words"], t["mrr"], s=46, color=c, zorder=3,
                   edgecolor="white", linewidth=0.7)
        off = COST_LABEL.get(t["name"], (0, 7, "center"))
        ax.annotate(PRETTY.get(t["name"], t["name"]),
                    (t["context_words"], t["mrr"]), xytext=off[:2],
                    textcoords="offset points", ha=off[2], fontsize=6.3,
                    color=INK if t["name"].startswith("docstruct") else "#71717a",
                    fontweight="bold" if t["name"].startswith("docstruct") else "normal")
    ax.set_xlabel("context words returned per query (top-5)")
    ax.set_ylabel("MRR  (span relevance)")
    ax.grid(color="#e4e4e7", lw=0.5, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cost.pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    style()
    os.makedirs(OUT, exist_ok=True)
    fig_pipeline()
    inv = fig_mode_inversion()
    fig_threshold_sweep()
    fig_size_confound()
    fig_cost()
    print("wrote:", ", ".join(sorted(os.listdir(OUT))))
    print("rank check (must be 1st/1st/6th vs 1st/4th/5th):")
    for t, r in inv.items():
        print(f"  {t:<14} page={r['page']} span={r['span']} region={r['region']}")
    return 0




# ---------------------------------------------------------------- figure 1
def fig_pipeline(pdf="data/raw-pdfs/doc1.pdf", page=1,
                 weights="weights/yolov8m-doclaynet.pt"):
    """Teaser: one real page at three stages. Shares its renderer with the README
    animation, so the paper and the repo front page cannot disagree."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mrg", os.path.join(ROOT, "scripts", "make_readme_gif.py"))
    mrg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mrg)

    from docstruct.pipeline import run_pipeline
    from docstruct.schema import Source
    from PIL import ImageDraw

    w = weights if os.path.exists(os.path.join(ROOT, weights)) else None
    res = run_pipeline(os.path.join(ROOT, pdf), weights=os.path.join(ROOT, w) if w else None)
    blocks = sorted([b for b in res.blocks if b.page_num == page],
                    key=lambda b: b.reading_order)
    base, zoom = mrg.render_page(os.path.join(ROOT, pdf), page, 620)

    def sc(b):
        return [b.bbox.x0 * zoom, b.bbox.y0 * zoom, b.bbox.x1 * zoom, b.bbox.y1 * zoom]

    panels = [base]

    im = mrg.faded(base, 0.5)
    d = ImageDraw.Draw(im)
    for b in blocks:
        d.rectangle(sc(b), outline=mrg.LABEL_COLOR.get(
            str(getattr(b.label, "value", b.label)), mrg.DEFAULT_COLOR), width=3)
    panels.append(im)

    im = mrg.faded(base, 0.5)
    d = ImageDraw.Draw(im)
    chunk_of = {bid: i for i, ch in enumerate(res.chunks) for bid in ch.source_block_ids}
    pts = []
    for b in blocks:
        ci = chunk_of.get(b.block_id)
        if ci is None:
            continue
        x0, y0, x1, y1 = sc(b)
        d.rectangle([x0, y0, x1, y1],
                    outline=mrg.CHUNK_CYCLE[ci % len(mrg.CHUNK_CYCLE)], width=4)
        pts.append(((x0 + x1) / 2, (y0 + y1) / 2))
    if len(pts) > 1:
        d.line(pts, fill=(37, 99, 235), width=2)
    panels.append(im)

    conf = sum(1 for b in blocks if b.source == Source.CONFIRMED)
    titles = [
        "(a) input page",
        f"(b) fused blocks ({conf} seen by both detectors)",
        f"(c) reading order + {len({chunk_of[b.block_id] for b in blocks if b.block_id in chunk_of})} chunks",
    ]
    # Crop to the top of the page: at column width a whole page is an unreadable
    # grey rectangle, and the point of the figure is that the boxes are legible.
    crop = int(panels[0].height * 0.52)
    panels = [im.crop((0, 0, im.width, crop)) for im in panels]

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.9))
    for ax, img, t in zip(axes, panels, titles):
        ax.imshow(img)
        ax.set_title(t, fontsize=7.5, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#d4d4d8"); sp.set_linewidth(0.6)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(OUT, "pipeline.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  pipeline: {len(blocks)} blocks, {conf} confirmed, weights={'on' if w else 'OFF'}")


if __name__ == "__main__":
    sys.exit(main())
