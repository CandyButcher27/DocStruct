"""Render the README animation: one real page walked through the real pipeline.

    python scripts/make_readme_gif.py data/raw-pdfs/doc1.pdf --page 0 --out docs/demo.gif

Every box drawn comes from an actual `run_pipeline()` result -- nothing here is a
mock-up. The frames are: the page as shipped, the fused blocks with their labels and
provenance, the reading order, and the chunks the blocks were assembled into.
"""
from __future__ import annotations

import argparse
import os
import sys

import fitz
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docstruct.pipeline import run_pipeline  # noqa: E402
from docstruct.schema import Source  # noqa: E402

INK = (24, 24, 27)
MUTED = (113, 113, 122)
PAPER = (250, 250, 249)

# The five labels the schema actually defines (docstruct/schema.py: Label).
LABEL_COLOR = {
    "header": (190, 24, 93),
    "text": (37, 99, 235),
    "table": (5, 150, 105),
    "figure": (217, 119, 6),
    "caption": (161, 98, 7),
}
DEFAULT_COLOR = (100, 116, 139)
CHUNK_CYCLE = [(37, 99, 235), (5, 150, 105), (217, 119, 6), (190, 24, 93), (124, 58, 237)]


def font(size, bold=False):
    for name in (("seguisb.ttf", "segoeuib.ttf") if bold else ("segoeui.ttf",)):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_page(pdf_path, page_num, width):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    zoom = width / page.rect.width
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img, zoom


def faded(img, amount=0.55):
    return Image.blend(img, Image.new("RGB", img.size, (255, 255, 255)), amount)


def chip(draw, xy, text, color, f):
    x, y = xy
    tw = draw.textlength(text, font=f)
    th = f.size + 4
    y = max(0, y - th - 1)
    draw.rectangle([x, y, x + tw + 8, y + th], fill=color)
    draw.text((x + 4, y + 1), text, fill=(255, 255, 255), font=f)


def banner(img, title, subtitle, f_title, f_sub, bar=46):
    w, h = img.size
    out = Image.new("RGB", (w, h + bar), PAPER)
    out.paste(img, (0, bar))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, w, bar - 1], fill=PAPER)
    d.line([0, bar - 1, w, bar - 1], fill=(228, 228, 231), width=1)
    d.text((14, 8), title, fill=INK, font=f_title)
    d.text((14, 8 + f_title.size + 3), subtitle, fill=MUTED, font=f_sub)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--out", default="docs/demo.gif")
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--weights", default="weights/yolov8m-doclaynet.pt")
    ap.add_argument("--ms", type=int, default=1750)
    args = ap.parse_args()

    print(f"parsing {args.pdf} ...", flush=True)
    weights = args.weights if args.weights and os.path.exists(args.weights) else None
    if not weights:
        print("  NOTE: no weights -- geometry only, frame 2 will say so")
    result = run_pipeline(args.pdf, weights=weights)
    blocks = [b for b in result.blocks if b.page_num == args.page]
    blocks.sort(key=lambda b: b.reading_order)
    print(f"  {len(result.blocks)} blocks, {len(result.chunks)} chunks; "
          f"{len(blocks)} blocks on page {args.page}")
    if not blocks:
        print("no blocks on that page")
        return 1

    base, zoom = render_page(args.pdf, args.page, args.width)
    f_t, f_s, f_c = font(17, True), font(13), font(11, True)

    def scaled(b):
        return [b.bbox.x0 * zoom, b.bbox.y0 * zoom, b.bbox.x1 * zoom, b.bbox.y1 * zoom]

    frames = []

    # 1 -- the page as the PDF ships it
    frames.append(banner(
        base, "1. A born-digital PDF",
        "Two columns. The order the text sits in the file is not the order a human reads it.",
        f_t, f_s))

    # 2 -- fused blocks, coloured by label, tagged by which detector saw them
    im = faded(base, 0.45)
    d = ImageDraw.Draw(im)
    for b in blocks:
        col = LABEL_COLOR.get(str(getattr(b.label, "value", b.label)), DEFAULT_COLOR)
        dashed = b.source == Source.DISPUTED
        d.rectangle(scaled(b), outline=col, width=3 if not dashed else 2)
    # Chip only blocks big enough to read, so the frame stays legible.
    chipped = sorted(blocks, key=lambda b: -(b.bbox.x1 - b.bbox.x0) * (b.bbox.y1 - b.bbox.y0))
    for b in chipped[:8]:
        col = LABEL_COLOR.get(str(getattr(b.label, "value", b.label)), DEFAULT_COLOR)
        x0, y0, _, _ = scaled(b)
        chip(d, (x0, y0), str(getattr(b.label, "value", b.label)), col, f_c)
    conf = sum(1 for b in blocks if b.source == Source.CONFIRMED)
    uni_g = sum(1 for b in blocks if b.source == Source.UNILATERAL_GEOMETRY)
    uni_m = sum(1 for b in blocks if b.source == Source.UNILATERAL_MODEL)
    sub = (f"Geometry reads PDF primitives; vision reads the raster. "
           f"{conf} confirmed by both, {uni_g} geometry-only, {uni_m} vision-only."
           if weights else
           "Geometry reads PDF primitives directly. No model weights present, so the "
           "vision detector is off -- the pipeline still runs.")
    frames.append(banner(im, "2. Two blind detectors, fused", sub, f_t, f_s))

    # 3 -- reading order
    im = faded(base, 0.62)
    d = ImageDraw.Draw(im)
    pts = []
    for i, b in enumerate(blocks, 1):
        x0, y0, x1, y1 = scaled(b)
        d.rectangle([x0, y0, x1, y1], outline=(203, 213, 225), width=2)
        pts.append(((x0 + x1) / 2, (y0 + y1) / 2))
    if len(pts) > 1:
        d.line(pts, fill=(37, 99, 235), width=3)
    for i, (cx, cy) in enumerate(pts, 1):
        r = 13
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(37, 99, 235))
        t = str(i)
        d.text((cx - d.textlength(t, font=f_c) / 2, cy - f_c.size / 2 - 1),
               t, fill=(255, 255, 255), font=f_c)
    frames.append(banner(
        im, "3. Column-aware reading order",
        "Left column top-to-bottom, then right. Not the order the bytes appear in.",
        f_t, f_s))

    # 4 -- chunks: colour every block by the chunk it was assembled into
    chunk_of = {}
    for ci, ch in enumerate(result.chunks):
        for bid in ch.source_block_ids:
            chunk_of[bid] = ci
    im = faded(base, 0.45)
    d = ImageDraw.Draw(im)
    on_page = []
    for b in blocks:
        ci = chunk_of.get(b.block_id)
        if ci is None:
            continue
        on_page.append(ci)
        col = CHUNK_CYCLE[ci % len(CHUNK_CYCLE)]
        x0, y0, x1, y1 = scaled(b)
        d.rectangle([x0, y0, x1, y1], outline=col, width=4)
    seen, order = set(), []
    for ci in on_page:
        if ci not in seen:
            seen.add(ci)
            order.append(ci)
    for ci in order[:10]:
        ch = result.chunks[ci]
        bs = [b for b in blocks if chunk_of.get(b.block_id) == ci]
        if not bs:
            continue
        x0 = min(scaled(b)[0] for b in bs)
        y0 = min(scaled(b)[1] for b in bs)
        sp = ch.section_path
        # deepest heading available -- h1 is the paper title on every chunk,
        # which says nothing about where the chunk sits.
        # h1 is the paper title on every chunk, so it says nothing about where
        # the chunk sits. Use the deepest real heading, else the chunk type.
        sec = sp.h3 or sp.h2 or ch.chunk_type
        chip(d, (x0, y0), f"{sec[:26]}", CHUNK_CYCLE[ci % len(CHUNK_CYCLE)], f_c)
    frames.append(banner(
        im, f"4. {len(order)} retrieval-ready chunks on this page",
        "Merged under a size floor, tagged with their section path. Same PDF in, same chunks out.",
        f_t, f_s))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    w = max(f.width for f in frames)
    h = max(f.height for f in frames)
    frames = [f if f.size == (w, h) else
              (lambda c: (c.paste(f, (0, 0)), c)[1])(Image.new("RGB", (w, h), PAPER))
              for f in frames]
    durations = [args.ms, args.ms + 700, args.ms + 700, args.ms + 1400]
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1024:.0f} KB, "
          f"{len(frames)} frames, {w}x{h})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
