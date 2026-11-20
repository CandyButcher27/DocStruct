"""Render the README animation: one real page walked through the real pipeline.

    python scripts/make_readme_gif.py data/raw-pdfs/doc1.pdf --page 1 --out docs/demo.gif

Every box drawn comes from an actual `run_pipeline()` result -- nothing here is a
mock-up. Four stages, each animated: the page as shipped, the fused blocks appearing
in reading order, the reading-order path drawing itself, and the chunks filling in one
at a time. Stages crossfade so the loop has no hard cuts.
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
PATH = (37, 99, 235)
GHOST = (203, 213, 225)

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

FRAME_MS = 70       # animated frames
FADE_FRAMES = 3     # crossfade length between stages


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
    d.line([0, bar - 1, w, bar - 1], fill=(228, 228, 231), width=1)
    d.text((14, 8), title, fill=INK, font=f_title)
    d.text((14, 8 + f_title.size + 3), subtitle, fill=MUTED, font=f_sub)
    return out


def label_of(b):
    return str(getattr(b.label, "value", b.label))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--out", default="docs/demo.gif")
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--weights", default="weights/yolov8m-doclaynet.pt")
    ap.add_argument("--hold", type=int, default=1800, help="ms each finished stage stays up")
    args = ap.parse_args()

    print(f"parsing {args.pdf} ...", flush=True)
    weights = args.weights if args.weights and os.path.exists(args.weights) else None
    if not weights:
        print("  NOTE: no weights -- geometry only, stage 2 will say so")
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

    frames, durations = [], []

    def add(img, ms):
        frames.append(img)
        durations.append(ms)

    def fade_to(img, bar=46):
        # The banner switches at once; only the page crossfades, so two titles
        # never overlap mid-fade.
        prev = frames[-1].crop((0, bar, img.width, img.height))
        new = img.crop((0, bar, img.width, img.height))
        for i in range(1, FADE_FRAMES + 1):
            f = img.copy()
            f.paste(Image.blend(prev, new, i / (FADE_FRAMES + 1)), (0, bar))
            add(f, FRAME_MS)

    # 1 -- the page as the PDF ships it
    t1 = ("1. A born-digital PDF",
          "Two columns. The order the text sits in the file is not the order a human reads it.")
    page_frame = banner(base, *t1, f_t, f_s)
    add(page_frame, args.hold)

    # 2 -- fused blocks appear in reading order, coloured by label
    conf = sum(1 for b in blocks if b.source == Source.CONFIRMED)
    uni_g = sum(1 for b in blocks if b.source == Source.UNILATERAL_GEOMETRY)
    uni_m = sum(1 for b in blocks if b.source == Source.UNILATERAL_MODEL)
    t2 = ("2. Two blind detectors, fused",
          f"Geometry reads PDF primitives; vision reads the raster. "
          f"{conf} confirmed by both, {uni_g} geometry-only, {uni_m} vision-only."
          if weights else
          "Geometry reads PDF primitives directly. No model weights present, so the "
          "vision detector is off -- the pipeline still runs.")
    blocks_bg = faded(base, 0.45)

    def blocks_frame(k, chips=False):
        im = blocks_bg.copy()
        d = ImageDraw.Draw(im)
        for b in blocks[:k]:
            width = 2 if b.source == Source.DISPUTED else 3
            d.rectangle(scaled(b), outline=LABEL_COLOR.get(label_of(b), DEFAULT_COLOR), width=width)
        if chips:
            # Chip only the largest blocks so the frame stays legible.
            big = sorted(blocks, key=lambda b: -(b.bbox.x1 - b.bbox.x0) * (b.bbox.y1 - b.bbox.y0))
            for b in big[:8]:
                chip(d, scaled(b)[:2], label_of(b), LABEL_COLOR.get(label_of(b), DEFAULT_COLOR), f_c)
        return banner(im, *t2, f_t, f_s)

    fade_to(blocks_frame(0))
    step = max(1, len(blocks) // 14)
    for k in range(step, len(blocks) + step, step):
        add(blocks_frame(min(k, len(blocks))), FRAME_MS)
    add(blocks_frame(len(blocks), chips=True), args.hold)

    # 3 -- the reading-order path draws itself
    t3 = ("3. Column-aware reading order",
          "Left column top-to-bottom, then right. Not the order the bytes appear in.")
    order_bg = faded(base, 0.62)
    d = ImageDraw.Draw(order_bg)
    pts = []
    for b in blocks:
        x0, y0, x1, y1 = scaled(b)
        d.rectangle([x0, y0, x1, y1], outline=GHOST, width=2)
        pts.append(((x0 + x1) / 2, (y0 + y1) / 2))

    def order_frame(k):
        im = order_bg.copy()
        d = ImageDraw.Draw(im)
        if k > 1:
            d.line(pts[:k], fill=PATH, width=3)
        for i, (cx, cy) in enumerate(pts[:k], 1):
            r = 13
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=PATH)
            t = str(i)
            d.text((cx - d.textlength(t, font=f_c) / 2, cy - f_c.size / 2 - 1),
                   t, fill=(255, 255, 255), font=f_c)
        return banner(im, *t3, f_t, f_s)

    fade_to(order_frame(0))
    for k in range(1, len(pts) + 1):
        add(order_frame(k), FRAME_MS + 30)
    durations[-1] = args.hold

    # 4 -- chunks fill in one at a time, each block coloured by its chunk
    chunk_of = {}
    for ci, ch in enumerate(result.chunks):
        for bid in ch.source_block_ids:
            chunk_of[bid] = ci
    order = list(dict.fromkeys(chunk_of[b.block_id] for b in blocks if b.block_id in chunk_of))
    t4 = (f"4. {len(order)} retrieval-ready chunks on this page",
          "Merged under a size floor, tagged with their section path. Same PDF in, same chunks out.")
    chunks_bg = faded(base, 0.45)

    def chunks_frame(k):
        im = chunks_bg.copy()
        d = ImageDraw.Draw(im)
        shown = order[:k]
        for b in blocks:
            ci = chunk_of.get(b.block_id)
            if ci in shown:
                d.rectangle(scaled(b), outline=CHUNK_CYCLE[ci % len(CHUNK_CYCLE)], width=4)
        for ci in shown[:10]:
            ch = result.chunks[ci]
            bs = [b for b in blocks if chunk_of.get(b.block_id) == ci]
            x0 = min(scaled(b)[0] for b in bs)
            y0 = min(scaled(b)[1] for b in bs)
            # h1 is the paper title on every chunk, so use the deepest real heading.
            sec = ch.section_path.h3 or ch.section_path.h2 or ch.chunk_type
            chip(d, (x0, y0), sec[:26], CHUNK_CYCLE[ci % len(CHUNK_CYCLE)], f_c)
        return banner(im, *t4, f_t, f_s)

    fade_to(chunks_frame(0))
    for k in range(1, len(order) + 1):
        add(chunks_frame(k), 260)
    durations[-1] = args.hold + 900

    fade_to(page_frame)

    # One shared palette, no dithering: consecutive frames then differ only where
    # something changed, so the GIF stores small deltas instead of full pages.
    palette = frames[len(frames) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1024:.0f} KB, "
          f"{len(frames)} frames, {sum(durations) / 1000:.1f}s, {frames[0].width}x{frames[0].height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
