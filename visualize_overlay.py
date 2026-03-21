#!/usr/bin/env python3
"""Visual overlay utility for DocStruct JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pdfplumber
from PIL import Image, ImageDraw, ImageFont

BLOCK_COLORS = {
    "text": (41, 128, 185),
    "header": (39, 174, 96),
    "table": (211, 84, 0),
    "figure": (142, 68, 173),
    "caption": (192, 57, 43),
}
HEADER_HEIGHT = 24
PANEL_GAP = 20


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pdf_to_image_coords(
    bbox: Dict[str, float],
    img_width: int,
    img_height: int,
    page_width: float,
    page_height: float,
) -> Tuple[float, float, float, float]:
    sx = img_width / page_width
    sy = img_height / page_height
    left = bbox["x0"] * sx
    right = bbox["x1"] * sx
    top = img_height - (bbox["y1"] * sy)
    bottom = img_height - (bbox["y0"] * sy)
    return left, top, right, bottom


def _draw_label(draw: ImageDraw.ImageDraw, text: str, x: float, y: float, color: Tuple[int, int, int], font) -> None:
    padding = 2
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle([left - padding, top - padding, right + padding, bottom + padding], fill=color)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def _render_page_overlay(rendered, page_payload: Dict[str, Any], font) -> Image.Image:
    image = rendered.convert("RGB")
    draw = ImageDraw.Draw(image)
    page_width = float(page_payload.get("dimensions", {}).get("width", image.width))
    page_height = float(page_payload.get("dimensions", {}).get("height", image.height))

    for block in page_payload.get("blocks", []):
        bbox = block.get("bbox")
        if not bbox:
            continue
        block_type = block.get("block_type", "text")
        color = BLOCK_COLORS.get(block_type, (0, 0, 0))
        left, top, right, bottom = _pdf_to_image_coords(
            bbox,
            image.width,
            image.height,
            page_width,
            page_height,
        )
        draw.rectangle([left, top, right, bottom], outline=color, width=2)
        conf = block.get("confidence", {}).get("final_confidence")
        label = f"{block_type} {'n/a' if conf is None else f'{float(conf):.2f}'}"
        _draw_label(draw, label, left + 2, max(0, top - 12), color, font)
    return image


def _compose_side_by_side(left_img: Image.Image, right_img: Image.Image, left_title: str, right_title: str, font) -> Image.Image:
    width = left_img.width + right_img.width + PANEL_GAP
    height = max(left_img.height, right_img.height) + HEADER_HEIGHT
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 4), left_title, fill=(0, 0, 0), font=font)
    draw.text((left_img.width + PANEL_GAP + 10, 4), right_title, fill=(0, 0, 0), font=font)
    canvas.paste(left_img, (0, HEADER_HEIGHT))
    canvas.paste(right_img, (left_img.width + PANEL_GAP, HEADER_HEIGHT))
    return canvas


def create_overlays(input_pdf: Path, input_json: Path, output_dir: Path, dpi: int = 150) -> None:
    data = _load_json(input_json)
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    with pdfplumber.open(str(input_pdf)) as pdf:
        for page_payload in data.get("pages", []):
            page_num = page_payload["page_num"]
            if page_num < 0 or page_num >= len(pdf.pages):
                continue
            rendered = pdf.pages[page_num].to_image(resolution=dpi).original
            overlay = _render_page_overlay(rendered, page_payload, font)
            overlay.save(output_dir / f"overlay_page_{page_num + 1:03d}.png", format="PNG")


def create_comparison_overlays(
    input_pdf: Path,
    left_json: Path,
    right_json: Path,
    output_dir: Path,
    dpi: int = 150,
) -> None:
    left_data = _load_json(left_json)
    right_data = _load_json(right_json)
    left_pages = {page["page_num"]: page for page in left_data.get("pages", [])}
    right_pages = {page["page_num"]: page for page in right_data.get("pages", [])}
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    with pdfplumber.open(str(input_pdf)) as pdf:
        for page_num in range(len(pdf.pages)):
            rendered = pdf.pages[page_num].to_image(resolution=dpi).original
            left_overlay = _render_page_overlay(rendered.copy(), left_pages.get(page_num, {"blocks": []}), font)
            right_overlay = _render_page_overlay(rendered.copy(), right_pages.get(page_num, {"blocks": []}), font)
            comparison = _compose_side_by_side(
                left_overlay,
                right_overlay,
                f"LEFT: {left_json.name}",
                f"RIGHT: {right_json.name}",
                font,
            )
            comparison.save(output_dir / f"compare_page_{page_num + 1:03d}.png", format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay DocStruct JSON blocks onto source PDF pages")
    parser.add_argument("input_pdf", type=Path, help="Path to source PDF")
    parser.add_argument("input_json", nargs="?", type=Path, help="Path to single JSON overlay input")
    parser.add_argument("--left-json", type=Path, help="Left JSON for side-by-side comparison")
    parser.add_argument("--right-json", type=Path, help="Right JSON for side-by-side comparison")
    parser.add_argument("--output-dir", type=Path, default=Path("overlay_output"))
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if not args.input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {args.input_pdf}")

    if args.left_json and args.right_json:
        if not args.left_json.exists():
            raise FileNotFoundError(f"Left JSON not found: {args.left_json}")
        if not args.right_json.exists():
            raise FileNotFoundError(f"Right JSON not found: {args.right_json}")
        create_comparison_overlays(args.input_pdf, args.left_json, args.right_json, args.output_dir, dpi=args.dpi)
        print(f"Comparison overlays written to: {args.output_dir}")
        return

    if args.input_json is None:
        raise ValueError("Provide input_json for single overlay mode or both --left-json and --right-json for comparison mode")
    if not args.input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")

    create_overlays(args.input_pdf, args.input_json, args.output_dir, dpi=args.dpi)
    print(f"Overlay images written to: {args.output_dir}")


if __name__ == "__main__":
    main()
