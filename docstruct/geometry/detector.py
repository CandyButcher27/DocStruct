"""Rules-based geometry detector (pdfplumber).

Produces ``List[Proposal]`` from a PDF using document-agnostic rules: font-size
ranking for headers (no hardcoded section names), whitespace-gap column
splitting (no fixed column count), ruled-table extraction, and graphic-cluster
figure detection. Coordinates are top-left (pdfplumber ``top``/``bottom``).
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List

from docstruct import config
from docstruct.errors import open_pdf
from docstruct.schema import BoundingBox, Proposal
from docstruct.utils.geometry import bbox_overlap

_CAPTION_RE = re.compile(config.CAPTION_PREFIX_PATTERN, re.IGNORECASE)


@dataclass
class _Line:
    words: List[Dict[str, Any]]
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    bold: bool
    text: str


def _to_bbox(x0, top, x1, bottom, page) -> BoundingBox:
    return BoundingBox(x0, top, x1, bottom, page.width, page.height)


def _group_words_into_lines(words: List[Dict[str, Any]]) -> List[_Line]:
    ordered = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines: List[List[Dict[str, Any]]] = []
    for w in ordered:
        if lines and abs(w["top"] - lines[-1][0]["top"]) <= config.LINE_Y_TOLERANCE:
            lines[-1].append(w)
        else:
            lines.append([w])

    result: List[_Line] = []
    for group in lines:
        group.sort(key=lambda w: w["x0"])
        sizes = [w.get("size", 0.0) for w in group if w.get("size")]
        bolds = [
            any(m in (w.get("fontname", "") or "").lower() for m in config.BOLD_FONT_MARKERS)
            for w in group
        ]
        result.append(
            _Line(
                words=group,
                x0=min(w["x0"] for w in group),
                x1=max(w["x1"] for w in group),
                top=min(w["top"] for w in group),
                bottom=max(w["bottom"] for w in group),
                size=statistics.median(sizes) if sizes else 0.0,
                bold=sum(bolds) > len(bolds) / 2,
                text=" ".join(w["text"] for w in group),
            )
        )
    return result


def _split_columns(lines: List[_Line], page_width: float) -> List[List[_Line]]:
    if len(lines) < 2:
        return [lines]

    def center(l: _Line) -> float:
        return (l.x0 + l.x1) / 2

    ordered = sorted(lines, key=center)
    gaps = [
        (idx, center(ordered[idx]) - center(ordered[idx - 1]))
        for idx in range(1, len(ordered))
    ]
    split_index, max_gap = max(gaps, key=lambda g: g[1])

    if max_gap < page_width * config.COLUMN_GAP_RATIO:
        return [lines]

    left = ordered[:split_index]
    right = ordered[split_index:]
    columns = [left, right]
    columns.sort(key=lambda col: sum((l.x0 + l.x1) / 2 for l in col) / len(col))
    return columns


def _line_kind(line: _Line, body_median: float) -> str:
    if body_median > 0 and line.size > body_median * config.HEADER_SIZE_RATIO:
        return "header"
    # A short, fully-bold line at body size is a numbered/styled section header.
    if (
        line.bold
        and len(line.words) <= config.HEADER_MAX_WORDS
        and (body_median == 0 or line.size >= body_median)
    ):
        return "header"
    return "body"


def _group_lines_into_blocks(
    lines: List[_Line], body_median: float, median_line_height: float
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None

    for line in sorted(lines, key=lambda l: l.top):
        kind = _line_kind(line, body_median)
        if current is None:
            current = {"kind": kind, "lines": [line]}
            continue

        gap = line.top - current["lines"][-1].bottom
        same_kind = kind == current["kind"]
        close_gap = gap <= config.PARAGRAPH_GAP_FACTOR * median_line_height
        header_ok = current["kind"] != "header" or len(current["lines"]) < config.HEADER_MAX_LINES

        if same_kind and close_gap and header_ok:
            current["lines"].append(line)
        else:
            blocks.append(current)
            current = {"kind": kind, "lines": [line]}

    if current is not None:
        blocks.append(current)
    return blocks


def _cluster_graphics(boxes: List[tuple], gap: float) -> List[tuple]:
    clusters: List[List[float]] = []
    for x0, top, x1, bottom in boxes:
        placed = False
        for c in clusters:
            if not (x1 < c[0] - gap or x0 > c[2] + gap or bottom < c[1] - gap or top > c[3] + gap):
                c[0] = min(c[0], x0)
                c[1] = min(c[1], top)
                c[2] = max(c[2], x1)
                c[3] = max(c[3], bottom)
                placed = True
                break
        if not placed:
            clusters.append([x0, top, x1, bottom])
    return [tuple(c) for c in clusters]


def detect_page(page, page_num: int) -> List[Proposal]:
    """Detect layout proposals on a single pdfplumber page."""
    proposals: List[Proposal] = []
    counter = 0
    page_area = page.width * page.height

    def _add(label: str, bbox: BoundingBox, conf: float) -> None:
        nonlocal counter
        proposals.append(
            Proposal(
                label=label,
                confidence=round(min(conf, config.GEOMETRY_CONFIDENCE_CEIL), 4),
                bbox=bbox,
                source="geometry",
                page_num=page_num,
                proposal_id=f"geo_{page_num}_{counter}",
            )
        )
        counter += 1

    # Tables first; their region masks out text and graphics.
    table_boxes: List[BoundingBox] = []
    for table in page.find_tables():
        if len(table.rows) < config.TABLE_MIN_ROWS:
            continue
        x0, top, x1, bottom = table.bbox
        bbox = _to_bbox(x0, top, x1, bottom, page)
        table_boxes.append(bbox)
        _add("table", bbox, config.GEOMETRY_CONFIDENCE["table"])

    def _in_table(bbox: BoundingBox) -> bool:
        return any(bbox_overlap(bbox, tb) > 0.5 for tb in table_boxes)

    # Drop rotated/vertical text (e.g. arXiv sidebars); it is not reading flow.
    upright_page = page.filter(lambda obj: obj.get("upright", True))
    words = upright_page.extract_words(extra_attrs=["fontname", "size"])
    lines = _group_words_into_lines(words)
    lines = [l for l in lines if not _in_table(_to_bbox(l.x0, l.top, l.x1, l.bottom, page))]

    if lines:
        sizes = [l.size for l in lines if l.size]
        body_median = statistics.median(sizes) if sizes else 0.0
        heights = [l.bottom - l.top for l in lines if l.bottom > l.top]
        median_line_height = statistics.median(heights) if heights else 12.0

        for column in _split_columns(lines, page.width):
            for block in _group_lines_into_blocks(column, body_median, median_line_height):
                blines = block["lines"]
                bbox = _to_bbox(
                    min(l.x0 for l in blines),
                    min(l.top for l in blines),
                    max(l.x1 for l in blines),
                    max(l.bottom for l in blines),
                    page,
                )
                text = " ".join(l.text for l in blines)
                if block["kind"] == "header":
                    conf = config.GEOMETRY_CONFIDENCE["header"]
                    if any(l.bold for l in blines):
                        conf += config.HEADER_BOLD_BONUS
                    _add("header", bbox, conf)
                elif _CAPTION_RE.match(text):
                    _add("caption", bbox, config.GEOMETRY_CONFIDENCE["caption"])
                else:
                    _add("text", bbox, config.GEOMETRY_CONFIDENCE["text"])

    # Figures: raster images plus clustered vector graphics.
    graphic_boxes = [
        (g["x0"], g["top"], g["x1"], g["bottom"])
        for g in (page.images + page.curves + page.rects + page.lines)
    ]
    for x0, top, x1, bottom in _cluster_graphics(graphic_boxes, config.FIGURE_CLUSTER_GAP):
        bbox = _to_bbox(x0, top, x1, bottom, page)
        if bbox.area < page_area * config.FIGURE_MIN_AREA_RATIO:
            continue
        if _in_table(bbox):
            continue
        text_overlap = sum(
            1
            for l in lines
            if bbox_overlap(bbox, _to_bbox(l.x0, l.top, l.x1, l.bottom, page)) > 0
        )
        if lines and text_overlap / max(len(lines), 1) > config.FIGURE_MAX_TEXT_OVERLAP:
            continue
        _add("figure", bbox, config.GEOMETRY_CONFIDENCE["figure"])

    return proposals


def detect(pdf_path: str, *, password: str | None = None) -> List[Proposal]:
    """Detect layout proposals across every page of a PDF."""
    proposals: List[Proposal] = []
    with open_pdf(pdf_path, password=password) as pdf:
        for page_num, page in enumerate(pdf.pages):
            proposals.extend(detect_page(page, page_num))
    return proposals
