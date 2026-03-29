"""Ground-truth loaders for evaluation modes."""

import json
from pathlib import Path
from typing import Dict, List

from utils.logging import setup_logger

logger = setup_logger(__name__)


HF_DOCLAYNET_DATASET_ID = "docling-project/DocLayNet-v1.2"


def load_doclaynet_local(data_dir: str, max_docs: int = 200) -> List[Dict]:
    """Load DocLayNet annotations from a locally downloaded JSONL file."""
    ann_path = Path(data_dir) / "annotations.jsonl"
    if not ann_path.exists():
        logger.error(f"DocLayNet annotation file not found at {ann_path}")
        return []

    docs = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            record = json.loads(line.strip())
            docs.append(record)

    logger.info(f"Loaded {len(docs)} DocLayNet pages from {ann_path}")
    return docs


def load_doclaynet_hf(max_docs: int = 50) -> List[Dict]:
    """Load the DocLayNet dataset from Hugging Face and parse ground truths."""
    from datasets import load_dataset

    logger.info(f"Loading Hugging Face dataset: {HF_DOCLAYNET_DATASET_ID}")
    ds = load_dataset(HF_DOCLAYNET_DATASET_ID, split="validation", streaming=True)

    # Class maps based on DocLayNet schema
    doclaynet_classes = ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']

    # Map to DocStruct variants
    label_remap = {
        'Caption': 'caption',
        'Footnote': 'text',
        'Formula': 'text',
        'List-item': 'text',
        'Page-footer': 'text',
        'Page-header': 'header',
        'Picture': 'figure',
        'Section-header': 'header',
        'Table': 'table',
        'Text': 'text',
        'Title': 'header'
    }

    docs = []
    for i, item in enumerate(ds):
        if i >= max_docs:
            break

        gts = []
        metadata = item.get("metadata", {})
        page_height = float(metadata.get("coco_height", 1025.0))

        categories = item.get("category_id", [])
        bboxes = item.get("bboxes", [])

        for cat_id, bbox in zip(categories, bboxes):
            raw_type = doclaynet_classes[cat_id - 1]  # category_id is 1-based
            block_type = label_remap.get(raw_type, "text")

            # Bbox is [x_min, y_min, width, height]
            x_min, y_min, width, height = bbox

            # Convert to bottom-left origin
            bl_y0 = page_height - (y_min + height)
            bl_y1 = page_height - y_min

            gts.append({
                "block_type": block_type,
                "bbox": {"x0": x_min, "y0": bl_y0, "x1": x_min + width, "y1": bl_y1}
            })

        docs.append({
            "image_id": i,
            "ground_truths": gts,
            "image": item.get("image"),
            "dataset_id": HF_DOCLAYNET_DATASET_ID
        })

    logger.info(f"Loaded {len(docs)} documents with ground truths from {HF_DOCLAYNET_DATASET_ID}")
    return docs
