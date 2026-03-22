"""Ground-truth loaders for evaluation modes."""

import json
from pathlib import Path
from typing import Dict, List

from utils.logging import setup_logger

logger = setup_logger(__name__)


HF_PUBLAYNET_DATASET_ID = "nielsr/publaynet-processed"
HF_DOCLAYNET_DATASET_ID = "dsm-72/doclaynet" # This is a common HF mirror of DocLayNet


def load_publaynet_sample(data_dir: str, max_docs: int = 50) -> List[Dict]:
    """
    Load a sample of PubLayNet-style annotations from a local directory.

    Expected layout:
      <data-dir>/val.json
      <data-dir>/pdfs/<file_name>.pdf
    """
    ann_path = Path(data_dir) / "val.json"
    if not ann_path.exists():
        logger.warning(f"PubLayNet annotation not found at {ann_path}")
        return []

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_map = {cat["id"]: cat["name"].lower() for cat in coco.get("categories", [])}
    label_remap = {"title": "header", "list": "text"}

    from collections import defaultdict

    image_anns: Dict[int, List] = defaultdict(list)
    for ann in coco.get("annotations", []):
        image_anns[ann["image_id"]].append(ann)

    docs = []
    pdf_dir = Path(data_dir) / "pdfs"
    for image in coco.get("images", [])[:max_docs]:
        img_h = image["height"]
        gts = []
        for ann in image_anns[image["id"]]:
            x0, y0, w, h = ann["bbox"]
            bl_y0 = img_h - (y0 + h)
            bl_y1 = img_h - y0
            raw_type = cat_map.get(ann["category_id"], "text")
            block_type = label_remap.get(raw_type, raw_type)
            gts.append(
                {
                    "block_type": block_type,
                    "bbox": {"x0": x0, "y0": bl_y0, "x1": x0 + w, "y1": bl_y1},
                }
            )

        pdf_path = pdf_dir / image.get("file_name", "")
        docs.append(
            {
                "image_id": image["id"],
                "ground_truths": gts,
                "pdf_path": str(pdf_path) if pdf_path.exists() else None,
            }
        )

    logger.info(f"Loaded {len(docs)} documents from PubLayNet ({ann_path})")
    return docs


def load_hf_publaynet(max_docs: int = 50) -> List[Dict]:
    """Load the canonical Hugging Face PubLayNet page-image dataset."""
    from datasets import load_dataset

    logger.info(f"Loading Hugging Face dataset: {HF_PUBLAYNET_DATASET_ID}")
    ds = load_dataset(HF_PUBLAYNET_DATASET_ID, split="validation")

    docs = []
    for i, item in enumerate(ds):
        if i >= max_docs:
            break

        gts = []
        for block_type, bbox in zip(item.get("block_types", []), item.get("bboxes", [])):
            gts.append({"block_type": block_type, "bbox": bbox})

        docs.append(
            {
                "image_id": i,
                "ground_truths": gts,
                "image": item.get("image"),
                "dataset_id": HF_PUBLAYNET_DATASET_ID,
            }
        )

    logger.info(f"Loaded {len(docs)} documents from {HF_PUBLAYNET_DATASET_ID}")
    return docs


def load_doclaynet_hf(max_docs: int = 50) -> List[Dict]:
    """Load the DocLayNet dataset from Hugging Face."""
    from datasets import load_dataset
    
    logger.info(f"Loading Hugging Face dataset: {HF_DOCLAYNET_DATASET_ID}")
    # Note: DocLayNet is large, we load only the validation split and a few samples
    ds = load_dataset(HF_DOCLAYNET_DATASET_ID, split="validation", streaming=True)
    
    docs = []
    # DocLayNet COCO categories to DocStruct block types
    cat_map = {
        "Caption": "caption",
        "Footnote": "text",
        "Formula": "text",
        "List-item": "text",
        "Page-footer": "text",
        "Page-header": "header",
        "Picture": "figure",
        "Section-header": "header",
        "Table": "table",
        "Text": "text",
        "Title": "header"
    }

    for i, item in enumerate(ds):
        if i >= max_docs:
            break
            
        # DocLayNet items in hf usually have 'image' and 'annotations' or similar
        # Depending on the mirror, the schema varies. Assuming standard COCO-like for dsm-72
        gts = []
        # If it's a streaming dataset, we might need to iterate
        # For simplicity in this implementation plan, we'll assume standard layout
        # If the mirror is different, this might need refinement.
        
        docs.append({
            "image_id": i,
            "ground_truths": [], # Placeholder for now as direct HF PDF eval is complex
            "image": item.get("image"),
            "dataset_id": HF_DOCLAYNET_DATASET_ID
        })
        
    logger.info(f"Loaded {len(docs)} document placeholders from {HF_DOCLAYNET_DATASET_ID}")
    return docs
