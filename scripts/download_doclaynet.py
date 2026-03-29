"""Download DocLayNet from HuggingFace and save images + annotations.jsonl.

Uses the same class mapping as DOC_LAYOUT_LABEL_MAP in models/doclaynet_detector.py
and the same coordinate convention as load_doclaynet_hf in evaluation/ground_truth.py.

Usage:
    python scripts/download_doclaynet.py --out-dir ./data/doclaynet --max-docs 200
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

DOCLAYNET_CLASSES = [
    'Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer',
    'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title',
]

LABEL_REMAP = {
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
    'Title': 'header',
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DocLayNet and write annotations.jsonl")
    parser.add_argument('--out-dir', default='./data/doclaynet', help='Output directory')
    parser.add_argument('--split', default='validation', help='Dataset split (validation/train/test — test has no annotations)')
    parser.add_argument('--max-docs', type=int, default=200, help='Max number of pages to save')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    images_dir = out_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading DocLayNet split='{args.split}' (streaming) ...")
    ds = load_dataset('docling-project/DocLayNet-v1.2', split=args.split, streaming=True)

    saved = 0
    with open(out_dir / 'annotations.jsonl', 'w', encoding='utf-8') as f:
        for i, item in enumerate(ds):
            if i >= args.max_docs:
                break

            image = item['image']
            img_path = images_dir / f'{i:06d}.png'
            image.save(str(img_path))

            # Height is nested under metadata.coco_height
            metadata = item.get('metadata', {})
            page_height = float(metadata.get('coco_height', image.height))

            gts = []
            categories = item.get('category_id', [])
            bboxes = item.get('bboxes', [])
            for cat_id, bbox in zip(categories, bboxes):
                raw_type = DOCLAYNET_CLASSES[cat_id - 1]  # category_id is 1-based
                block_type = LABEL_REMAP.get(raw_type, 'text')
                x_min, y_min, width, height = bbox
                # Convert top-left origin → bottom-left origin
                bl_y0 = page_height - (y_min + height)
                bl_y1 = page_height - y_min
                gts.append({
                    'block_type': block_type,
                    'bbox': {
                        'x0': float(x_min),
                        'y0': float(bl_y0),
                        'x1': float(x_min + width),
                        'y1': float(bl_y1),
                    },
                })

            record = {
                'image_id': i,
                'image_path': str(img_path),
                'image_width': image.width,
                'image_height': image.height,
                'ground_truths': gts,
            }
            f.write(json.dumps(record) + '\n')
            saved += 1

            if i == 0 and not gts:
                print(f'WARNING: first item has 0 GT boxes — check that split="{args.split}" includes annotations.')
            if i % 10 == 0:
                print(f'  saved {i} / {args.max_docs}')

    print(f'Done. {saved} records written to {out_dir / "annotations.jsonl"}')


if __name__ == '__main__':
    main()
