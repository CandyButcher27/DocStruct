import json
import pdfplumber
from PIL import ImageDraw
from pathlib import Path

INPUT_PDF = "arxiv_test/pdfs/2302.04062.pdf"
INPUT_JSON = "arxiv_test/outputs/2302.04062.json"
OUTPUT_DIR = Path("arxiv_test/overlays/2302.04062")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "text": "blue",
    "header": "red",
    "table": "green",
    "figure": "orange",
    "caption": "purple",
}

def main():
    with open(INPUT_JSON, encoding="utf-8") as f:
        data = json.load(f)

    with pdfplumber.open(INPUT_PDF) as pdf:
        for page_index, page_data in enumerate(data["pages"]):
            page = pdf.pages[page_index]

            # Render page at higher resolution
            im = page.to_image(resolution=150)
            image = im.original
            scale = im.scale  # critical for DPI correction

            draw = ImageDraw.Draw(image)

            for block in page_data["blocks"]:
                bbox = block["bbox"]

                # Scale coordinates from PDF space (72 DPI) → image space
                x0 = bbox["x0"] * scale
                x1 = bbox["x1"] * scale
                y0 = bbox["y0"] * scale
                y1 = bbox["y1"] * scale

                color = COLORS.get(block["block_type"], "black")

                # Convert bottom-left PDF origin → top-left image origin
                image_height = image.height
                y0_img = image_height - y1
                y1_img = image_height - y0

                draw.rectangle(
                    [x0, y0_img, x1, y1_img],
                    outline=color,
                    width=2
                )

                draw.text(
                    (x0, y0_img),
                    f"{block['reading_order']}:{block['block_type']}",
                    fill=color
                )

            output_path = OUTPUT_DIR / f"page_{page_index+1}.png"
            image.save(output_path)
            print(f"Saved overlay: {output_path}")

if __name__ == "__main__":
    main()