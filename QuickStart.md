# DocStruct - Quick Start Guide

## Installation (30 seconds)

```bash
# Navigate to project
cd docstruct

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- 50 MB disk space
- No GPU needed (stub detector)

## First Run (1 minute)

```bash
# Create a test PDF or use your own
# Run DocStruct
python main.py your_document.pdf output.json

# View results
cat output.json | python -m json.tool | head -50
```

## Example Output

```json
{
  "metadata": {
    "filename": "document.pdf",
    "num_pages": 3,
    "processing_version": "1.0.0"
  },
  "pages": [
    {
      "page_num": 0,
      "dimensions": {
        "width": 612,
        "height": 792
      },
      "blocks": [
        {
          "block_id": "block_0_0",
          "block_type": "header",
          "bbox": {
            "x0": 72,
            "y0": 50,
            "x1": 540,
            "y1": 80
          },
          "text": "Introduction",
          "confidence": {
            "rule_score": 0.9,
            "model_score": 0.0,
            "geometric_score": 1.0,
            "final_confidence": 0.47
          },
          "reading_order": 0
        }
      ]
    }
  ]
}
```

## Common Tasks

### Extract All Text

```python
import json

with open("output.json") as f:
    doc = json.load(f)

for page in doc["pages"]:
    for block in page["blocks"]:
        if block["text"]:
            print(block["text"])
```

### Find All Tables

```python
import json

with open("output.json") as f:
    doc = json.load(f)

for page in doc["pages"]:
    for block in page["blocks"]:
        if block["block_type"] == "table":
            print(f"Table on page {page['page_num']}")
            if block["table_data"]["has_ruling"]:
                rows = block["table_data"]["num_rows"]
                cols = block["table_data"]["num_cols"]
                print(f"  Grid: {rows}x{cols}")
```

### Filter by Confidence

```python
import json

with open("output.json") as f:
    doc = json.load(f)

high_confidence = []
for page in doc["pages"]:
    for block in page["blocks"]:
        if block["confidence"]["final_confidence"] > 0.8:
            high_confidence.append(block)

print(f"Found {len(high_confidence)} high-confidence blocks")
```

## Running Tests

```bash
# All tests
pytest

# Specific test
pytest tests/test_schema.py -v

# See coverage
pytest --cov=. --cov-report=term-missing
```

## Troubleshooting

### Error: "FileNotFoundError"
- Check PDF path is correct
- Ensure PDF exists

### Error: "ValidationError"
- PDF may be corrupted
- Enable verbose mode: `python main.py --verbose input.pdf output.json`
- Check logs for specific validation failure

### Warning: "No text extracted"
- PDF may be scanned-only (no text layer)
- Try OCR preprocessing first

### Low Confidence Scores
- Expected for stub detector (model_score = 0)
- Replace with real detector for better scores

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Run example.py** to see more usage patterns
3. **Replace detector** with real model for production

