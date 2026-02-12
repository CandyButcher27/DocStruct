#!/usr/bin/env python3
"""
Example usage of DocStruct pipeline.

This script demonstrates how to use DocStruct programmatically
to extract structured content from PDFs.
"""

import json
from pathlib import Path
from main import process_pdf


def example_basic_usage():
    """Basic example: process a PDF and save JSON."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Define paths
    input_pdf = "example_input.pdf"
    output_json = "example_output.json"
    
    # Process PDF
    try:
        process_pdf(input_pdf, output_json, detector_type="stub")
        print(f"✓ Processed {input_pdf}")
        print(f"✓ Output saved to {output_json}")
    except FileNotFoundError:
        print(f"✗ File not found: {input_pdf}")
        print("  Create a PDF file named 'example_input.pdf' to test")


def example_analyze_output():
    """Example: analyze the output JSON."""
    print("\n" + "=" * 60)
    print("Example 2: Analyzing Output")
    print("=" * 60)
    
    output_file = "example_output.json"
    
    if not Path(output_file).exists():
        print(f"✗ Output file not found: {output_file}")
        print("  Run example_basic_usage() first")
        return
    
    # Load output
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    # Print summary
    print(f"\nDocument: {data['metadata']['filename']}")
    print(f"Pages: {data['metadata']['num_pages']}")
    
    # Count block types
    block_counts = {}
    all_blocks = []
    
    for page in data['pages']:
        for block in page['blocks']:
            block_type = block['block_type']
            block_counts[block_type] = block_counts.get(block_type, 0) + 1
            all_blocks.append(block)
    
    print(f"\nTotal blocks: {len(all_blocks)}")
    print("\nBlock distribution:")
    for block_type, count in sorted(block_counts.items()):
        print(f"  {block_type:10s}: {count:3d}")
    
    # Show high-confidence blocks
    print("\nHigh-confidence blocks (>0.8):")
    high_conf = [b for b in all_blocks if b['confidence']['final_confidence'] > 0.8]
    for block in high_conf[:5]:  # Show first 5
        conf = block['confidence']['final_confidence']
        text = block.get('text', '[image]')[:50]
        print(f"  [{conf:.2f}] {block['block_type']:10s}: {text}...")


def example_filter_tables():
    """Example: extract only tables from output."""
    print("\n" + "=" * 60)
    print("Example 3: Filtering Tables")
    print("=" * 60)
    
    output_file = "example_output.json"
    
    if not Path(output_file).exists():
        print(f"✗ Output file not found: {output_file}")
        return
    
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    # Find all tables
    tables = []
    for page in data['pages']:
        for block in page['blocks']:
            if block['block_type'] == 'table':
                tables.append({
                    'page': page['page_num'],
                    'bbox': block['bbox'],
                    'table_data': block.get('table_data', {}),
                    'confidence': block['confidence']['final_confidence']
                })
    
    print(f"\nFound {len(tables)} tables")
    
    for i, table in enumerate(tables):
        print(f"\nTable {i + 1}:")
        print(f"  Page: {table['page']}")
        print(f"  Confidence: {table['confidence']:.2f}")
        
        table_data = table['table_data']
        if table_data.get('has_ruling'):
            print(f"  Grid: {table_data['num_rows']}x{table_data['num_cols']}")
        else:
            print(f"  Grid: unruled (structure unknown)")


def main():
    """Run all examples."""
    print("\nDocStruct Examples")
    print("==================\n")
    
    # Example 1: Basic usage
    example_basic_usage()
    
    # Example 2: Analyze output
    example_analyze_output()
    
    # Example 3: Filter tables
    example_filter_tables()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()