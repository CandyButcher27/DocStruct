#!/usr/bin/env python3
"""
DocStruct - Layout-first, deterministic PDF understanding pipeline.

Main entry point for CLI usage.
"""

import sys
import json
import argparse
from pathlib import Path

from pipeline.decomposition import decompose_pdf
from pipeline.layout import process_page_layout
from pipeline.classification import classify_blocks
from pipeline.reading_order import assign_reading_order
from pipeline.tables_figures import refine_blocks
from pipeline.confidence import ensure_confidence_scores
from pipeline.validator import validate_and_build_document
from models.detector import create_detector
from utils.logging import setup_logger, log_pipeline_stage


logger = setup_logger(__name__)


def process_pdf(pdf_path: str, output_path: str, detector_type: str = "stub") -> None:
    """
    Process a PDF through the complete DocStruct pipeline.
    
    Pipeline stages:
    1. Decomposition: Extract raw text spans, images, lines
    2. Layout Formation: Merge spans into blocks
    3. Classification: Classify blocks using hybrid approach
    4. Reading Order: Sort blocks and attach captions
    5. Refinement: Extract table/figure structure
    6. Confidence: Validate confidence scores
    7. Schema Validation: Build and validate final Document
    
    Args:
        pdf_path: Path to input PDF
        output_path: Path to output JSON
        detector_type: Type of detector to use ("stub" or "tablenet")
    """
    logger.info(f"Processing PDF: {pdf_path}")
    logger.info(f"Output will be written to: {output_path}")
    
    # Create detector
    detector = create_detector(detector_type)
    logger.info(f"Using detector: {detector.get_model_name()}")
    
    # Stage 1: Decomposition
    log_pipeline_stage(logger, "Decomposition")
    pages_data = decompose_pdf(pdf_path)
    
    # Process each page through remaining stages
    all_pages_results = []
    
    for page_data in pages_data:
        page_num = page_data.page_num
        logger.info(f"Processing page {page_num}")
        
        # Stage 2: Layout Formation
        log_pipeline_stage(logger, "Layout Formation", page_num)
        layout_blocks, image_blocks = process_page_layout(page_data)
        
        # Stage 3: Classification
        log_pipeline_stage(logger, "Classification", page_num)
        # Get detections from model (empty for stub)
        detections = detector.detect(
            b"",  # Would pass page image in production
            page_data.width,
            page_data.height
        )
        
        classified_blocks = classify_blocks(
            layout_blocks,
            image_blocks,
            detections,
            page_data.width,
            page_data.height
        )
        
        # Stage 4: Reading Order
        log_pipeline_stage(logger, "Reading Order", page_num)
        ordered_blocks = assign_reading_order(classified_blocks, page_data.width)
        
        # Stage 5: Refinement
        log_pipeline_stage(logger, "Table/Figure Refinement", page_num)
        refined_blocks = refine_blocks(ordered_blocks, page_data)
        
        # Stage 6: Confidence Validation
        log_pipeline_stage(logger, "Confidence Validation", page_num)
        validated_blocks = ensure_confidence_scores(refined_blocks)
        
        all_pages_results.append((page_data, validated_blocks))
    
    # Stage 7: Schema Validation
    log_pipeline_stage(logger, "Schema Validation")
    document = validate_and_build_document(
        all_pages_results,
        Path(pdf_path).name
    )
    
    # Write output
    logger.info(f"Writing output to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(document.to_dict(), f, indent=2, ensure_ascii=False)
    
    logger.info("Pipeline complete!")
    
    # Print summary
    total_blocks = len(document.get_all_blocks())
    block_types = {}
    for block in document.get_all_blocks():
        block_types[block.block_type] = block_types.get(block.block_type, 0) + 1
    
    logger.info(f"Summary: {total_blocks} blocks extracted")
    for block_type, count in sorted(block_types.items()):
        logger.info(f"  {block_type}: {count}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DocStruct - Layout-first PDF understanding pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input.pdf output.json
  python main.py --detector stub document.pdf result.json
        """
    )
    
    parser.add_argument(
        "input_pdf",
        type=str,
        help="Path to input PDF file"
    )
    
    parser.add_argument(
        "output_json",
        type=str,
        help="Path to output JSON file"
    )
    
    parser.add_argument(
        "--detector",
        type=str,
        default="stub",
        choices=["stub", "tablenet"],
        help="Detector type to use (default: stub)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)
    
    # Validate input file exists
    if not Path(args.input_pdf).exists():
        logger.error(f"Input file not found: {args.input_pdf}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        process_pdf(args.input_pdf, args.output_json, args.detector)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()