"""
Logging utilities for DocStruct pipeline.

Provides consistent logging across all modules.
"""

import logging
import sys
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def log_pipeline_stage(logger: logging.Logger, stage: str, page_num: Optional[int] = None) -> None:
    """
    Log pipeline stage entry.
    
    Args:
        logger: Logger instance
        stage: Name of pipeline stage
        page_num: Optional page number being processed
    """
    if page_num is not None:
        logger.info(f"[Stage: {stage}] Processing page {page_num}")
    else:
        logger.info(f"[Stage: {stage}] Starting")