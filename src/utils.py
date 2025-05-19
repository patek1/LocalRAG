"""
Utility functions for the Local RAG Pipeline.

This module contains helper functions used throughout the application,
including logging setup, timing utilities, and other common operations.
"""

import logging
import os
import sys
from typing import Optional

# Set environment variable to suppress HuggingFace tokenizer warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure the root logger with console and optional file output.
    
    Args:
        level: The logging level to set (default: logging.INFO)
        log_file: Optional path to a log file to write to
        
    Returns:
        The configured root logger instance
    """
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Clear any existing handlers to avoid duplicate logs
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Set the global logging level
    root_logger.setLevel(level)
    
    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create a formatter with timestamp, level, and message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add the console handler to the root logger
    root_logger.addHandler(console_handler)
    
    # Optionally add a file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Silence third-party libraries (set them to ERROR level)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("ragas").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("langchain").setLevel(logging.ERROR)
    logging.getLogger("chromadb").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("filelock").setLevel(logging.ERROR)
    logging.getLogger("fsspec").setLevel(logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("evaluate").setLevel(logging.ERROR)
    logging.getLogger("rouge_score").setLevel(logging.ERROR)
    logging.getLogger("tqdm").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("numba").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    # Return the configured logger
    return root_logger 