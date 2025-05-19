"""
Common data loading utilities for the Local RAG Pipeline.

This module contains general-purpose functions for dataset manipulation:
- Utilities for accessing nested dictionaries
- Re-exports base data loading functions
"""

import logging
from typing import Dict, Any

from src.data_loader.base import (
    load_hf_dataset,
    load_retrieval_ground_truth
)

# Get logger for this module
logger = logging.getLogger(__name__)


def get_nested_value(data_dict: Dict, field_path: str, default=None) -> Any:
    """
    Get a value from a nested dictionary using a dot-separated path.
    
    Args:
        data_dict: Dictionary to extract value from
        field_path: Dot-separated path (e.g., "output.0.answer")
        default: Value to return if path doesn't exist
        
    Returns:
        Value at the specified path or default if not found
    """
    if not data_dict:
        return default
        
    # Handle array indexing in the path (e.g., "output.0.answer")
    parts = []
    for part in field_path.split('.'):
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(part)
    
    # Traverse the dictionary/list
    current = data_dict
    try:
        for part in parts:
            current = current[part]
        return current
    except (KeyError, IndexError, TypeError):
        return default 