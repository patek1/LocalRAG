"""
Data loading utilities for the Local RAG Pipeline.

This module is deprecated. Please import directly from the data_loader package instead.

Example:
    from src.data_loader import process_clapnq_for_rag, load_hf_dataset
"""

import logging
import warnings

# Re-export data loader functions for backward compatibility
from src.data_loader.utils import get_nested_value
from src.data_loader.base import (
    load_hf_dataset, 
    load_retrieval_ground_truth
)
from src.data_loader.clapnq_processor import process_clapnq_for_rag
from src.data_loader.triviaqa_processor import process_triviaqa_for_rag
from src.data_loader.hotpotqa_processor import process_hotpotqa_for_rag

# Set up logger for deprecation warnings
logger = logging.getLogger(__name__)

# Issue a deprecation warning
warnings.warn(
    "The src.data_loader module is deprecated. Please import directly from the "
    "src.data_loader package instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'load_hf_dataset',
    'load_retrieval_ground_truth',
    'get_nested_value',
    'process_clapnq_for_rag',
    'process_triviaqa_for_rag',
    'process_hotpotqa_for_rag'
] 