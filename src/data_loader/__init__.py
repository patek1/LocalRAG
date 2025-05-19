"""
Data loading utilities for the Local RAG Pipeline.

This module contains functions for loading and preparing data from various sources:
- Generic Hugging Face dataset loader
- Dataset-specific processing adapters for ClapNQ, TriviaQA, and HotpotQA
- Retrieval ground truth loading from TSV files
"""

import logging

# Set up logger for the package
logger = logging.getLogger(__name__)
logger.info("Initializing data_loader package...")

# Import common utilities
from src.data_loader.utils import get_nested_value

# Import base dataset loading functions
from src.data_loader.base import (
    load_hf_dataset,
    load_retrieval_ground_truth
)

# Import dataset processors
from src.data_loader.clapnq_processor import process_clapnq_for_rag
from src.data_loader.triviaqa_processor import process_triviaqa_for_rag
from src.data_loader.hotpotqa_processor import process_hotpotqa_for_rag

# For backward compatibility with existing imports
__all__ = [
    'load_hf_dataset',
    'load_retrieval_ground_truth',
    'get_nested_value',
    'process_clapnq_for_rag',
    'process_triviaqa_for_rag',
    'process_hotpotqa_for_rag'
] 