"""
Base data loading utilities for the Local RAG Pipeline.

This module contains core dataset loading functions:
- Generic Hugging Face dataset loader
- Retrieval ground truth loading from TSV files
"""

import csv
import logging
from typing import Dict, Optional

import datasets
from datasets import Dataset

from src.config import settings

# Get logger for this module
logger = logging.getLogger(__name__)


def load_hf_dataset(
    dataset_identifier: str, 
    dataset_config_name: Optional[str], 
    split: str
) -> Dataset:
    """
    Load a dataset from Hugging Face Hub.
    
    Args:
        dataset_identifier: Name or path of the dataset on Hugging Face Hub.
        dataset_config_name: Specific configuration/subset of the dataset (e.g., "rc" for trivia_qa).
        split: The dataset split to load (e.g., "train", "validation", "dev").
    
    Returns:
        A Hugging Face Dataset object.
    
    Raises:
        FileNotFoundError: If the dataset or split is not found.
        Exception: For other Hugging Face datasets library errors.
    """
    logger.info(
        f"Loading dataset '{dataset_identifier}'"
        f"{f' (config: {dataset_config_name})' if dataset_config_name else ''}"
        f" split '{split}' from Hugging Face Hub..."
    )
    try:
        dataset = datasets.load_dataset(
            dataset_identifier,
            name=dataset_config_name,
            split=split,
            trust_remote_code=True  # As per Spec.md
        )
        logger.info(f"Successfully loaded dataset '{dataset_identifier}' split '{split}'. Rows: {len(dataset)}")
        return dataset
    except FileNotFoundError as e:
        logger.error(
            f"Dataset or split not found: '{dataset_identifier}' ({dataset_config_name=}, {split=}). Error: {e}"
        )
        raise
    except Exception as e:
        logger.error(f"Error loading dataset '{dataset_identifier}' from Hugging Face: {e}")
        raise


def load_retrieval_ground_truth(
    answerable_qrels_filepath: Optional[str] = None,
    unanswerable_qrels_filepath: Optional[str] = None
) -> Dict[str, Dict[str, int]]:
    """
    Load retrieval ground truth (qrels) from TSV files.
    
    Args:
        answerable_qrels_filepath: Path to TSV file with ground truth for answerable questions.
        unanswerable_qrels_filepath: Path to TSV file with ground truth for unanswerable questions.
    
    Returns:
        Dictionary mapping question_id to {passage_id: relevance_score}.
    """
    qrels = {}
    
    # Load answerable qrels if filepath provided
    if answerable_qrels_filepath:
        try:
            logger.info(f"Loading answerable qrels from {answerable_qrels_filepath}")
            with open(answerable_qrels_filepath, 'r', encoding='utf-8') as f:
                tsv_reader = csv.reader(f, delimiter='\t')
                for line in tsv_reader:
                    if len(line) >= 3:
                        q_id = line[0]
                        p_id = line[1]
                        rel = int(line[2])
                        
                        if q_id not in qrels:
                            qrels[q_id] = {}
                        qrels[q_id][p_id] = rel
            logger.info(f"Loaded {len(qrels)} answerable question qrels")
        except (IOError, ValueError) as e:
            logger.error(f"Error loading answerable qrels: {e}")
    
    # Load unanswerable qrels if filepath provided
    if unanswerable_qrels_filepath:
        try:
            logger.info(f"Loading unanswerable qrels from {unanswerable_qrels_filepath}")
            with open(unanswerable_qrels_filepath, 'r', encoding='utf-8') as f:
                tsv_reader = csv.reader(f, delimiter='\t')
                unanswerable_count = 0
                for line in tsv_reader:
                    if len(line) >= 3:
                        q_id = line[0]
                        p_id = line[1]
                        rel = int(line[2])
                        
                        if q_id not in qrels:
                            qrels[q_id] = {}
                            unanswerable_count += 1
                        qrels[q_id][p_id] = rel
            logger.info(f"Loaded {unanswerable_count} unanswerable question qrels")
        except (IOError, ValueError) as e:
            logger.error(f"Error loading unanswerable qrels: {e}")
    
    return qrels 