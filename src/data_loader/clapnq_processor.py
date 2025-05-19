"""
ClapNQ dataset processing for the Local RAG Pipeline.

This module contains functions specifically for processing the ClapNQ dataset
from the Hugging Face Hub and its associated qrels files.
"""

import logging
from typing import Dict, List, Tuple, Any

from datasets import Dataset

from src.config import DatasetConfig

# Set up logger for this module
logger = logging.getLogger(__name__)


def process_clapnq_for_rag(
    raw_question_data: Dataset, # From PrimeQA/clapnq
    raw_corpus_data: Dataset,   # From PrimeQA/clapnq_passages
    dataset_config: DatasetConfig,
    qrels_data: Dict[str, Dict[str, int]] # Loaded from TSV files
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process raw ClapNQ data from Hugging Face into standardized formats
    for corpus passages and questions.
    
    Args:
        raw_question_data: Raw Hugging Face Dataset for ClapNQ questions.
        raw_corpus_data: Raw Hugging Face Dataset for ClapNQ passages.
        dataset_config: The DatasetConfig for ClapNQ.
        qrels_data: Ground truth relevance data mapping question_id to {passage_id: score}.
    
    Returns:
        A tuple containing:
        - processed_corpus_passages: List of dicts, each {'passage_id': ..., 'text': ..., 'title': ...}
        - processed_questions: List of dicts, each {'question_id': ..., 'question_text': ..., 
                                                   'gold_answer': ..., 'is_answerable': ..., 
                                                   'gold_passage_ids': [...]}
    """
    logger.info(f"Processing ClapNQ data for RAG...")
    
    # Initialize lists to store processed data
    processed_corpus_passages: List[Dict[str, Any]] = []
    processed_questions: List[Dict[str, Any]] = []
    
    # Get field mappings from config
    field_mappings = dataset_config.field_mappings
    
    # =========================================================================
    # Step 1: Process corpus passages (from PrimeQA/clapnq_passages)
    # =========================================================================
    logger.info(f"Processing corpus passages from {len(raw_corpus_data)} raw entries...")
    
    # Track passage IDs to detect duplicates
    corpus_passage_ids_seen = set()
    
    # Process each passage in the raw corpus data
    for item in raw_corpus_data:
        # Extract passage_id using field mapping
        passage_id = str(item.get(field_mappings.passage_id, ""))
        
        if not passage_id:
            logger.warning(f"Skipping passage with missing ID")
            continue
            
        # Check for duplicate passage IDs
        if passage_id in corpus_passage_ids_seen:
            logger.warning(f"Duplicate passage ID found: {passage_id}. Skipping.")
            continue
            
        # Extract text content using field mapping
        text = ""
        for field in field_mappings.passage_content_fields:
            if field in item:
                text += str(item[field]) + " "
        text = text.strip()
        
        if not text:
            logger.warning(f"Skipping passage {passage_id} with empty text")
            continue
            
        # Extract title using field mapping
        title = ""
        if field_mappings.passage_title_field:
            for field in field_mappings.passage_title_field:
                if field in item:
                    title += str(item[field]) + " "
            title = title.strip()
        
        # Add to processed corpus passages
        processed_corpus_passages.append({
            'passage_id': passage_id,
            'text': text,
            'title': title
        })
        
        # Track processed passage ID
        corpus_passage_ids_seen.add(passage_id)
    
    logger.info(f"Processed {len(processed_corpus_passages)} unique corpus passages")
    
    # =========================================================================
    # Step 2: Process questions (from PrimeQA/clapnq)
    # =========================================================================
    logger.info(f"Processing questions from {len(raw_question_data)} raw entries...")
    
    # Process each question in the raw question data
    for item in raw_question_data:
        # Extract question_id using field mapping
        question_id = str(item.get(field_mappings.question_id, ""))
        
        if not question_id:
            logger.warning(f"Skipping question with missing ID")
            continue
            
        # Extract question_text using field mapping
        question_text = str(item.get(field_mappings.question_text, ""))
        
        if not question_text:
            logger.warning(f"Skipping question {question_id} with empty text")
            continue
            
        # Extract gold_answer and determine if answerable
        gold_answer_text = None
        is_answerable = False
        
        # Handle nested answer field for ClapNQ
        try:
            if item.get("output") and len(item["output"]) > 0 and item["output"][0].get("answer") is not None:
                gold_answer_text = item["output"][0].get("answer", "")
                # Convert None to empty string for consistency
                gold_answer_text = "" if gold_answer_text is None else gold_answer_text
                is_answerable = bool(gold_answer_text and gold_answer_text.strip())
        except (IndexError, KeyError) as e:
            logger.warning(f"Error extracting gold answer for question {question_id}: {e}")
            
        # Determine gold_passage_ids from qrels data
        gold_pids = []
        current_qrels = qrels_data.get(str(question_id), {})
        
        for pid, score in current_qrels.items():
            if score > 0:  # Only relevant passages
                if pid in corpus_passage_ids_seen:
                    gold_pids.append(pid)
                else:
                    logger.warning(f"Gold passage ID {pid} from qrels not found in corpus for question {question_id}")
        
        # Add to processed questions
        processed_questions.append({
            'question_id': question_id,
            'question_text': question_text,
            'gold_answer': gold_answer_text or "",  # Ensure not None
            'is_answerable': is_answerable,
            'gold_passage_ids': gold_pids
        })
        
    logger.info(f"Processed {len(processed_questions)} questions")
    logger.info(f"Complete. Processed {len(processed_corpus_passages)} corpus passages and {len(processed_questions)} questions.")
    
    return processed_corpus_passages, processed_questions 