"""
HotpotQA dataset processing for the Local RAG Pipeline.

This module contains functions specifically for processing the HotpotQA dataset
from the Hugging Face Hub.
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any

from datasets import Dataset

from src.config import DatasetConfig
from src.data_loader.utils import get_nested_value

# Set up logger for this module
logger = logging.getLogger(__name__)

# Define constants
TEMP_PASSAGE_DIR_NAME = "temp_hotpotqa_intermediate_passages"
CHUNK_SIZE_ITEMS = 5000  # Number of source items to process before writing a chunk file
LOG_INTERVAL_ITEMS = 1000  # How often to log progress during processing


def _ensure_cleanup_temp_dir(temp_dir_path: Path):
    """Clean up temporary directory if it exists."""
    if temp_dir_path.exists():
        logger.info(f"Cleaning up temporary directory: {temp_dir_path}")
        shutil.rmtree(temp_dir_path)


def process_hotpotqa_for_rag(
    raw_question_dataset_eval: Dataset,  # e.g., validation split
    raw_corpus_source_dataset: Dataset,  # e.g., train split
    dataset_config: DatasetConfig
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Processes raw HotpotQA data using a memory-efficient chunked approach.
    
    Builds a global corpus from paragraphs in 'context' of 'raw_corpus_source_dataset'.
    Processes questions from 'raw_question_dataset_eval', linking 'supporting_facts'
    to passage_ids in the global corpus.
    
    This implementation uses a three-phase approach:
    1. Extract passages in chunks and write to temporary files
    2. Deduplicate passages globally
    3. Process evaluation questions and link to the deduplicated corpus
    
    Args:
        raw_question_dataset_eval: Raw Hugging Face Dataset for HotpotQA questions (validation split).
        raw_corpus_source_dataset: Raw Hugging Face Dataset for building corpus (train split).
        dataset_config: The DatasetConfig for HotpotQA.
        
    Returns:
        A tuple containing:
        - processed_corpus_passages: List of dicts, each {'passage_id': ..., 'text': ..., 'title': ...}
        - processed_questions: List of dicts, each {'question_id': ..., 'question_text': ..., 
                                                   'gold_answer': ..., 'is_answerable': ..., 
                                                   'gold_passage_ids': [...]}
    """
    logger.info(f"Processing HotpotQA data (eval: {len(raw_question_dataset_eval)} items, corpus_source: {len(raw_corpus_source_dataset)} items)...")
    fm = dataset_config.field_mappings
    
    # Create a path for temporary files
    temp_dir_base = Path(".")  # Current dir, or choose a better location
    temp_passages_path = temp_dir_base / TEMP_PASSAGE_DIR_NAME
    
    # Clean up any existing temp directory
    _ensure_cleanup_temp_dir(temp_passages_path)
    temp_passages_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using temporary directory for intermediate passages: {temp_passages_path}")

    # Initialize variables for Phase 1
    intermediate_passage_files = []
    passage_counter_temp = 0  # For preliminary IDs in chunks

    # == PHASE 1: Chunked Passage Extraction ==
    logger.info(f"PHASE 1: Extracting passages from {len(raw_corpus_source_dataset)} HotpotQA source items into chunks...")
    current_chunk_passages = []

    for item_idx, item in enumerate(raw_corpus_source_dataset):
        if (item_idx + 1) % LOG_INTERVAL_ITEMS == 0:
            logger.info(f"  PHASE 1: Processing source item {item_idx + 1}/{len(raw_corpus_source_dataset)}")

        # Get context structure from the item
        context = item.get('context', {})
        titles = context.get('title', [])
        sentence_sets = context.get('sentences', [])
        
        if len(titles) != len(sentence_sets):
            logger.warning(f"  PHASE 1: Mismatch between titles and sentences for item {item_idx}. Skipping.")
            continue
            
        # Process each document in the context
        for doc_idx, (doc_title, sentences) in enumerate(zip(titles, sentence_sets)):
            # Join all sentences to form the document text
            passage_text = " ".join(sentences).strip()
            if not passage_text:
                continue
                
            # Create a content hash for later deduplication
            content_hash = hashlib.md5(passage_text.encode('utf-8')).hexdigest()
            
            # Create a preliminary passage ID
            passage_counter_temp += 1
            preliminary_passage_id = f"temp_hpqa_p_{passage_counter_temp}"
            
            # Add passage to current chunk
            current_chunk_passages.append({
                'prelim_id': preliminary_passage_id,
                'text': passage_text,
                'title': str(doc_title),
                'content_hash': content_hash
            })

        # Write chunk to file if we've reached CHUNK_SIZE_ITEMS or we're at the end
        if (item_idx + 1) % CHUNK_SIZE_ITEMS == 0 or (item_idx + 1) == len(raw_corpus_source_dataset):
            if current_chunk_passages:
                chunk_file_num = (item_idx // CHUNK_SIZE_ITEMS) + 1
                chunk_file_path = temp_passages_path / f"chunk_{chunk_file_num}.jsonl"
                with open(chunk_file_path, 'w', encoding='utf-8') as cf:
                    for p_dict in current_chunk_passages:
                        cf.write(json.dumps(p_dict) + '\n')
                intermediate_passage_files.append(chunk_file_path)
                logger.info(f"  PHASE 1: Wrote {len(current_chunk_passages)} passages to {chunk_file_path}")
                current_chunk_passages = []  # Reset for next chunk
    
    logger.info(f"PHASE 1: Completed. Extracted passages into {len(intermediate_passage_files)} chunk files.")

    # == PHASE 2: Global Deduplication ==
    logger.info("PHASE 2: Performing global deduplication of passages...")
    processed_corpus_passages = []
    passage_content_hash_to_id_map = {}
    doc_title_to_passage_id = {}  # Map to help link supporting facts to passages
    final_passage_counter = 0
    
    for chunk_idx, chunk_file in enumerate(intermediate_passage_files):
        logger.info(f"  PHASE 2: Deduplicating passages from {chunk_file} ({chunk_idx+1}/{len(intermediate_passage_files)})...")
        with open(chunk_file, 'r', encoding='utf-8') as cf:
            for line_idx, line in enumerate(cf):
                try:
                    passage_dict = json.loads(line)
                    content_hash = passage_dict['content_hash']
                    
                    if content_hash not in passage_content_hash_to_id_map:
                        final_passage_counter += 1
                        global_passage_id = f"hotpotqa_passage_{final_passage_counter}"
                        passage_content_hash_to_id_map[content_hash] = global_passage_id
                        processed_corpus_passages.append({
                            'passage_id': global_passage_id,
                            'text': passage_dict['text'],
                            'title': passage_dict['title']
                        })
                        
                        # Update title to passage mapping
                        doc_title_to_passage_id[passage_dict['title']] = global_passage_id
                        
                except json.JSONDecodeError:
                    logger.warning(f"  PHASE 2: Skipping malformed JSON line {line_idx+1} in {chunk_file}")
                    continue
                
    logger.info(f"PHASE 2: Completed. Built global corpus with {len(processed_corpus_passages)} unique passages.")
    logger.info(f"Mapped {len(doc_title_to_passage_id)} document titles to passage IDs.")
    
    # Clean up temporary files
    _ensure_cleanup_temp_dir(temp_passages_path)

    # == PHASE 3: Process Evaluation Questions ==
    logger.info(f"PHASE 3: Processing {len(raw_question_dataset_eval)} HotpotQA evaluation questions...")
    processed_questions = []
    questions_with_gold_passages = 0
    
    for item_idx, item in enumerate(raw_question_dataset_eval):
        # Extract question data
        question_id = str(get_nested_value(item, fm.question_id, f"hotpotqa_q_{item_idx}"))
        question_text = str(get_nested_value(item, fm.question_text, ""))
        answer = str(get_nested_value(item, fm.answer_text, ""))
        
        if not question_text:
            logger.warning(f"Skipping question {question_id} with empty text")
            continue
            
        # Process supporting facts to find gold passages
        gold_passage_ids = set()
        supporting_facts = item.get('supporting_facts', {})
        sf_titles = supporting_facts.get('title', [])
        
        # First, ensure all context docs from this question are in the corpus
        # This is necessary because supporting facts refer to titles in this question's context
        context = item.get('context', {})
        eval_titles = context.get('title', [])
        eval_sentence_sets = context.get('sentences', [])
        
        if len(eval_titles) != len(eval_sentence_sets):
            logger.warning(f"Mismatch between titles and sentences for eval item {question_id}. Skipping.")
            continue
            
        # Build a local mapping for this question's docs
        question_doc_title_to_passage_id = {}
        
        # Process each document in this question's context
        for doc_idx, (doc_title, sentences) in enumerate(zip(eval_titles, eval_sentence_sets)):
            passage_text = " ".join(sentences).strip()
            if not passage_text:
                continue
                
            content_hash = hashlib.md5(passage_text.encode('utf-8')).hexdigest()
            
            # Check if we've seen this content before in the global corpus
            if content_hash in passage_content_hash_to_id_map:
                passage_id = passage_content_hash_to_id_map[content_hash]
            else:
                # This is a new passage not in the corpus
                final_passage_counter += 1
                passage_id = f"hotpotqa_passage_{final_passage_counter}"
                passage_content_hash_to_id_map[content_hash] = passage_id
                
                # Add to corpus
                processed_corpus_passages.append({
                    'passage_id': passage_id,
                    'text': passage_text,
                    'title': str(doc_title)
                })
                logger.debug(f"Added new passage {passage_id} from eval question {question_id}")
                
            # Update local mapping
            question_doc_title_to_passage_id[str(doc_title)] = passage_id
            
            # Also update global mapping if needed
            if doc_title not in doc_title_to_passage_id:
                doc_title_to_passage_id[doc_title] = passage_id
        
        # Now identify gold passages from supporting facts
        for sf_title in sf_titles:
            if sf_title in question_doc_title_to_passage_id:
                gold_passage_ids.add(question_doc_title_to_passage_id[sf_title])
            elif sf_title in doc_title_to_passage_id:
                gold_passage_ids.add(doc_title_to_passage_id[sf_title])
            else:
                logger.warning(f"Supporting fact title '{sf_title}' not found in mappings for question {question_id}")
        
        # Create processed question
        processed_question = {
            'question_id': question_id,
            'question_text': question_text,
            'gold_answer': answer,
            'is_answerable': True,  # HotpotQA questions are generally answerable
            'gold_passage_ids': sorted(list(gold_passage_ids))  # Convert set to sorted list
        }
        
        processed_questions.append(processed_question)
        
        if gold_passage_ids:
            questions_with_gold_passages += 1
            
        # Log progress
        if (item_idx + 1) % LOG_INTERVAL_ITEMS == 0:
            logger.info(f"  PHASE 3: Processed {item_idx + 1} questions...")
    
    logger.info(f"PHASE 3: Processed {len(processed_questions)} HotpotQA questions")
    logger.info(f"Questions with gold passage IDs: {questions_with_gold_passages}")
    logger.info(f"Final corpus size: {len(processed_corpus_passages)} passages")
    
    return processed_corpus_passages, processed_questions 