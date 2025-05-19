"""
TriviaQA dataset processing for the Local RAG Pipeline.

This module contains functions specifically for processing the TriviaQA dataset
from the Hugging Face Hub.
"""

import hashlib
import logging
from typing import Dict, List, Tuple, Any

from datasets import Dataset

from src.config import DatasetConfig

# Set up logger for this module
logger = logging.getLogger(__name__)


def process_triviaqa_for_rag(
    raw_question_data: Dataset, # From trivia_qa (validation split)
    raw_corpus_data: Dataset,   # From trivia_qa (train split)
    dataset_config: DatasetConfig
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process raw TriviaQA data from Hugging Face into standardized formats
    for corpus passages and questions.
    
    For TriviaQA dataset, we:
    1. Extract unique passages from both entity_pages.wiki_context and search_results.search_context
    2. Assign unique passage IDs based on content hashing
    3. For each question, identify which passages were originally associated with it
       These become gold_passage_ids for retrieval evaluation
    
    Args:
        raw_question_data: Raw Hugging Face Dataset for TriviaQA questions.
        raw_corpus_data: Raw Hugging Face Dataset for TriviaQA corpus (usually train split).
        dataset_config: The DatasetConfig for TriviaQA.
    
    Returns:
        A tuple containing:
        - processed_corpus_passages: List of dicts, each {'passage_id': ..., 'text': ..., 'title': ...}
        - processed_questions: List of dicts, each {'question_id': ..., 'question_text': ..., 
                                                   'gold_answer': ..., 'is_answerable': ..., 
                                                   'gold_passage_ids': [...]}
    """
    logger.info(f"Processing TriviaQA data for RAG...")
    
    # Initialize lists to store processed data
    processed_corpus_passages: List[Dict[str, Any]] = []
    processed_questions: List[Dict[str, Any]] = []
    
    # Get field mappings from config
    field_mappings = dataset_config.field_mappings
    
    # Map to store passage content to ID for deduplication
    passage_content_to_id: Dict[str, str] = {}
    passage_id_to_idx: Dict[str, int] = {}  # Maps passage_id to its index in processed_corpus_passages
    
    # =========================================================================
    # Step 1: Process corpus to extract unique passages 
    # =========================================================================
    logger.info(f"Processing corpus passages from {len(raw_corpus_data)} raw entries...")
    
    passage_count = 0
    wiki_passage_count = 0
    search_passage_count = 0
    
    # Function to create a passage ID from content
    def get_passage_id(content: str, source_type: str, idx: int) -> str:
        """Create a deterministic passage ID based on content hash."""
        if not content:
            return f"triviaqa_{source_type}_{idx}"
            
        # Create a hash of the content for deduplication
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:10]
        return f"triviaqa_{source_type}_{content_hash}"
    
    # Function to extract and process passages from a single item
    def extract_passages_from_item(item: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """
        Extract passages from entity_pages and search_results.
        Returns a list of tuples (passage_id, text, title).
        """
        passages = []
        
        # Process entity_pages (Wikipedia articles)
        if 'entity_pages' in item and 'wiki_context' in item['entity_pages']:
            wiki_contexts = item['entity_pages']['wiki_context']
            wiki_titles = item['entity_pages'].get('title', [])
            
            # Handle case where wiki_context is a list
            if isinstance(wiki_contexts, list):
                for i, context in enumerate(wiki_contexts):
                    if context:  # Skip empty contexts
                        title = wiki_titles[i] if i < len(wiki_titles) else ""
                        pid = get_passage_id(context, "wiki", i)
                        passages.append((pid, context, title))
            # Handle case where wiki_context is a string
            elif isinstance(wiki_contexts, str) and wiki_contexts:
                title = wiki_titles[0] if wiki_titles else ""
                pid = get_passage_id(wiki_contexts, "wiki", 0)
                passages.append((pid, wiki_contexts, title))
        
        # Process search_results
        if 'search_results' in item and 'search_context' in item['search_results']:
            search_contexts = item['search_results']['search_context']
            search_titles = item['search_results'].get('title', [])
            
            for i, context in enumerate(search_contexts):
                if context:  # Skip empty contexts
                    title = search_titles[i] if i < len(search_titles) else ""
                    pid = get_passage_id(context, "search", i)
                    passages.append((pid, context, title))
        
        return passages
    
    # Process the corpus data to extract all unique passages
    for idx, item in enumerate(raw_corpus_data):
        extracted_passages = extract_passages_from_item(item)
        
        for pid, text, title in extracted_passages:
            # Skip if we've seen this exact text before
            if text in passage_content_to_id:
                continue
                
            passage_content_to_id[text] = pid
            passage_id_to_idx[pid] = len(processed_corpus_passages)
            
            processed_corpus_passages.append({
                'passage_id': pid,
                'text': text,
                'title': title
            })
            
            passage_count += 1
            if "wiki" in pid:
                wiki_passage_count += 1
            elif "search" in pid:
                search_passage_count += 1
                
            # Log progress periodically
            if passage_count % 1000 == 0:
                logger.info(f"Processed {passage_count} unique passages so far...")
    
    logger.info(f"Processed {passage_count} unique corpus passages")
    logger.info(f"  - Wiki passages: {wiki_passage_count}")
    logger.info(f"  - Search passages: {search_passage_count}")
    
    # =========================================================================
    # Step 2: Process questions and associate them with gold passages
    # =========================================================================
    logger.info(f"Processing questions from {len(raw_question_data)} raw entries...")
    
    questions_processed = 0
    questions_with_gold_passages = 0
    
    for idx, item in enumerate(raw_question_data):
        # Extract question ID and text
        question_id = str(item.get(field_mappings.question_id, ""))
        if not question_id:
            logger.warning(f"Skipping question with missing ID at index {idx}")
            continue
            
        question_text = str(item.get(field_mappings.question_text, ""))
        if not question_text:
            logger.warning(f"Skipping question {question_id} with empty text")
            continue
        
        # Extract answer - handles nested fields with dot notation
        answer_field = field_mappings.answer_text
        if '.' in answer_field:
            parts = answer_field.split('.')
            answer_obj = item
            for part in parts:
                if part in answer_obj:
                    answer_obj = answer_obj[part]
                else:
                    answer_obj = None
                    break
            gold_answer = str(answer_obj) if answer_obj is not None else ""
        else:
            gold_answer = str(item.get(answer_field, ""))
        
        # Clean up none answer
        gold_answer = "" if gold_answer is None else gold_answer.strip()
        
        # For TriviaQA, all questions with a non-empty answer are considered answerable
        is_answerable = bool(gold_answer)
        
        # Find the gold passages for this question
        # The gold passages are those that were originally part of this question's context
        gold_passage_ids = []
        extracted_passages = extract_passages_from_item(item)
        
        for pid, text, _ in extracted_passages:
            # Check if this passage exists in our processed corpus
            if text in passage_content_to_id:
                gold_passage_ids.append(passage_content_to_id[text])
            else:
                # This is a new passage not seen in the corpus building phase
                # We'll add it to the corpus
                new_pid = get_passage_id(text, "question", len(processed_corpus_passages))
                processed_corpus_passages.append({
                    'passage_id': new_pid,
                    'text': text,
                    'title': ""  # No title available for this new passage
                })
                passage_content_to_id[text] = new_pid
                passage_id_to_idx[new_pid] = len(processed_corpus_passages) - 1
                gold_passage_ids.append(new_pid)
        
        # Create the processed question entry
        processed_questions.append({
            'question_id': question_id,
            'question_text': question_text,
            'gold_answer': gold_answer,
            'is_answerable': is_answerable,
            'gold_passage_ids': gold_passage_ids
        })
        
        questions_processed += 1
        if gold_passage_ids:
            questions_with_gold_passages += 1
            
        # Log progress periodically
        if questions_processed % 500 == 0:
            logger.info(f"Processed {questions_processed} questions so far...")
    
    logger.info(f"Processed {questions_processed} questions")
    logger.info(f"Questions with gold passage IDs: {questions_with_gold_passages}")
    logger.info(f"Answerable questions: {sum(1 for q in processed_questions if q['is_answerable'])}")
    
    return processed_corpus_passages, processed_questions 