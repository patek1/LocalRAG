"""
Data loading utilities for the Local RAG Pipeline.

This module contains functions for loading and preparing data from various sources:
- ClapNQ passages corpus from Hugging Face
- ClapNQ dev questions (answerable and unanswerable) from JSONL files
- ClapNQ retrieval ground truth from TSV files
"""

import csv
import json
import logging
from typing import Dict, List, Union, Any

import datasets
from datasets import Dataset

from src.config import settings

# Get logger for this module
logger = logging.getLogger(__name__)


def load_corpus_passages() -> Union[Dataset, List[Dict[str, Any]]]:
    """
    Load the ClapNQ passages corpus from Hugging Face.
    
    Uses the dataset name and split specified in the configuration settings.
    
    Returns:
        Dataset or list of dictionaries containing passages with id, text, and title fields
    
    Raises:
        RuntimeError: If the dataset cannot be loaded after retries
    """
    logger.info(f"Loading corpus passages from {settings.corpus_dataset_name}, "
                f"split '{settings.corpus_dataset_split}'...")
    
    try:
        # Load the dataset from Hugging Face
        dataset = datasets.load_dataset(
            settings.corpus_dataset_name,
            split=settings.corpus_dataset_split,
            trust_remote_code=True
        )
        
        # Validate essential fields exist in the dataset
        required_fields = ["id", "text", "title"]
        missing_fields = [field for field in required_fields if field not in dataset.column_names]
        
        if missing_fields:
            error_msg = f"Dataset missing required fields: {missing_fields}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log success
        passage_count = len(dataset)
        logger.info(f"Successfully loaded {passage_count} passages from corpus")
        
        # Return the dataset as-is (keeping it as a Dataset object for efficiency)
        return dataset
        
    except (FileNotFoundError, ConnectionError) as e:
        error_msg = f"Error loading corpus from Hugging Face: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    except Exception as e:
        error_msg = f"Unexpected error loading corpus: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def load_development_questions() -> List[Dict[str, Any]]:
    """
    Load and parse ClapNQ development questions from JSONL files.
    
    Reads both answerable and unanswerable questions from the JSONL files
    specified in the settings. Extracts question_id, question_text, gold_answer,
    and determines is_answerable flag based on the answer being non-empty.
    
    Returns:
        List of dictionaries containing question_id, question_text, gold_answer,
        and is_answerable flag for each question.
    
    Raises:
        RuntimeError: If neither file can be loaded
    """
    questions = []
    files_loaded = 0
    
    # Define file paths from settings
    answerable_path = settings.dev_answerable_jsonl_path
    unanswerable_path = settings.dev_unanswerable_jsonl_path
    
    # Load answerable questions
    try:
        logger.info(f"Loading answerable questions from {answerable_path}")
        with open(answerable_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    
                    # Extract gold answer, defaulting to None if missing
                    gold_answer = None
                    try:
                        if item.get('output') and len(item['output']) > 0:
                            gold_answer = item['output'][0].get('answer', None)
                    except (IndexError, KeyError) as e:
                        logger.warning(f"Error extracting gold answer for question {item.get('id', 'unknown')}: {str(e)}")
                    
                    # Determine if question is answerable (has non-empty answer)
                    is_answerable = gold_answer is not None and gold_answer.strip() != ""
                    
                    # Store question data
                    question_data = {
                        'question_id': item.get('id', ''),
                        'question_text': item.get('input', ''),
                        'gold_answer': gold_answer,
                        'is_answerable': is_answerable
                    }
                    
                    questions.append(question_data)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line in {answerable_path}: {str(e)}")
                    continue
        
        files_loaded += 1
        logger.info(f"Successfully loaded {len(questions)} answerable questions")
        
    except FileNotFoundError:
        logger.error(f"Answerable questions file not found: {answerable_path}")
    
    # Load unanswerable questions
    try:
        logger.info(f"Loading unanswerable questions from {unanswerable_path}")
        answerable_count = len(questions)  # Track current count to calculate unanswerable count later
        
        with open(unanswerable_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    
                    # Extract gold answer (should be null or empty for unanswerable questions)
                    gold_answer = None
                    try:
                        if item.get('output') and len(item['output']) > 0:
                            gold_answer = item['output'][0].get('answer', None)
                    except (IndexError, KeyError):
                        # This is expected for unanswerable questions, so just use None
                        pass
                    
                    # For explicit clarity, force is_answerable to False for unanswerable file
                    is_answerable = False
                    
                    # Store question data
                    question_data = {
                        'question_id': item.get('id', ''),
                        'question_text': item.get('input', ''),
                        'gold_answer': gold_answer,
                        'is_answerable': is_answerable
                    }
                    
                    questions.append(question_data)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line in {unanswerable_path}: {str(e)}")
                    continue
        
        files_loaded += 1
        unanswerable_count = len(questions) - answerable_count
        logger.info(f"Successfully loaded {unanswerable_count} unanswerable questions")
        
    except FileNotFoundError:
        logger.error(f"Unanswerable questions file not found: {unanswerable_path}")
    
    # Check if any files were loaded
    if files_loaded == 0:
        error_msg = "No question files could be loaded. Check file paths and permissions."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Log total
    logger.info(f"Total questions loaded: {len(questions)} ({files_loaded}/2 files processed)")
    
    return questions


def load_retrieval_ground_truth() -> Dict[str, Dict[str, int]]:
    """
    Load retrieval ground truth data from TSV files for ranx evaluation.
    
    Reads both answerable and unanswerable ground truth TSV files.
    Extracts question IDs and their corresponding relevant passage IDs.
    Formats the data as a qrels dictionary for use with ranx.
    
    Returns:
        Dictionary mapping question IDs to inner dictionaries, where inner dictionaries 
        map passage IDs to relevance scores (always 1 for binary relevance).
        Format: {'question_id': {'passage_id_1': 1, 'passage_id_2': 1, ...}, ...}
    
    Raises:
        RuntimeError: If neither ground truth file can be loaded
    """
    qrels = {}
    files_loaded = 0
    total_questions = 0
    
    # Define file paths to process
    tsv_files = [
        settings.retrieval_answerable_tsv_path,
        settings.retrieval_unanswerable_tsv_path
    ]
    
    # Process each TSV file
    for file_path in tsv_files:
        try:
            logger.info(f"Loading retrieval ground truth from {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                # Create TSV reader
                tsv_reader = csv.reader(f, delimiter='\t')
                
                # Skip header row
                next(tsv_reader, None)
                
                # Initialize counter for this file
                file_questions = 0
                
                # Process each data row
                for row in tsv_reader:
                    try:
                        # Ensure row has enough columns
                        if len(row) < 3:
                            logger.warning(f"Skipping row with insufficient columns: {row}")
                            continue
                        
                        # Extract data
                        question_id = row[0].strip()
                        doc_id_list_str = row[2].strip()
                        
                        # Skip if missing question ID
                        if not question_id:
                            logger.warning("Skipping row with empty question ID")
                            continue
                        
                        # Skip if missing document IDs
                        if not doc_id_list_str:
                            logger.warning(f"No relevant passages for question {question_id}, adding empty entry")
                            qrels[question_id] = {}
                            file_questions += 1
                            continue
                        
                        # Split document ID string into a list
                        passage_ids = doc_id_list_str.split()
                        
                        # Create inner dictionary with binary relevance (score=1)
                        relevant_passages = {passage_id: 1 for passage_id in passage_ids if passage_id}
                        
                        # Check for duplicate question ID
                        if question_id in qrels:
                            logger.warning(f"Duplicate question ID found: {question_id}. Overwriting.")
                        
                        # Add to qrels dictionary
                        qrels[question_id] = relevant_passages
                        file_questions += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing row in {file_path}: {str(e)}")
                        continue
                
                # Log success for this file
                logger.info(f"Loaded {file_questions} questions with ground truth from {file_path}")
                total_questions += file_questions
                files_loaded += 1
                
        except FileNotFoundError:
            logger.error(f"Ground truth file not found: {file_path}")
    
    # Check if any files were loaded
    if files_loaded == 0:
        error_msg = "No ground truth files could be loaded. Check file paths and permissions."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Log total
    logger.info(f"Total questions with ground truth: {len(qrels)} from {files_loaded}/2 files")
    
    return qrels 