#!/usr/bin/env python3
"""
Script for processing datasets and creating search indices.

This script handles the loading, processing, and indexing of datasets
for the LocalRAG system. It supports different datasets configured
in src/config.py and creates persistent vector stores.
"""

import argparse
import json
import logging
import os
import sys
import shutil
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from src.config import Settings
from src.utils import setup_logging
from src.data_loader import (
    load_hf_dataset, 
    process_clapnq_for_rag, 
    process_triviaqa_for_rag,
    process_hotpotqa_for_rag,
    load_retrieval_ground_truth
)
from src.embedding import get_embedding_model
from src.vector_store import get_vector_store
from src.indexing import index_corpus

# Set up logging
logger = setup_logging()

def parse_arguments():
    """
    Parse command-line arguments for the create_index script.
    
    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Process and index a dataset for the LocalRAG pipeline."
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name key from settings.dataset_configs (e.g., 'ClapNQ', 'TriviaQA', 'HotpotQA')"
    )
    
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-processing and re-indexing if an index already exists for the dataset"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with limited data samples"
    )
    
    return parser.parse_args()

def save_to_jsonl(data, file_path):
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to the output file
    """
    try:
        logger.info(f"Saving data to {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Successfully saved {len(data)} items to {file_path}")
    except IOError as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise

def main():
    """
    Main execution function for dataset processing and indexing.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Initialize settings
    settings = Settings()
    
    # Get dataset configuration
    try:
        active_dataset_config = settings.get_active_dataset_config(args.dataset)
    except ValueError as e:
        logger.error(f"Error: {e}")
        available_datasets = ", ".join(settings.dataset_configs.keys())
        logger.error(f"Available datasets: {available_datasets}")
        sys.exit(1)
    
    # Log dataset information
    logger.info(f"Processing dataset: {args.dataset}")
    logger.info(f"Dataset config: hf_id={active_dataset_config.hf_id}, "
                f"config={active_dataset_config.hf_config_name}, "
                f"eval_split={active_dataset_config.hf_split_eval}")
    
    # Step 1: Load raw Hugging Face dataset
    # Define corpus and question dataset details based on active configuration
    corpus_hf_id = active_dataset_config.corpus_hf_id or active_dataset_config.hf_id
    corpus_hf_config = active_dataset_config.corpus_hf_config_name or active_dataset_config.hf_config_name
    corpus_hf_split = active_dataset_config.corpus_hf_split or "train"
    
    question_hf_id = active_dataset_config.hf_id
    question_hf_config = active_dataset_config.hf_config_name
    question_hf_split = active_dataset_config.hf_split_eval
    
    # Limit data in debug mode
    if args.debug:
        logger.info("Debug mode: Limiting dataset size")
        corpus_split = f"{corpus_hf_split}[:20]"  # Limit to 20 samples in debug mode
        question_split = f"{question_hf_split}[:10]"  # Limit to 10 samples in debug mode
    else:
        corpus_split = corpus_hf_split
        question_split = question_hf_split
    
    # Load the corpus data
    logger.info(f"Loading corpus data from {corpus_hf_id}...")
    raw_corpus_data = load_hf_dataset(
        corpus_hf_id, 
        corpus_hf_config, 
        corpus_split
    )
    
    # Load the question data
    logger.info(f"Loading question data from {question_hf_id}...")
    raw_question_data = load_hf_dataset(
        question_hf_id, 
        question_hf_config, 
        question_split
    )
    
    # Step 2: Process raw data using dataset-specific adapter
    processed_corpus_passages = []
    processed_questions = []
    
    if args.dataset == "ClapNQ":
        # Load qrels data for ClapNQ
        logger.info("Loading ClapNQ QRELs from original TSV files...")
        clapnq_fm = active_dataset_config.field_mappings
        qrels_data = load_retrieval_ground_truth(
            clapnq_fm.qrels_ans_path, 
            clapnq_fm.qrels_unans_path
        )
        
        logger.info("Processing ClapNQ data using the adapter...")
        processed_corpus_passages, processed_questions = process_clapnq_for_rag(
            raw_question_data, 
            raw_corpus_data, 
            active_dataset_config, 
            qrels_data
        )
    elif args.dataset == "TriviaQA":
        logger.info("Processing TriviaQA data using the adapter...")
        processed_corpus_passages, processed_questions = process_triviaqa_for_rag(
            raw_question_data,
            raw_corpus_data,
            active_dataset_config
        )
    elif args.dataset == "HotpotQA":
        logger.info("Processing HotpotQA data using the adapter...")
        processed_corpus_passages, processed_questions = process_hotpotqa_for_rag(
            raw_question_data,
            raw_corpus_data,
            active_dataset_config
        )
    else:
        logger.error(f"Processing for dataset {args.dataset} not yet implemented.")
        sys.exit(1)
    
    logger.info(f"Processed {len(processed_corpus_passages)} corpus passages and {len(processed_questions)} questions.")
    
    # Step 3: Save processed corpus and questions to files
    logger.info("Saving processed corpus and questions to files...")
    
    # Get file paths for saving
    corpus_file_path = active_dataset_config.get_processed_corpus_file_path(
        Path(settings.base_data_dir), 
        args.dataset
    )
    
    questions_file_path = active_dataset_config.get_processed_questions_file_path(
        Path(settings.base_data_dir), 
        args.dataset
    )
    
    # Save processed corpus and questions
    if processed_corpus_passages:
        save_to_jsonl(processed_corpus_passages, corpus_file_path)
    else:
        logger.warning(f"No corpus passages to save for {args.dataset}")
        
    if processed_questions:
        save_to_jsonl(processed_questions, questions_file_path)
    else:
        logger.warning(f"No questions to save for {args.dataset}")

    # Step 4: Initialize embedding model
    logger.info("Initializing embedding model...")
    embedder = get_embedding_model()

    # Step 5: Initialize vector store
    logger.info("Initializing vector store...")
    vector_store_path = active_dataset_config.get_vector_store_dir(
        Path(settings.base_vector_store_dir), args.dataset
    )
    vector_store_path_str = str(vector_store_path)
    collection_name = f"{args.dataset.lower()}{active_dataset_config.vector_store_collection_name_suffix}"

    # Handle reindexing by deleting existing vector store if needed
    if args.reindex:
        if vector_store_path.exists():
            logger.info(f"Reindex flag set. Deleting existing vector store at {vector_store_path_str}")
            shutil.rmtree(vector_store_path_str)
    
    # Ensure directory exists for vector store
    vector_store_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize vector store
    vector_store = get_vector_store(
        embedding_model=embedder,
        persist_directory=vector_store_path_str,
        collection_name=collection_name
    )
        
    # Step 6: Load processed corpus from file for indexing
    logger.info(f"Loading processed corpus from {corpus_file_path} for indexing...")
    corpus_to_index = []
    with open(corpus_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            corpus_to_index.append(json.loads(line))
    logger.info(f"Loaded {len(corpus_to_index)} passages for indexing.")

    # Step 7: Prepare corpus data for the indexing API by renaming passage_id to id
    logger.info("Preparing corpus data for indexing...")
    corpus_for_indexing_api = [
        {"id": p["passage_id"], "text": p["text"], "title": p["title"]} 
        for p in corpus_to_index
    ]

    # Step 8: Index corpus into vector store
    logger.info("Starting corpus indexing...")
    index_corpus(
        corpus_data=corpus_for_indexing_api, 
        vector_store=vector_store, 
        embedder=embedder
    )
    
    # Optional: Check document count in the vector store after indexing
    try:
        doc_count = vector_store._collection.count()
        logger.info(f"Vector store now contains {doc_count} documents")
    except Exception as e:
        logger.warning(f"Could not check final document count: {e}")
    
    logger.info(f"Processing for dataset {args.dataset} complete.")

if __name__ == "__main__":
    main() 