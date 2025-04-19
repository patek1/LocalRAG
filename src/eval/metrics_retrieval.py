"""
Retrieval metrics evaluation module.

This module provides functions to evaluate retrieval performance
using ranx for standard information retrieval metrics.
"""

import logging
from typing import Dict, List, Any

import ranx


# Set up logger
logger = logging.getLogger(__name__)

# Constants
MAX_QUESTIONS_TO_LOG = 3  # Log details for the first few questions - moved from hardcoded value

def calculate_retrieval_metrics(
    qrels: Dict[str, Dict[str, int]], 
    results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate retrieval metrics using ranx library.
    
    Args:
        qrels: Ground truth dictionary mapping question IDs to relevant passage IDs with scores.
               Format: {'question_id': {'passage_id_1': 1, 'passage_id_2': 1, ...}, ...}
        results: List of result dictionaries from pipeline runs, each containing at least:
                - 'question_id': ID of the question
                - 'retrieved_ids': List of retrieved passage IDs
    
    Returns:
        Dictionary containing calculated metrics (ndcg@10, precision@10, recall@10)
    
    Raises:
        ValueError: If results list is empty or doesn't contain required fields
    """
    # Validate inputs
    if not results:
        error_msg = "Empty results list provided to calculate_retrieval_metrics"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Prepare the run dictionary for ranx
    run_dict = {}
    missing_question_ids = 0
    missing_retrieved_ids = 0
    questions_processed_for_logging = 0  # Counter for logging
    
    for result in results:
        # Get question_id and retrieved_ids, with error handling
        try:
            question_id = result.get('question_id')
            if not question_id:
                logger.warning("Result missing question_id, skipping")
                missing_question_ids += 1
                continue
                
            retrieved_ids = result.get('retrieved_ids')
            if retrieved_ids is None:
                logger.warning(f"Retrieved IDs is None for question {question_id}, treating as empty.")
                retrieved_ids = []  # Treat None as empty list
            
            if not retrieved_ids:  # Handles empty list or None after the above check
                logger.debug(f"No retrieved_ids for question {question_id}, adding empty entry to run_dict")
                missing_retrieved_ids += 1
                inner_dict = {}  # Empty dict for ranx
            else:
                inner_dict = {str(passage_id): 1.0 for passage_id in retrieved_ids if passage_id is not None}  # Score=1.0
            
            # Add to run dictionary
            run_dict[question_id] = inner_dict
            
            # Debug Logging 
            if questions_processed_for_logging < MAX_QUESTIONS_TO_LOG:
                logger.info(f"--- Debugging ranx input for QID: {question_id} ---")
                qrels_entry = qrels.get(question_id, "QID_NOT_FOUND_IN_QRELS")
                run_entry = run_dict.get(question_id, "QID_NOT_FOUND_IN_RUN")  # Use run_dict directly

                # Log the ground truth passage IDs for this question (ensure keys are strings)
                if isinstance(qrels_entry, dict):
                    logger.info(f"Qrels (Ground Truth) Passage IDs for {question_id}: {[str(k) for k in qrels_entry.keys()]}")
                else:
                    logger.warning(f"Could not find Qrels entry for {question_id}")

                # Log the retrieved passage IDs for this question (keys from inner_dict are already strings)
                if isinstance(run_entry, dict):
                    logger.info(f"Run (Retrieved) Passage IDs for {question_id}: {list(run_entry.keys())}")
                    # Check overlap
                    if isinstance(qrels_entry, dict):
                        overlap = set(qrels_entry.keys()).intersection(set(run_entry.keys()))
                        logger.info(f"  Overlap with Qrels: {list(overlap)}")
                    else:
                        logger.info("  Overlap with Qrels: Cannot check (Qrels entry missing)")
                else:
                    logger.warning(f"Could not find Run entry for {question_id} in run_dict")  # Should not happen if logic above is correct

                logger.info(f"--- End Debugging for QID: {question_id} ---")
                questions_processed_for_logging += 1
            
        except Exception as e:
            logger.warning(f"Error processing result for retrieval metrics: {str(e)}")
            continue
    
    # Log preparation results
    logger.info(f"Prepared retrieval run with {len(run_dict)} questions for ranx evaluation.")
    if missing_question_ids > 0:
        logger.warning(f"Skipped {missing_question_ids} results with missing question IDs")
    if missing_retrieved_ids > 0:
        logger.warning(f"Processed {missing_retrieved_ids} questions with no retrieved passages")
    
    # Define metrics to calculate
    metrics = ['ndcg@10', 'precision@10', 'recall@10'] # For k=10

    # Filter for common question IDs between qrels and run_dict
    qrels_keys = set(qrels.keys())
    run_keys = set(run_dict.keys())
    common_keys = qrels_keys.intersection(run_keys)

    if not common_keys:
        logger.error("FATAL: No common question IDs found between qrels and run_dict. Cannot evaluate.")
        return {metric: 0.0 for metric in metrics}

    # IMPORTANT CHANGE: Filter out questions with empty ground truth
    # This is key to fixing the Numba conversion issue in ranx
    qids_with_relevant_docs = {
        qid for qid in common_keys if qrels.get(qid) and len(qrels.get(qid)) > 0
    }
    
    logger.info(f"Found {len(qids_with_relevant_docs)} questions with non-empty ground truth out of {len(common_keys)} common questions.")
    
    if not qids_with_relevant_docs:
        logger.error("No questions with non-empty ground truth found. Cannot calculate metrics.")
        return {metric: 0.0 for metric in metrics}
    
    # Create filtered dictionaries with only questions that have ground truth documents
    filtered_run_dict = {qid: run_dict[qid] for qid in qids_with_relevant_docs}
    filtered_qrels = {
        qid: {str(k): v for k, v in qrels[qid].items()} 
        for qid in qids_with_relevant_docs
    }
    
    logger.info(f"Evaluating on {len(filtered_qrels)} questions with ground truth.")
    
    try:
        # Calculate metrics using ranx
        logger.info(f"Calling ranx.evaluate with {len(filtered_qrels)} qrels entries and {len(filtered_run_dict)} run entries.")
        # Log sample of final input to ranx
        if len(filtered_qrels) > 0 and len(filtered_run_dict) > 0:
            sample_qid = list(filtered_qrels.keys())[0]
            logger.debug(f"Sample ranx input - Qrels[{sample_qid}]: {filtered_qrels[sample_qid]}")
            logger.debug(f"Sample ranx input - Run[{sample_qid}]: {filtered_run_dict[sample_qid]}")
        
        scores = ranx.evaluate(filtered_qrels, filtered_run_dict, metrics)  # Use filtered dicts
        
        # Log results
        logger.info("Retrieval metrics calculated successfully")
        for metric, value in scores.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return scores
        
    except Exception as e:
        logger.error(f"Error calculating retrieval metrics with ranx: {str(e)}", exc_info=True)
        # Return empty dict with metrics as keys to avoid downstream errors
        return {metric: 0.0 for metric in metrics} 