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
MAX_QUESTIONS_TO_LOG = 2  # Log details for fewer questions

def calculate_retrieval_metrics(
    qrels: Dict[str, Dict[str, int]], 
    results: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Calculate retrieval metrics using ranx library.
    
    Args:
        qrels: Ground truth dictionary mapping question IDs to relevant passage IDs with scores.
               Format: {'question_id': {'passage_id_1': 1, 'passage_id_2': 1, ...}, ...}
        results: Dictionary mapping question IDs to retrieved passage IDs with scores.
               Format: {'question_id': {'passage_id_1': score_1, 'passage_id_2': score_2, ...}, ...}
    
    Returns:
        Dictionary containing calculated metrics (ndcg@10, precision@10, recall@10)
    """
    # Validate inputs
    if not results:
        error_msg = "Empty results dictionary provided to calculate_retrieval_metrics"
        logger.error(error_msg)
        return {'ndcg@10': 0.0, 'precision@10': 0.0, 'recall@10': 0.0}
    
    # Debug input types - only if needed for debugging
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Type of qrels: {type(qrels)}, Type of results: {type(results)}")
        if len(results) > 0:
            sample_qid = next(iter(results))
            logger.debug(f"Sample question ID: {sample_qid}, Type: {type(sample_qid)}")
            logger.debug(f"Sample result value type: {type(results[sample_qid])}")
    
    # Ensure all keys in both dicts are strings
    qrels = {str(k): {str(pk): v for pk, v in pdict.items()} for k, pdict in qrels.items()}
    
    # Count non-empty ground truth entries
    non_empty_qrels = sum(1 for qrel_dict in qrels.values() if qrel_dict)
    logger.info(f"Non-empty ground truth entries: {non_empty_qrels}/{len(qrels)}")
    
    # Prepare the run dictionary for ranx - ensure all keys are strings and values are properly formatted
    run_dict = {}
    missing_question_ids = 0
    missing_retrieved_ids = 0
    questions_processed_for_logging = 0
    
    for qid, retrieved_dict in results.items():
        try:
            qid_str = str(qid)
            
            # Skip if retrieved_dict isn't a dictionary
            if not isinstance(retrieved_dict, dict):
                logger.warning(f"Retrieved value for question {qid} is not a dictionary, it's a {type(retrieved_dict)}. Skipping.")
                missing_retrieved_ids += 1
                continue
            
            # Create a properly formatted inner dict with string keys
            inner_dict = {str(passage_id): float(score) for passage_id, score in retrieved_dict.items() if passage_id is not None}
            
            # Add to run dictionary
            run_dict[qid_str] = inner_dict
            
            # Debug Logging - limited to first few questions only when debugging
            if questions_processed_for_logging < MAX_QUESTIONS_TO_LOG and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"--- Debugging ranx input for QID: {qid_str} ---")
                qrels_entry = qrels.get(qid_str, "QID_NOT_FOUND_IN_QRELS")
                
                # Log the ground truth passage IDs for this question
                if isinstance(qrels_entry, dict):
                    logger.debug(f"Qrels (Ground Truth) Passage IDs for {qid_str}: {list(qrels_entry.keys())}")
                    
                    # Check if this is an empty qrels entry
                    if not qrels_entry:
                        logger.debug(f"  Note: This question has an empty ground truth entry")
                else:
                    logger.debug(f"Could not find Qrels entry for {qid_str}")

                # Log the retrieved passage IDs for this question
                logger.debug(f"Run (Retrieved) Passage IDs for {qid_str}: {list(inner_dict.keys())}")
                
                # Check overlap
                if isinstance(qrels_entry, dict) and qrels_entry:  # Only check if non-empty
                    overlap = set(qrels_entry.keys()).intersection(set(inner_dict.keys()))
                    logger.debug(f"  Overlap with Qrels: {list(overlap)}")
                    if overlap:
                        logger.debug(f"  Number of relevant docs retrieved: {len(overlap)}/{len(qrels_entry)} ({(len(overlap) / len(qrels_entry)) * 100:.1f}%)")
                else:
                    logger.debug("  Overlap with Qrels: Cannot check (Qrels entry missing or empty)")

                logger.debug(f"--- End Debugging for QID: {qid_str} ---")
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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"qrels keys (sample of 5): {list(qrels_keys)[:5]}")
            logger.debug(f"run_dict keys (sample of 5): {list(run_keys)[:5]}")
        return {metric: 0.0 for metric in metrics}

    # Filter out questions with empty ground truth
    qids_with_relevant_docs = {
        qid for qid in common_keys if qrels.get(qid) and len(qrels.get(qid)) > 0
    }
    
    logger.info(f"Found {len(qids_with_relevant_docs)} questions with non-empty ground truth out of {len(common_keys)} common questions.")
    
    if not qids_with_relevant_docs:
        logger.error("No questions with non-empty ground truth found. Cannot calculate metrics.")
        return {metric: 0.0 for metric in metrics}
    
    # Create filtered dictionaries with only questions that have ground truth documents
    filtered_run_dict = {qid: run_dict[qid] for qid in qids_with_relevant_docs}
    filtered_qrels = {qid: qrels[qid] for qid in qids_with_relevant_docs}
    
    logger.info(f"Evaluating on {len(filtered_qrels)} questions with ground truth.")
    
    try:
        # Calculate metrics using ranx
        logger.info(f"Calling ranx.evaluate with {len(filtered_qrels)} qrels entries and {len(filtered_run_dict)} run entries.")
        # Log sample of final input to ranx only when debugging
        if logger.isEnabledFor(logging.DEBUG) and len(filtered_qrels) > 0 and len(filtered_run_dict) > 0:
            sample_qid = list(filtered_qrels.keys())[0]
            logger.debug(f"Sample ranx input - Qrels[{sample_qid}]: {filtered_qrels[sample_qid]}")
            logger.debug(f"Sample ranx input - Run[{sample_qid}]: {filtered_run_dict[sample_qid]}")
        
        scores = ranx.evaluate(filtered_qrels, filtered_run_dict, metrics)
        
        # Log results
        logger.info("Retrieval metrics calculated successfully")
        for metric, value in scores.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return scores
        
    except Exception as e:
        logger.error(f"Error calculating retrieval metrics with ranx: {str(e)}", exc_info=True)
        # Return empty dict with metrics as keys to avoid downstream errors
        return {metric: 0.0 for metric in metrics} 