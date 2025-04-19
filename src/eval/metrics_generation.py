"""
Generation metrics evaluation module.

This module provides functions to evaluate generated answer quality
using ROUGE scores and unanswerable question accuracy.
"""

import logging
from typing import Dict, List, Any

import evaluate


# Set up logger
logger = logging.getLogger(__name__)

def calculate_rouge_metrics(
    results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate ROUGE scores for answerable questions.
    
    Args:
        results: List of result dictionaries from pipeline runs, each containing at least:
               - 'generated_answer': The model-generated answer
               - 'gold_answer': The ground truth answer
               - 'is_answerable': Boolean flag indicating if question is answerable
    
    Returns:
        Dictionary containing calculated ROUGE scores (rouge1, rouge2, rougeL)
    
    Notes:
        - Only answerable questions are included in the calculation
        - Returns empty scores if no answerable questions are found
    """
    # Validate inputs
    if not results:
        logger.warning("Empty results list provided to calculate_rouge_metrics")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    # Filter for answerable questions
    answerable_results = [r for r in results if r.get('is_answerable', False)]
    
    # Return default scores if no answerable questions
    if not answerable_results:
        logger.warning("No answerable questions found for ROUGE calculation")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    try:
        # Extract predictions and references
        predictions = []
        references = []
        skipped = 0
        
        for r in answerable_results:
            # Ensure both answers exist
            gen_answer = r.get('generated_answer')
            gold_answer = r.get('gold_answer')
            
            if not gen_answer or not gold_answer:
                skipped += 1
                continue
                
            # Convert to strings and add to lists
            predictions.append(str(gen_answer))
            references.append(str(gold_answer))
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} answerable results with missing answers")
        
        if not predictions:
            logger.warning("No valid answer pairs found for ROUGE calculation")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        # Load ROUGE metric
        rouge = evaluate.load('rouge')
        
        # Compute ROUGE scores
        scores = rouge.compute(predictions=predictions, references=references)
        
        # Log results
        logger.info(f"ROUGE metrics calculated on {len(predictions)} answerable questions")
        for metric, value in scores.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return scores
        
    except Exception as e:
        logger.error(f"Error calculating ROUGE metrics: {str(e)}")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}


def calculate_unanswerable_accuracy(
    results: List[Dict[str, Any]]
) -> float:
    """
    Calculate accuracy for unanswerable questions.
    
    Accuracy is defined as the proportion of unanswerable questions where
    the model correctly responded with "unanswerable" (case-insensitive).
    
    Args:
        results: List of result dictionaries from pipeline runs, each containing at least:
               - 'generated_answer': The model-generated answer
               - 'is_answerable': Boolean flag indicating if question is answerable
    
    Returns:
        Accuracy score as a float between 0.0 and 1.0
        Returns 0.0 if no unanswerable questions are found
    """
    # Validate inputs
    if not results:
        logger.warning("Empty results list provided to calculate_unanswerable_accuracy")
        return 0.0
    
    # Filter for unanswerable questions
    unanswerable_results = [r for r in results if not r.get('is_answerable', True)]
    
    # Return 0.0 if no unanswerable questions
    if not unanswerable_results:
        logger.warning("No unanswerable questions found for accuracy calculation")
        return 0.0
    
    try:
        # Count matches where generated answer is "unanswerable"
        matches = 0
        total = len(unanswerable_results)
        
        for r in unanswerable_results:
            gen_answer = r.get('generated_answer', '')
            
            # Skip if missing generated answer
            if gen_answer is None:
                total -= 1
                continue
                
            # Check if answer matches "unanswerable" (case-insensitive, ignoring whitespace)
            if gen_answer.strip().lower() == "unanswerable":
                matches += 1
        
        # Calculate accuracy
        accuracy = matches / total if total > 0 else 0.0
        
        # Log result
        logger.info(f"Unanswerable accuracy: {accuracy:.4f} ({matches}/{total})")
        
        return accuracy
        
    except Exception as e:
        logger.error(f"Error calculating unanswerable accuracy: {str(e)}")
        return 0.0 