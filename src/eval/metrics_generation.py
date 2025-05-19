"""
Generation metrics evaluation module.

This module provides functions to evaluate generated answer quality
using ROUGE scores and unanswerable question accuracy.
"""

import logging
import re
from typing import Dict, List, Any

import evaluate


# Set up logger
logger = logging.getLogger(__name__)

def calculate_rouge_metrics(
    gold_answers: Dict[str, str],
    generated_answers: Dict[str, str]
) -> Dict[str, float]:
    """
    Calculate ROUGE scores for answerable questions.
    
    Args:
        gold_answers: Dictionary mapping question IDs to gold standard answers
        generated_answers: Dictionary mapping question IDs to generated answers
    
    Returns:
        Dictionary containing calculated ROUGE scores (rouge1, rouge2, rougeL)
    
    Notes:
        - Only questions with both gold and generated answers are included
        - Returns empty scores if no valid answer pairs are found
    """
    # Validate inputs
    if not gold_answers or not generated_answers:
        logger.warning("Empty dictionaries provided to calculate_rouge_metrics")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0}
    
    try:
        # Extract predictions and references
        predictions = []
        references = []
        
        # Find common question IDs between gold and generated answers
        common_qids = set(gold_answers.keys()) & set(generated_answers.keys())
        
        for qid in common_qids:
            gold_answer = gold_answers[qid]
            gen_answer = generated_answers[qid]
            
            # Skip if either answer is None or empty
            if not gold_answer or not gen_answer:
                continue
                
            # Convert to strings and add to lists
            predictions.append(str(gen_answer))
            references.append(str(gold_answer))
        
        if not predictions:
            logger.warning("No valid answer pairs found for ROUGE calculation")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0}
        
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
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0}


def calculate_unanswerable_accuracy(
    generated_answers: Dict[str, str]
) -> Any:
    """
    Calculate accuracy for unanswerable questions.
    
    Accuracy is defined as the proportion of unanswerable questions where
    the model correctly responded with "unanswerable" (case-insensitive),
    allowing for non-alphanumeric characters after the word.
    
    Args:
        generated_answers: Dictionary mapping unanswerable question IDs to generated answers
    
    Returns:
        Accuracy score as a float between 0.0 and 1.0, or "N/A" if no unanswerable questions found
    """
    # Validate inputs
    if not generated_answers:
        logger.warning("Empty dictionary provided to calculate_unanswerable_accuracy")
        return "N/A"
    
    try:
        # Count matches where generated answer is "unanswerable"
        matches = 0
        total = len(generated_answers)
        
        for gen_answer in generated_answers.values():
            # Skip if missing generated answer
            if not gen_answer:
                total -= 1
                continue
                
            # Check if answer matches "unanswerable" pattern (allowing for non-alphanumeric characters at the end)
            if re.match(r'^unanswerable\W*$', gen_answer.strip().lower()):
                matches += 1
        
        # Calculate accuracy
        accuracy = matches / total if total > 0 else 0.0
        
        # Log result
        logger.info(f"Unanswerable accuracy: {accuracy:.4f} ({matches}/{total})")
        
        return accuracy
        
    except Exception as e:
        logger.error(f"Error calculating unanswerable accuracy: {str(e)}")
        return 0.0 