"""
Latency metrics calculation and file saving module.

This module provides functions to calculate performance statistics
and save metrics to output files.
"""

import json
import logging
import os
import numpy as np
from typing import Dict, List, Optional

from src.config import settings

# Set up logger
logger = logging.getLogger(__name__)

def calculate_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """
    Calculate statistical metrics for a list of latency values.
    
    Computes mean, median, 95th percentile, and 99th percentile of latencies.
    
    Args:
        latencies: List of latency values in seconds
        
    Returns:
        Dictionary containing the calculated statistics:
            - 'mean': Average latency
            - 'median': Median latency
            - 'p95': 95th percentile latency
            - 'p99': 99th percentile latency
            
    Notes:
        - Returns zeros if the latencies list is empty
    """
    # Initialize default values
    default_stats = {'mean': 0.0, 'median': 0.0, 'p95': 0.0, 'p99': 0.0}
    
    # Return defaults if latencies list is empty
    if not latencies:
        logger.warning("Empty latencies list provided to calculate_latency_stats")
        return default_stats
    
    try:
        # Convert latencies to numpy array for calculations
        latencies_array = np.array(latencies)
        
        # Calculate statistics
        mean = float(np.mean(latencies_array))
        median = float(np.median(latencies_array))
        p95 = float(np.percentile(latencies_array, 95))
        p99 = float(np.percentile(latencies_array, 99))
        
        # Create stats dictionary
        stats = {
            'mean': mean,
            'median': median,
            'p95': p95,
            'p99': p99
        }
        
        # Log results
        logger.info(f"Latency statistics calculated for {len(latencies)} values")
        logger.info(f"Mean: {mean:.4f}s, Median: {median:.4f}s, P95: {p95:.4f}s, P99: {p99:.4f}s")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating latency statistics: {str(e)}")
        return default_stats


def save_latency_metrics(
    retrieval_stats: Dict[str, float], 
    generation_stats: Dict[str, float]
) -> None:
    """
    Save retrieval and generation latency metrics to a JSON file.
    
    Args:
        retrieval_stats: Dictionary containing retrieval latency statistics
        generation_stats: Dictionary containing generation latency statistics
        
    Returns:
        None
        
    Notes:
        - Creates the results directory if it doesn't exist
        - Saves the metrics to settings.results_dir/settings.latency_metrics_file
    """
    try:
        # Create the combined metrics dictionary
        metrics = {
            'retrieval': retrieval_stats,
            'generation': generation_stats
        }
        
        # Ensure the results directory exists
        os.makedirs(settings.results_dir, exist_ok=True)
        
        # Define the output path
        output_path = os.path.join(settings.results_dir, settings.latency_metrics_file)
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Latency metrics saved to {output_path}")
        
    except IOError as e:
        logger.error(f"Error saving latency metrics to file: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error saving latency metrics: {str(e)}")


def save_quality_metrics(
    retrieval_scores: Optional[Dict[str, float]],
    rouge_scores: Optional[Dict[str, float]],
    unanswerable_accuracy: Optional[float],
    ragas_scores: Optional[Dict[str, float]]
) -> None:
    """
    Save quality metrics (retrieval, ROUGE, unanswerable accuracy, RAGAs) to a JSON file.
    
    Args:
        retrieval_scores: Dictionary containing retrieval quality metrics (ndcg@10, precision@10, recall@10)
        rouge_scores: Dictionary containing ROUGE scores for answerable questions
        unanswerable_accuracy: Accuracy score for unanswerable questions
        ragas_scores: Dictionary containing RAGAs metrics (faithfulness, answer_relevancy)
        
    Returns:
        None
        
    Notes:
        - Creates the results directory if it doesn't exist
        - Saves the metrics to settings.results_dir/settings.quality_metrics_file
        - Handles None values gracefully, replacing them with appropriate defaults
    """
    try:
        # Create the combined metrics dictionary with appropriate default values for None inputs
        all_quality_metrics = {
            "retrieval": retrieval_scores if retrieval_scores is not None else {},
            "generation_rouge": rouge_scores if rouge_scores is not None else {},
            "unanswerable_accuracy": unanswerable_accuracy if unanswerable_accuracy is not None else None,
            "ragas": ragas_scores if ragas_scores is not None else {},
        }
        
        # Ensure the results directory exists
        os.makedirs(settings.results_dir, exist_ok=True)
        
        # Define the output path using settings
        output_path = settings.get_quality_metrics_path()
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(all_quality_metrics, f, indent=4)
        
        logger.info(f"Quality metrics saved to {output_path}")
        
    except IOError as e:
        logger.error(f"Error saving quality metrics to file: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error saving quality metrics: {str(e)}") 