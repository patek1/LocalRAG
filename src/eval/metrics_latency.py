"""
Latency metrics calculation and file saving module.

This module provides functions to calculate performance statistics
and save metrics to output files.
"""

import json
import logging
import os
import numpy as np
import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.config import settings

# Set up logger
logger = logging.getLogger(__name__)

def calculate_latency_stats(
    retrieval_latencies: List[float], 
    generation_latencies: List[float]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistical metrics for retrieval and generation latencies.
    
    Computes mean, median, 95th percentile, and 99th percentile of latencies.
    
    Args:
        retrieval_latencies: List of retrieval latency values in seconds
        generation_latencies: List of generation latency values in seconds
        
    Returns:
        Dictionary containing statistics for both retrieval and generation:
        {
            'retrieval': {
                'mean': float, 'median': float, 'p95': float, 'p99': float
            },
            'generation': {
                'mean': float, 'median': float, 'p95': float, 'p99': float
            }
        }
    """
    # Calculate individual statistics
    retrieval_stats = _calculate_single_latency_stats(retrieval_latencies, "retrieval")
    generation_stats = _calculate_single_latency_stats(generation_latencies, "generation")
    
    # Combine and return
    return {
        'retrieval': retrieval_stats,
        'generation': generation_stats
    }


def _calculate_single_latency_stats(latencies: List[float], latency_type: str) -> Dict[str, float]:
    """
    Calculate statistical metrics for a list of latency values.
    
    Computes mean, median, 95th percentile, and 99th percentile of latencies.
    
    Args:
        latencies: List of latency values in seconds
        latency_type: Type of latency ("retrieval" or "generation") for logging
        
    Returns:
        Dictionary containing the calculated statistics:
            - 'mean': Average latency
            - 'median': Median latency
            - 'p95': 95th percentile latency
            - 'p99': 99th percentile latency
    """
    # Initialize default values
    default_stats = {'mean': 0.0, 'median': 0.0, 'p95': 0.0, 'p99': 0.0}
    
    # Return defaults if latencies list is empty
    if not latencies:
        logger.warning(f"Empty {latency_type} latencies list provided to calculate_latency_stats")
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
        logger.info(f"{latency_type.capitalize()} latency statistics calculated for {len(latencies)} values")
        logger.info(f"{latency_type.capitalize()} - Mean: {mean:.4f}s, Median: {median:.4f}s, P95: {p95:.4f}s, P99: {p99:.4f}s")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating {latency_type} latency statistics: {str(e)}")
        return default_stats


def save_latency_metrics(
    latency_stats: Dict[str, Dict[str, float]], 
    output_path: str,
    current_run_subset_size: int = 0
) -> None:
    """
    Save latency metrics to a JSON file with overwrite logic based on subset size.
    
    Args:
        latency_stats: Dictionary containing both retrieval and generation latency statistics
        output_path: Path where to save the metrics file
        current_run_subset_size: Number of questions processed in this run (for overwrite logic)
        
    Returns:
        None
    """
    try:
        # Ensure the parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp and subset size to metrics
        metrics_to_save = {
            **latency_stats,
            'timestamp': datetime.datetime.now().isoformat(),
            'source_subset_size': current_run_subset_size
        }
        
        # Implement overwrite logic based on source_subset_size
        save_this_run = True
        if Path(output_path).exists():
            try:
                with open(output_path, 'r') as f:
                    existing_metrics = json.load(f)
                existing_subset_size = existing_metrics.get('source_subset_size', 0)
                if current_run_subset_size < existing_subset_size:
                    save_this_run = False
                    logger.info(
                        f"Current run subset size ({current_run_subset_size}) is smaller than "
                        f"existing latency metrics source_subset_size ({existing_subset_size}). "
                        f"Skipping save for {output_path}."
                    )
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read or parse existing latency metrics at {output_path}. Will overwrite. Error: {e}")
        
        # Save to JSON file if appropriate
        if save_this_run:
            with open(output_path, 'w') as f:
                json.dump(metrics_to_save, f, indent=4)
            logger.info(f"Latency metrics saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving latency metrics: {str(e)}")


def save_quality_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Save quality metrics to a JSON file.
    
    Args:
        metrics: Dictionary containing all quality metrics
        output_path: Path where to save the metrics file
        
    Returns:
        None
    """
    try:
        # Ensure the parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp
        if "test_date" in metrics and not metrics["test_date"]:
            metrics["test_date"] = datetime.datetime.now().isoformat()
        
        # Process the metrics to handle NaN values
        processed_metrics = _process_metrics_for_json(metrics)
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(processed_metrics, f, indent=4)
        
        logger.info(f"Quality metrics saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving quality metrics: {str(e)}")


def _process_metrics_for_json(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process metrics dictionary to handle NaN values for JSON serialization.
    
    Args:
        metrics: Dictionary containing metrics data
        
    Returns:
        Processed dictionary with NaN values converted to strings
    """
    if isinstance(metrics, dict):
        return {k: _process_metrics_for_json(v) for k, v in metrics.items()}
    elif isinstance(metrics, list):
        return [_process_metrics_for_json(item) for item in metrics]
    elif isinstance(metrics, float) and np.isnan(metrics):
        return "NaN"  # Convert NaN to string representation
    else:
        return metrics


def save_latency_metrics_v0(
    latency_stats: Dict[str, Dict[str, float]],
    output_path: str = None
) -> None:
    """
    Save latency metrics to a JSON file.
    
    Args:
        latency_stats: Dictionary containing both retrieval and generation latency statistics
        output_path: Path where to save the metrics file (defaults to settings.get_latency_metrics_path())
        
    Returns:
        None
    """
    try:
        # Use default path if none provided
        if output_path is None:
            output_path = settings.get_latency_metrics_path()
        
        # Ensure the parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to metrics
        metrics_to_save = {
            **latency_stats,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        
        logger.info(f"Latency metrics saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving latency metrics: {str(e)}")


def save_quality_metrics_v0(
    metrics: Dict[str, Optional[Dict[str, float]]],
    output_path: str = None
) -> None:
    """
    Save quality metrics to a JSON file.
    
    Args:
        metrics: Dictionary containing retrieval, rouge, and ragas metrics
        output_path: Path where to save the metrics file (defaults to settings.get_quality_metrics_path())
        
    Returns:
        None
    """
    try:
        # Use default path if none provided
        if output_path is None:
            output_path = settings.get_quality_metrics_path()
        
        # Ensure the parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp
        metrics_to_save = {
            **metrics,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        
        logger.info(f"Quality metrics saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving quality metrics: {str(e)}") 