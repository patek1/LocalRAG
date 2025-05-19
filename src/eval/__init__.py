"""
Evaluation module for the Local RAG Pipeline.

This package provides functions to evaluate the performance of the RAG pipeline
using various metrics for retrieval quality, generation quality, and system performance.
"""

# Import all the evaluation functions to make them available at the package level
from src.eval.metrics_retrieval import calculate_retrieval_metrics
from src.eval.metrics_generation import (
    calculate_rouge_metrics,
    calculate_unanswerable_accuracy
)
from src.eval.metrics_ragas import (
    get_evaluator_llm,
    calculate_ragas_metrics
)
from src.eval.metrics_latency import (
    calculate_latency_stats,
    save_latency_metrics,
    save_quality_metrics
)

# Re-export all the functions for backward compatibility
__all__ = [
    'calculate_retrieval_metrics',
    'calculate_rouge_metrics',
    'calculate_unanswerable_accuracy',
    'get_evaluator_llm',
    'calculate_ragas_metrics',
    'calculate_latency_stats',
    'save_latency_metrics',
    'save_quality_metrics'
] 