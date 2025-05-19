"""
Command line argument parsing for the Local RAG Pipeline.

This module provides functions for parsing command line arguments
used by the main script and other tools in the project.
"""

import argparse


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        An argparse.Namespace object containing the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run the Local RAG Pipeline evaluation on Apple Silicon"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specifies the Ollama model tag (e.g., gemma3:1b)."
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Specifies the dataset key from settings.dataset_configs (e.g., ClapNQ, TriviaQA)."
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions to process (for testing)"
    )
    
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAs evaluation to save money on API calls."
    )
    
    return parser.parse_args() 