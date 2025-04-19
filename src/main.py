"""
Main execution script for the Local RAG Pipeline.

This script orchestrates the complete RAG pipeline process:
1. Loading of data (passages, questions, ground truth)
2. Initializing components (embedder, vector store, generator LLM, evaluator LLM)
3. Indexing corpus passages (if needed)
4. Running the RAG pipeline for each question
5. Evaluating results using various metrics
6. Saving performance metrics
"""

import argparse
import sys
from dotenv import load_dotenv

# Adjust import paths for running as a script
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.embedding import get_embedding_model
from src.vector_store import get_vector_store
from src.indexing import index_corpus
from src.data_loader import (
    load_corpus_passages,
    load_development_questions,
    load_retrieval_ground_truth
)
from src.utils import setup_logging
from src.generation import get_generator_llm
from src.eval import (
    calculate_retrieval_metrics,
    calculate_rouge_metrics,
    calculate_unanswerable_accuracy,
    get_evaluator_llm,
    calculate_ragas_metrics,
    calculate_latency_stats,
    save_latency_metrics,
    save_quality_metrics
)
from src.pipeline import run_rag_chain

# Set up logging
logger = setup_logging()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Local RAG Pipeline evaluation on Apple Silicon"
    )
    
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing of corpus passages even if index exists"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions to process (for testing)"
    )
    
    return parser.parse_args()


def main():
    """Main execution function for the RAG pipeline."""
    args = parse_arguments()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Verify OpenAI API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    logger.info(f"OpenAI API Key Found: {openai_api_key is not None}")
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY not found in environment. RAGAS evaluation will likely fail.")
    
    logger.info("Starting Local RAG Pipeline evaluation")
    logger.info("Initializing components...")
    
    # Initialize embedding model
    try:
        embedder = get_embedding_model()
    except Exception as e:
        logger.critical(f"Failed to initialize embedding model: {e}")
        sys.exit(1)
    
    # Initialize vector store
    try:
        vector_store = get_vector_store(embedding_model=embedder)
    except Exception as e:
        logger.critical(f"Failed to initialize vector store: {e}")
        sys.exit(1)
    
    # Load corpus passages
    try:
        logger.info("Loading corpus passages...")
        corpus_passages = load_corpus_passages()
    except Exception as e:
        logger.critical(f"Failed to load corpus passages: {e}")
        sys.exit(1)
    
    # Index corpus passages if needed or if --reindex flag is used
    current_docs_count = vector_store._collection.count()
    
    if args.reindex or current_docs_count == 0:
        logger.info(f"Indexing corpus passages (force_reindex={args.reindex})...")
        try:
            index_corpus(
                corpus_data=corpus_passages,
                vector_store=vector_store,
                embedder=embedder,
                force_reindex=args.reindex
            )
        except Exception as e:
            logger.critical(f"Failed to index corpus passages: {e}")
            sys.exit(1)
    else:
        logger.info(
            f"Using existing index with {current_docs_count} documents. "
            f"Use --reindex to force re-indexing."
        )
    
    # Load development questions and ground truth
    try:
        logger.info("Loading development questions and ground truth...")
        dev_questions = load_development_questions()
        qrels = load_retrieval_ground_truth()
    except Exception as e:
        logger.critical(f"Failed to load development questions or ground truth: {e}")
        sys.exit(1)
    
    # Limit the number of questions if specified
    if args.limit is not None and args.limit > 0:
        logger.info(f"Limiting to {args.limit} questions")
        dev_questions = dev_questions[:args.limit]
    
    # Initialize the generator LLM
    try:
        logger.info("Initializing generator LLM...")
        generator_llm = get_generator_llm()
    except Exception as e:
        logger.critical(f"Failed to initialize generator LLM: {e}")
        sys.exit(1)
    
    # Process questions
    logger.info(f"Processing {len(dev_questions)} questions...")
    
    # Collection variables for results
    results_list = []
    all_retrieval_latencies = []
    all_generation_latencies = []
    
    # Important message for manual resource monitoring
    logger.warning(
        "IMPORTANT: Please manually monitor resource usage (CPU, RAM, GPU) during execution "
        "using asitop, mactop, or Activity Monitor. These metrics are critical for "
        "performance evaluation on Apple Silicon."
    )
    
    # Main execution loop for processing questions
    logger.info(f"Starting RAG pipeline for {len(dev_questions)} questions...")
    for i, question_data in enumerate(dev_questions):
        question_id = question_data['question_id']
        question_text = question_data['question_text']
        logger.info(f"Processing question {i+1}/{len(dev_questions)}: ID {question_id}")

        try:
            # Call the RAG chain function
            rag_result = run_rag_chain(
                question_text=question_text,
                vector_store=vector_store,
                embedder=embedder,
                generator_llm=generator_llm
            )

            # Combine input data and RAG result
            combined_result = {
                **question_data,  # Includes question_id, question_text, gold_answer, is_answerable
                **rag_result      # Includes generated_answer, retrieved_contexts, retrieved_ids, latencies
            }
            results_list.append(combined_result)

            # Collect latencies separately for aggregation
            if rag_result.get('retrieval_latency') is not None:
                all_retrieval_latencies.append(rag_result['retrieval_latency'])
            if rag_result.get('generation_latency') is not None:
                all_generation_latencies.append(rag_result['generation_latency'])

        except Exception as e:
            logger.error(f"Error processing question ID {question_id}: {e}", exc_info=True)
            # Append partial results or error info to results_list
            results_list.append({
                **question_data,
                'error': str(e),
                'generated_answer': None,
                'retrieved_contexts': [],
                'retrieved_ids': [],
                'retrieval_latency': 0.0,
                'generation_latency': 0.0,
            })

    logger.info("RAG pipeline processing complete.")
    
    # Evaluation section
    logger.info("Starting evaluation...")

    if not results_list:
        logger.warning("No results collected, skipping evaluation.")
        sys.exit(1)

    # Initialize variables to store metrics (with default values)
    retrieval_scores = None
    rouge_scores = None
    unanswerable_acc = None
    ragas_scores = None

    # --- Retrieval Evaluation ---
    logger.info("Calculating retrieval metrics (ranx)...")
    try:
        retrieval_scores = calculate_retrieval_metrics(qrels, results_list)
        logger.info(f"Retrieval Metrics (ranx): {retrieval_scores}")
    except Exception as e:
        logger.error(f"Failed to calculate retrieval metrics: {e}", exc_info=True)

    # --- Generation Evaluation (ROUGE, Accuracy) ---
    logger.info("Calculating generation metrics (ROUGE, Unanswerable Accuracy)...")
    try:
        rouge_scores = calculate_rouge_metrics(results_list)
        logger.info(f"ROUGE Scores (Answerable): {rouge_scores}")
    except Exception as e:
        logger.error(f"Failed to calculate ROUGE scores: {e}", exc_info=True)

    try:
        unanswerable_acc = calculate_unanswerable_accuracy(results_list)
        logger.info(f"Unanswerable Accuracy: {unanswerable_acc:.4f}")
    except Exception as e:
        logger.error(f"Failed to calculate unanswerable accuracy: {e}", exc_info=True)

    # --- RAGAs Evaluation ---
    logger.info("Initializing RAGAs evaluator LLM (OpenAI)...")
    try:
        evaluator_llm = get_evaluator_llm()
        
        logger.info("Calculating RAGAs metrics (Faithfulness, Answer Relevancy)...")
        ragas_scores = calculate_ragas_metrics(results_list, evaluator_llm)
        logger.info(f"RAGAs Scores (Answerable): {ragas_scores}")
    except Exception as e:
        logger.error(f"Failed to calculate RAGAs metrics: {e}", exc_info=True)

    # --- Latency Evaluation ---
    logger.info("Calculating latency statistics...")
    try:
        retrieval_latency_stats = calculate_latency_stats(all_retrieval_latencies)
        logger.info(f"Retrieval Latency Stats: {retrieval_latency_stats}")
        generation_latency_stats = calculate_latency_stats(all_generation_latencies)
        logger.info(f"Generation Latency Stats: {generation_latency_stats}")

        logger.info("Saving latency metrics...")
        save_latency_metrics(retrieval_latency_stats, generation_latency_stats)
    except Exception as e:
        logger.error(f"Failed to calculate or save latency stats: {e}", exc_info=True)
    
    # --- Save Quality Metrics ---
    logger.info("Saving quality metrics...")
    try:
        save_quality_metrics(
            retrieval_scores=retrieval_scores,
            rouge_scores=rouge_scores,
            unanswerable_accuracy=unanswerable_acc,
            ragas_scores=ragas_scores
        )
    except Exception as e:
        logger.error(f"Failed to save quality metrics: {e}", exc_info=True)

    logger.info("Evaluation complete.")
    # Add final reminder for manual resource monitoring
    logger.info("REMINDER: Remember to note manual observations of CPU/RAM/GPU usage during the run.")
    
    logger.info("RAG Pipeline execution completed")


if __name__ == "__main__":
    main() 