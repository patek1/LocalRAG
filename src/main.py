"""
Main execution script for the Local RAG Pipeline.

This script orchestrates the complete RAG pipeline process:
1. Loading of data (passages, questions, ground truth)
2. Initializing components (embedder, vector store, generator LLM, evaluator LLM)
3. Running the RAG pipeline for each question
4. Evaluating results using various metrics
5. Saving performance metrics
"""

import sys
import json
import logging
import random
import re
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any, List

# Adjust import paths for running as a script
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from src.embedding import get_embedding_model
from src.vector_store import get_vector_store
from src.utils import setup_logging
from src.generation import get_generator_llm
from src.config import Settings
from src.cli import parse_arguments
from src.analysis_utils import classify_llm_output_for_unanswerable
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
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Set up logging
logger = setup_logging()


def generate_synthetic_gold_passage_ids(
    question_data: Dict[str, Any],
    gold_answer: str,
    vector_store: Chroma,
    embedder: HuggingFaceEmbeddings,
    k: int = 5
) -> List[str]:
    """
    Generate synthetic gold passage IDs by finding passages that likely contain the gold answer.
    This is used when a question doesn't have explicit gold passage IDs.
    
    Args:
        question_data: Dictionary containing question data
        gold_answer: The expected gold answer text
        vector_store: Initialized vector store
        embedder: Initialized embedding model
        k: Number of passages to retrieve (top-k)
        
    Returns:
        List of passage IDs (strings) that likely contain the answer
    """
    if not gold_answer:
        return []
    
    # Combine question and answer to find relevant passages
    question_text = question_data.get('question_text', '')
    combined_query = f"{question_text} {gold_answer}"
    
    # Perform similarity search to find relevant passages
    docs = vector_store.similarity_search_by_vector(
        embedding=embedder.embed_query(combined_query),
        k=k
    )
    
    # Extract passage IDs from search results
    passage_ids = []
    for i, doc in enumerate(docs):
        pid = doc.metadata.get('original_id', None)
        if pid:
            passage_ids.append(str(pid))
    
    return passage_ids


def main() -> None:
    """Main execution function for the RAG pipeline."""
    args = parse_arguments()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Load settings
    settings = Settings()
    
    # Create directory for unanswerable analysis
    analysis_output_dir = Path(settings.results_dir) / "unanswerable_analysis_details"
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize a dictionary to store file paths for each model
    model_analysis_files = {}
    
    # Initialize dictionary to store unanswerable analysis data
    current_run_unanswerable_analysis_data = {}
    
    # Get active dataset configuration
    try:
        active_dataset_config = settings.get_active_dataset_config(args.dataset)
        logger.info(f"Running evaluation for dataset: {args.dataset} with model: {args.model}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Active dataset config: {active_dataset_config}")
    except ValueError as e:
        logger.error(f"Error: {e}. Available datasets are: {list(settings.dataset_configs.keys())}")
        sys.exit(1)
    
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
    
    # Get paths for vector store and collection name
    vector_store_dir = active_dataset_config.get_vector_store_dir(
        Path(settings.base_vector_store_dir), 
        args.dataset
    )
    collection_name = f"{args.dataset.lower()}{active_dataset_config.vector_store_collection_name_suffix}"
    
    # Check if vector store exists and has documents
    logger.info(f"Checking for existing vector store at: {vector_store_dir} with collection: {collection_name}")
    try:
        # Try to access the ChromaDB client and collection to check if it exists
        client = chromadb.PersistentClient(path=str(vector_store_dir))
        try:
            coll = client.get_collection(name=collection_name)
            doc_count = coll.count()
            
            if doc_count == 0:
                logger.error(
                    f"Vector store at {vector_store_dir} for collection {collection_name} is empty. "
                    f"Please run 'python scripts/create_index.py --dataset {args.dataset}' first."
                )
                sys.exit(1)
            logger.info(f"Found existing vector store with {doc_count} documents.")
        except Exception:
            # Collection doesn't exist
            logger.error(
                f"Collection '{collection_name}' not found in vector store at {vector_store_dir}. "
                f"Please run 'python scripts/create_index.py --dataset {args.dataset}' first."
            )
            sys.exit(1)
    except Exception as e:
        logger.error(
            f"Could not access vector store at {vector_store_dir}. Error: {e}. "
            f"Please run 'python scripts/create_index.py --dataset {args.dataset}' first."
        )
        sys.exit(1)
    
    # Initialize vector store for actual use with embeddings
    try:
        vector_store = get_vector_store(
            embedding_model=embedder,
            persist_directory=str(vector_store_dir),
            collection_name=collection_name
        )
        logger.info(f"Successfully initialized vector store for search")
    except Exception as e:
        logger.critical(f"Failed to initialize vector store for search: {e}")
        sys.exit(1)
    
    # Get the processed corpus and questions paths
    processed_corpus_path = active_dataset_config.get_processed_corpus_file_path(
        Path(settings.base_data_dir),
        args.dataset
    )
    processed_questions_path = active_dataset_config.get_processed_questions_file_path(
        Path(settings.base_data_dir),
        args.dataset
    )
    
    # Check if processed data exists
    if not processed_corpus_path.exists() or not processed_questions_path.exists():
        logger.critical(
            f"Processed data not found at expected locations. "
            f"Run 'python scripts/create_index.py --dataset {args.dataset}' first."
        )
        sys.exit(1)
    
    # Load processed questions from file
    try:
        logger.info(f"Loading processed questions from {processed_questions_path}...")
        dev_questions = []
        with open(processed_questions_path, 'r', encoding='utf-8') as f:
            for line in f:
                dev_questions.append(json.loads(line))
        logger.info(f"Loaded {len(dev_questions)} questions")
    except Exception as e:
        logger.critical(f"Failed to load processed questions: {e}")
        sys.exit(1)
    
    # Construct QRELs for retrieval evaluation from processed questions
    try:
        logger.info("Constructing QRELs for retrieval evaluation from processed questions...")
        qrels = {}
        explicit_gold_count = 0
        synthetic_gold_count = 0
        
        for q_data in dev_questions:
            question_id = str(q_data['question_id'])
            
            # Check if question has explicit gold passage IDs
            if 'gold_passage_ids' in q_data and q_data['gold_passage_ids']:
                # Use explicit gold passage IDs
                qrels[question_id] = {str(pid): 1 for pid in q_data['gold_passage_ids']}
                explicit_gold_count += 1
            else:
                # Get gold answer
                gold_answer = q_data.get('gold_answer', '')
                
                # Generate synthetic gold passage IDs for answerable questions
                if gold_answer and q_data.get('is_answerable', True):
                    # Generate synthetic gold passage IDs
                    gold_passage_ids = generate_synthetic_gold_passage_ids(q_data, gold_answer, vector_store, embedder)
                    if gold_passage_ids:
                        qrels[question_id] = {str(pid): 1 for pid in gold_passage_ids}
                        synthetic_gold_count += 1
                    else:
                        qrels[question_id] = {}  # No gold passages found
                else:
                    qrels[question_id] = {}  # Question might be unanswerable
        
        # Log statistics about the qrels
        logger.info(f"Constructed QRELs for {len(qrels)} questions.")
        logger.info(f"Questions with explicit gold passage IDs: {explicit_gold_count}")
        logger.info(f"Questions with synthetic gold passage IDs: {synthetic_gold_count}")
        
        if len(qrels) > 0:
            non_empty_qrels = sum(1 for qrel_dict in qrels.values() if qrel_dict)
            logger.info(f"Questions with non-empty gold passages: {non_empty_qrels}/{len(qrels)}")
            
            # Only log detailed qrels information in debug level
            if logger.isEnabledFor(logging.DEBUG) and non_empty_qrels > 0:
                sample_qid = next(qid for qid, qrel_dict in qrels.items() if qrel_dict)
                sample_qrels = qrels[sample_qid]
                logger.debug(f"Sample qrels - QID: {sample_qid}, has {len(sample_qrels)} gold passages")
                if sample_qrels:
                    sample_pid = next(iter(sample_qrels))
                    logger.debug(f"Sample qrels format - passage ID: {sample_pid}, type: {type(sample_pid)}, relevance: {sample_qrels[sample_pid]}")
                
                # Log just a couple of examples instead of all
                logger.debug(f"QRELS details for sample questions:")
                for qid in list(qrels.keys())[:2]:  # Only log first 2 questions 
                    passage_ids = list(qrels[qid].keys()) if qrels[qid] else []
                    logger.debug(f"  QID {qid}: {len(passage_ids)} gold passages {passage_ids[:3] if passage_ids else '[]'}")
    except Exception as e:
        logger.error(f"Error constructing QRELs: {e}")
        qrels = {}  # Continue with empty qrels, which will affect retrieval metrics
    
    # Apply limit if specified
    if args.limit and args.limit < len(dev_questions):
        logger.info(f"Limiting to a random subset of {args.limit} questions from {len(dev_questions)} total.")
        
        # Check for a saved subset file
        subset_qids_file = active_dataset_config.get_subset_qids_file_path(
            Path(settings.base_data_dir),
            args.dataset,
            args.limit
        )
        
        if subset_qids_file.exists():
            # Load previous subset to ensure consistent testing across runs
            try:
                logger.info(f"Loading previous subset from {subset_qids_file}")
                with open(subset_qids_file, 'r') as f:
                    selected_qids = json.load(f)
                    # Filter questions to only keep those in the selected qids
                    dev_questions = [q for q in dev_questions if str(q['question_id']) in selected_qids]
                    logger.info(f"Loaded {len(selected_qids)} selected QIDs from {subset_qids_file}.")
            except Exception as e:
                logger.error(f"Error loading subset file, generating new one: {e}")
                # Fall through to generation
        
        if not subset_qids_file.exists() or len(dev_questions) != args.limit:
            # Either no file exists or we couldn't load it properly
            logger.info(f"No existing subset QID file found. Generating and saving to: {subset_qids_file}")
            # Randomly select questions
            dev_questions = random.sample(dev_questions, args.limit)
            # Save the selected question IDs for future runs
            selected_qids = [str(q['question_id']) for q in dev_questions]
            with open(subset_qids_file, 'w') as f:
                json.dump(selected_qids, f)
                logger.info(f"Saved {len(selected_qids)} selected QIDs to {subset_qids_file}.")
    
    logger.info(f"Proceeding with {len(dev_questions)} questions after subset selection.")
    
    # Initialize generator LLM
    logger.info("Initializing generator LLM...")
    try:
        generator_llm = get_generator_llm(args.model)
        logger.info("Generator LLM initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize generator LLM: {e}")
        sys.exit(1)
    
    # Process questions
    logger.info(f"Processing {len(dev_questions)} questions...")
    logger.warning(
        "IMPORTANT: Please manually monitor resource usage (CPU, RAM, GPU) "
        "during execution using asitop, mactop, or Activity Monitor. "
        "These metrics are critical for performance evaluation on Apple Silicon."
    )
    
    retrieval_run = {}  # Will store retrieved passages for each question
    retrieval_latencies = []  # Will store retrieval latencies
    generation_latencies = []  # Will store generation latencies
    generated_answers = {}  # Will store generated answers for evaluation
    question_text_map = {}  # Maps question ID to question text
    gold_answers = {}  # Maps question ID to gold answers
    answerable_qids = []  # Stores IDs of answerable questions
    unanswerable_qids = []  # Stores IDs of unanswerable questions
    
    # Additional data for RAGAs evaluation
    ragas_contexts = {}  # Stores contexts for each question
    
    # Start the RAG pipeline
    logger.info(f"Starting RAG pipeline for {len(dev_questions)} questions...")
    
    for idx, question_data in enumerate(dev_questions, start=1):
        question_id = str(question_data['question_id'])
        question_text = question_data['question_text']
        
        # Store data for later evaluation
        question_text_map[question_id] = question_text
        
        # Get gold answer, ensuring it's a non-empty string
        # The gold answer is in the 'gold_answer' field, not 'answer_text'
        gold_answer = question_data.get('gold_answer', '')
        if gold_answer:
            gold_answers[question_id] = gold_answer.strip()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Gold answer for {question_id}: {gold_answers[question_id][:50]}...")
        
        # Track answerability for metrics
        is_answerable = question_data.get('is_answerable', True)  # Default to True if not specified
        if is_answerable:
            answerable_qids.append(question_id)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Question {question_id} marked as answerable")
        else:
            unanswerable_qids.append(question_id)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Question {question_id} marked as unanswerable")
        
        logger.info(f"Processing question {idx}/{len(dev_questions)}: ID {question_id}")
        
        # Run RAG chain (retrieval + generation)
        response = run_rag_chain(
            question_text=question_text,
            vector_store=vector_store,
            embedder=embedder,
            generator_llm=generator_llm
        )
        
        # Save results for evaluation
        if 'retrieved_ids' in response:
            # Format retrieval run dict properly for ranx:
            # Each question ID maps to a dict of {passage_id: score}
            # where score is 1/rank position (so first result has score 1.0, etc.)
            retrieval_run[question_id] = {
                str(doc_id): (1.0 / (idx + 1)) for idx, doc_id in enumerate(response['retrieved_ids'])
            }
            
            # Log the format for debugging - only for the first question
            if idx == 1 and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Sample retrieval_run format - Question {question_id}: {retrieval_run[question_id]}")
        
        retrieval_latencies.append(response['retrieval_latency'])
        generation_latencies.append(response['generation_latency'])
        generated_answers[question_id] = response['generated_answer']
        ragas_contexts[question_id] = response['retrieved_contexts']
        
        # Conditional Logic for ClapNQ Unanswerable Analysis
        if args.dataset == "ClapNQ" and not question_data.get('is_answerable', True):
            model_tag_for_file = args.model.replace(":", "_")  # Safe filename
            analysis_file_path = analysis_output_dir / f"analysis_unans_{model_tag_for_file}_{args.dataset.lower()}.json"

            # Open file if not already open, prepare for JSON structure (list of entries after a summary)
            if args.model not in model_analysis_files:
                # Initialize the model's analysis list if not already done
                if args.model not in current_run_unanswerable_analysis_data:
                    current_run_unanswerable_analysis_data[args.model] = []
                
                # Store placeholder for file path, actual writing will be at the end
                model_analysis_files[args.model] = analysis_file_path

            # Perform classification
            llm_output_class = classify_llm_output_for_unanswerable(response['generated_answer'])
            
            # Check correctness with the NEW FLEXIBLE definition
            # This requires the updated calculate_unanswerable_accuracy or equivalent logic
            # For now, let's use the regex directly here for clarity in this specific logging.
            # The main metrics will use the updated calculate_unanswerable_accuracy.
            is_correct_flexible = bool(re.match(r"^\s*unanswerable\W*\s*$", response['generated_answer'], re.IGNORECASE))

            analysis_entry = {
                "question_id": question_id,
                "question_text": question_text,
                "gold_answer_expected": "unanswerable",
                "raw_llm_output": response['generated_answer'],
                "is_correctly_unanswerable_by_main_eval": is_correct_flexible, # Reflects what main eval will count
                "llm_output_classification": llm_output_class
            }
            current_run_unanswerable_analysis_data[args.model].append(analysis_entry)
    
    logger.info("RAG pipeline processing complete.")
    
    # Start evaluation
    logger.info("Starting evaluation...")
    
    # Retrieval metrics
    logger.info("Calculating retrieval metrics (ranx)...")
    retrieval_metrics = calculate_retrieval_metrics(qrels, retrieval_run)
    logger.info(f"Retrieval Metrics (ranx): {retrieval_metrics}")
    
    # NLG metrics
    logger.info("Calculating generation metrics (ROUGE, Unanswerable Accuracy)...")
    
    # Filter for answerable questions that have both gold and generated answers
    valid_answerable_qids = [qid for qid in answerable_qids 
                             if qid in generated_answers and qid in gold_answers]
    logger.info(f"Found {len(valid_answerable_qids)}/{len(answerable_qids)} answerable questions with both gold and generated answers")
    
    rouge_scores_ans = calculate_rouge_metrics(
        {qid: gold_answers[qid] for qid in valid_answerable_qids},
        {qid: generated_answers[qid] for qid in valid_answerable_qids}
    )
    logger.info(f"ROUGE Scores (Answerable): {rouge_scores_ans}")
    
    # Calculate accuracy only if we have unanswerable questions
    unanswerable_accuracy = calculate_unanswerable_accuracy(
        {qid: generated_answers.get(qid, '') for qid in unanswerable_qids}
    )
    logger.info(f"Unanswerable Accuracy: {unanswerable_accuracy}")
    
    # Initialize ragas_metrics before the conditional block
    ragas_metrics = {}
    
    # RAGAs evaluation (requires OpenAI API key)
    if not args.skip_ragas:
        try:
            logger.info("Initializing RAGAs evaluator LLM (OpenAI)...")
            evaluator_llm = get_evaluator_llm()
            
            # Prepare data for RAGAs using only questions with gold answers
            ragas_input = {}
            for question_id in valid_answerable_qids:
                if question_id in gold_answers and question_id in generated_answers and question_id in ragas_contexts:
                    ragas_input[question_id] = {
                        'question': question_text_map[question_id],
                        'ground_truths': [gold_answers[question_id]],
                        'answer': generated_answers[question_id],
                        'contexts': ragas_contexts[question_id]
                    }
                    # Log first question data at debug level
                    if question_id == valid_answerable_qids[0] and logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Sample RAGAs input for question {question_id}:")
                        logger.debug(f"  Question: {question_text_map[question_id][:50]}...")
                        logger.debug(f"  Answer: {generated_answers[question_id][:50]}...")
                        logger.debug(f"  Ground Truth: {gold_answers[question_id][:50]}...")
                        logger.debug(f"  Contexts: {len(ragas_contexts[question_id])} passages")
            
            logger.info(f"Calculating RAGAs metrics for {len(ragas_input)} questions...")
            ragas_metrics = calculate_ragas_metrics(ragas_input, evaluator_llm)
            logger.info(f"RAGAs Metrics: {ragas_metrics}")
        except Exception as e:
            logger.error(f"Failed to calculate RAGAs metrics: {e}", exc_info=True)
    else:
        logger.info("Skipping RAGAs evaluation as per --skip-ragas flag.")
    
    # Latency statistics
    logger.info("Calculating latency statistics...")
    latency_stats = calculate_latency_stats(retrieval_latencies, generation_latencies)
    logger.info(f"Latency Stats: {latency_stats}")
    
    # Save metrics to files
    logger.info(f"Saving metrics to results/{args.model.replace(':', '_')}/{args.dataset.lower()}...")
    
    # Create directory for results
    results_dir = active_dataset_config.get_results_dir(
        Path(settings.results_dir),
        args.model,
        args.dataset
    )
    
    # Construct the metrics object
    quality_metrics = {
        "dataset": args.dataset,
        "model": args.model,
        "metrics": {
            "retrieval": retrieval_metrics,
            "rouge": rouge_scores_ans,
            "unanswerable_accuracy": unanswerable_accuracy,
            "ragas": ragas_metrics
        },
        "sample_size": len(dev_questions),
        "test_date": "",  # Will be added by the save function
        "dataset_config": {
            "hf_id": active_dataset_config.hf_id,
            "corpus_hf_id": active_dataset_config.corpus_hf_id,
            "has_explicit_unanswerables": active_dataset_config.has_explicit_unanswerables,
        }
    }
    
    # Save metrics to files
    save_latency_metrics(
        latency_stats=latency_stats, 
        output_path=str(results_dir / "latency_metrics.json"),
        current_run_subset_size=len(dev_questions)  # Pass the number of questions processed
    )
    save_quality_metrics(quality_metrics, str(results_dir / f"{args.limit or 'all'}_quality_metrics.json"))
    
    # Write detailed unanswerable analysis files
    if current_run_unanswerable_analysis_data:
        for model_tag_key, entries_list in current_run_unanswerable_analysis_data.items():
            if not entries_list: continue

            analysis_file_path = model_analysis_files[model_tag_key]  # Get the path stored earlier
            
            # Calculate summary statistics for classification
            classification_summary = {}
            for entry in entries_list:
                cls = entry["llm_output_classification"]
                classification_summary[cls] = classification_summary.get(cls, 0) + 1
            
            # Prepare final JSON structure for the file
            output_data_for_file = {
                "model_tag": model_tag_key,  # The actual model tag from the loop
                "dataset": args.dataset,     # args.dataset from the run
                "run_sample_size": args.limit if args.limit else len(dev_questions),  # Number of questions in this main.py run
                "analyzed_unanswerable_count": len(entries_list),
                "classification_summary": classification_summary,
                "detailed_entries": entries_list
            }

            with open(analysis_file_path, 'w', encoding='utf-8') as f_out:
                json.dump(output_data_for_file, f_out, indent=4)
            logger.info(f"Saved detailed unanswerable analysis for {model_tag_key} on {args.dataset} to {analysis_file_path}")
    
    logger.info(f"All metrics saved to {results_dir}")
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main() 