"""
RAGAs metrics evaluation module.

This module provides functions to evaluate RAG system performance
using the RAGAs framework for metrics like faithfulness and answer relevancy.
"""

import logging
import os
from typing import Dict, List, Any

from langchain_openai import ChatOpenAI

from src.config import settings

# Set up logger
logger = logging.getLogger(__name__)

def get_evaluator_llm() -> ChatOpenAI:
    """
    Initialize the ChatOpenAI for RAGAs evaluation.
    
    Uses settings from config to initialize a ChatOpenAI instance with
    the appropriate model and parameters. Uses OpenAI API for better
    evaluation performance compared to local models.
    
    Returns:
        ChatOpenAI: Initialized language model ready for RAGAs evaluation
        
    Raises:
        RuntimeError: If the OpenAI API key is missing or invalid
    """
    try:
        # Initialize the ChatOpenAI with settings from config
        logger.info(f"Initializing ChatOpenAI with model: {settings.evaluator_gpt_model}")
        
        # Configure ChatOpenAI - it will automatically use OPENAI_API_KEY from environment
        evaluator_llm = ChatOpenAI(
            model=settings.evaluator_gpt_model,
            temperature=0.0
        )
        
        logger.info("Evaluator LLM (OpenAI) initialized successfully")
        return evaluator_llm
        
    except Exception as e:
        logger.critical(f"Failed to initialize ChatOpenAI for evaluation: {e}")
        raise RuntimeError(f"Failed to initialize evaluator LLM: {e}") from e


def calculate_ragas_metrics(
    results: List[Dict[str, Any]], 
    evaluator_llm: ChatOpenAI
) -> Dict[str, float]:
    """
    Calculate RAGAs metrics (faithfulness and answer relevancy) for answerable questions.
    
    Args:
        results: List of result dictionaries from pipeline runs, each containing at least:
                - 'question_text': The original question text
                - 'generated_answer': The model-generated answer 
                - 'retrieved_contexts': List of retrieved passage texts
                - 'gold_answer': The ground truth answer
                - 'is_answerable': Boolean flag indicating if question is answerable
        evaluator_llm: The ChatOpenAI instance to use for RAGAs evaluation
    
    Returns:
        Dictionary containing calculated RAGAs metrics (faithfulness, answer_relevancy)
        
    Notes:
        - Only answerable questions are included in the evaluation
        - Returns empty scores if no answerable questions are found or if evaluation fails
    """
    # Validate inputs
    if not results:
        logger.warning("Empty results list provided to calculate_ragas_metrics")
        return {'faithfulness': 0.0, 'answer_relevancy': 0.0}
    
    if evaluator_llm is None:
        logger.error("No evaluator LLM provided to calculate_ragas_metrics")
        return {'faithfulness': 0.0, 'answer_relevancy': 0.0}
    
    # Filter for answerable questions
    answerable_results = [r for r in results if r.get('is_answerable', False)]
    
    # Return default scores if no answerable questions
    if not answerable_results:
        logger.warning("No answerable questions found for RAGAs evaluation")
        return {'faithfulness': 0.0, 'answer_relevancy': 0.0}
    
    try:
        # Prepare the dataset in the format needed by RAGAs
        evaluation_data = []
        skipped = 0
        
        for r in answerable_results:
            # Check for missing fields
            question_text = r.get('question_text')
            gen_answer = r.get('generated_answer')
            retrieved_contexts = r.get('retrieved_contexts')
            gold_answer = r.get('gold_answer')
            
            # Skip if any required field is missing (adjusting for RAGAs error message)
            # Faithfulness needs user_input, response, retrieved_contexts
            # AnswerRelevancy likely needs user_input, response (and maybe ground_truth)
            if not question_text or not gen_answer or not retrieved_contexts:
                logger.warning(f"Skipping QID {r.get('question_id')} due to missing required RAGAs fields (question/answer/contexts).")
                skipped += 1
                continue
            
            # Ensure contexts are a list of strings
            if not isinstance(retrieved_contexts, list) or not retrieved_contexts:
                logger.warning(f"Skipping QID {r.get('question_id')} due to invalid or empty retrieved contexts.")
                skipped += 1
                continue
                
            # --- REVERTED MAPPING: Use keys specified in the ValueError ---
            evaluation_data.append({
                "user_input": question_text,       # As required by the error
                "response": gen_answer,           # As required by the error
                "retrieved_contexts": retrieved_contexts, # As required by the error
                "ground_truth": gold_answer       # Include ground_truth as it's standard
            })
            # --- END REVERTED MAPPING ---
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} answerable results with missing or invalid fields")
        
        # Return default scores if no valid data
        if len(evaluation_data) == 0:
            logger.warning("No valid data found for RAGAs evaluation after filtering")
            return {'faithfulness': 0.0, 'answer_relevancy': 0.0}
        
        # Create RAGAs evaluation dataset
        from ragas import EvaluationDataset
        # Check if the dataset object expects specific column names during creation
        try:
            evaluation_dataset = EvaluationDataset.from_list(evaluation_data)
            # Log dataset creation success and sample data
            logger.info(f"RAGAs dataset created successfully with {len(evaluation_data)} entries")
            if evaluation_data:
                logger.debug(f"Sample data keys: {list(evaluation_data[0].keys())}")
        except Exception as ds_err:
            logger.error(f"Failed to create RAGAs dataset: {ds_err}", exc_info=True)
            return {'faithfulness': 0.0, 'answer_relevancy': 0.0}
        
        # Initialize metrics
        from ragas.metrics import Faithfulness, AnswerRelevancy
        metrics_to_run = [Faithfulness(), AnswerRelevancy()]
        
        # Set environment variable to silence tokenizer warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Wrap the LLM for RAGAs
        from ragas.llms import LangchainLLMWrapper
        wrapped_llm = LangchainLLMWrapper(evaluator_llm)
        
        # Log start of evaluation
        logger.info(f"Starting RAGAs evaluation on {len(evaluation_data)} answerable questions")
        
        try:
            # Execute evaluation following RAGAs documentation pattern
            from ragas import evaluate as ragas_evaluate
            evaluation_result = ragas_evaluate(
                dataset=evaluation_dataset,
                metrics=metrics_to_run,
                llm=wrapped_llm
            )
            
            # --- ADDED: Log the raw result object ---
            logger.info(f"Raw RAGAs evaluation result object: {evaluation_result}")
            logger.info(f"Type of RAGAs result: {type(evaluation_result)}")
            # --- END ADDED ---
            
            # Extract scores correctly from the EvaluationResult object
            scores_dict = {}
            
            # Based on the logs, RAGAs returns a EvaluationResult object that looks like a dict
            try:
                # First check if it's a dictionary-like object that supports direct item access
                try:
                    # Try item access directly - this works for both dict and EvaluationResult
                    faithfulness_score = evaluation_result['faithfulness']
                    answer_relevancy_score = evaluation_result['answer_relevancy']
                    
                    logger.info(f"Extracted raw scores - faithfulness: {faithfulness_score}, answer_relevancy: {answer_relevancy_score}")
                    
                    # Handle array/Series values properly
                    if faithfulness_score is not None:
                        # Try to convert to float directly
                        try:
                            scores_dict['faithfulness'] = float(faithfulness_score)
                            logger.info(f"Converted faithfulness score: {scores_dict['faithfulness']}")
                        except (ValueError, TypeError):
                            # If it's a complex type like Series, this will handle it
                            if hasattr(faithfulness_score, 'mean'):
                                scores_dict['faithfulness'] = float(faithfulness_score.mean())
                                logger.info(f"Used mean() for faithfulness: {scores_dict['faithfulness']}")
                    else:
                        scores_dict['faithfulness'] = 0.0
                        logger.warning("Faithfulness score was None")
                            
                    if answer_relevancy_score is not None:
                        # Try to convert to float directly
                        try:
                            scores_dict['answer_relevancy'] = float(answer_relevancy_score)
                            logger.info(f"Converted answer_relevancy score: {scores_dict['answer_relevancy']}")
                        except (ValueError, TypeError):
                            # If it's a complex type like Series, this will handle it
                            if hasattr(answer_relevancy_score, 'mean'):
                                scores_dict['answer_relevancy'] = float(answer_relevancy_score.mean())
                                logger.info(f"Used mean() for answer_relevancy: {scores_dict['answer_relevancy']}")
                    else:
                        scores_dict['answer_relevancy'] = 0.0
                        logger.warning("Answer relevancy score was None")
                        
                    logger.info(f"Successfully extracted scores from EvaluationResult: {scores_dict}")
                    
                except (KeyError, TypeError) as access_err:
                    # If direct access fails, log detailed info but continue to backup method
                    logger.warning(f"Failed to access scores directly: {access_err}")
                
                # If the above failed, try string parsing as a reliable fallback
                if not scores_dict or all(v == 0.0 for v in scores_dict.values()):
                    result_str = str(evaluation_result)
                    logger.info(f"Using string parsing fallback on: {result_str}")
                    
                    # Parse the string representation - this has been working reliably
                    import re
                    faith_matches = re.findall(r"'faithfulness':\s*([\d\.]+)", result_str)
                    rel_matches = re.findall(r"'answer_relevancy':\s*([\d\.]+)", result_str)
                    
                    if faith_matches:
                        scores_dict['faithfulness'] = float(faith_matches[0])
                    if rel_matches:
                        scores_dict['answer_relevancy'] = float(rel_matches[0])
                    
                    if scores_dict and not all(v == 0.0 for v in scores_dict.values()):
                        logger.info(f"Successfully extracted scores via string parsing: {scores_dict}")
                
                # If we still don't have values, use zeros
                if not scores_dict or all(v == 0.0 for v in scores_dict.values()):
                    logger.warning("All extraction methods failed, using default values")
                    scores_dict = {'faithfulness': 0.0, 'answer_relevancy': 0.0}
                
            except Exception as e_extract:
                logger.error(f"Error during score extraction: {e_extract}", exc_info=True)
                scores_dict = {'faithfulness': 0.0, 'answer_relevancy': 0.0}
            
            # Log results
            logger.info("RAGAs evaluation completed (score extraction attempted)")
            for metric, value in scores_dict.items():
                logger.info(f"{metric}: {value:.4f}")
            
            return scores_dict
            
        except Exception as e_eval:
            logger.error(f"Error during RAGAs evaluate() call: {str(e_eval)}", exc_info=True)
            return {'faithfulness': 0.0, 'answer_relevancy': 0.0}
        
    except Exception as e_prep:
        logger.error(f"Error preparing data for RAGAs evaluation: {str(e_prep)}", exc_info=True)
        return {'faithfulness': 0.0, 'answer_relevancy': 0.0} 