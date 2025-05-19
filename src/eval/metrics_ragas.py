"""
RAGAs metrics evaluation module.

This module provides functions to evaluate RAG system performance
using the RAGAs framework for metrics like faithfulness and answer relevancy.
"""

import logging
import os
from typing import Dict, List, Any
import numpy as np

from langchain_openai import ChatOpenAI

from src.config import settings

# Set up logger
logger = logging.getLogger(__name__)

def get_evaluator_llm() -> Any:
    """
    Initialize and return an OpenAI LLM for RAGAS evaluation.
    
    Returns:
        An initialized ChatOpenAI model for evaluation
    """
    try:
        # Ensure OpenAI API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            return None
            
        # Initialize ChatOpenAI with appropriate settings
        from langchain_openai import ChatOpenAI
        
        evaluator_llm = ChatOpenAI(
            model=settings.evaluator_gpt_model,
            temperature=0,
            request_timeout=120,
        )
        
        return evaluator_llm
        
    except Exception as e:
        logger.error(f"Failed to initialize evaluator LLM: {e}")
        return None


def calculate_ragas_metrics(
    evaluation_data: Dict[str, Dict[str, Any]],
    evaluator_llm: Any = None,
) -> Dict[str, float]:
    """
    Calculate RAGAS metrics (faithfulness, answer relevancy) for the generated responses.
    
    Args:
        evaluation_data: Dictionary with question IDs as keys, each containing:
            - 'question': The question text
            - 'ground_truths': List of expected answers
            - 'answer': The LLM-generated response
            - 'contexts': List of retrieved context passages
        evaluator_llm: The LLM to use for evaluation (OpenAI)
        
    Returns:
        Dictionary with calculated RAGAS metrics
    """
    # Default metrics
    metrics = {
        'faithfulness': 0.0,
        'answer_relevancy': 0.0,
    }
    
    if not evaluation_data:
        logger.warning("No evaluation data provided to calculate_ragas_metrics")
        return metrics
        
    try:
        # Import RAGAS dependencies here to avoid loading them if not needed
        try:
            from ragas import EvaluationDataset
            from ragas.metrics import faithfulness, answer_relevancy
            # Note: We're removing the critique import since it might not be available
            # and it's not used directly in the evaluation
            import pandas as pd
            logger.info("Successfully imported RAGAS dependencies")
        except ImportError as e:
            logger.error(f"Error importing RAGAS dependencies: {e}")
            return metrics
            
        # Convert the dict to a list of dicts for pandas
        data_list = []
        for qid, data in evaluation_data.items():
            record = {
                # Map to the column names that RAGAS expects
                'user_input': data['question'],  # Use user_input instead of question
                'response': data['answer'],      # Use response instead of answer
                'retrieved_contexts': data['contexts'],  # Use retrieved_contexts instead of contexts
            }
            if 'ground_truths' in data:
                record['ground_truth'] = data['ground_truths'][0] if data['ground_truths'] else ""
                
            data_list.append(record)
            
        # Log the data we're using for RAGAS
        logger.info(f"Prepared {len(data_list)} records for RAGAS evaluation")
        if len(data_list) > 0:
            logger.info(f"Sample record keys: {data_list[0].keys()}")
            
        # Ensure we have data to process
        if not data_list:
            logger.warning("No valid records found after processing evaluation data")
            return metrics
            
        # Create a pandas DataFrame
        df = pd.DataFrame(data_list)
        logger.info(f"Created DataFrame with columns: {df.columns.tolist()}")
        
        # Create the RAGAS EvaluationDataset
        try:
            evaluation_dataset = EvaluationDataset.from_pandas(df)
            logger.info("Successfully created RAGAS EvaluationDataset")
        except Exception as e:
            logger.error(f"Error creating RAGAS EvaluationDataset: {e}")
            return metrics
        
        # Define the metrics to use
        metrics_list = [
            faithfulness, 
            answer_relevancy
        ]
        
        # Run the evaluation
        try:
            from ragas import evaluate as ragas_evaluate
            logger.info("Starting RAGAS evaluation...")
            evaluation_result = ragas_evaluate(
                dataset=evaluation_dataset,
                metrics=metrics_list,
                llm=evaluator_llm
            )
            
            logger.info(f"RAGAS evaluation completed. Result type: {type(evaluation_result)}")
            
            # Extract scores from the scores attribute
            if hasattr(evaluation_result, 'scores'):
                scores_values = evaluation_result.scores
                logger.info(f"Scores attribute found with value: {scores_values}")
                
                # Process scores based on their type
                scores_dict = {}
                
                # If it's a list of dictionaries (common format)
                if isinstance(scores_values, list) and len(scores_values) > 0:
                    # Calculate the average for each metric across all records
                    metric_sums = {}
                    metric_counts = {}
                    
                    # Process each record's scores
                    for record_scores in scores_values:
                        if isinstance(record_scores, dict):
                            for metric, value in record_scores.items():
                                if metric not in metric_sums:
                                    metric_sums[metric] = 0.0
                                    metric_counts[metric] = 0
                                
                                try:
                                    # Skip NaN values when calculating the average
                                    if isinstance(value, float) and np.isnan(value):
                                        continue
                                    
                                    metric_sums[metric] += float(value)
                                    metric_counts[metric] += 1
                                except (ValueError, TypeError):
                                    logger.warning(f"Could not convert metric value to float: {value}")
                    
                    # Calculate averages
                    for metric, total in metric_sums.items():
                        if metric_counts[metric] > 0:
                            scores_dict[metric] = total / metric_counts[metric]
                        else:
                            scores_dict[metric] = 0.0
                    
                    logger.info(f"Calculated average metrics: {scores_dict}")
                    return scores_dict
                
                # If it's a DataFrame with a mean method
                elif hasattr(scores_values, 'mean'):
                    # Get mean scores across all questions
                    scores_dict = scores_values.mean().to_dict()
                    logger.info(f"Successfully extracted scores from DataFrame: {scores_dict}")
                    return scores_dict
                
                # If it's a dictionary already
                elif isinstance(scores_values, dict):
                    logger.info(f"Using scores dictionary directly: {scores_values}")
                    return scores_values
                
                else:
                    logger.warning(f"Unhandled scores format: {type(scores_values)}")
            
            # Legacy approach - try to extract individual metrics if scores not found
            scores_dict = {}
            
            # Extract faithfulness
            if hasattr(evaluation_result, 'faithfulness'):
                faithfulness_score = evaluation_result.faithfulness
                logger.info(f"Faithfulness score found: {faithfulness_score}, type: {type(faithfulness_score)}")
                try:
                    if isinstance(faithfulness_score, list):
                        scores_dict['faithfulness'] = sum(faithfulness_score) / len(faithfulness_score)
                    else:
                        scores_dict['faithfulness'] = float(faithfulness_score)
                except Exception as e:
                    logger.error(f"Error processing faithfulness score: {e}")
                    scores_dict['faithfulness'] = 0.0
            else:
                logger.warning("No faithfulness attribute found in evaluation result")
            
            # Extract answer_relevancy
            if hasattr(evaluation_result, 'answer_relevancy'):
                answer_relevancy_score = evaluation_result.answer_relevancy
                logger.info(f"Answer relevancy score found: {answer_relevancy_score}, type: {type(answer_relevancy_score)}")
                try:
                    if isinstance(answer_relevancy_score, list):
                        scores_dict['answer_relevancy'] = sum(answer_relevancy_score) / len(answer_relevancy_score)
                    else:
                        scores_dict['answer_relevancy'] = float(answer_relevancy_score)
                except Exception as e:
                    logger.error(f"Error processing answer_relevancy score: {e}")
                    scores_dict['answer_relevancy'] = 0.0
            else:
                logger.warning("No answer_relevancy attribute found in evaluation result")
            
            return scores_dict
            
        except Exception as e_eval:
            logger.error(f"Error during RAGAs evaluate() call: {str(e_eval)}", exc_info=True)
            return metrics
            
    except Exception as e:
        logger.error(f"Error in calculate_ragas_metrics: {str(e)}", exc_info=True)
        return metrics 